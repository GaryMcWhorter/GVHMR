import torch
from torch.cuda.amp import autocast
from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)

import hmr4d.utils.matrix as matrix
from hmr4d.utils.ik.ccd_ik import CCD_IK
from hmr4d.utils.geo_transform import get_sequence_cammat, transform_mat, apply_T_on_points
from hmr4d.utils.net_utils import gaussian_smooth
from hmr4d.model.gvhmr.utils.endecoder import EnDecoder
from hmr4d.model.gvhmr.utils.static_cam_solve import solve_static_cam_root, chordal_mean
from hmr4d.model.gvhmr.utils.foot_lock import contact_lock_targets, two_bone_ik
from hmr4d.model.gvhmr.utils.rot_filter import smooth_axis_angle

from hmr4d.utils.wis3d_utils import make_wis3d, add_motion_as_lines


@autocast(enabled=False)
def pp_static_joint(outputs, endecoder: EnDecoder):
    # Global FK
    pred_w_j3d = endecoder.fk_v2(**outputs["pred_smpl_params_global"])
    L = pred_w_j3d.shape[1]
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    pred_j3d_static = pred_w_j3d.clone()[:, :, joint_ids]  # (B, L, J, 3)

    ######## update overall movement with static info, and make displacement ~[0,0,0]
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)

    static_conf_logits = outputs["static_conf_logits"][:, :-1].clone()
    static_label_ = static_conf_logits > 0  # (B, L-1, J) # avoid non-contact frame
    static_conf_logits = static_conf_logits.float() - (~static_label_ * 1e6)  # fp16 cannot go through softmax
    is_static = static_label_.sum(dim=-1) > 0  # (B, L-1)

    pred_disp = pred_j_disp * static_conf_logits[..., None].softmax(dim=-2)  # (B, L-1, J, 3)
    pred_disp = pred_disp * is_static[..., None, None]  # (B, L-1, J, 3)
    pred_disp = pred_disp.sum(-2)  # (B, L-1, 3)
    ####################

    # Overwrite results:
    if False:  # for-loop
        post_w_transl = outputs["pred_smpl_params_global"]["transl"].clone()  # (B, L, 3)
        for i in range(1, L):
            post_w_transl[:, i:] -= pred_disp[:, i - 1 : i]
    else:  # vectorized
        pred_w_transl = outputs["pred_smpl_params_global"]["transl"].clone()  # (B, L, 3)
        pred_w_disp = pred_w_transl[:, 1:] - pred_w_transl[:, :-1]  # (B, L-1, 3)
        pred_w_disp_new = pred_w_disp - pred_disp
        post_w_transl = torch.cumsum(torch.cat([pred_w_transl[:, :1], pred_w_disp_new], dim=1), dim=1)
        post_w_transl[..., 0] = gaussian_smooth(post_w_transl[..., 0], dim=-1)
        post_w_transl[..., 2] = gaussian_smooth(post_w_transl[..., 2], dim=-1)

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    post_w_j3d = pred_w_j3d - pred_w_transl.unsqueeze(-2) + post_w_transl.unsqueeze(-2)
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,)  Minimum y value
    post_w_transl[..., 1] -= ground_y

    return post_w_transl


@autocast(enabled=False)
def pp_static_joint_cam(outputs, endecoder: EnDecoder):
    """Use static joint and static camera assumption to postprocess the global transl"""
    # input
    pred_smpl_params_incam = outputs["pred_smpl_params_incam"].copy()
    pred_smpl_params_global = outputs["pred_smpl_params_global"]
    static_conf_logits = outputs["static_conf_logits"].clone()[:, :-1]  # (B, L-1, J)
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    B, L = pred_smpl_params_incam["transl"].shape[:2]
    assert B == 1

    # FK
    pred_w_j3d = endecoder.fk_v2(**pred_smpl_params_global)  # (B, L, J, 3)
    # smooth incam results, as this could be noisy
    pred_smpl_params_incam["transl"] = gaussian_smooth(pred_smpl_params_incam["transl"], sigma=5, dim=-2)
    pred_c_j3d = endecoder.fk_v2(**pred_smpl_params_incam)  # (B, L, J, 3)

    # compute T_c2w (static) from first frame
    R_gv = axis_angle_to_matrix(pred_smpl_params_global["global_orient"][:, 0])  # (B, 3, 3)
    R_c = axis_angle_to_matrix(pred_smpl_params_incam["global_orient"][:, 0])  # (B, 3, 3)
    R_c2w = R_gv @ R_c.mT  # (B, 3, 3)
    t_c2w = pred_w_j3d[:, 0, 0] - torch.einsum("bij,bj->bi", R_c2w, pred_c_j3d[:, 0, 0])  # (B, 3)
    T_c2w = transform_mat(R_c2w, t_c2w)  # (B, 4, 4)
    pred_c_j3d_in_w = apply_T_on_points(pred_c_j3d, T_c2w[:, None])

    # 1. Make transl similar to incam
    post_w_transl = pred_smpl_params_global["transl"].clone()  # (B, L, 3)
    post_w_j3d = pred_w_j3d.clone()  # (B, L, J, 3)
    cp_thr = torch.tensor([0.25, 0.25, 0.25]).to(post_w_j3d)  # Only update very bad pred
    for i in range(1, L):
        cp_diff = post_w_j3d[:, i, 0] - pred_c_j3d_in_w[:, i, 0]  # (B, 3)
        cp_diff = cp_diff * ~((cp_diff > -cp_thr) * (cp_diff < cp_thr))
        cp_diff = torch.clamp(cp_diff, -0.02, 0.02)
        post_w_transl[:, i:] -= cp_diff
        post_w_j3d[:, i:] -= (cp_diff)[:, None, None]

    # 1. Make stationary joint stay stationary
    # pred_j3d_static = pred_w_j3d.clone()[:, :, joint_ids]  # (B, L, J, 3)
    pred_j3d_static = post_w_j3d[:, :, joint_ids]  # (B, L, J, 3)
    pred_j_disp = pred_j3d_static[:, 1:] - pred_j3d_static[:, :-1]  # (B, L-1, J, 3)

    static_label = static_conf_logits.sigmoid() > 0.8  # (B, L-1, J)
    static_label_sumJ = static_label.sum(-1, keepdim=True)  # (B, L-1, 1)
    static_label_sumJ = torch.clamp_min(static_label_sumJ, 1)  # replace 0 with 1
    pred_disp_sumJ = (pred_j_disp * static_label[..., None]).sum(-2)  # (B, L-1, 3)
    pred_disp = pred_disp_sumJ / static_label_sumJ  # (B, L-1, 3)
    pred_disp[:, :, 1] = 0  # do not modify y

    # Overwrite results (for-loop)
    for i in range(1, L):
        post_w_transl[:, i:] -= pred_disp[:, [i - 1]]
        post_w_j3d[:, i:] -= pred_disp[:, [i - 1], None]

    # Put the sequence on the ground by -min(y), this does not consider foot height, for o3d vis
    ground_y = post_w_j3d[..., 1].flatten(-2).min(dim=-1)[0]  # (B,)  Minimum y value
    post_w_transl[..., 1] -= ground_y

    return post_w_transl


@autocast(enabled=False)
def pp_static_cam_lsq(outputs, endecoder: EnDecoder, **solver_kwargs):
    """Static camera: solve the global root trajectory from the in-camera translation, the world
    velocity rollout, contact and ground constraints at once (see static_cam_solve.py).
    Returns transl (B, L, 3) in the global (y-up) frame, ground at y = 0.
    """
    incam = outputs["pred_smpl_params_incam"]
    glob = outputs["pred_smpl_params_global"]
    B, L = glob["transl"].shape[:2]
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    foot_idx = [1, 3]  # positions of L_foot / R_foot in joint_ids: these touch the ground when planted

    # world joints with zero translation: root joint sits at the skeleton's rest pelvis position
    w_j3d0 = endecoder.fk_v2(glob["body_pose"], glob["betas"], glob["global_orient"])  # (B, L, 22, 3)
    offsets = w_j3d0 - w_j3d0[:, :, :1]
    pelvis_c = endecoder.fk_v2(**incam)[:, :, 0]  # (B, L, 3)
    vel = glob["transl"][:, 1:] - glob["transl"][:, :-1]
    vel = torch.cat([vel, vel[:, -1:]], dim=1)
    R_w = axis_angle_to_matrix(glob["global_orient"])
    R_c = axis_angle_to_matrix(incam["global_orient"])
    R_c2w_per_frame = R_w @ R_c.mT  # (B, L, 3, 3); constant for a static camera, so average it
    prob = outputs["static_conf_logits"].sigmoid()

    out = []
    for b in range(B):
        R_c2w = chordal_mean(R_c2w_per_frame[b].double().cpu().numpy())
        p, _ = solve_static_cam_root(
            pelvis_c[b].double().cpu().numpy(),
            offsets[b].double().cpu().numpy(),
            vel[b].double().cpu().numpy(),
            prob[b].double().cpu().numpy(),
            joint_ids,
            R_c2w,
            foot_joints=foot_idx,
            **solver_kwargs,
        )
        out.append(torch.from_numpy(p).to(glob["transl"]) - w_j3d0[b, :, 0])  # transl = pelvis - rest root
    return torch.stack(out)


@autocast(enabled=False)
def process_ik(outputs, endecoder):
    static_conf = outputs["static_conf_logits"].sigmoid()  # (B, L, J)
    post_w_j3d, local_mat, post_w_mat = endecoder.fk_v2(**outputs["pred_smpl_params_global"], get_intermediate=True)

    # sebas rollout merge
    joint_ids = [7, 10, 8, 11, 20, 21]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]
    post_target_j3d = post_w_j3d.clone()
    for i in range(1, post_w_j3d.size(1)):
        prev = post_target_j3d[:, i - 1, joint_ids]
        this = post_w_j3d[:, i, joint_ids]
        c_prev = static_conf[:, i - 1, :, None]
        post_target_j3d[:, i, joint_ids] = prev * c_prev + this * (1 - c_prev)

    # ik
    global_rot = matrix.get_rotation(post_w_mat)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    left_leg_chain = [0, 1, 4, 7, 10]
    right_leg_chain = [0, 2, 5, 8, 11]
    left_hand_chain = [9, 13, 16, 18, 20]
    right_hand_chain = [9, 14, 17, 19, 21]

    def ik(local_mat, target_pos, target_rot, target_ind, chain):
        local_mat = local_mat.clone()
        IK_solver = CCD_IK(
            local_mat,
            parents,
            target_ind,
            target_pos,
            target_rot,
            kinematic_chain=chain,
            max_iter=2,
        )

        chain_local_mat = IK_solver.solve()
        chain_rotmat = matrix.get_rotation(chain_local_mat)
        local_mat[:, :, chain[1:], :-1, :-1] = chain_rotmat[:, :, 1:]  # (B, L, J, 3, 3)
        return local_mat

    local_mat = ik(local_mat, post_target_j3d[:, :, [7, 10]], global_rot[:, :, [7, 10]], [3, 4], left_leg_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [8, 11]], global_rot[:, :, [8, 11]], [3, 4], right_leg_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [20]], global_rot[:, :, [20]], [4], left_hand_chain)
    local_mat = ik(local_mat, post_target_j3d[:, :, [21]], global_rot[:, :, [21]], [4], right_hand_chain)

    body_pose = matrix_to_axis_angle(matrix.get_rotation(local_mat[:, :, 1:]))  # (B, L, J-1, 3, 3)
    body_pose = body_pose.flatten(2)  # (B, L, (J-1)*3)

    return body_pose


@autocast(enabled=False)
def process_footlock(outputs, endecoder, pin_to_floor=True, **lock_kwargs):
    """Lock planted feet in place with analytic leg IK (replaces process_ik's leaky target blend).

    Each contact interval of each ankle / toe joint gets one fixed world position; toes are
    pinned to the floor (y = 0) when pin_to_floor, which matches pp_static_cam_lsq's ground.
    Returns body_pose (B, L, 63).
    """
    glob_p = outputs["pred_smpl_params_global"]
    prob = outputs["static_conf_logits"].sigmoid()  # (B, L, 6) for joints below
    joint_ids = [7, 10, 8, 11]  # L_ankle, L_foot, R_ankle, R_foot (wrists are left alone)
    w_j3d, local_mat, _ = endecoder.fk_v2(**glob_p, get_intermediate=True)
    parents = endecoder.parents
    B, L = w_j3d.shape[:2]
    tgt = torch.zeros(B, L, 4, 3).to(w_j3d)
    wt = torch.zeros(B, L, 4).to(w_j3d)
    for b in range(B):
        pos = w_j3d[b, :, joint_ids].double().cpu().numpy()
        pb = prob[b, :, :4].double().cpu().numpy()
        t_np, w_np = contact_lock_targets(pos, pb, ground=(), **lock_kwargs)
        if pin_to_floor:
            # drop each planted toe onto the floor and move its ankle's lock by the same amount,
            # so the foot keeps its length and the toe can actually reach the floor
            t_floor, _ = contact_lock_targets(pos, pb, ground=(1, 3), **lock_kwargs)
            for ia, it in ((0, 1), (2, 3)):
                dy = t_floor[:, it, 1] - t_np[:, it, 1]
                t_np[:, it, 1] = t_floor[:, it, 1]
                t_np[:, ia, 1] += dy * (w_np[:, ia] > 0)
        tgt[b] = torch.from_numpy(t_np).to(tgt)
        wt[b] = torch.from_numpy(w_np).to(wt)
    legs = (((1, 4, 7, 10), 0, 1), ((2, 5, 8, 11), 2, 3))
    for (hip, knee, ankle, toe), ia, it in legs:  # 1. put the ankles on their locks
        local_mat = two_bone_ik(local_mat, parents, hip, knee, ankle, toe, tgt[:, :, ia], wt[:, :, ia])
    glob = matrix.forward_kinematics(local_mat, parents)
    for (hip, knee, ankle, toe), ia, it in legs:  # 2. aim each toe from where its ankle now is
        p_a, p_t = glob[..., ankle, :3, 3], glob[..., toe, :3, 3]
        toe_tgt = tgt[:, :, it].clone()
        if pin_to_floor:
            # a toe on the floor at bone length from the ankle, in the locked toe's direction
            bone = (p_t - p_a).norm(dim=-1)
            h = torch.sqrt((bone**2 - p_a[..., 1] ** 2).clamp(min=1e-8))
            d = toe_tgt - p_a
            d[..., 1] = 0
            d = torch.nn.functional.normalize(d, dim=-1)
            on_floor = p_a + d * h[..., None]
            on_floor[..., 1] = 0.0
            reachable = (p_a[..., 1] < bone)[..., None]
            toe_tgt = torch.where(reachable, on_floor, toe_tgt)
        local_mat = two_bone_ik(local_mat, parents, hip, knee, ankle, toe,
                                p_a, torch.zeros_like(wt[:, :, ia]), toe_tgt, wt[:, :, it])
    body_pose = matrix_to_axis_angle(matrix.get_rotation(local_mat[:, :, 1:]))
    return body_pose.flatten(2)


@autocast(enabled=False)
def smooth_pose_rotations(outputs, **kw):
    """Speed-adaptive, zero-lag smoothing of joint and root rotations (in place).
    Body pose is shared by the in-camera and global results; each root orientation is smoothed
    in its own frame."""
    B, L = outputs["pred_smpl_params_global"]["body_pose"].shape[:2]
    bp = outputs["pred_smpl_params_global"]["body_pose"].reshape(B, L, 21, 3)
    bp = smooth_axis_angle(bp, **kw).reshape(B, L, 63)
    for key in ("pred_smpl_params_global", "pred_smpl_params_incam"):
        outputs[key]["body_pose"] = bp
        outputs[key]["global_orient"] = smooth_axis_angle(outputs[key]["global_orient"], **kw)
    return outputs
