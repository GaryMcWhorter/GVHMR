import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from .utils import focal_length_from_mm
from .matcher_wrapper import Matcher
from .solver_two_view import TwoPairSolver, CameraParams, interpolate_missing_frames
from .rotation_averaging import average_rotations
from tqdm import tqdm

from hmr4d.utils.video_io_utils import get_video_lwh, read_video_np


class SimpleVO:
    def __init__(
        self, video_path, scale=0.5, step=8, method="sift", f_mm=None, num_workers=1, skips=(1, 2, 4), min_inliers=30
    ):
        """
        skips: keyframe offsets to match. (1,) reproduces the original chained VO exactly.
            Anything wider matches extra pairs (i, i+k) and solves robust rotation averaging,
            which bounds drift and outvotes bad RANSAC pairs (needs 3+ offsets to outvote).
        min_inliers: pairs with fewer RANSAC inliers are dropped from averaging.
        """
        self.video_path = video_path
        self.scale = scale
        self.step = step
        self.method = method
        self.f_mm = 24 if f_mm is None else f_mm  # fullframe camera focal length in mm
        self.num_workers = num_workers
        self.skips = tuple(sorted(set(skips)))
        assert self.skips[0] == 1, "skips must include 1 (consecutive keyframes)"
        self.min_inliers = min_inliers
        self.last_info = None  # filled by the averaging path: edge residuals, dropped pairs

    def compute(self):
        # Read video
        frames = read_video_np(self.video_path, scale=self.scale)

        # Downsample frames, and interpolate missing frames
        F_all = frames.shape[0]
        sample_idxs = np.arange(0, F_all, self.step)
        if sample_idxs[-1] != F_all - 1:
            sample_idxs = np.concatenate([sample_idxs, [F_all - 1]])
        frames = frames[sample_idxs]
        F, H, W, C = frames.shape
        print(f"[SimpleVO] Choosen frames shape: {frames.shape}")

        matcher: Matcher = Matcher(self.method)
        camera_params = CameraParams(W, H, focal_length=focal_length_from_mm(W, H, self.f_mm))
        solver: TwoPairSolver = TwoPairSolver(camera_params, solver="pycolmap")

        # TODO:We should use different pipelines for different methods
        if self.skips == (1,):
            T_w2c_list = self.process_video_T_w2c_list_np(frames, matcher, solver)
        else:
            T_w2c_list = self.process_video_T_w2c_list_averaged(frames, matcher, solver)

        # Interpolate missing frames
        T_w2c_list = interpolate_missing_frames(T_w2c_list, sample_idxs)

        return T_w2c_list

    def process_video_T_w2c_list_np(self, frames, matcher: Matcher, solver: TwoPairSolver):
        if self.num_workers <= 1:
            # Serial path: identical to the pre-parallel implementation, including
            # incremental accumulation (so a mid-loop exception leaves a partial
            # T_w2c_list, matching prior behavior).
            T_w2c_list = [np.eye(4)]
            prev_frame = frames[0]
            for frame_idx in tqdm(range(1, len(frames))):
                curr_frame = frames[frame_idx]
                pts0, pts1 = matcher.match_np(prev_frame, curr_frame)
                T_delta = solver.solve(pts0, pts1)
                T_w2c_list.append(T_delta @ T_w2c_list[-1])
                prev_frame = curr_frame
            return T_w2c_list

        # Parallel path: each thread owns its own Matcher/TwoPairSolver to avoid
        # sharing cv2.SIFT / pycolmap internals across threads. Per-thread
        # instances are cloned from the passed-in objects so configuration
        # (matcher type/args, solver backend, camera params) carries over.
        #
        # Note: this is NOT bit-for-bit equivalent to the serial path. pycolmap
        # RANSAC is non-deterministic, and accumulation is deferred until all
        # deltas are collected, so partial-failure behavior also differs.
        n_pairs = len(frames) - 1
        tls = threading.local()

        def get_worker_state():
            if not hasattr(tls, "matcher"):
                tls.matcher = matcher.clone()
                tls.solver = solver.clone()
            return tls.matcher, tls.solver

        def solve_pair(i):
            m, s = get_worker_state()
            pts0, pts1 = m.match_np(frames[i], frames[i + 1])
            return s.solve(pts0, pts1)

        T_deltas = [None] * n_pairs
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for i, T in tqdm(
                zip(range(n_pairs), ex.map(solve_pair, range(n_pairs))),
                total=n_pairs,
            ):
                T_deltas[i] = T

        T_w2c_list = [np.eye(4)]
        for T_delta in T_deltas:
            T_w2c_list.append(T_delta @ T_w2c_list[-1])
        return T_w2c_list

    def process_video_T_w2c_list_averaged(self, frames, matcher: Matcher, solver: TwoPairSolver):
        """Match every (i, i+k) for k in self.skips, average rotations, then chain translations.

        Only rotations are averaged. GVHMR consumes camera rotation alone (cam_angvel); the
        two-view translations are unit-length (scale-free), so they are chained from the
        consecutive pairs as before, but rotated by the averaged orientations.
        """
        n = len(frames)
        pairs = [(i, i + k) for k in self.skips for i in range(n - k)]
        tls = threading.local()

        def get_worker_state():
            if not hasattr(tls, "matcher"):
                tls.matcher = matcher.clone() if self.num_workers > 1 else matcher
                tls.solver = solver.clone() if self.num_workers > 1 else solver
            return tls.matcher, tls.solver

        def solve_pair(ij):
            m, s = get_worker_state()
            pts0, pts1 = m.match_np(frames[ij[0]], frames[ij[1]])
            return s.solve_with_info(pts0, pts1)

        if self.num_workers <= 1:
            results = [solve_pair(ij) for ij in tqdm(pairs)]
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                results = list(tqdm(ex.map(solve_pair, pairs), total=len(pairs)))

        T_pair = {}
        dropped = []
        for ij, (T, n_inl) in zip(pairs, results):
            if T is None or n_inl < self.min_inliers:
                dropped.append((ij, n_inl))
            else:
                T_pair[ij] = T

        R, info = average_rotations(n, {ij: T[:3, :3] for ij, T in T_pair.items()}, return_info=True)
        info["dropped_pairs"] = dropped
        info["pairs"] = T_pair
        self.last_info = info

        # translations: t_{i+1} = R_{i,i+1} t_i + t_delta, with R_{i,i+1} from the averaged result
        T_w2c_list = []
        t = np.zeros(3)
        for i in range(n):
            if i > 0:
                R_rel = R[i] @ R[i - 1].T
                t_delta = T_pair[(i - 1, i)][:3, 3] if (i - 1, i) in T_pair else np.zeros(3)
                t = R_rel @ t + t_delta
            T = np.eye(4)
            T[:3, :3] = R[i]
            T[:3, 3] = t
            T_w2c_list.append(T)
        return T_w2c_list
