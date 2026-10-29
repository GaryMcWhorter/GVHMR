"""
Compare GVHMR's gravity-view (GV) frame construction against a shortest-arc
("rotor between two vectors") construction, on synthetic camera paths.

GV frame (current, hmr_global.get_R_c2gv):
    y_gv = gravity in camera coords
    x_gv = normalize(y_gv x z_cam)        <- degenerate when view axis || gravity
    z_gv = x_gv x y_gv
Rotor frame (proposed):
    R_c2gv = shortest-arc rotation taking gravity-in-camera to +y.
    In GA terms the rotor is (1 + y g)/|1 + y g|; it is singular only when
    g = -y, i.e. the camera is rolled upside down.

For a person standing still in the world, global_orient_gv changes from frame
to frame by exactly the yaw of the GV frame between frames. That yaw is what
the network must regress from one image crop, so we measure it.
"""
import numpy as np

def rot(axis, ang):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K

def gv_current(R_w2c, g_w=np.array([0, 0, -1.0])):
    y = R_w2c @ g_w
    z_c = np.array([0, 0, 1.0])
    x = np.cross(y, z_c); n = np.linalg.norm(x)
    x = x / (n + 1e-5) if n >= 1e-5 else np.array([1.0, 0, 0])
    z = np.cross(x, y)
    return np.stack([x, y, z], -1).T  # R_c2gv

def gv_rotor(R_w2c, g_w=np.array([0, 0, -1.0])):
    g = R_w2c @ g_w                       # gravity in camera coords
    y = np.array([0, 1.0, 0])
    v = np.cross(g, y); c = g @ y        # rotor (1 + y g) <-> quaternion (1 + c, v)
    q = np.array([1 + c, *v]); q /= np.linalg.norm(q)
    w, a, b, d = q
    return np.array([[1-2*(b*b+d*d), 2*(a*b-w*d), 2*(a*d+w*b)],
                     [2*(a*b+w*d), 1-2*(a*a+d*d), 2*(b*d-w*a)],
                     [2*(a*d-w*b), 2*(b*d+w*a), 1-2*(a*a+b*b)]])

def yaw_of(R):
    """R must map +y to +y; returns its rotation angle about y."""
    return np.arctan2(R[0, 2], R[0, 0])

def camera_path(n, pitch_fn, roll_fn, yaw_fn):
    # world: z up (gravity -z). Level camera looks along +x_w, image-down = -z_w.
    base = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0.0]])   # R_w2c for a level camera
    Rs = []
    for t in range(n):
        R_c2w = rot([0, 0, 1], yaw_fn(t)) @ base.T @ rot([1, 0, 0], -pitch_fn(t)) @ rot([0, 0, 1], roll_fn(t))
        Rs.append(R_c2w.T)
    return np.array(Rs)

def frame_yaw_jitter(Rs, builder):
    """Per-frame yaw of the GV frame, in degrees, for a world-static subject."""
    out = []
    for t in range(len(Rs) - 1):
        A = builder(Rs[t]); B = builder(Rs[t + 1])
        # orientation of a world-static body in GV: R_c2gv @ R_w2c @ R_body_w
        # change between frames = B R_w2c(t+1) (A R_w2c(t))^T, a pure yaw
        M = (B @ Rs[t + 1]) @ (A @ Rs[t]).T
        assert abs(M[1, 1] - 1) < 1e-6, "frame change must be pure yaw"
        out.append(np.degrees(yaw_of(M)))
    return np.array(out)

def cam_step_deg(Rs):
    return np.array([np.degrees(np.arccos(np.clip((np.trace(Rs[t+1] @ Rs[t].T) - 1) / 2, -1, 1)))
                     for t in range(len(Rs) - 1)])

def report(name, Rs):
    cur = frame_yaw_jitter(Rs, gv_current); new = frame_yaw_jitter(Rs, gv_rotor)
    cs = cam_step_deg(Rs)
    print(f"\n{name}")
    print(f"  camera rotation per frame   max {cs.max():7.3f} deg")
    print(f"  GV frame yaw/frame current  max {np.abs(cur).max():7.3f}  mean {np.abs(cur).mean():7.3f} deg")
    print(f"  GV frame yaw/frame rotor    max {np.abs(new).max():7.3f}  mean {np.abs(new).mean():7.3f} deg")
    return cur, new

if __name__ == "__main__":
    np.random.seed(0)
    N = 300
    d = np.radians

    # 1. Ordinary handheld: 10 deg pitch, no roll, slow pan. Constructions should agree.
    Rs = camera_path(N, lambda t: d(10), lambda t: 0.0, lambda t: d(0.2) * t)
    cur, new = report("1. Handheld, pitched 10 deg, no roll, panning", Rs)
    A = np.array([gv_current(R) @ R for R in Rs]); B = np.array([gv_rotor(R) @ R for R in Rs])
    print(f"  max |R_gv2w difference| between constructions: {np.abs(A - B).max():.2e}")

    # 2. Handheld with roll wobble (walking cameraman).
    Rs = camera_path(N, lambda t: d(15 + 3*np.sin(t/7)), lambda t: d(4*np.sin(t/5)), lambda t: d(0.2)*t)
    report("2. Handheld, 15 deg pitch, +/-4 deg roll wobble", Rs)

    # 3. Drone / overhead: pitch sweeping toward nadir with small wobble.
    Rs = camera_path(N, lambda t: d(80 + 9.9*t/N + 0.3*np.sin(t/3)),
                     lambda t: d(1.0*np.sin(t/4)), lambda t: d(0.05)*t)
    report("3. Drone tilting from 80 to ~90 deg down, +/-1 deg roll", Rs)

    # 4. Straight-down camera with tiny jitter (ceiling cam, phone on a mount).
    Rs = camera_path(N, lambda t: d(89.7 + 0.2*np.random.randn()),
                     lambda t: d(0.2*np.random.randn()), lambda t: 0.0)
    report("4. Near-nadir static camera, 0.2 deg sensor jitter", Rs)
