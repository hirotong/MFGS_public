import numpy as np


def min_line_dist(rays_o, rays_d):
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
    b_i = -A_i @ rays_o
    pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
    return pt_mindist


def get_cam_center(poses):
    rays_o = poses[:, :3, 3:4]
    rays_d = poses[:, :3, 2:3]

    pt_mindist = min_line_dist(rays_o, rays_d)
    return pt_mindist  # set this to be the center of the world


# def normalize_poses(w2cs, ref_w2c):
#     p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

#     # make sure the first pose is the same position 0
#     R0 = poses[0, :3, :3]  # use the same rotation of the first pose
#     C0 = poses[0, :3, 3]

#     cam_pose = np.empty((3, 4))
#     cam_pose[:3, :3] = R0
#     cam_pose[:3, 3] = C0 - center

#     cam_poses = []
#     obj_poses = []
#     for i in range(len(poses)):
#         obj_pose = np.empty((3, 4))

#         R = poses[i, :3, :3]
#         C = poses[i, :3, 3]

#         obj_pose[:3, :3] = R0 @ R.T
#         obj_pose[:3, 3] = R0 @ R.T @ (center - C) + (C0 - center)

#         obj_poses.append(obj_pose)
#         cam_poses.append(cam_pose)

#     return np.stack(cam_poses, axis=0), np.stack(obj_poses, axis=0)


def normalize_poses(w2cs, ref_w2c):
    obj_poses = []
    ref_c2w = np.linalg.inv(ref_w2c)
    for i in range(len(w2cs)):
        w2c = w2cs[i]

        obj_pose = ref_c2w @ w2c
        obj_poses.append(obj_pose)

    return np.stack(obj_poses, axis=0)
