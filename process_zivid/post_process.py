#!/usr/bin/env python3
# Author: hiro
# Date: 2024-06-24 22:05:41
# LastEditTime: 2024-07-08 23:26:09
# Description:

import argparse
import json
import os
from glob import glob

import cv2
import imageio.v2 as iio
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from utils.mesh_utils import post_process_mesh
from utils.pose_utils import get_cam_center, normalize_poses


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="detect markers and corners of charuco board, estimate pose of charuco" "board", add_help=False
    )
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")

    parser.add_argument("-d", "--data", help="Dataset root folder", default="", action="store", dest="data_folder")

    parser.add_argument("-s", "--stage", required=True, help="Things to do", action="store", dest="stage")

    parser.add_argument("-m", "--mask", help="mask folder", action="store", dest="mask_folder")

    parser.add_argument(
        "-c", help="Input file with calibrated camera parameters", default="", action="store", dest="cam_param"
    )

    args = parser.parse_args()

    return args


def tsdf_fusion(cameras, rgbs, depths, voxel_size=0.001, sdf_trunc=0.005, depth_trunc=2):
    print("Running tsdf volume integration ...")
    print(f"voxel_size: {voxel_size}")
    print(f"sdf_trunc: {sdf_trunc}")
    print(f"depth_truc: {depth_trunc}")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i, cam_o3d in tqdm(enumerate(cameras), desc="TSDF integration progress"):
        rgb = rgbs[i]
        depth = depths[i]

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb),
            o3d.geometry.Image(depth),
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0,
        )

        volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

    mesh = volume.extract_triangle_mesh()
    return mesh


def split_dataset(meta, train_ratio=0.6, val_ratio=0.2):
    meta_train, meta_val, meta_test = {}, {}, {}
    for key, value in meta.items():
        if key.startswith("frames"):
            frames = meta["frames"]
            num_frames = len(frames)
            num_train = int(num_frames * train_ratio)
            num_val = int(num_frames * val_ratio)
            idx = np.random.permutation(num_frames)
            meta_train["frames"] = [frames[i] for i in idx[:num_train]]
            meta_val["frames"] = [frames[i] for i in idx[num_train : num_train + num_val]]
            meta_test["frames"] = [frames[i] for i in idx[num_train + num_val :]]
        else:
            meta_train[key] = value
            meta_val[key] = value
            meta_test[key] = value
    return meta_train, meta_val, meta_test


def load_ply(path):
    plydata = PlyData.read(path)
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    return xyz


def save_ply(path, xyzs, text=False):
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]

    points = np.empty(xyzs.shape[0], dtype=dtype)
    points[:] = list(map(tuple, xyzs))
    vertex_element = PlyElement.describe(points, "vertex")

    PlyData([vertex_element], text=text).write(path)


def get_pcds_all(arg):
    pcd_dir = os.path.join(arg.data_folder, "pc")
    camera_filename = "cameras.npz"
    cameras = np.load(os.path.join(arg.data_folder, camera_filename))

    xyzs_world = []
    for cam in tqdm(cameras):
        # if "JTN004" not in cam:
        #     continue
        cam_id, img_id = cam.split("_")[-2:]

        ply_file = os.path.join(pcd_dir, f"{cam_id}_{img_id}.ply")
        xyz = load_ply(ply_file)
        # xyz = xyz / 1000.0

        # mask = np.logical_and(xyz[..., 2] > 0.200, xyz[..., 2] < 0.800)  # roughly filter the points
        # xyz = xyz[mask]

        c2w = np.linalg.inv(cameras[cam])

        xyz_world = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1) @ c2w.T
        xyz_world /= xyz_world[:, 3:4]
        xyzs_world.append(xyz_world)
        # save_ply(os.path.join(data_dir, cam_id + ".ply"), xyz_world[..., :3])
    xyzs_world = np.concatenate(xyzs_world, axis=0)

    mask = np.all(xyzs_world > 0, axis=1)  # all xyzs should be larger than 0 to be above the calibration board

    xyzs_world = xyzs_world[mask]

    save_path = os.path.join(arg.data_folder, "point_cloud_all.ply")
    save_ply(save_path, xyzs_world[::10, :3], text=False)

    return xyzs_world


def reformat_pose(args):
    camera_filename = "cameras.npz"
    pcd_filename = "point_cloud_processed.ply"
    intrinsic_filename = "intrinsic.txt"

    intrinsic_matrix = np.loadtxt(os.path.join(args.data_folder, intrinsic_filename))

    # shift and rescale the poses
    pcd_roi = load_ply(os.path.join(args.data_folder, pcd_filename))

    centroid = pcd_roi.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(pcd_roi - centroid, axis=1).mean()
    # scale = 1.0  # translation only

    normalization_matrix = np.eye(4).astype(np.float32)

    # normalization_matrix[0, 0] = scale
    # normalization_matrix[1, 1] = scale
    # normalization_matrix[2, 2] = scale

    normalization_matrix[:3, 3] = centroid

    cameras = np.load(os.path.join(args.data_folder, camera_filename))

    # normalize the cameras by the center of the pointcloud
    new_cameras = {}
    for cam in cameras:
        if cam.startswith("world_mat"):
            w2c = cameras[cam]
            new_w2c = w2c @ normalization_matrix
            new_cameras[cam] = new_w2c

    camera_dict = {}
    for new_cam in new_cameras:
        cam_id, img_id = new_cam.split("_")[-2:]

        w2c = new_cameras[new_cam]

        if cam_id not in camera_dict:
            camera_dict[cam_id] = {img_id: w2c}
        else:
            camera_dict[cam_id].update({img_id: w2c})

    reformat_cameras = {}
    reformat_cameras["scale"] = scale
    reformat_cameras["intrinsic"] = intrinsic_matrix
    reformat_cameras["focal_x"] = intrinsic_matrix[0, 0]
    reformat_cameras["focal_y"] = intrinsic_matrix[1, 1]
    print("object scale factor: {}".format(scale))

    out_dict = {"focal_x": intrinsic_matrix[0, 0], "focal_y": intrinsic_matrix[1, 1], "scale": float(scale)}
    frames = []

    for cam_id in camera_dict.keys():
        ref_w2c = camera_dict[cam_id]["000"]
        ref_c2w = np.linalg.inv(ref_w2c)

        w2cs_ = np.stack(list(camera_dict[cam_id].values()), axis=0)

        obj_pose = normalize_poses(w2cs_, ref_w2c)

        for i, img_id in enumerate(camera_dict[cam_id].keys()):
            if img_id == "final":
                continue
            frame = {}
            reformat_cameras[f"world_mat_{cam_id}_{img_id}"] = ref_w2c
            reformat_cameras[f"object_pose_{cam_id}_{img_id}"] = obj_pose[i]
            reformat_cameras[f"original_world_mat_{cam_id}_{img_id}"] = camera_dict[cam_id][img_id]

            frame["file_path"] = f"images/{cam_id}_{img_id}"
            ref_c2w_ = ref_c2w.copy()
            ref_c2w_[:3, 1:3] *= -1  # opencv to blender
            frame["transform_matrix"] = ref_c2w_.tolist()
            frame["object"] = obj_pose[i].tolist()

            frames.append(frame)

    out_dict["frames"] = frames

    reformat_camera_filename = "cameras_reformat.npz"
    np.savez(os.path.join(args.data_folder, reformat_camera_filename), **reformat_cameras)

    with open(os.path.join(args.data_folder, "transforms.json"), "w") as f:
        json.dump(out_dict, f, indent=4)

    meta_train, meta_val, meta_test = split_dataset(out_dict, train_ratio=0.8, val_ratio=0.0)

    with open(os.path.join(args.data_folder, "transforms_train.json"), "w") as f:
        json.dump(meta_train, f, indent=4)

    with open(os.path.join(args.data_folder, "transforms_test.json"), "w") as f:
        json.dump(meta_test, f, indent=4)

    # also save to transformation file

    # pose_files = {}
    # for i in range(1, 5):
    #     cam_id = f"JTN00{i}"
    #     pose_dict[cam_id] = []print(os.path.join(args.data_folder, case_name + ".mp4"))
    #         pose = np.loadtxt(pose_file)
    #         pose_dict[cam_id].append(pose)
    #     pose_dict[cam_id] = np.stack(pose_dict[cam_id], axis=0)

    # # use JTN004 for normalization
    # center = get_cam_center(pose_dict["JTN004"])

    # for k in pose_files.keys():
    #     poses = pose_dict[k]
    #     cam_poses, obj_poses = normalize_poses(poses, center)
    #     for i, pose_file in enumerate(pose_files[k]):
    #         pose_id = os.path.basename(pose_file)[:-4]
    #         pose_fn = os.path.join(save_folder, pose_id + ".npz")
    #         np.savez(pose_fn, cam=cam_poses[i], obj=obj_poses[i])


def resize_images(args):
    case_name = os.path.basename(args.data_folder)
    final_img_folder = os.path.join(args.data_folder, "processed_images")
    dst_img_folder = os.path.join(args.data_folder, "images")
    if os.path.exists(dst_img_folder):
        # key = input("Do you want to remove the folder? (y/n)")
        key = "y"
        if key.lower() == "y":
            import shutil

            shutil.rmtree(dst_img_folder)
        else:
            raise RuntimeError(f"Destination folder {dst_img_folder} exists, please rename it.")
    os.makedirs(dst_img_folder)

    imgs = [im for im in os.listdir(final_img_folder) if "final" not in im]
    imgs = sorted(imgs)

    # write a video for track anything
    tmp_img = cv2.imread(os.path.join(final_img_folder, imgs[0]), -1)
    dst_size = (tmp_img.shape[1] // 2, tmp_img.shape[0] // 2)
    # video_writer = cv2.VideoWriter(
    #     os.path.join(data_dir, case_name + ".mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, dst_size
    # )
    video_writer = iio.get_writer(
        os.path.join(args.data_folder, case_name + ".mp4"), format="FFMPEG", fps=30.0, codec="libx264", quality=10
    )

    for iname in tqdm(imgs):
        img_path = os.path.join(final_img_folder, iname)
        img = cv2.imread(img_path, -1)
        resized_img = cv2.resize(img, dst_size, interpolation=cv2.INTER_AREA)
        video_writer.append_data(resized_img[..., ::-1])
        cv2.imwrite(os.path.join(dst_img_folder, iname), resized_img)
    video_writer.close()

    # also resize the intrinsic matrix
    intrinsic = np.loadtxt(os.path.join(args.data_folder, "original_intrinsic.txt"))
    intrinsic[0, 0] *= 0.5
    intrinsic[1, 1] *= 0.5
    intrinsic[0, 2] = dst_size[0] / 2
    intrinsic[1, 2] = dst_size[1] / 2
    np.savetxt(os.path.join(args.data_folder, "intrinsic.txt"), intrinsic)


def convert_mask(args):
    case_name = os.path.basename(args.data_folder)
    img_folder = os.path.join(args.data_folder, "images")
    dst_mask_folder = os.path.join(args.data_folder, "masks")
    os.makedirs(dst_mask_folder, exist_ok=True)
    for i in range(1, 16, 2):
        os.makedirs(dst_mask_folder + f"_{i:d}", exist_ok=True)

    mask_operation_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (mask_operation_size, mask_operation_size))

    img_names = os.listdir(img_folder)
    img_names.sort()
    masks = os.listdir(args.mask_folder)
    masks.sort()

    for idx, mask_img in tqdm(enumerate(masks)):
        img_name = img_names[idx]
        mask_img_name = img_name

        mask_img = np.load(os.path.join(args.mask_folder, mask_img))
        # mask_img = mask_img * 255
        mask_img = (mask_img > 0.5).astype(np.uint8) * 255

        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, element, iterations=3)

        for i in range(1, 16, 2):
            mask_img_ = cv2.morphologyEx(mask_img, cv2.MORPH_DILATE, element, iterations=i)
            # mask_img_ = cv2.dilate(mask_img, element, iterations=i)
            cv2.imwrite(os.path.join(dst_mask_folder + f"_{i:d}", mask_img_name), mask_img_)

        img = cv2.imread(os.path.join(img_folder, img_name), -1)[..., :3]
        img = np.concatenate([img, np.clip(mask_img[..., None], 0, 255)], axis=2)

        cv2.imwrite(os.path.join(dst_mask_folder, mask_img_name), mask_img)
        cv2.imwrite(os.path.join(img_folder, img_name), img)


def depth_fusion(args):
    depth_img_dir = os.path.join(args.data_folder, "depth")
    mask_img_dir = os.path.join(args.data_folder, "masks")
    rgb_img_dir = os.path.join(args.data_folder, "processed_images")

    images = os.listdir(rgb_img_dir)
    images.sort()

    img_h, img_w = cv2.imread(os.path.join(rgb_img_dir, images[0]), -1).shape[:2]

    intrinsic_matrix = np.loadtxt(os.path.join(args.data_folder, "original_intrinsic.txt"))
    cameras_params = np.load(os.path.join(args.data_folder, "cameras_reformat.npz"))

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=img_w, height=img_h, intrinsic_matrix=intrinsic_matrix)

    rgbs, depths, cameras = [], [], []

    for cam in tqdm(cameras_params):
        if not cam.startswith("original_world_mat"):
            continue
        iname = cam.split("_", 3)[-1]
        img = f"{iname}.png"
        rgb = cv2.imread(os.path.join(rgb_img_dir, img), -1)
        depth = cv2.imread(os.path.join(depth_img_dir, img), -1)
        mask = cv2.imread(os.path.join(mask_img_dir, img), -1)
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        mask = ((mask / 255.0) > 0.5).astype(np.float32)

        depth = depth * mask / 10000.0  # convert to meters, depth scale factor is 10000

        extrinsic = cameras_params[cam]

        camera = o3d.camera.PinholeCameraParameters()
        camera.intrinsic = intrinsic
        camera.extrinsic = extrinsic

        rgbs.append(np.ascontiguousarray(rgb[..., ::-1]))
        depths.append(depth)
        cameras.append(camera)

    mesh = tsdf_fusion(cameras, rgbs, depths, voxel_size=0.0003, sdf_trunc=0.0015, depth_trunc=1.0)
    mesh_post = post_process_mesh(mesh, cluster_to_keep=1)

    eval_pcd = mesh_post.sample_points_poisson_disk(number_of_points=100_000)

    o3d.io.write_point_cloud(os.path.join(args.data_folder, "eval_points.ply"), eval_pcd)
    o3d.io.write_triangle_mesh(os.path.join(args.data_folder, "mesh.ply"), mesh)
    o3d.io.write_triangle_mesh(os.path.join(args.data_folder, "eval_mesh.ply"), mesh_post)


if __name__ == "__main__":
    args = parse_arguments()

    if args.stage == "pcd":
        all_pcds = get_pcds_all(args)

    elif args.stage == "resize":
        resize_images(args)

    elif args.stage == "reformat":
        reformat_pose(args)

    elif args.stage == "mask":
        convert_mask(args)

    elif args.stage == "fusion":
        depth_fusion(args)
    else:
        raise ValueError("Unknown stage")
