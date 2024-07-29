#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import rembg
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    normal_image: np.array
    alpha_mask: np.array

    obj_R: np.array = None
    obj_T: np.array = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

    train_vcameras: list = None
    test_vcameras: list = None


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                )
            )

    return cam_infos


def readCamerasFromRotRealTransforms(
    path, transformsfile, white_background, extension=".png", mask_level=-1, apply_mask=False
):
    cam_infos = []
    vcam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if "camera_angle_x" not in contents.keys():
            fovx = None
        else:
            fovx = contents["camera_angle_x"]
        scale = contents["scale"]  # the scale of the object
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_file_path = frame["file_path"]
            cam_id, img_id = os.path.basename(cam_file_path).split("_")[-2:]
            # matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            # R = -np.transpose(matrix[:3,:3])
            # R[:,0] = -R[:,0]
            # T = -matrix[:3, 3]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            o2w = np.array(frame["object"])
            vc2w = np.linalg.inv(o2w) @ c2w
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            vc2w[:3, 1:3] *= -1
            # o2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            # w2c = np.linalg.inv(c2w)
            w2c = np.linalg.inv(c2w)
            w2o = np.linalg.inv(o2w)
            w2vc = np.linalg.inv(vc2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            obj_R = np.transpose(w2o[:3, :3])
            obj_T = w2o[:3, 3]

            vR = np.transpose(w2vc[:3, :3])
            vT = w2vc[:3, 3]

            image_path = os.path.join(path, cam_file_path + extension)
            image_name = Path(image_path).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0

            if mask_level == -1:  # no mask
                alpha_mask = np.ones_like(norm_data[:, :, 3]).astype(np.float32)
            elif mask_level == 0:  # mask level 0
                alpha_mask = (norm_data[:, :, 3] > 0.5).astype(np.float32)
            else:
                mask_folder = f"masks_{mask_level}"
                if not os.path.exists(mask_folder):
                    os.makedirs(mask_folder)
                    mask_operation_size = 3
                    element = cv2.getStructuringElement(cv2.MORPH_RECT, (mask_operation_size, mask_operation_size))

                    alpha_mask = (norm_data[:, :, 3] > 0.5).astype(np.float32)
                    alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_DILATE, element, iterations=mask_level)
                    alpha_mask_img = Image.fromarray(np.array(alpha_mask * 255.0, dtype=np.byte), "L")
                    alpha_mask_img.save(os.path.join(path, mask_folder, image_name + ".png"))
                else:
                    alpha_mask = Image.open(os.path.join(path, mask_folder, image_name + ".png"))
                    alpha_mask.convert("L")
                    alpha_mask = ((np.array(alpha_mask) / 255.0) > 0.5).astype(np.float32)

            if apply_mask:
                arr = norm_data[:, :, :3] * alpha_mask[..., None] + bg * (1 - alpha_mask[..., None])
            else:
                arr = norm_data[:, :, :3]

            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            # alpha_mask = (alpha_mask > 0.5).astype(np.float32)
            alpha_mask = Image.fromarray(np.array(alpha_mask * 255.0, dtype=np.byte), "L")

            normal_cam_name = os.path.join(path, frame["file_path"] + "_normal" + extension)
            normal_image_path = os.path.join(path, normal_cam_name)
            if os.path.exists(normal_image_path):
                normal_image = Image.open(normal_image_path)

                normal_im_data = np.array(normal_image.convert("RGBA"))
                normal_bg_mask = (normal_im_data == 128).sum(-1) == 3
                normal_norm_data = normal_im_data / 255.0
                normal_arr = normal_norm_data[:, :, :3] * normal_norm_data[:, :, 3:4] + bg * (
                    1 - normal_norm_data[:, :, 3:4]
                )
                normal_arr[normal_bg_mask] = 0
                normal_image = Image.fromarray(np.array(normal_arr * 255.0, dtype=np.byte), "RGB")
            else:
                normal_image = None

            if fovx == None:
                focal_x = contents["focal_x"]
                focal_y = contents["focal_y"]
                FovY = focal2fov(focal_y, image.size[1])
                FovX = focal2fov(focal_x, image.size[0])
            else:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    normal_image=normal_image,
                    alpha_mask=alpha_mask,
                    obj_R=obj_R,
                    obj_T=obj_T,
                )
            )

            vcam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=vR,
                    T=vT,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    normal_image=normal_image,
                    alpha_mask=alpha_mask,
                )
            )

    return cam_infos, vcam_infos, scale


def readCamerasFromRotReal(path, camera_file, white_background, extension=".png", apply_mask=False):
    cam_infos = []
    vcam_infos = []

    # remove background
    session = rembg.new_session()

    cameras = np.load(os.path.join(path, camera_file))
    scale = cameras["scale"]
    intrinsic_matrix = cameras["intrinsic"]

    focal_x = intrinsic_matrix[0, 0]
    focal_y = intrinsic_matrix[1, 1]

    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    fovx = focal2fov(focal_x, cx * 2)
    fovy = focal2fov(focal_y, cy * 2)

    for idx, cam_name in tqdm(enumerate(cameras)):
        if cam_name.startswith("world_mat"):
            cam_id, img_id = cam_name.split("_")[-2:]

            w2c = cameras[cam_name]
            c2w = np.linalg.inv(w2c)
            o2w = cameras[f"object_pose_{cam_id}_{img_id}"]
            w2o = np.linalg.inv(o2w)
            vc2w = w2o @ c2w
            w2vc = np.linalg.inv(vc2w)

            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            obj_R = np.transpose(w2o[:3, :3])
            obj_T = w2o[:3, 3]

            vR = np.transpose(w2vc[:3, :3])
            vT = w2vc[:3, 3]

            image_path = os.path.join(path, "images", f"{cam_id}_{img_id}{extension}")
            image = Image.open(image_path)

            mask_path = os.path.join(path, "masks", f"{cam_id}_{img_id}.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
            else:
                fg_mask_path = os.path.join(path, "fg_masks", f"{cam_id}_{img_id}.png")
                fg_mask = Image.open(fg_mask_path)
                fg_mask = (np.array(fg_mask.convert("L")) / 255.0).astype(np.float32)
                fg_image = Image.fromarray((np.array(image.convert("RGB")) * fg_mask[..., None]).astype(np.uint8))
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                mask = rembg.remove(fg_image, session=session, only_mask=True)
                mask.save(mask_path)

            mask = mask.convert("L")
            mask = ((np.array(mask) / 255.0) > 0.0).astype(np.float32)

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            im_data = np.array(image.convert("RGB"))
            norm_data = im_data / 255.0

            if apply_mask:
                arr = norm_data * mask[..., None] + bg * (1 - mask[..., None])
            else:
                arr = norm_data

            image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")
            alpha_mask = Image.fromarray(np.array(mask * 255.0, dtype=np.uint8), "L")

            # add normal / depth images
            normal_image = None

            FovX = fovx
            FovY = fovy

            cam_infos.append(
                CameraInfo(
                    uid=f"{cam_id}_{img_id}",
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=f"{cam_id}_{img_id}",
                    width=image.size[0],
                    height=image.size[1],
                    normal_image=normal_image,
                    alpha_mask=alpha_mask,
                    obj_R=obj_R,
                    obj_T=obj_T,
                )
            )

            vcam_infos.append(
                CameraInfo(
                    uid=f"{cam_id}_{img_id}",
                    R=vR,
                    T=vT,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=f"{cam_id}_{img_id}",
                    width=image.size[0],
                    height=image.size[1],
                    normal_image=normal_image,
                    alpha_mask=alpha_mask,
                    obj_R=obj_R,
                    obj_T=obj_T,
                )
            )

    del session

    return cam_infos, vcam_infos, scale


def readCamerasFromRotTransforms(
    path, transformsfile, white_background, extension=".png", mask_level=-1, apply_mask=False
):
    cam_infos = []
    vcam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if "camera_angle_x" not in contents.keys():
            fovx = None
        else:
            fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            # R = -np.transpose(matrix[:3,:3])
            # R[:,0] = -R[:,0]
            # T = -matrix[:3, 3]

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            o2w = np.array(frame["object"])
            vc2w = np.linalg.inv(o2w) @ c2w
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            vc2w[:3, 1:3] *= -1
            # o2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            # w2c = np.linalg.inv(c2w)
            w2c = np.linalg.inv(c2w)
            w2o = np.linalg.inv(o2w)
            w2vc = np.linalg.inv(vc2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            obj_R = np.transpose(w2o[:3, :3])
            obj_T = w2o[:3, 3]

            vR = np.transpose(w2vc[:3, :3])
            vT = w2vc[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0

            if mask_level == -1:  # no mask
                alpha_mask = np.ones_like(norm_data[:, :, 3]).astype(np.float32)
            elif mask_level == 0:  # mask level 0
                alpha_mask = (norm_data[:, :, 3] > 0.5).astype(np.float32)

                mask_folder = "masks"
                # alpha_mask = Image.open(os.path.join(path, mask_folder, image_name + ".png"))
                os.makedirs(os.path.join(path, mask_folder), exist_ok=True)

                if not os.path.exists(os.path.join(path, mask_folder, image_name + ".png")):
                    alpha_mask_img = Image.fromarray(np.array(alpha_mask * 255.0, dtype=np.byte), "L")
                    alpha_mask_img.save(os.path.join(path, mask_folder, image_name + ".png"))
            else:
                mask_folder = f"masks_{mask_level}"
                os.makedirs(os.path.join(path, mask_folder), exist_ok=True)
                if not os.path.exists(os.path.join(path, mask_folder, image_name + ".png")):
                    mask_operation_size = 3
                    element = cv2.getStructuringElement(cv2.MORPH_RECT, (mask_operation_size, mask_operation_size))

                    alpha_mask = (norm_data[:, :, 3] > 0.5).astype(np.float32)
                    alpha_mask = cv2.morphologyEx(alpha_mask, cv2.MORPH_DILATE, element, iterations=mask_level)
                    alpha_mask_img = Image.fromarray(np.array(alpha_mask * 255.0, dtype=np.byte), "L")
                    alpha_mask_img.save(os.path.join(path, mask_folder, image_name + ".png"))
                else:
                    alpha_mask = Image.open(os.path.join(path, mask_folder, image_name + ".png"))
                    alpha_mask = ((np.array(alpha_mask) / 255.0) > 0.5).astype(np.float32)

            if apply_mask:
                arr = norm_data[:, :, :3] * alpha_mask[..., None] + bg * (1 - alpha_mask[..., None])
            else:
                arr = norm_data[:, :, :3]

            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
            # alpha_mask = (alpha_mask > 0.5).astype(np.float32)
            alpha_mask = Image.fromarray(np.array(alpha_mask * 255.0, dtype=np.byte), "L")
            # arr = np.concatenate([arr, norm_data[:, :, 3:4]], axis=-1)
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGBA")

            normal_cam_name = os.path.join(path, frame["file_path"] + "_normal" + extension)
            normal_image_path = os.path.join(path, normal_cam_name)
            if os.path.exists(normal_image_path):
                normal_image = Image.open(normal_image_path)

                normal_im_data = np.array(normal_image.convert("RGBA"))
                normal_bg_mask = (normal_im_data == 128).sum(-1) == 3
                normal_norm_data = normal_im_data / 255.0
                normal_arr = normal_norm_data[:, :, :3] * normal_norm_data[:, :, 3:4] + bg * (
                    1 - normal_norm_data[:, :, 3:4]
                )
                normal_arr[normal_bg_mask] = 0
                normal_image = Image.fromarray(np.array(normal_arr * 255.0, dtype=np.byte), "RGB")
            else:
                normal_image = None

            if fovx == None:
                focal_length = contents["fl_x"]
                FovY = focal2fov(focal_length, image.size[1])
                FovX = focal2fov(focal_length, image.size[0])
            else:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy
                FovX = fovx

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    normal_image=normal_image,
                    alpha_mask=alpha_mask,
                    obj_R=obj_R,
                    obj_T=obj_T,
                )
            )

            vcam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=vR,
                    T=vT,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    normal_image=normal_image,
                    alpha_mask=alpha_mask,
                )
            )

    return cam_infos, vcam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png", linear=False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def readRotNerfSyntheticInfo(
    path, white_background, eval, extension=".png", mask_level=-1, apply_mask=False
) -> SceneInfo:
    print("Reading Training Transforms")
    train_cam_infos, train_vcam_infos = readCamerasFromRotTransforms(
        path, "transforms_train.json", white_background, extension, mask_level, apply_mask
    )
    print("Reading Test Transforms")
    test_cam_infos, test_vcam_infos = readCamerasFromRotTransforms(
        path, "transforms_test.json", white_background, extension, mask_level, apply_mask
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        train_vcam_infos.extend(test_vcam_infos)
        test_cam_infos = []
        test_vcam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    nerf_normalization["scale"] = 1.3

    ply_path = os.path.join(path, "points3d.ply")
    # ply_path = os.path.join(path, "visual_hull_points.ply")

    # if not os.path.exists(ply_path):
    if True:
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        radius = 1.3
        xyz = np.random.random((num_pts, 3)) * 2 * radius - radius
        shs = np.random.random((num_pts, 3)) / 255.0
        # Initialize points outside the bound
        xyz_bg = np.random.random((num_pts // 2, 3)) * 2 - 1
        shs_bg = np.random.random((num_pts // 2, 3)) / 255.0
        xyz_bg = xyz_bg / np.linalg.norm(xyz_bg, axis=-1, keepdims=True) * 4 * radius

        xyz = np.concatenate([xyz, xyz_bg], axis=0)
        shs = np.concatenate([shs, shs_bg], axis=0)

        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        train_vcameras=train_vcam_infos,
        test_vcameras=test_vcam_infos,
    )
    return scene_info


def readRotRealInfo(path, white_background, eval, extension=".png", mask_level=-1, apply_mask=False) -> SceneInfo:
    print("Reading  Transforms")

    train_cam_infos, train_vcam_infos, scale = readCamerasFromRotRealTransforms(
        path, "transforms_train.json", white_background, extension, mask_level, apply_mask
    )
    print("Reading Test Transforms")
    test_cam_infos, test_vcam_infos, scale = readCamerasFromRotRealTransforms(
        path, "transforms_test.json", white_background, extension, mask_level, apply_mask
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        train_vcam_infos.extend(test_vcam_infos)
        test_cam_infos = []
        test_vcam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # ply_path = os.path.join(path, "visual_hull_points.ply")

    # if not os.path.exists(ply_path):
    if True:
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        radius = 4 * scale
        xyz = np.random.random((num_pts, 3)) * 2 * radius - radius
        shs = np.random.random((num_pts, 3)) / 255.0
        # Initialize points outside the bound
        xyz_bg = np.random.random((num_pts // 2, 3)) * 2 - 1
        shs_bg = np.random.random((num_pts // 2, 3)) / 255.0
        xyz_bg = xyz_bg / np.linalg.norm(xyz_bg, axis=-1, keepdims=True) * 4 * radius

        xyz = np.concatenate([xyz, xyz_bg], axis=0)
        shs = np.concatenate([shs, shs_bg], axis=0)

        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    nerf_normalization["scale"] = 4 * scale

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        train_vcameras=train_vcam_infos,
        test_vcameras=test_vcam_infos,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "RotBlender": readRotNerfSyntheticInfo,
    "RotReal": readRotRealInfo,
}
