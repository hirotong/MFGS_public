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

import os
from argparse import ArgumentParser
from os import makedirs
from typing import List

import numpy as np
import open3d as o3d
import torch
import torchvision
import trimesh
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.mesh_utils import GaussianExtractor, post_process_mesh, to_cam_open3d
from utils.render_utils import create_videos, generate_path
from utils.visualize_utils import render_trimesh


def open3d_to_trimesh(src):
    if not src.has_vertex_normals:
        src.compute_vertex_normals()
    dst = trimesh.Trimesh()
    dst.vertices = src.vertices
    dst.faces = src.triangles
    dst.vertex_normals = src.vertex_normals
    # dst.visual.vertex_colors = (np.asarray(src.vertex_colors) * 255.0).astype(np.uint8)

    return dst


def to_cam_poses(viewpoint_stack: List[Camera]):
    camera_poses = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([[W / 2, 0, 0, (W - 1) / 2], [0, H / 2, 0, (H - 1) / 2], [0, 0, 0, 1]]).float().cuda().T
        intrins = (viewpoint_cam.projection_matrix @ ndc2pix)[:3, :3].T

        intrins = intrins.cpu().numpy()

        focal_x, focal_y = intrins[0, 0], intrins[1, 1]

        extrinsic = np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        pose = np.linalg.inv(extrinsic)
        pose[:3, 1:3] *= -1
        camera_poses.append(pose)

    return H, W, focal_x, focal_y, camera_poses


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--scale_factor", default=1, type=int, help="Scale factor for the mesh")
    parser.add_argument("--mesh_path", type=str, default="", help="Path to the mesh file")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help="Mesh: voxel size for TSDF")
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help="Mesh: Max depth range for TSDF")
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help="Mesh: truncation value for TSDF")
    parser.add_argument("--num_cluster", default=50, type=int, help="Mesh: number of connected clusters to export")
    parser.add_argument("--unbounded", action="store_true", help="Mesh: using unbounded mode for meshing")
    parser.add_argument("--mesh_res", default=1024, type=int, help="Mesh: resolution for unbounded mesh extraction")
    parser.add_argument("--bbox", type=float, default=-1, help="Bounding box for the scene")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_dir = os.path.join(args.model_path, "train", "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, "test", "ours_{}".format(scene.loaded_iter))

    method_name = os.path.basename(args.model_path)

    mesh_path = os.path.join(train_dir, "fuse_post.ply") if args.mesh_path == "" else args.mesh_path
    pred_mesh = trimesh.load(mesh_path, process=False, force="mesh")
    # pred_mesh = o3d.io.read_triangle_mesh(mesh_path)
    # pred_mesh = open3d_to_trimesh(pred_mesh)

    # render train cameras
    viewpoints = scene.getTrainVCameras() if scene.getTrainVCameras() is not None else scene.getTrainCameras()
    H, W, focal_x, focal_y, viewposes = to_cam_poses(viewpoints)
    output_dir = (
        os.path.join(train_dir, "pred_mesh")
        if args.mesh_path == ""
        else os.path.join(train_dir, os.path.basename(os.path.dirname(os.path.dirname(args.mesh_path))))
    )
    os.makedirs(output_dir, exist_ok=True)
    render_trimesh(output_dir, pred_mesh, viewposes, focal_x, focal_y, (W, H), scale_factor=8, offscreen=True)

    if method_name == "2dgs_m_0":
        try:
            gt_mesh_path = os.path.join(dataset.source_path, "mesh_gt.obj")
            gt_mesh = trimesh.load(gt_mesh_path, process=False, force="mesh")
        except:
            gt_mesh_path = os.path.join(dataset.source_path, "eval_mesh.ply")
            gt_mesh = trimesh.load(gt_mesh_path, process=False, force="mesh")
        # gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
        # gt_mesh = open3d_to_trimesh(gt_mesh)

        gt_output_dir = os.path.join(train_dir, "gt_mesh")
        os.makedirs(gt_output_dir, exist_ok=True)
        render_trimesh(gt_output_dir, gt_mesh, viewposes, focal_x, focal_y, (W, H), scale_factor=8, offscreen=True)
