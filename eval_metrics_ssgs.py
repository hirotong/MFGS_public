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
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torchvision.transforms.functional as tf
import trimesh
from PIL import Image
from tqdm import tqdm

from arguments import get_config_args
from lpipsPyTorch import lpips
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.mesh_utils import MeshEvaluator, tsdf_integration


def get_mesh_eval_points(scene: Scene, source_dir):
    viewpoints = scene.getTrainVCameras()
    if viewpoints is None:
        viewpoints = scene.getTrainCameras()

    rgbs, depths = [], []
    for cam in viewpoints:
        rgb = cam.original_image
        image_name = cam.image_name
        depth_path = os.path.join(source_dir, "images", f"{image_name}_depth.png")

        depth_img = Image.open(depth_path)
        depth = np.asarray(depth_img) / 1000.0

        depth = torch.from_numpy(depth).unsqueeze(0).float()

        rgbs.append(rgb)
        depths.append(depth)

    mesh = tsdf_integration(viewpoints, rgbs, depths, voxel_size=0.002, sdf_trunc=0.01, depth_trunc=1)

    eval_pcd = mesh.sample_points_poisson_disk(number_of_points=100_000)

    return mesh, eval_pcd


def convert_to_serializable(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    return obj


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    masks = []
    image_names = []
    for fname in os.listdir(renders_dir):
        try:
            render = Image.open(renders_dir / fname)
            gt = Image.open(gt_dir / fname)
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            masks.append(tf.to_tensor(gt).unsqueeze(0)[:, 3:4, :, :].cuda())
            image_names.append(fname)
        except:
            print(renders_dir / fname)
            raise RuntimeError("Unable to read image")
    return renders, gts, masks, image_names


def find_roi(mask):
    mask = mask.squeeze()

    indices = torch.nonzero(mask)

    min_h, min_w = indices.min(dim=0).values.chunk(2)
    max_h, max_w = indices.max(dim=0).values.chunk(2)

    return min_h, min_w, max_h, max_w


def evaluate(model_paths, apply_mask=False):
    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        # try:
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method in os.listdir(test_dir):
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}
            full_dict_polytopeonly[scene_dir][method] = {}
            per_view_dict_polytopeonly[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            renders, gts, masks, image_names = readImages(renders_dir, gt_dir)

            if apply_mask:
                rois = [find_roi(m) for m in masks]
                renders = [r * m for r, m in zip(renders, masks)]
                gts = [g * m for g, m in zip(gts, masks)]

                renders = [
                    r[:, :, min_h : max_h + 1, min_w : max_w + 1]
                    for r, (min_h, min_w, max_h, max_w) in zip(renders, rois)
                ]
                gts = [
                    g[:, :, min_h : max_h + 1, min_w : max_w + 1] for g, (min_h, min_w, max_h, max_w) in zip(gts, rois)
                ]

            ssims = []
            psnrs = []
            lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type="vgg"))

            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("")

            full_dict[scene_dir][method].update({
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
            })
            per_view_dict[scene_dir][method].update({
                "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
            })

        with open(scene_dir + "/results.json", "w") as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", "w") as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)
    # except:
    #     print("Unable to compute metrics for model", scene_dir)


def evaluate_mesh(model_paths, n_points, visualize_pcd=True):
    full_dict = {}
    mesh_evaluator = MeshEvaluator(n_points)

    for scene_dir in model_paths:
        full_dict[scene_dir] = {}
        print("Scene:", scene_dir)

        cfg_args = get_config_args(scene_dir)
        train_dir = Path(scene_dir) / "train"
        source_path = Path(cfg_args.source_path)

        if not (source_path / "eval_points.ply").exists():
            gaussians = GaussianModel(cfg_args.sh_degree)
            scene = Scene(cfg_args, gaussians, shuffle=False)

            mesh_gt, pcd_gt = get_mesh_eval_points(scene, source_path)

            o3d.io.write_triangle_mesh((source_path / "eval_mesh.ply").as_posix(), mesh_gt)
            o3d.io.write_point_cloud((source_path / "eval_points.ply").as_posix(), pcd_gt)

        else:
            pcd_gt = o3d.io.read_point_cloud((source_path / "eval_points.ply").as_posix())

        for method in os.listdir(train_dir):
            print("Method:", method)
            method_dir = train_dir / method

            mesh_pred = trimesh.load(method_dir / "fuse_post.ply", process=False, force="mesh")
            # pcd_pred = o3d.io.read_point_cloud((method_dir / "pred_points.ply").as_posix())

            # pointcloud_pred = np.asarray(pcd_pred.points)
            pointcloud_tgt = np.asarray(pcd_gt.points)

            mesh_eval_dict, pred2gt_pcd, gt2pred_pcd = mesh_evaluator.eval_mesh(
                mesh_pred, pointcloud_tgt, None, visualize_pcd=visualize_pcd
            )
            # mesh_eval_dict, pred2gt_pcd, gt2pred_pcd = mesh_evaluator.eval_pointcloud(
            #     pointcloud_pred, pointcloud_tgt, None, visualize_pcd=visualize_pcd
            # )

            if visualize_pcd:
                save_dir = method_dir / "vis_pcd"

                if not save_dir.exists():
                    save_dir.mkdir()

                o3d.io.write_point_cloud((save_dir / "pred2gt.ply").as_posix(), pred2gt_pcd)
                o3d.io.write_point_cloud((save_dir / "gt2pred.ply").as_posix(), gt2pred_pcd)

            mesh_eval_dict["n_points"] = n_points

            full_dict[scene_dir][method] = mesh_eval_dict

            print("  Mesh evaluation results:")
            print("    Chamfer distance L1: {:>12.7f}".format(mesh_eval_dict["chamfer-L1"]))

        serializable_data = convert_to_serializable(full_dict[scene_dir])

        with open(scene_dir + "/results_mesh.json", "w") as fp:
            json.dump(serializable_data, fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--skip_metric", action="store_true")
    parser.add_argument("--apply_mask", action="store_true")
    parser.add_argument("--n_points", "-n", type=int, default=100_000)
    args = parser.parse_args()

    if not args.skip_metric:
        evaluate(args.model_paths, args.apply_mask)

    if not args.skip_mesh:
        evaluate_mesh(args.model_paths, args.n_points)
