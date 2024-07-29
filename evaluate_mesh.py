#!/usr/bin/env python3
# Author: hiro
# Date: 2022-07-14 13:02:17
# LastEditors: Please set LastEditors
# Copyright (c) 2022 by hiro jinguang.tong@anu.edu.au, All Rights Reserved.

import argparse
import os

# from utils import open3d_utils
import numpy as np
import open3d as o3d
import pandas as pd
import torch
import trimesh

from utils.mesh_utils import MeshEvaluator

# from utils.visualize import render_trimesh

if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    parser = argparse.ArgumentParser(description="Evaluate Mesh")
    parser.add_argument("--mesh_path", type=str, default=None, help="Path to mesh file")
    parser.add_argument("--eval_pcd_path", type=str, default=None, help="Path to gt pointcloud file")
    parser.add_argument("--n_points", "-n", type=int, default=100_000)
    # parser.add_argument("--sample_ratio", type=float, default=0.2)

    args = parser.parse_args()

    mesh_path = os.path.expanduser(args.mesh_path)
    eval_pcd_path = os.path.expanduser(args.eval_pcd_path)

    n_points = args.n_points
    # sample_ratio = args.sample_ratio

    try:
        mesh_pred = trimesh.load(mesh_path, preprocess=False)
        pcd_gt = o3d.io.read_point_cloud(eval_pcd_path)
        pointcloud_tgt = np.asarray(pcd_gt.points)

    except:
        raise RuntimeError("Load mesh data failed.")

    # evaluate mesh
    mesh_evaluator = MeshEvaluator(n_points=n_points)

    # incase colmap point cloud
    if isinstance(mesh_pred, trimesh.PointCloud):
        if mesh_pred.shape[0] <= n_points:
            pointcloud_eval = np.array(mesh_pred.vertices).astype(np.float32)
            mesh_eval_dict = mesh_evaluator.eval_pointcloud(pointcloud_eval, pointcloud_tgt)
        else:
            vertices = np.asarray(mesh_pred.vertices)
            index = np.random.choice(mesh_pred.shape[0], n_points)
            pointcloud_eval = np.array(vertices[index]).astype(np.float32)
            mesh_eval_dict = mesh_evaluator.eval_pointcloud(pointcloud_eval, pointcloud_tgt)
    else:
        # debug show
        # trimesh_clean.show()

        # render

        mesh_eval_dict = mesh_evaluator.eval_mesh(mesh_pred, pointcloud_tgt, None, None, None)

    mesh_eval_dict["mesh_path"] = mesh_path
    mesh_eval_dict["mesh_gt_path"] = eval_pcd_path
    mesh_eval_dict["num_sample"] = n_points

    out_file_csv = mesh_path.replace(".ply", ".csv")

    # create pandas dataframe and save
    eval_df = pd.DataFrame(mesh_eval_dict, index=[0])
    eval_df.to_csv(out_file_csv, index=False, float_format="%f")

    print(mesh_eval_dict)
