#!/usr/bin/env python3
# Author: hiro.tong

import argparse
import os
import shutil
import subprocess

import numpy as np

from colmap_utils.read_write_model import read_model, write_model
from colmap_utils.utils import (
    load_camera_from_json,
    load_camera_from_npy,
    load_image_from_aruco,
    load_image_from_json,
    load_image_from_rotblender_json,
)

COLMAP = "/home/ton068/code/colmap/build/src/colmap/exe/colmap"  # path to colmap executable
num_threads = 8

# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense


def run_colmap(basedir, image_dir="images", match_type="exhaustive_matcher"):
    logfile_name = os.path.join(basedir, "colmap_output.txt")
    logfile = open(logfile_name, "w")

    feature_extractor_args = [
        COLMAP,
        "feature_extractor",
        "--database_path",
        os.path.join(basedir, "database.db"),
        "--image_path",
        os.path.join(basedir, image_dir),
        "--ImageReader.camera_model",
        "SIMPLE_PINHOLE",
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_params",
        "1111.11,400,400",
    ]
    feat_output = subprocess.check_output(feature_extractor_args, text=True)
    logfile.write(feat_output)
    print("Features extracted")

    exhaustive_matcher_args = [
        COLMAP,
        match_type,
        "--database_path",
        os.path.join(basedir, "database.db"),
    ]

    match_output = subprocess.check_output(exhaustive_matcher_args, text=True)
    logfile.write(match_output)
    print("Features matched")

    p = os.path.join(basedir, "sparse")
    if not os.path.exists(p):
        os.makedirs(p)

    # $ colmap mapper \
    #     --database_path $DATASET_PATH/database.db \
    #     --image_path $DATASET_PATH/images \
    #     --output_path $DATASET_PATH/sparse

    mapper_args = [
        COLMAP,
        "mapper",
        "--database_path",
        os.path.join(basedir, "database.db"),
        "--image_path",
        os.path.join(basedir, image_dir),
        "--output_path",
        p,
        "--Mapper.init_min_tri_angle",
        "4",
        "--Mapper.multiple_models",
        "1",
        "--Mapper.extract_colors",
        "0",
        "--Mapper.ba_refine_focal_length",
        "0",
    ]

    map_out = subprocess.check_output(mapper_args, text=True)
    logfile.write(map_out)
    logfile.close()
    print("Sparse map created")

    print("Finished running COLMAP, see {} for logs".format(logfile_name))


def run_colmap_with_pose(
    basedir, image_dir, width, height, data_type, pose_file, match_type="exhaustive_matcher", mask=None
):
    if data_type == "blender" or data_type == "rotblender":
        cameras = load_camera_from_json(basedir, pose_file, width=width, height=height)
    else:
        cameras = load_camera_from_npy(basedir, "intrinsic.txt", width, height)
    camera = cameras[1]

    logfile_name = os.path.join(basedir, "colmap_output.txt")
    logfile = open(logfile_name, "w")

    feature_extractor_args = [
        COLMAP,
        "feature_extractor",
        "--database_path",
        os.path.join(basedir, "database.db"),
        "--image_path",
        os.path.join(basedir, image_dir),
        "--ImageReader.camera_model",
        camera.model,
        "--ImageReader.single_camera",
        "1",
        "--ImageReader.camera_params",
        ",".join([str(camera.params[i]) for i in range(len(camera.params))]),
    ]
    if mask is not None:
        feature_extractor_args.extend(["--ImageReader.mask_path", os.path.join(basedir, mask)])
    print(feature_extractor_args)

    feat_output = subprocess.check_output(feature_extractor_args, text=True)
    logfile.write(feat_output)
    print("Features extracted")

    exhaustive_matcher_args = [
        COLMAP,
        match_type,
        "--database_path",
        os.path.join(basedir, "database.db"),
    ]

    match_output = subprocess.check_output(exhaustive_matcher_args, text=True)
    logfile.write(match_output)
    logfile.flush()
    print("Features matched")

    pose_dir = os.path.join(basedir, "known_pose")
    sparse_dir = os.path.join(basedir, "sparse") if mask is None else os.path.join(basedir, f"sparse_{mask}")
    dense_dir = os.path.join(basedir, "dense") if mask is None else os.path.join(basedir, f"dense_{mask}")
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)

    if data_type == "blender":
        images = load_image_from_json(basedir, pose_file)
    elif data_type == "rotblender" or data_type == "rotreal":
        images = load_image_from_rotblender_json(basedir, pose_file)
    else:
        images = load_image_from_aruco(basedir, pose_file)
    write_model(cameras, images, {}, pose_dir, ext=".txt")

    # triangulator
    point_triangulator_args = [
        COLMAP,
        "point_triangulator",
        "--database_path",
        os.path.join(basedir, "database.db"),
        "--image_path",
        os.path.join(basedir, image_dir),
        "--input_path",
        pose_dir,
        "--output_path",
        sparse_dir,
    ]

    tri_output = subprocess.check_output(point_triangulator_args, text=True)
    logfile.write(tri_output)
    print("Points triangulated.")

    # iamge undistort

    # $ colmap image_undistorter \
    # --image_path $DATASET_PATH/images \
    # --input_path $DATASET_PATH/sparse/0 \
    # --output_path $DATASET_PATH/dense \
    # --output_type COLMAP \
    # --max_image_size 2000

    image_undistorter_args = [
        COLMAP,
        "image_undistorter",
        "--image_path",
        os.path.join(basedir, image_dir),
        "--input_path",
        sparse_dir,
        "--output_path",
        dense_dir,
        "--output_type",
        "COLMAP",
        "--max_image_size",
        "2000",
    ]

    undistort_output = subprocess.check_output(image_undistorter_args, text=True)
    logfile.write(undistort_output)
    print("Image undistorted.")

    # $ colmap patch_match_stereo \
    # --workspace_path $DATASET_PATH/dense \
    # --workspace_format COLMAP \
    # --PatchMatchStereo.geom_consistency true

    patch_match_stereo_args = [
        COLMAP,
        "patch_match_stereo",
        "--workspace_path",
        dense_dir,
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true",
        # '--PatchMatchStereo.gpu_index', '0,0,0,0,1,1,1,1'
    ]

    patch_output = subprocess.check_output(patch_match_stereo_args, text=True)
    logfile.write(patch_output)
    print("Patch matched.")

    # $ colmap stereo_fusion \
    # --workspace_path $DATASET_PATH/dense \
    # --workspace_format COLMAP \
    # --input_type geometric \
    # --output_path $DATASET_PATH/dense/fused.ply

    stereo_fusion_args = [
        COLMAP,
        "stereo_fusion",
        "--workspace_path",
        dense_dir,
        "--workspace_format",
        "COLMAP",
        "--input_type",
        "geometric",
        "--output_path",
        os.path.join(dense_dir, "fused.ply"),
        "--StereoFusion.num_threads",
        f"{num_threads}",
    ]

    if mask is not None:
        stereo_fusion_args.extend(["--StereoFusion.mask_path", os.path.join(basedir, mask)])

    stereo_output = subprocess.check_output(stereo_fusion_args, text=True)
    logfile.write(stereo_output)
    print("Stereo fused.")

    # $ colmap poisson_mesher \
    # --input_path $DATASET_PATH/dense/fused.ply \
    # --output_path $DATASET_PATH/dense/meshed-poisson.ply

    poisson_mesher_args = [
        COLMAP,
        "poisson_mesher",
        "--input_path",
        os.path.join(dense_dir, "fused.ply"),
        "--output_path",
        os.path.join(dense_dir, "meshed-poisson-trim0.ply"),
        "--PoissonMeshing.trim",
        "0",
        "--PoissonMeshing.num_threads",
        f"{num_threads}",
    ]

    poisson_output = subprocess.check_output(poisson_mesher_args, text=True)
    logfile.write(poisson_output)
    print("Poisson meshed.")

    # poisson_mesher_trim_args = [
    #     COLMAP,
    #     "poisson_mesher",
    #     "--input_path",
    #     os.path.join(dense_dir, "fused.ply"),
    #     "--output_path",
    #     os.path.join(dense_dir, "meshed-poisson-trim7.ply"),
    #     "--PoissonMeshing.trim",
    #     "7",
    # ]

    # poisson_trim_output = subprocess.check_output(poisson_mesher_trim_args, text=True)
    # logfile.write(poisson_trim_output)
    # print("Poisson trim meshed.")


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    parser = argparse.ArgumentParser(description="Run COLMAP on a dataset")

    parser.add_argument("--basedir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--imagedir", type=str, default="images", help="Path to the images directory")
    parser.add_argument("--width", type=int, default=800, dest="width", help="Image width")
    parser.add_argument("--height", type=int, default=800, dest="height", help="Image height")
    parser.add_argument("--data_type", type=str, default="blender", help="Data type")
    parser.add_argument("--pose_file", type=str, default="transforms.json", help="Pose file")
    parser.add_argument("--match_type", type=str, default="exhaustive_matcher", help="Matching type")
    parser.add_argument("--mask", type=str, default=None, help="Mask file")

    args = parser.parse_args()

    run_colmap_with_pose(
        args.basedir,
        image_dir=args.imagedir,
        width=args.width,
        height=args.height,
        data_type=args.data_type,
        pose_file=args.pose_file,
        match_type=args.match_type,
        mask=args.mask,
    )
