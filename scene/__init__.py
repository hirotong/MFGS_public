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
import random

from arguments import ModelParams
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration


class Scene:
    gaussians: GaussianModel

    def __init__(
        self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]
    ):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.train_vcameras = {}
        self.test_vcameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)

        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            if os.path.exists(os.path.join(args.source_path, "cameras_reformat.npz")):
                print("Found cameras_reformat.npz file, assuming Zivid data set!")
                scene_info = sceneLoadTypeCallbacks["RotReal"](
                    args.source_path,
                    args.white_background,
                    args.eval,
                    mask_level=args.mask_level if hasattr(args, "mask_level") else 0,
                    apply_mask=args.apply_mask if hasattr(args, "apply_mask") else False,
                )
                if not args.rotation:
                    scene_info = scene_info._replace(
                        train_cameras=scene_info.train_vcameras,
                        test_cameras=scene_info.test_vcameras,
                        train_vcameras=None,
                        test_vcameras=None,
                    )

            elif os.path.exists(os.path.join(args.source_path, "envmap.exr")):
                print("Found envmap.exr file, assuming RotBlender synthetic dataset with rotation!")
                scene_info = sceneLoadTypeCallbacks["RotBlender"](
                    args.source_path,
                    args.white_background,
                    args.eval,
                    mask_level=args.mask_level if hasattr(args, "mask_level") else 0,
                    apply_mask=args.apply_mask if hasattr(args, "apply_mask") else False,
                )
                if not args.rotation:
                    scene_info = scene_info._replace(
                        train_cameras=scene_info.train_vcameras,
                        test_cameras=scene_info.test_vcameras,
                        train_vcameras=None,
                        test_vcameras=None,
                    )
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)

        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            # also shuffle the virtual cameras

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.fg_scale = scene_info.nerf_normalization["scale"] if "scale" in scene_info.nerf_normalization else 1.3

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )
            if args.rotation:
                self.train_vcameras[resolution_scale] = cameraList_from_camInfos(
                    scene_info.train_vcameras, resolution_scale, args
                )
                self.test_vcameras[resolution_scale] = cameraList_from_camInfos(
                    scene_info.test_vcameras, resolution_scale, args
                )
            else:
                self.train_vcameras[resolution_scale] = None
                self.test_vcameras[resolution_scale] = None

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply")
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.fg_scale)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getTrainVCameras(self, scale=1.0):
        return self.train_vcameras[scale]

    def getTestVCameras(self, scale=1.0):
        return self.test_vcameras[scale]
