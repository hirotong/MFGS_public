#!/usr/bin/env python3
# Author: hiro
# Date: 2022-11-01 14:33:28
# Description:

import json
import os
import os.path as osp

from colmap_utils.database import COLMAPDatabase
from colmap_utils.read_write_model import *


def load_camera_from_json(basedir, filename="transforms.json", width=800, height=800):
    # camera models
    cameras = {}
    camera_id = 1
    model_name = "SIMPLE_PINHOLE"

    json_file = osp.join(basedir, filename)
    with open(json_file, "r") as fp:
        metas = json.load(fp)
    camera_angle_x = float(metas["camera_angle_x"])
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
    params = np.array(tuple(map(float, [focal, width / 2, height / 2])))

    cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=params)

    return cameras


def load_camera_from_npy(basedir, filename="intrinsic.txt", width=800, height=800):
    cameras = {}
    camera_id = 1
    model_name = "PINHOLE"

    if filename.endswith(".npy"):
        intrinsic = np.load(osp.join(basedir, filename))
    elif filename.endswith(".txt"):
        intrinsic = np.loadtxt(osp.join(basedir, filename))
    else:
        raise ValueError("Unknown file format")

    params = np.array((intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]))

    cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=params)

    return cameras


def load_image_from_json(
    basedir,
    filename="transforms.json",
):
    json_file = osp.join(basedir, filename)
    with open(json_file, "r") as fp:
        metas = json.load(fp)

    db = COLMAPDatabase.connect(osp.join(basedir, "database.db"))
    name_dict = {}
    for image_id, name in db.execute("SELECT image_id, name FROM images"):
        name_dict[name] = image_id

    # poses
    images = {}
    frames = metas["frames"]
    frames.sort(key=lambda x: x["file_path"].split(os.sep)[-1])

    for i, frame in enumerate(frames, start=1):
        image_name = frame["file_path"].split(os.sep)[-1]
        if not image_name.endswith(".png"):
            image_name = image_name + ".png"
        c2w_mat = np.array(frame["transform_matrix"])
        # from (right, up, backward) to (right, down, forward)
        c2w_colmap = np.stack((c2w_mat[:, 0], -c2w_mat[:, 1], -c2w_mat[:, 2], c2w_mat[:, 3]), axis=1)
        w2c_colmap = np.linalg.inv(c2w_colmap)
        qvec = rotmat2qvec(w2c_colmap[:3, :3])
        tvec = w2c_colmap[:3, 3]
        image_id = name_dict[image_name]
        images[image_id] = Image(
            id=image_id, qvec=qvec, tvec=tvec, camera_id=1, name=image_name, xys=np.array([]), point3D_ids=np.array([])
        )

    return images


def load_image_from_rotblender_json(
    basedir,
    filename="transforms.json",
):
    json_file = osp.join(basedir, filename)
    with open(json_file, "r") as fp:
        metas = json.load(fp)

    db = COLMAPDatabase.connect(osp.join(basedir, "database.db"))
    name_dict = {}
    for image_id, name in db.execute("SELECT image_id, name FROM images"):
        name_dict[name] = image_id

    # poses
    images = {}
    frames = metas["frames"]
    frames.sort(key=lambda x: x["file_path"].split(os.sep)[-1])

    for i, frame in enumerate(frames, start=1):
        image_name = frame["file_path"].split(os.sep)[-1]
        if not image_name.endswith(".png"):
            image_name = image_name + ".png"
        c2w_mat = np.array(frame["transform_matrix"])
        o2w_mat = np.array(frame["object"])
        c2w_mat = np.linalg.inv(o2w_mat) @ c2w_mat
        # from (right, up, backward) to (right, down, forward)
        c2w_colmap = np.stack((c2w_mat[:, 0], -c2w_mat[:, 1], -c2w_mat[:, 2], c2w_mat[:, 3]), axis=1)
        w2c_colmap = np.linalg.inv(c2w_colmap)
        qvec = rotmat2qvec(w2c_colmap[:3, :3])
        tvec = w2c_colmap[:3, 3]
        image_id = name_dict[image_name]
        images[image_id] = Image(
            id=image_id, qvec=qvec, tvec=tvec, camera_id=1, name=image_name, xys=np.array([]), point3D_ids=np.array([])
        )

    return images


def load_image_from_aruco(basedir, filename="transforms.json"):
    json_file = osp.join(basedir, filename)
    with open(json_file, "r") as fp:
        metas = json.load(fp)

    db = COLMAPDatabase.connect(osp.join(basedir, "database.db"))
    name_dict = {}
    for image_id, name in db.execute("SELECT image_id, name FROM images"):
        name_dict[name] = image_id

    # poses
    images = {}
    frames = metas["frames"]
    frames.sort(key=lambda x: x["file_path"].split(os.sep)[-1])

    for i, frame in enumerate(frames, start=1):
        image_name = frame["file_path"].split(os.sep)[-1]
        if not image_name.endswith(".png"):
            image_name = image_name + ".png"
        c2w_colmap = np.array(frame["transform_matrix"])
        # from (right, up, backward) to (right, down, forward)
        # c2w_colmap = np.stack((c2w_mat[:, 0], -c2w_mat[:, 1], -c2w_mat[:, 2], c2w_mat[:, 3]), axis=1)
        w2c_colmap = np.linalg.inv(c2w_colmap)
        qvec = rotmat2qvec(w2c_colmap[:3, :3])
        tvec = w2c_colmap[:3, 3]
        image_id = name_dict[image_name]
        images[image_id] = Image(
            id=image_id, qvec=qvec, tvec=tvec, camera_id=1, name=image_name, xys=np.array([]), point3D_ids=np.array([])
        )

    return images
