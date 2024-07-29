#!/usr/bin/env python3
# Author: hiro.tong

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import cv2
import matplotlib
import torch
import trimesh

matplotlib.use("Agg")
from itertools import cycle

import matplotlib.pyplot as plt
import open3d as o3d
import pyglet
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.vis.plotly_vis import get_camera_wireframe

pyglet.options["shadow_window"] = False

# from utils.open3d_utils import trimesh_to_open3d
import time

import imageio
import numpy as np
import pyrender
from pyrender import (
    DirectionalLight,
    Mesh,
    MetallicRoughnessMaterial,
    Node,
    OffscreenRenderer,
    PerspectiveCamera,
    PointLight,
    Primitive,
    RenderFlags,
    Scene,
    SpotLight,
    Viewer,
)
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

cycol = cycle("bgrcmk")


def plot_cameras(ax, cameras, color: str = "blue"):
    cam_wires_canonical = get_camera_wireframe()[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    for l in range(len(cam_wires_trans)):
        wire = cam_wires_trans[l][2]
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        ax.text(x_, y_, z_, str(l + 1))
    return plot_handles


def plot_camera_scene(cameras_dict, status: str):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.clear()
    ax.set_title(status)
    labels_handles = {}
    for key, cams in cameras_dict.items():
        handle_cam = plot_cameras(ax, cams, color=next(cycol))
        labels_handles[key] = handle_cam[0]
    plot_radius = 6
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([-plot_radius, 3 + plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    ax.legend(labels_handles.values(), labels_handles.keys(), loc="upper center", bbox_to_anchor=(0.5, 0))
    # plt.show()
    return fig


def pose_to_rot_and_trans(pose):
    if pose.shape[1] != 4:
        homo = torch.tensor([[[0, 0, 0, 1]]], device=pose.device).tile(pose.shape[0], 1, 1)
        pose = torch.cat([pose, homo], dim=1)
    w2c = pose.inverse()
    R = w2c[:, :3, :3]
    T = w2c[:, :3, 3]
    return R, T


def visualize_camera_poses(poses_dict, status: str, skip=None):
    cameras_dict = {}
    for k, pose in poses_dict.items():
        if skip is not None:
            pose = pose[::skip]
        R, T = pose_to_rot_and_trans(pose)
        cameras = PerspectiveCameras(R=R.transpose(1, 2), T=T)
        cameras_dict[k] = cameras
    fig = plot_camera_scene(cameras_dict, status=status)
    plt.savefig(f"{status}.png", format="png")
    plt.close(fig)


def render_trimesh(output_path, mesh, poses, focal_x, focal_y, img_size, scale_factor=1, offscreen=False):
    # if scale_factor != 1:
    #     mesh.apply_scale(scale_factor)
    try:
        mesh.visual = trimesh.visual.ColorVisuals()
        # vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    except:
        # trimesh.visual.ColorVisuals()
        raise RuntimeError("Failed to set color.")

    model_mesh = Mesh.from_trimesh(mesh, smooth=True)
    # model_mesh = pyrender.Mesh.from_trimesh(
    #         mesh,
    #         material=pyrender.MetallicRoughnessMaterial(
    #             alphaMode="BLEND",
    #             baseColorFactor=[1.0, 1.0, 1.0, 1.0],
    #             metallicFactor=0.2,
    #             roughnessFactor=0.8,
    #         ),
    #     )
    # print(model_mesh.vertex_normals)
    model_node = Node(mesh=model_mesh)

    if scale_factor != 1:
        focal_x *= scale_factor
        focal_y *= scale_factor
        img_size = [int(img_size[0] * scale_factor), int(img_size[1] * scale_factor)]

    viewport_size = [int(img_size[0]), int(img_size[1])]
    cam = PerspectiveCamera(yfov=2 * np.arctan2(img_size[1] / 2, focal_y))

    # scene = Scene(ambient_light=np.array([0.5, 0.5, 0.5]))
    # scene.add_node(model_node)
    # # scene.add_node(plane_node)
    # cam_node = scene.add(cam)  # keep the camera node for later manipulation

    # light = DirectionalLight(color=np.ones(3), intensity=6.0)
    # light_node = scene.add(light)
    # light2 = PointLight(color=np.ones(3), intensity=1.0)
    # light_pose2 = np.eye(4)
    # light_pose2[2, 3] = -3
    # light_node2 = scene.add(light2, pose=light_pose2)
    # light_pose3 = np.eye(4)
    # light_pose3[2, 3] = 3
    # light3 = PointLight(color=np.ones(3), intensity=1.0)
    # light_node3 = scene.add(light3, pose=light_pose3)

    # flags = RenderFlags.ALL_SOLID | RenderFlags.SHADOWS_DIRECTIONAL

    scene = Scene(ambient_light=[0.2, 0.2, 0.2])
    scene.add_node(model_node)
    # scene.add_node(plane_node)
    cam_node = scene.add(cam)  # keep the camera node for later manipulation

    light = DirectionalLight(color=np.ones(3), intensity=8.0)

    light_node = scene.add(light)
    light2 = PointLight(color=np.ones(3), intensity=20.0)
    light_pose2 = np.eye(4)
    light_pose2[2, 3] = -3
    light_node2 = scene.add(light2, pose=light_pose2)

    flags = RenderFlags.ALL_SOLID | RenderFlags.SHADOWS_DIRECTIONAL

    if not offscreen:
        # #==============================================================================
        # # Using the viewer with a default camera
        # #==============================================================================
        # v = Viewer(scene, shadows=True, viewport_size=viewport_size)

        # ==============================================================================
        # Using the viewer with a pre-specified camera
        # ==============================================================================
        v = Viewer(scene, central_node=model_node, viewport_size=viewport_size)
    else:
        ren = OffscreenRenderer(viewport_width=viewport_size[0], viewport_height=viewport_size[1])
        K = np.array([[focal_x, 0.0, img_size[0] / 2], [0.0, focal_y, img_size[1] / 2], [0.0, 0.0, 1.0]])

        for i, pose in tqdm(enumerate(poses)):
            file_name = os.path.join(output_path, "render_%04d.jpg" % i)
            light_node.matrix = pose
            cam_node.matrix = pose
            color, depth = ren.render(scene, flags)

            # save images
            # color = cv2.resize(color, None, fx=1 / scale_factor, fy=1 / scale_factor)
            imageio.imsave(file_name, color)


def render_trimesh2(output_path, mesh, poses, focal_x, focal_y, img_size, scale_factor=1, offscreen=False):
    # if scale_factor != 1:
    #     mesh.apply_scale(scale_factor)
    try:
        mesh.visual = trimesh.visual.ColorVisuals()
        # vertex_colors = mesh.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    except:
        # trimesh.visual.ColorVisuals()
        raise RuntimeError("Failed to set color.")

    model_mesh = Mesh.from_trimesh(mesh, smooth=True)
    # print(model_mesh.vertex_normals)
    model_node = Node(mesh=model_mesh)

    # if scale_factor != 1:
    #     focal_x *= scale_factor
    #     focal_y *= scale_factor
    #     img_size = [int(img_size[0] * scale_factor), int(img_size[1] * scale_factor)]

    viewport_size = [int(img_size[0]), int(img_size[1])]
    cam = PerspectiveCamera(yfov=2 * np.arctan2(img_size[1] / 2, focal_y))

    # scene = Scene(ambient_light=np.array([0.5, 0.5, 0.5]))
    # scene.add_node(model_node)
    # # scene.add_node(plane_node)
    # cam_node = scene.add(cam)  # keep the camera node for later manipulation

    # light = DirectionalLight(color=np.ones(3), intensity=6.0)
    # light_node = scene.add(light)
    # light2 = PointLight(color=np.ones(3), intensity=1.0)
    # light_pose2 = np.eye(4)
    # light_pose2[2, 3] = -3
    # light_node2 = scene.add(light2, pose=light_pose2)
    # light_pose3 = np.eye(4)
    # light_pose3[2, 3] = 3
    # light3 = PointLight(color=np.ones(3), intensity=1.0)
    # light_node3 = scene.add(light3, pose=light_pose3)

    # flags = RenderFlags.ALL_SOLID | RenderFlags.SHADOWS_DIRECTIONAL

    scene = Scene(ambient_light=np.array([0.0, 0.0, 0.0, 0.0]))
    scene.add_node(model_node)
    # scene.add_node(plane_node)
    cam_node = scene.add(cam)  # keep the camera node for later manipulation

    light = DirectionalLight(color=np.ones(3), intensity=8.0)
    light_node = scene.add(light)
    light2 = PointLight(color=np.ones(3), intensity=20.0)
    light_pose2 = np.eye(4)
    light_pose2[2, 3] = -3
    light_node2 = scene.add(light2, pose=light_pose2)

    flags = RenderFlags.ALL_SOLID | RenderFlags.SHADOWS_DIRECTIONAL

    if not offscreen:
        # #==============================================================================
        # # Using the viewer with a default camera
        # #==============================================================================
        # v = Viewer(scene, shadows=True, viewport_size=viewport_size)

        # ==============================================================================
        # Using the viewer with a pre-specified camera
        # ==============================================================================
        v = Viewer(scene, central_node=model_node, viewport_size=viewport_size)
    else:
        ren = OffscreenRenderer(
            viewport_width=viewport_size[0], viewport_height=viewport_size[1], point_size=1.0 / scale_factor
        )
        K = np.array([[focal_x, 0.0, img_size[0] / 2], [0.0, focal_y, img_size[1] / 2], [0.0, 0.0, 1.0]])

        for i, pose in tqdm(enumerate(poses)):
            file_name = os.path.join(output_path, "render_%04d.jpg" % i)

            light_node.matrix = pose
            cam_node.matrix = pose
            color, depth = ren.render(scene, flags)

            # save images
            # color = cv2.resize(color, None, fx=1 / scale_factor, fy=1 / scale_factor)
            imageio.imsave(file_name, color)


def render_mesh(output_paths, meshes, poses, focal, img_size, scale_factor=1, offscreen=False, visualizer=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_size[0], height=img_size[1])

    for output_path, mesh in zip(output_paths, meshes):
        if scale_factor != 1:
            mesh.apply_scale(scale_factor)
        if isinstance(mesh, trimesh.Trimesh):
            mesh_o3d = trimesh_to_open3d(mesh)
            # o3d.visualization.draw([mesh_o3d])
        else:
            mesh_o3d = mesh
        mesh_o3d.compute_vertex_normals()

        vis.add_geometry(mesh_o3d)
        vis.get_render_option().load_from_json("./renderoption.json")
        cameraParam = o3d.camera.PinholeCameraParameters()
        cameraParam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            img_size[0], img_size[1], focal, focal, img_size[0] / 2 - 0.5, img_size[1] / 2 - 0.5
        )

        for i, pose in enumerate(poses):
            pose[:, 1] = pose[:, 1] * (-1)
            pose[:, 2] = pose[:, 2] * (-1)
            cameraParam.extrinsic = np.linalg.inv(pose)
            vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Default
            vis.get_view_control().convert_from_pinhole_camera_parameters(cameraParam, False)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(1)
            vis.capture_screen_image(os.path.join(output_path, "color_{:04d}.png".format(i)), False)
            # normal
            vis.get_render_option().mesh_color_option = o3d.visualization.MeshColorOption.Normal
            vis.poll_events()
            vis.update_renderer()
            time.sleep(1)
            vis.capture_screen_image(os.path.join(output_path, "normal_{:04d}.png".format(i)), False)
        vis.remove_geometry(mesh_o3d)
        # vis.destroy_window()
    vis.destroy_window()
    # del vis


if __name__ == "__main__":
    cameras = PerspectiveCameras()
    camera_dict = {"gt": cameras, "est": cameras}
    fig = plot_camera_scene(camera_dict, "test")
    plt.savefig("test.png")
    lines = get_camera_wireframe(scale=1)[None]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca(projection="3d")
    ax.clear()
    for wire in lines:
        x, z, y = wire.detach().cpu().numpy().T.astype(float)
        ax.plot(x, y, z, color="r", linewidth=0.3)
    ax.set_xlim3d([-3, 3])
    ax.set_ylim3d([-5, 5])
    ax.set_zlim3d([-2, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    plt.savefig("cameras.png")
    plt.close()
