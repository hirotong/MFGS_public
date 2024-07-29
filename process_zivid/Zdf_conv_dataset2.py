# 0. Run 'Zdf_conv_dataset.py' to convert '.zdf' files for a dataset path with single folders
# Store output point clouds and registered rgb and depth images of the entire dataset in another dataset_conv folder
# Zdf_conv_dataset.py [dataset_file_path]
# ex: Zdf_capture_fn.py /home/mut027/Zivid/data9_noisemodel
# output: Store rgb - /home/mut027/Zivid/data9_noisemodel_conv/rgb
#         Store depth - /home/mut027/Zivid/data9_noisemodel_conv/depth
#         Store dept_raw - /home/mut027/Zivid/data9_noisemodel_conv/depth_raw
#         Store pointclouds - /home/mut027/Zivid/data9_noisemodel_conv/pc

# pip uninstall --yes zivid  (For zivid version error run commands in the environment)
# pip install zivid

import glob
import os

import cv2
import numpy as np
import scipy
import zivid
from PIL import Image


def _point_cloud_to_cv_z(point_cloud: zivid.PointCloud) -> np.ndarray:
    """Get depth map from frame.

    Args:
        point_cloud: Zivid point cloud

    Returns:
        depth_map_color_map: Depth map (HxWx1 ndarray)

    """
    depth_map = point_cloud.copy_data("z")
    depth_map_uint8 = ((depth_map - np.nanmin(depth_map)) / (np.nanmax(depth_map) - np.nanmin(depth_map)) * 255).astype(
        np.uint8
    )

    depth_map_color_map = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_VIRIDIS)

    # Setting nans to black
    depth_map_color_map[np.isnan(depth_map)[:, :]] = 0

    return depth_map_color_map


def _point_cloud_to_cv_bgr(point_cloud: zivid.PointCloud) -> np.ndarray:
    """Get bgr image from frame.

    Args:
        point_cloud: Zivid point cloud

    Returns:
        bgr: BGR image (HxWx3 ndarray)

    """
    rgba = point_cloud.copy_data("rgba")
    # Applying color map
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    return bgr


def _convert_2_ply(frame: zivid.Frame, file_name: str) -> None:
    """Convert from frame to ply.

    Args:
        frame: A frame captured by a Zivid camera
        file_name: File name without extension

    """
    print(f"Saving the frame to {file_name}.ply")
    frame.save(f"{file_name}.ply")


def _flatten_point_cloud(point_cloud: zivid.PointCloud, lower, upper) -> np.ndarray:
    """Convert from point cloud to flattened point cloud (with numpy).
    Args:
        point_cloud: A handle to point cloud in the GPU memory
    Returns:
        A 2D numpy array, with 8 columns and npixels rows
    """
    # Convert to numpy 3D array
    point_cloud = np.dstack([point_cloud.copy_data("xyz"), point_cloud.copy_data("rgba"), point_cloud.copy_data("snr")])
    # Flattening the point cloud
    # flattened_point_cloud = point_cloud.reshape(-1, 8)
    nd = point_cloud[:, :, 2]
    mask = np.ones(nd.shape)

    mask[np.isnan(nd)] = 0
    nd[np.isnan(nd)] = 0

    mask[nd > upper] = 0
    nd[nd > upper] = 0

    mask[nd < lower] = 0
    nd[nd < lower] = 0

    # Removing nans
    # return flattened_point_cloud[~np.isnan(flattened_point_cloud[:, 0]), :]
    return nd, mask


data_dir = "/home/mut027/Zivid/data27_Mar1_noise_model/Dino"
lower = 300
upper = 800

zdffiles = sorted(glob.glob(os.path.join(data_dir, "*.zdf")))
dest_folder_pc = os.path.join(data_dir + "_conv/", "pc/")
dest_folder_rgb = os.path.join(data_dir + "_conv/", "rgb/")
dest_folder_depth = os.path.join(data_dir + "_conv/", "depth/")
dest_folder_depth_raw = os.path.join(data_dir + "_conv/", "depth_raw/")
dest_folder_normals_raw = os.path.join(data_dir + "_conv/", "normals_raw/")
dest_folder_depth_16bit_masked = os.path.join(data_dir + "_conv/", "depth16_masked/")
dest_folder_rgb_masked = os.path.join(data_dir + "_conv/", "rgb_masked/")
dest_folder_rgb_masked_white = os.path.join(data_dir + "_conv/", "rgb_masked_white/")
dest_folder_fg_mask = os.path.join(data_dir + "_conv/", "fg_mask/")
if not os.path.exists(dest_folder_pc):
    os.makedirs(dest_folder_pc)
    os.makedirs(dest_folder_rgb)
    os.makedirs(dest_folder_depth)
    os.makedirs(dest_folder_depth_raw)
if not os.path.exists(dest_folder_normals_raw):
    os.makedirs(dest_folder_normals_raw)
if not os.path.exists(dest_folder_fg_mask):
    os.makedirs(dest_folder_fg_mask)
    os.makedirs(dest_folder_rgb_masked)
    os.makedirs(dest_folder_depth_16bit_masked)
if not os.path.exists(dest_folder_rgb_masked_white):
    os.makedirs(dest_folder_rgb_masked_white)

for zdffile in zdffiles:
    with zivid.Application():
        print(f"Reading point cloud from file: {zdffile}")
        frame = zivid.Frame(zdffile)
        point_cloud = frame.point_cloud()
        _convert_2_ply(frame, os.path.join(dest_folder_pc, os.path.basename(zdffile).rsplit(".", 1)[0]))

        print("Converting to BGR image in OpenCV format")
        bgr = _point_cloud_to_cv_bgr(point_cloud)
        cv2.imwrite(os.path.join(dest_folder_rgb, os.path.basename(zdffile).rsplit(".", 1)[0] + ".png"), bgr)

        print("Converting to Depth map in OpenCV format")
        z_color_map = _point_cloud_to_cv_z(point_cloud)
        cv2.imwrite(os.path.join(dest_folder_depth, os.path.basename(zdffile).rsplit(".", 1)[0] + ".png"), z_color_map)

        nd, mask = _flatten_point_cloud(point_cloud, lower, upper)
        np.save(os.path.join(dest_folder_depth_raw, os.path.basename(zdffile).rsplit(".", 1)[0] + ".npy"), nd)

        mask = scipy.ndimage.morphology.binary_fill_holes(mask)
        mask = scipy.ndimage.binary_dilation(mask, structure=np.ones((12, 12)))
        mask_img = Image.fromarray(mask.astype(dtype=bool))

        mask_img.save(os.path.join(dest_folder_fg_mask, os.path.basename(zdffile).rsplit(".", 1)[0] + ".bmp"))

        nd16 = (nd * 10).astype(np.uint16)
        nd16_image = Image.fromarray(nd16, mode="I;16")
        nd16_image.save(
            os.path.join(dest_folder_depth_16bit_masked, os.path.basename(zdffile).rsplit(".", 1)[0] + ".png")
        )

        mask3 = np.expand_dims(mask, axis=2).astype(np.uint8)
        mask3d = np.append(mask3, mask3, axis=2)
        mask3d = np.append(mask3d, mask3, axis=2)

        bgr_mask = np.multiply(mask3d, bgr)
        cv2.imwrite(
            os.path.join(dest_folder_rgb_masked, os.path.basename(zdffile).rsplit(".", 1)[0] + ".png"), bgr_mask
        )

        bgr_mask = bgr_mask + 255
        cv2.imwrite(
            os.path.join(dest_folder_rgb_masked_white, os.path.basename(zdffile).rsplit(".", 1)[0] + ".png"), bgr_mask
        )

        normals = point_cloud.copy_data("normals")
        np.save(os.path.join(dest_folder_normals_raw, os.path.basename(zdffile).rsplit(".", 1)[0] + ".npy"), normals)

        z = 1
