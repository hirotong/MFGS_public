#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add the project root to sys.path to import colmap_loader
sys.path.append(os.getcwd())

from scene.colmap_loader import (
    read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text,
    qvec2rotmat
)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COLMAP sparse model to MFGS Zivid/RotReal format")
    parser.add_argument("-s", "--source_path", required=True, help="Path to COLMAP dataset root (contains 'sparse' and 'images')")
    parser.add_argument("-o", "--output_path", help="Path to save converted data (defaults to source_path)")
    return parser.parse_args()

def main():
    args = parse_args()
    source_path = args.source_path
    output_path = args.output_path if args.output_path else source_path
    
    os.makedirs(output_path, exist_ok=True)
    
    # 1. Find sparse model
    sparse_path = os.path.join(source_path, "sparse/0")
    if not os.path.exists(sparse_path):
        sparse_path = os.path.join(source_path, "sparse")
        
    print(f"Reading COLMAP model from: {sparse_path}")
    
    try:
        if os.path.exists(os.path.join(sparse_path, "images.bin")):
            cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_path, "images.bin"))
            cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_path, "cameras.bin"))
        else:
            cam_extrinsics = read_extrinsics_text(os.path.join(sparse_path, "images.txt"))
            cam_intrinsics = read_intrinsics_text(os.path.join(sparse_path, "cameras.txt"))
    except Exception as e:
        print(f"Error reading COLMAP model: {e}")
        return

    # 2. Export intrinsic.txt (use the first camera)
    intr = list(cam_intrinsics.values())[0]
    intrinsic_matrix = np.eye(3)
    if intr.model in ["PINHOLE", "SIMPLE_PINHOLE"]:
        intrinsic_matrix[0, 0] = intr.params[0]
        intrinsic_matrix[1, 1] = intr.params[1] if intr.model == "PINHOLE" else intr.params[0]
        intrinsic_matrix[0, 2] = intr.params[2] if intr.model == "PINHOLE" else intr.params[1]
        intrinsic_matrix[1, 2] = intr.params[3] if intr.model == "PINHOLE" else intr.params[2]
    else:
        print(f"Warning: Camera model {intr.model} is not Pinhole. Intrinsic conversion might be inaccurate.")
        # Fallback for SIMPLE_RADIAL (f, cx, cy, k)
        intrinsic_matrix[0, 0] = intr.params[0]
        intrinsic_matrix[1, 1] = intr.params[0]
        intrinsic_matrix[0, 2] = intr.params[1]
        intrinsic_matrix[1, 2] = intr.params[2]

    np.savetxt(os.path.join(output_path, "intrinsic.txt"), intrinsic_matrix)
    print(f"Saved intrinsic.txt")

    # 3. Export cameras.npz and prepare images symlinks
    cameras_dict = {}
    images_dst_dir = os.path.join(output_path, "images")
    os.makedirs(images_dst_dir, exist_ok=True)
    
    colmap_images_dir = os.path.join(source_path, "images")
    if not os.path.exists(colmap_images_dir):
        # In some COLMAP projects, images are in the root
        colmap_images_dir = source_path

    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        
        # Convert qvec/tvec to W2C matrix
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = T
        
        # Reformat name to match camID_imgID expected by post_process.py
        # If name is "camera_1/0000.jpg", we want cam_id="1" and img_id="0000"
        # so that world_mat_1_0000 results in cam_id="1", img_id="0000"
        path_parts = Path(extr.name).parts
        if len(path_parts) > 1:
            cam_id = path_parts[0].replace("camera_", "")
            img_id = Path(path_parts[-1]).stem
            clean_name = f"{cam_id}_{img_id}"
        else:
            clean_name = Path(extr.name).stem
        
        npz_key = f"world_mat_{clean_name}"
        cameras_dict[npz_key] = w2c
        
        # Create symlink for the image in the flat output images folder
        src_image_path = os.path.join(colmap_images_dir, extr.name)
        dst_image_path = os.path.join(images_dst_dir, f"{clean_name}{Path(extr.name).suffix}")
        if not os.path.exists(dst_image_path):
            try:
                os.symlink(src_image_path, dst_image_path)
            except OSError:
                # Fallback to copy if symlinks are not supported
                import shutil
                shutil.copy(src_image_path, dst_image_path)

    np.savez(os.path.join(output_path, "cameras.npz"), **cameras_dict)
    print(f"Saved cameras.npz with {len(cameras_dict)} cameras and prepared images folder")

    # 4. Copy points3D.ply (or convert bin to ply if needed)
    ply_source = os.path.join(sparse_path, "points3D.ply")
    if not os.path.exists(ply_source):
        # In MFGS, we often have sparse/0/points3D.bin which was already converted to .ply
        # If it doesn't exist, we can't easily convert it without more dependencies in this script,
        # but the main MFGS pipeline usually creates it.
        pass
    
    if os.path.exists(ply_source):
        import shutil
        shutil.copy(ply_source, os.path.join(output_path, "points3D.ply"))
        print(f"Copied points3D.ply to output folder")

    print("\nConversion complete!")
    print("-" * 40)
    print("Next steps to use 'process_zivid/post_process.py':")
    print("1. Open your points3D.ply in CloudCompare.")
    print("2. Crop the object and save as 'point_cloud_processed.ply' in the output folder.")
    print(f"3. Run: python process_zivid/post_process.py -d {output_path} -s reformat")
    print("-" * 40)

if __name__ == "__main__":
    main()
