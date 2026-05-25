# Mathematical & Architectural Breakdown of the Zivid/RotReal Pre-processing Workflow

This document provides a rigorous mathematical and structural analysis of the Zivid pre-processing pipeline implemented in the `process_zivid` directory. It explains how raw scans, camera poses, global point clouds, and volumetric reconstructions are computed and synchronized for **Mask-free Gaussian Separation (MFGS)**.

---

## 1. End-to-End Data Flow

```
[Raw .zdf Scans]
       │  (Zdf_conv_dataset2.py)
       ▼
[RGB, Depth, single-view PCDs]
       │  (estimate_pose_charuco.py)
       ▼
[Extrinsics T_w2c & Intrinsics K in Board Frame]
       │  (post_process.py -s resize)
       ▼
[Downscaled Images & Scaled Intrinsics K_scaled]
       │  (post_process.py -s pcd)
       ▼
[Global Fused Scene Point Cloud point_cloud_all.ply]
       │  (Manual Crop in CloudCompare)
       ▼
[Object Point Cloud point_cloud_processed.ply]
       │  (post_process.py -s reformat)
       ▼
[Centered Object Poses transforms.json & cameras_reformat.npz]
       │  (post_process.py -s fusion)
       ▼
[Surface Reconstruction TSDF Mesh eval_mesh.ply]
```

---

## 2. Mathematical Formulations by Stage

### Stage 0: Converting ZDF raw scans (`Zdf_conv_dataset2.py`)
Extracts raw structured-light sensor data into standard formats:
* **RGB Color Image**: $\mathbf{I} \in \mathbb{R}^{W \times H \times 3}$
* **16-bit Depth Image**: $\mathbf{D} \in \mathbb{N}^{W \times H \times 1}$
* **Single-View 3D Point Cloud**: $\mathbf{P}_{\text{cam}} \in \mathbb{R}^{M \times 3}$

---

### Stage 1: Camera Pose Estimation (`estimate_pose_charuco.py`)
Computes camera extrinsics relative to a stationary ChArUco calibration board (which serves as the global coordinate frame).

1. **Intrinsics Matrix $\mathbf{K}$**:
   $$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
   The image is undistorted, and cropped so the optical center $\mathbf{c} = (c_x, c_y)$ is precisely positioned at the middle of the frame: $(W/2, H/2)$.

2. **Extrinsics PnP Solving**:
   OpenCV's `estimatePoseCharucoBoard` computes rotation vector $\mathbf{r}$ and translation vector $\mathbf{t}$ mapping world (board) coordinates to camera coordinates. The Rodrigues formula yields the rotation matrix $\mathbf{R}_{w2c} \in SO(3)$:
   $$\mathbf{R}_{w2c} = \text{Rodrigues}(\mathbf{r})$$
   The complete World-to-Camera ($w2c$) transformation matrix is:
   $$\mathbf{T}_{w2c} = \begin{bmatrix} \mathbf{R}_{w2c} & \mathbf{t}_{w2c} \\ \mathbf{0}^T & 1 \end{bmatrix} \in SE(3)$$

3. **Camera-to-World ($c2w$) Transform**:
   $$\mathbf{T}_{c2w} = \mathbf{T}_{w2c}^{-1} = \begin{bmatrix} \mathbf{R}_{w2c}^T & -\mathbf{R}_{w2c}^T \mathbf{t}_{w2c} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

---

### Stage 2: Image Downsampling (`post_process.py -s resize`)
Downsamples images by a scale factor $s = 0.5$ to reduce VRAM utilization.
* **Intrinsics Adaptation**:
  $$\mathbf{K}_{\text{resized}} = \begin{bmatrix} s \cdot f_x & 0 & s \cdot c_x \\ 0 & s \cdot f_y & s \cdot c_y \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} f_x / 2 & 0 & W / 4 \\ 0 & f_y / 2 & H / 4 \\ 0 & 0 & 1 \end{bmatrix}$$
  *Note: Extrinsics $\mathbf{T}_{w2c}$ remain unchanged as physical scale does not scale with 2D resolution.*

---

### Stage 3: Back-Projection & Global Fusion (`post_process.py -s pcd`)
Project local points from each camera frame into the static ChArUco board global frame to assemble a complete point cloud.

1. **Camera Frame Coordinate Projection**:
   For pixel $\mathbf{p} = [u, v, 1]^T$ in camera $i$ with a measured depth value $d$:
   $$\mathbf{X}_{\text{cam}} = \mathbf{K}^{-1} \cdot (d \cdot \mathbf{p}) = \begin{bmatrix} \frac{(u - c_x) \cdot d}{f_x} \\ \frac{(v - c_y) \cdot d}{f_y} \\ d \end{bmatrix}$$

2. **World space Transformation**:
   Using the Camera-to-World extrinsic matrix $\mathbf{T}_{c2w}^{(i)}$:
   $$\mathbf{X}_{\text{world}} = \mathbf{T}_{c2w}^{(i)} \cdot \begin{bmatrix} \mathbf{X}_{\text{cam}} \\ 1 \end{bmatrix}$$
   Fusing these across all frames generates the global scene point cloud `point_cloud_all.ply`.

---

### Stage 4: Centering & Pose Reformatting (`post_process.py -s reformat`)
* **Preparation**: The user manually crops the background (table, board) from `point_cloud_all.ply`, saving the object-only points as `point_cloud_processed.ply`.

1. **Centering Transform**:
   The centroid $\mathbf{t}_{\text{object}}$ and average spatial scale $\sigma$ of the cropped object points are calculated:
   $$\mathbf{t}_{\text{object}} = \frac{1}{N} \sum_{j=1}^{N} \mathbf{X}_{j}, \quad \sigma = \frac{1}{N} \sum_{j=1}^{N} \|\mathbf{X}_{j} - \mathbf{t}_{\text{object}}\|_2$$
   To align the origin of the world frame to the object center:
   $$\mathbf{T}_{\text{norm}} = \begin{bmatrix} \mathbf{I} & \mathbf{t}_{\text{object}} \\ \mathbf{0}^T & 1 \end{bmatrix}$$
   The centered world-to-camera matrix for each frame $i$ becomes:
   $$\mathbf{T}_{w2c\_new}^{(i)} = \mathbf{T}_{w2c}^{(i)} \cdot \mathbf{T}_{\text{norm}}$$

2. **Turntable Relative Motion Decomposition**:
   A specific frame is designated as the reference frame (with centered pose $\mathbf{T}_{w2c\_ref}$ and camera-to-world pose $\mathbf{T}_{c2w\_ref}$). For any frame $i$, the relative rotation and translation of the turntable ($\mathbf{T}_{\text{object}}^{(i)}$) is calculated as:
   $$\mathbf{T}_{\text{object}}^{(i)} = \mathbf{T}_{c2w\_ref} \cdot \mathbf{T}_{w2c}^{(i)}$$
   The camera-to-world transform in the *object's local frame* (where the object is static and the camera revolves around it) is:
   $$\mathbf{T}_{vc2w}^{(i)} = \left(\mathbf{T}_{\text{object}}^{(i)}\right)^{-1} \cdot \mathbf{T}_{c2w\_ref}$$

These centered physical cameras, virtual cameras, and scale values are written to `transforms.json`, `transforms_train.json`, `transforms_test.json`, and `cameras_reformat.npz`.

---

### Stage 6: Volumetric TSDF Integration (`post_process.py -s fusion`)
Uses Open3D's Scalable TSDF Volumetric Fusion to construct a highly accurate surface mesh from depth maps.

1. **Projection**:
   For each voxel coordinate $\mathbf{x} = [x, y, z]^T$ in the global grid, project it onto camera $i$'s screen space:
   $$\mathbf{p} = \pi\left(\mathbf{K} \cdot \mathbf{T}_{w2c}^{(i)} \cdot \begin{bmatrix} \mathbf{x} \\ 1 \end{bmatrix}\right)$$

2. **Signed Distance Calculation**:
   $$sdf_i = D_i(\mathbf{p}) - x_z$$
   where $D_i(\mathbf{p})$ is the depth value at pixel $\mathbf{p}$, and $x_z$ is the depth of $\mathbf{x}$ along the camera's optical axis.

3. **Cumulative Volume Updating**:
   If $sdf_i$ falls within the truncation limit $[-t_{\text{trunc}}, +t_{\text{trunc}}]$, update the cumulative signed distance function value $S(\mathbf{x})$ and weight $W(\mathbf{x})$:
   $$S(\mathbf{x}) \leftarrow \frac{W(\mathbf{x}) \cdot S(\mathbf{x}) + w_i \cdot \text{clamp}(sdf_i)}{W(\mathbf{x}) + w_i}$$
   $$W(\mathbf{x}) \leftarrow W(\mathbf{x}) + w_i$$
   Finally, Marching Cubes is run on the zero-crossings ($S(\mathbf{x}) = 0$) to generate `eval_mesh.ply`.
