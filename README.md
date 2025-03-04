# <img src="assets/pikachu_surprised_loop.gif" alt="logo" style="height: 1em;"> S$^2$GS: Self-supervised Gaussian Segmentation for 3D Reconstruction of Moving Object

<!-- [Project page](https://surfsplatting.github.io/) | [Paper](https://arxiv.org/pdf/2403.17888) | [Video](https://www.youtube.com/watch?v=oaHCtB6yiKU) | [Surfel Rasterizer (CUDA)](https://github.com/hbb1/diff-surfel-rasterization) | [Surfel Rasterizer (Python)](https://colab.research.google.com/drive/1qoclD7HJ3-o0O1R8cvV3PxLhoDCMsH8W?usp=sharing) | [DTU+COLMAP (3.5GB)](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9) | [SIBR Viewer Pre-built for Windows](https://drive.google.com/file/d/1DRFrtFUfz27QvQKOWbYXbRS2o2eSgaUT/view?usp=sharing)<br> -->


<!-- ![Teaser image](assets/ACCV2024_First%20figure.png) -->
<div align="center">
  <img src="assets/ACCV2024_First figure.png">
</div>

This repo contains the official implementation for the paper "S$^2$GS: Self-supervised Gaussian Segmentation for 3D Reconstruction of Moving Object".

## SIBR Viewer

> Note: This is from the original repo of [2DGS](https://github.com/hbb1/2d-gaussian-splatting), I haven't modified this to apply to our method. since we use the same representation as 2DGS, it should work but the visualization won't be good because the Gaussians are not transformed correctly.

The Pre-built Viewer for Windows can be found [here](https://drive.google.com/file/d/1DRFrtFUfz27QvQKOWbYXbRS2o2eSgaUT/view?usp=sharing). If you use Ubuntu or want to check the viewer usage, please refer to [GS Monitor](https://github.com/RongLiu-Leo/Gaussian-Splatting-Monitor).
### How to use
Firstly open the viewer, 
```shell
<path to downloaded/compiled viewer>/bin/SIBR_remoteGaussian_app_rwdi
```
and then
```shell
# Monitor the training process
python train.py -s <path to COLMAP or NeRF Synthetic dataset> 
# View the trained model
python view.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to trained model> 
```

## Installation

```bash
# download
git clone https://github.com:hirotong/Self-supervisedGaussianSplatting.git --recursive

# if you already clone the repo, you can initialize the submodules by
git submodule update --init --recursive

# if you have an environment used for 3dgs, use it
# if not, create a new environment
conda env create --file environment.yml
conda activate s2gs

# install 2dgs rasterizer
cd submodules/diff-surfel-rasterization
pip install .
cd ../simple-knn
pip install .
```

## Data preparation

Download the [Real](https://drive.google.com/file/d/1glBuy4Zc1Oy5RoR3DbBznKibU340RfCV/view?usp=sharing) and [Synthetic](https://drive.google.com/file/d/168zTU4n_yNdE4Rz-_6ip1ggdXybH6E1Y/view?usp=sharing) dataset from [Google Drive](https://drive.google.com/drive/folders/1eqllCExPDLphJK6cM33iDTZgeNOxmmNB?usp=sharing) and uncompress the dataset to your prefered location.

The data are organized as:
```
├── <case>
│   ├── <case>.mp4
│   ├── eval_mesh.ply
│   ├── eval_points.ply
│   ├── images
│   ├── intrinsic.txt
│   ├── sparse
│   ├── transforms.json
│   ├── transforms_test.json
│   ├── transforms_train.json
│   └── transforms_val.json
├── ...
```
where the `<case>.mp4` is a video we create for mask extraction with [Track-Anything](https://github.com/gaomingqi/Track-Anything).

### Prepare your own dataset (zivid only)

Please follow the [instuction](process_zivid/README.md) to prepare your own zivid dataset.

## Training
To train a scene, simply use
```bash
python train.py -s <path to dataset> --rotation
```
<details><summary><span style="font-weight: bold;">Commandline arguments for regularizations </span></summary>

```bash
--lambda_reg3d  # control the consistency loss
--lambda_zero_one   # control the 2D segmentation regularization
--lambda_zero_one_3d    # control the 3D segmentation regularization
```
The followings are from [2DGS](https://github.com/hbb1/2d-gaussian-splatting) paper. 
```bash
--lambda_normal  # hyperparameter for normal consistency
--lambda_distortion # hyperparameter for depth distortion
--depth_ratio # 0 for mean depth and 1 for median depth, 0 works for most cases
```
</details>

## Testing
### Bounded Mesh Extraction
To export a mesh within a bounded volume, simply use
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset> 
```

<details>
<summary><span style="font-weight: bold;">Commandline arguments you should adjust accordingly for meshing for bounded TSDF fusion </span></summary>

```bash
--bbox  # restrict the boudning box size
--depth_ratio # 0 for mean depth and 1 for median depth
--voxel_size # voxel size
--depth_trunc # depth truncation
```
If these arguments are not specified, the script will automatically estimate them using the camera information.
</details>
<br>

Some useful running scripts can be found in the `scripts` folder.

## Evaluation

To evaluate the reconstruction and novel-view synthesize metrics
```bash
python eval_metrics_ssgs.py -m <path to pre-trained model> --apply_mask
```
Before evaluating the nvs performance, we crop the images with the bounding box of the object mask and we only calculate the metrics within the bbox area for equal comparison. 


## Acknowledgements
This project is built upon [2DGS](https://github.com/hbb1/2d-gaussian-splatting). The TSDF fusion for extracting mesh is based on [Open3D](https://github.com/isl-org/Open3D). We thank all the authors for their great repos. 

