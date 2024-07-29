## Step 0 Convert raw data to images
Use the provided python [script](Zdf_conv_dataset2.py) to convert raw `zdf` files to images.
``` bash
python process_zivid/Zdf_conv_dataset2.py
```
## Step 1 Camera pose estimation
Change the parameters `w, h, sl, ml` and `d` according to the calibration board used.<br>
The intrinsic `camera_intrinsics_zivid.yml` is obtained from the hard-coded one according to [Zivid](https://support.zivid.com/en/latest/reference-articles/camera-intrinsics.html#camera-intrinsics)'s documents.

``` bash
python process_zivid/estimate_pose_charuco.py -i <path/to/dataset> -w 12 -h 9 -sl 0.03 -ml 0.023 -d 6 -c process_zivid/camera_intrinsics_zivid.yml
```


## Step 2 Resize the images and camera parameters
``` bash 
python process_zivid/post_process.py -d <path/to/dataset> -s resize 
```

## Step 3 Back project pointclouds from all views
``` bash 
python process_zivid/post_process.py -d <path/to/dataset> -s pcd 
```

## Step 4 reformat the camera poses

Before running the code, you need to edit the projected pointcloud file saved in the dataset folder. <br> 
You need to crop the pointcloud by removing other parts except the object, and save the pointcloud of the object renamed to `point_cloud_processed.ply`, we will read the cropped pointcloud in the next step. We suggest to do this with [cloudcompare](https://www.danielgm.net/cc/).

``` bash
python process_zivid/post_process.py -d <path/to/dataset> -s reformat
```

We transform the camera poses accroding to the pointcloud of the object, so that the origin of the camera coordinate system is roughly set to the center of the object.


## [Optional] Step 5 Process the masks and save

We adapt [Track-Anything](https://github.com/gaomingqi/Track-Anything) to generate object masks. After that, you can process the masks as following:

``` bash
python process_zivid/post_process.py -d <path/to/dataset> -s mask -m <path/to/masks>
```
> Note: this is not essential if you only want to run $s^2gs$. 
> 