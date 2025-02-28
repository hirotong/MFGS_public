# The following code is used to watch a video stream, detect a Charuco board, and use
# it to determine the posture of the camera in relation to the plane
# of markers.
#
# Assumes that all markers are on the same plane, for example on the same piece of paper
#
# Requires camera calibration (see the rest of the project for example calibration)

import argparse
import glob
import os
import pickle
import sys

import cv2
import cv2.aruco as aruco
import numpy as np
from tqdm import tqdm


def read_camera_parameters(filename):
    fs = cv2.FileStorage(cv2.samples.findFile(filename, False), cv2.FileStorage_READ)
    if fs.isOpened():
        cam_matrix = fs.getNode("camera_matrix").mat()
        dist_coefficients = fs.getNode("distortion_coefficients").mat()
        return True, cam_matrix, dist_coefficients
    return False, [], []


# Check for camera calibration data
# if not os.path.exists("./calibration.pckl"):
#     print("You need to calibrate the camera you'll be using. See calibration project directory for details.")
#     exit()
# else:
#     f = open("calibration.pckl", "rb")
#     (cameraMatrix, distCoeffs, _, _) = pickle.load(f)
#     f.close()
#     if cameraMatrix is None or distCoeffs is None:
#         print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera with CalibrateCamera.py.")
#         exit()

def matchImagePoints(charuco_board, charuco_corners, charuco_ids):
    objPoints = []
    imgPoints = []
    for i in range(0, len(charuco_ids)):
        index = charuco_ids[i]
        objPoints.append(charuco_board.getChessboardCorners()[index])
        objPoints[-1][0][1] = charuco_board.getRightBottomCorner()[1] - objPoints[-1][0][1]  # set old axis direction
        imgPoints.append(charuco_corners[i])
    return np.array(objPoints), np.array(imgPoints)


def main():
    # parse command line options
    parser = argparse.ArgumentParser(
        description="detect markers and corners of charuco board, estimate pose of charuco" "board", add_help=False
    )
    parser.add_argument("-H", "--help", help="show help", action="store_true", dest="show_help")

    parser.add_argument("-i", "--image", help="Input from image folder", default="", action="store", dest="img_folder")
    parser.add_argument("-w", help="Number of squares in X direction", default="3", action="store", dest="w", type=int)
    parser.add_argument("-h", help="Number of squares in Y direction", default="3", action="store", dest="h", type=int)
    parser.add_argument("-sl", help="Square side length", default="1.", action="store", dest="sl", type=float)
    parser.add_argument("-ml", help="Marker side length", default="0.5", action="store", dest="ml", type=float)
    parser.add_argument("-x", help="Image file extension", default="png", action="store", dest="x")
    parser.add_argument(
        "-d",
        help="dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,  DICT_4X4_1000=3,"
        "DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, DICT_6X6_50=8,"
        "DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12, DICT_7X7_100=13,"
        "DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}",
        default="0",
        action="store",
        dest="d",
        type=int,
    )
    parser.add_argument(
        "-c", help="Input file with calibrated camera parameters", default="", action="store", dest="cam_param"
    )

    args = parser.parse_args()

    show_help = args.show_help
    if show_help:
        parser.print_help()
        sys.exit()
    width = args.w
    height = args.h
    square_len = args.sl
    marker_len = args.ml
    aruco_dict = args.d
    base_folder = args.img_folder
    img_ext = args.x

    cam_param = args.cam_param
    cam_matrix = []
    dist_coeffs = []

    if cam_param != "":
        _, cam_matrix, dist_coeffs = read_camera_parameters(cam_param)

    # Create Aruco detector object
    ARUCO_DETECTOR_PARAMETERS = aruco.DetectorParameters()
    ARUCO_DETECTOR_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco_dict)
    aruco_detector = aruco.ArucoDetector(ARUCO_DICT, ARUCO_DETECTOR_PARAMETERS)

    # Create CharucoBoard detector
    CHARUCO_BOARD = aruco.CharucoBoard(
        size=(width, height),
        squareLength=square_len,
        markerLength=marker_len,
        dictionary=ARUCO_DICT,
    )
    CHARUCO_PARAMETERS = aruco.CharucoParameters()
    CHARUCO_PARAMETERS.tryRefineMarkers  = True
    CHARUCO_PARAMETERS.cameraMatrix = cam_matrix
    CHARUCO_PARAMETERS.distCoeffs = dist_coeffs
    CHARUCO_DETECTOR_PARAMETERS = aruco.DetectorParameters()
    CHARUCO_DETECTOR_PARAMETERS.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    charuco_detector = aruco.CharucoDetector(CHARUCO_BOARD, CHARUCO_PARAMETERS, CHARUCO_DETECTOR_PARAMETERS)


    # Create vectors we'll be using for rotations and translations for postures
    rvecs, tvecs = None, None

    # base_folder = os.path.join(img_folder, os.path.pardir)
    img_folder = os.path.join(base_folder, "rgb")
    vis_img_folder = os.path.join(base_folder, "visualize")
    undistort_img_folder = os.path.join(base_folder, "undistort_images")
    final_img_folder = os.path.join(base_folder, "processed_images")
    pose_folder = os.path.join(base_folder, "poses")
    fg_mask_folder = os.path.join(base_folder, "fg_masks")
    depth_img_folder = os.path.join(base_folder, "depth")
    os.makedirs(vis_img_folder, exist_ok=True)
    os.makedirs(undistort_img_folder, exist_ok=True)
    os.makedirs(final_img_folder, exist_ok=True)
    os.makedirs(pose_folder, exist_ok=True)
    os.makedirs(fg_mask_folder, exist_ok=True)
    os.makedirs(depth_img_folder, exist_ok=True)

    images = [i for i in os.listdir(img_folder) if i.endswith(img_ext)]
    print("Found {} images".format(len(images)))

    cam_dict = {}

    for iname in tqdm(images):
        # Capturing each frame of our video stream
        print(os.path.join(img_folder, iname))
        img = cv2.imread(os.path.join(img_folder, iname))
        fg_mask_img = cv2.imread(os.path.join(base_folder, "fg_mask", iname.replace("png", "bmp")), -1)
        img_copy = img.copy()
        # grayscale image
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect Aruco markers
        corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(img)

        # Outline all of the markers detected in our image
        img_copy = aruco.drawDetectedMarkers(img_copy, corners, ids, borderColor=(0, 0, 255))

        # Only try to find CharucoBoard if we found 10 markers or more
        if ids is not None and len(ids) > 10:
            # Get charuco corners and ids from detected aruco markers
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(
                image=img,
            )
            print(f"charuco_corners.shape: {charuco_corners.shape}")

            # Require more than 20 squares
            if charuco_corners is not None and charuco_corners.shape[0] > 20:
                objPoints, imgPoints = matchImagePoints(CHARUCO_BOARD, charuco_corners, charuco_ids)
                valid, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cam_matrix, dist_coeffs)
                if valid:
                    # Draw the camera posture calculated from the gridboard
                    img_copy = aruco.drawDetectedCornersCharuco(img_copy, charuco_corners, charuco_ids, (255, 0, 0))
                    img_copy = cv2.drawFrameAxes(img_copy, cam_matrix, dist_coeffs, rvec, tvec, 0.3)

                    rot, _ = cv2.Rodrigues(rvec)

                    w2c = np.eye(4)
                    w2c[:3, :3] = rot
                    w2c[:3, 3:4] = tvec

                    c2w = np.linalg.inv(w2c)

                    cam_dict[f"world_mat_{iname[:-4]}"] = w2c
                    # cam_dict[f"world_mat_inv_{iname[:-4]}"] = c2w

                    np.savetxt(os.path.join(pose_folder, os.path.splitext(iname)[0] + ".txt"), c2w)
        # Display our image
        # cv2.imshow('QueryImage', QueryImg)
        cv2.imwrite(os.path.join(vis_img_folder, iname), img_copy)
        # undistort the images, crop, resize
        undistort_img = cv2.undistort(img, cam_matrix, dist_coeffs)
        cv2.imwrite(os.path.join(undistort_img_folder, iname), undistort_img)

        undistort_mask_img = cv2.undistort(fg_mask_img, cam_matrix, dist_coeffs)

        depth_img_name = os.path.join(base_folder, "depth16_masked", iname)

        depth_img = cv2.imread(depth_img_name, -1)
        undistort_depth_img = cv2.undistort(depth_img, cam_matrix, dist_coeffs)

        # crop image
        img_h, img_w = undistort_img.shape[:2]
        cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]

        img_w2 = min((img_w - cx), cx)
        img_h2 = min((img_h - cy), cy)

        # make sure the image size is divisible by 16
        img_w2 = (img_w2 // 16) * 16
        img_h2 = (img_h2 // 16) * 16

        crop_box = (
            int(cx - img_w2),  # left
            int(cx + img_w2),  # right
            int(cy - img_h2),  # top
            int(cy + img_h2),  # bottom
        )

        crop_img = undistort_img[crop_box[2] : crop_box[3], crop_box[0] : crop_box[1]]

        crop_mask_img = undistort_mask_img[crop_box[2] : crop_box[3], crop_box[0] : crop_box[1]]
        crop_depth_img = undistort_depth_img[crop_box[2] : crop_box[3], crop_box[0] : crop_box[1]]
        # crop_img = undistort_img[0 : -img_h // 4, img_w // 4 : -img_w // 4]
        cv2.imwrite(os.path.join(final_img_folder, iname), crop_img)
        cv2.imwrite(os.path.join(fg_mask_folder, iname), crop_mask_img)
        cv2.imwrite(os.path.join(depth_img_folder, iname), crop_depth_img)

    # save camera parameters, poses
    new_cam_matrix = cam_matrix.copy()
    new_h, new_w = crop_img.shape[:2]
    new_cam_matrix[0, 2] = new_w / 2
    new_cam_matrix[1, 2] = new_h / 2
    # new_cam_matrix[0, 2] -= img_w // 4

    # save poses
    # np.savetxt(os.path.join(pose_folder, "camera.txt"), new_cam_matrix)
    # cam_dict["camera"] = cam_matrix
    np.savetxt(os.path.join(base_folder, "original_intrinsic.txt"), new_cam_matrix)
    np.savez(os.path.join(base_folder, "cameras.npz"), **cam_dict)
    # save new intrinsics


if __name__ == "__main__":
    main()

    # help
    # python process_zivid/estimate_pose_charuco.py -i zivid-scan/dragon/images -w 12 -h 9 -sl 0.03 -ml 0.023 -d 6 -c process_zivid/camera_intrinsics_zivid.yml
