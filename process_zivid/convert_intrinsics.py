import cv2
import numpy as np

# Camera intrinsics from Zivid library
zivid_intrinsics = {
    "CameraMatrix": {
        "CX": 959.119079589844,
        "CY": 573.3330078125,
        "FX": 1779.87915039062,
        "FY": 1780.30529785156,
    },
    "Distortion": {
        "K1": -0.0902933329343796,
        "K2": 0.130275040864944,
        "K3": -0.0610044673085213,
        "P1": 0.000627919391263276,
        "P2": 0.000284405949059874,
    },
}

# Convert to OpenCV format
camera_matrix = np.array([
    [zivid_intrinsics["CameraMatrix"]["FX"], 0, zivid_intrinsics["CameraMatrix"]["CX"]],
    [0, zivid_intrinsics["CameraMatrix"]["FY"], zivid_intrinsics["CameraMatrix"]["CY"]],
    [0, 0, 1],
])

distortion_coeffs = np.array([
    zivid_intrinsics["Distortion"]["K1"],
    zivid_intrinsics["Distortion"]["K2"],
    zivid_intrinsics["Distortion"]["P1"],
    zivid_intrinsics["Distortion"]["P2"],
    zivid_intrinsics["Distortion"]["K3"],
])

# Save to YAML file
fs = cv2.FileStorage("camera_intrinsics_zivid.yml", cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", camera_matrix)
fs.write("distortion_coefficients", distortion_coeffs)
fs.release()

print("Camera intrinsics have been saved to camera_intrinsics.yml")
