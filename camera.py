#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import pickle
import os


def calibrate_camera(image_list, inner_points_per_row, inner_points_per_col, cache_path=None):
    inner_points_coordinates = np.zeros((inner_points_per_row * inner_points_per_col, 3), np.float32)
    inner_points_coordinates[:, :2] = np.mgrid[0:inner_points_per_row, 0:inner_points_per_col].T.reshape(-1, 2)

    detected_obj_points = []
    detected_img_points = []

    image_size = None
    # Step through the list and search for chessboard corners
    for image_path in image_list:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            # image size in (w, h) format
            image_size = (gray.shape[1], gray.shape[0])

        assert (gray.shape[1] == image_size[0]) and (gray.shape[0] == image_size[1])

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (inner_points_per_row, inner_points_per_col), None)

        # If found all inner points, add object points and image points
        if ret:
            detected_obj_points.append(inner_points_coordinates)
            detected_img_points.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (inner_points_per_row, inner_points_per_col), corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    # Do camera calibration given object points and image points
    _, intrinsic, distortion, rvecs, tvecs = \
        cv2.calibrateCamera(detected_obj_points, detected_img_points, image_size, None, None)
    if cache_path is not None:
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        params = dict()
        params["intrinsic"] = intrinsic
        params["distortion"] = distortion
        with open(cache_path, "wb") as f:
            pickle.dump(params, f)
    return intrinsic, distortion


def load_cached_camera_parameters(cache_path):
    if cache_path is not None and os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            camera_params = pickle.load(f)
            return camera_params["intrinsic"], camera_params["distortion"]
    return None, None


def undistort_image(image, intrinsic, distortion):
    return cv2.undistort(image, intrinsic, distortion, None, intrinsic)
