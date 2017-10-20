#!/usr/bin/env python
# coding=utf8

import cv2
import glob
import numpy as np
import os
import cPickle

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

thresh_config = {
    "abs_sobel_x_thresh": (25, 200),
    "abs_sobel_y_thresh": (25, 180),
    "gradient_magnitude_thresh": (15, 200),
    "gradient_direction_thresh": (0.5, 1.3),
    "s_channel_tresh": (135, 250),
}

warp_config = {
    "src_rect": np.float32([[584, 457], [707, 457], [1102, 676], [299, 676]]),
    "dst_rect": np.float32([[320, 0], [950, 0], [950, 720], [320, 720]])
}

conv_det_config= {
    "width": 50,
    "height": 80,
    "margin": 100
}

window_det_config = {
    "height": 80,
    "margin": 80,
    "min_num_pixels": 50
}


def load_cached_camera_parameters(cache_path):
    if cache_path is not None and os.path.isfile(cache_path):
        with open(cache_path, "rb") as f:
            camera_params = cPickle.load(f)
            return camera_params["intrinsic"], camera_params["distortion"]
    return None


def calibrate_camera(image_list, inner_points_per_row, inner_points_per_col,
                     cache_path=None):
    loaded_params = load_cached_camera_parameters(cache_path)
    if loaded_params is not None:
        return loaded_params

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
            cPickle.dump(params, f)
    return intrinsic, distortion


def undistort_image(image, intrinsic, distortion):
    return cv2.undistort(image, intrinsic, distortion, None, intrinsic)


def sobel_filter(gray_image, sobel_kernel=3,
                 magnitude_thresh=None, direction_thresh=None,
                 abs_x_thresh=None, abs_y_thresh=None):
    abs_sobel_x = np.absolute(cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    abs_sobel_y = np.absolute(cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    sobel_xy_mask = np.ones_like(gray_image, dtype=np.bool)
    use_sobel_xy = False
    if abs_x_thresh is not None:
        scaled_sobel_x = 255 * abs_sobel_x / np.max(abs_sobel_x)
        scaled_sobel_x = scaled_sobel_x.astype(np.uint8)
        sobel_xy_mask = (sobel_xy_mask &
                         (scaled_sobel_x >= abs_x_thresh[0]) & (scaled_sobel_x <= abs_x_thresh[1]))
        use_sobel_xy = True
    if abs_y_thresh is not None:
        scaled_sobel_y = 255 * abs_sobel_y / np.max(abs_sobel_y)
        scaled_sobel_y = scaled_sobel_y.astype(np.uint8)
        sobel_xy_mask = (sobel_xy_mask &
                         (scaled_sobel_y >= abs_y_thresh[0]) & (scaled_sobel_y <= abs_y_thresh[1]))
        use_sobel_xy = True

    magnitude_direction_mask = np.ones_like(gray_image, dtype=np.bool)
    use_magnitude_direction = False
    if magnitude_thresh is not None:
        gradient_magnitude = np.sqrt(abs_sobel_x ** 2 + abs_sobel_y ** 2)
        scaled_magnitude = 255 * gradient_magnitude / np.max(gradient_magnitude)
        scaled_magnitude = scaled_magnitude.astype(np.uint8)
        magnitude_direction_mask = (magnitude_direction_mask
                                    & (scaled_magnitude >= magnitude_thresh[0])
                                    & (scaled_magnitude <= magnitude_thresh[1]))
        use_magnitude_direction = True
    if direction_thresh is not None:
        abs_grad_direction = np.arctan2(abs_sobel_y, abs_sobel_x)
        magnitude_direction_mask = (magnitude_direction_mask
                                    & (abs_grad_direction >= direction_thresh[0])
                                    & (abs_grad_direction <= direction_thresh[1]))
        use_magnitude_direction = True

    if use_sobel_xy and use_magnitude_direction:
        return sobel_xy_mask & magnitude_direction_mask
    elif use_sobel_xy:
        return sobel_xy_mask
    elif use_magnitude_direction:
        return magnitude_direction_mask
    else:
        return np.ones_like(gray_image, dtype=np.bool)


def hls_filter(rgb_image, s_thresh=None):
    hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
    channel_s = hls[:, :, 2]
    mask = (channel_s >= s_thresh[0]) & (channel_s <= s_thresh[1])
    return mask


def get_warp_matrix(src_rect, dst_rect):
    return cv2.getPerspectiveTransform(src_rect, dst_rect)


def warp_image(image, warp_matrix):
    image_h, image_w = image.shape[0:2]
    warped = cv2.warpPerspective(image, warp_matrix, (image_w, image_h), flags=cv2.INTER_LINEAR)
    return warped


def rgb_to_warped_binary(rgb_image, cam_intrinsic, cam_distortion):
    # undistort input image
    undistorted = undistort_image(rgb_image, cam_intrinsic, cam_distortion)
    gray_img = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    # filter by gradient
    sobel_mask = sobel_filter(gray_img,
                              abs_x_thresh=thresh_config["abs_sobel_x_thresh"],
                              magnitude_thresh=thresh_config["gradient_magnitude_thresh"],
                              direction_thresh=thresh_config["gradient_direction_thresh"])
    # filter by channel S
    hls_mask = hls_filter(undistorted, s_thresh=thresh_config["s_channel_tresh"])

    # combine together
    threshed_binary = (sobel_mask | hls_mask).astype(np.uint8)

    # warp the result
    M = get_warp_matrix(warp_config["src_rect"], warp_config["dst_rect"])
    warped_binary = warp_image(threshed_binary, M)
    return warped_binary


def sliding_window_conv_detect(warped_image, window_width, window_height, margin):
    nonzero = np.nonzero(warped_image)
    nonzero_x = nonzero[1]
    nonzero_y = nonzero[0]

    img_h, img_w = warped_image.shape[0:2]

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Create our window template that we will use for convolutions
    conv_weight = np.ones(window_width)

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped_image[int(3 * img_h / 4):, :int(img_w / 2)], axis=0)
    l_center = np.argmax(np.convolve(conv_weight, l_sum)) - window_width / 2
    win_top = img_h - window_height
    win_bottom = img_h
    left_win_left = max(0, int(l_center - window_width / 2))
    left_win_right = min(int(l_center + window_width / 2), img_w)
    good_left_inds = ((nonzero_y >= win_top) & (nonzero_y < win_bottom) &
                      (nonzero_x >= left_win_left) & (nonzero_x < left_win_right)).nonzero()[0]
    left_lane_inds.append(good_left_inds)

    r_sum = np.sum(warped_image[int(3 * img_h / 4):, int(img_w / 2):], axis=0)
    r_center = np.argmax(np.convolve(conv_weight, r_sum)) - window_width / 2 + int(img_w / 2)
    right_win_left = max(0, int(r_center - window_width / 2))
    right_win_right = min(int(r_center + window_width / 2), img_w)
    good_right_inds = ((nonzero_y >= win_top) & (nonzero_y < win_bottom) &
                       (nonzero_x >= right_win_left) & (nonzero_x < right_win_right)).nonzero()[0]
    right_lane_inds.append(good_right_inds)

    # Go through each layer looking for max pixel locations
    for level in range(1, int(img_h / window_height)):
        # convolve the window into the vertical slice of the image
        win_top = int(img_h - (level + 1) * window_height)
        win_bottom = int(img_h - level * window_height)
        image_layer = np.sum(warped_image[win_top:win_bottom, :], axis=0)
        conv_signal = np.convolve(conv_weight, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped_image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        left_win_left = max(0, int(l_center - window_width / 2))
        left_win_right = min(int(l_center + window_width / 2), img_w)
        good_left_inds = ((nonzero_y >= win_top) & (nonzero_y < win_bottom) &
                          (nonzero_x >= left_win_left) & (nonzero_x < left_win_right)).nonzero()[0]
        left_lane_inds.append(good_left_inds)

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped_image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        right_win_left = max(0, int(r_center - window_width / 2))
        right_win_right = min(int(r_center + window_width / 2), img_w)
        good_right_inds = ((nonzero_y >= win_top) & (nonzero_y < win_bottom) &
                           (nonzero_x >= right_win_left) & (nonzero_x < right_win_right)).nonzero()[0]
        right_lane_inds.append(good_right_inds)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]
    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def sliding_window_detect(warped_image, window_height, margin, min_num_pixels):
    img_h, img_w = warped_image.shape[0:2]
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_image[img_h * 2 / 3:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    mid_point = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:mid_point])
    rightx_base = np.argmax(histogram[mid_point:]) + mid_point

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    n_windows = int(img_h / window_height)
    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_image.shape[0] - (window + 1) * window_height
        win_y_high = warped_image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > min_num_pixels:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > min_num_pixels:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

calibration_image_dir = "camera_cal/"
# the list of calibration images
calibration_image_list = glob.glob(calibration_image_dir + 'calibration*.jpg')
camera_intrinsic, camera_distortion = calibrate_camera(calibration_image_list, 9, 6, "./camera_params.p")
