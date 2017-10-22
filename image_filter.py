#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np

import configure as cfg

from camera import undistort_image


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


def hls_filter(rgb_image, s_thresh, l_thresh=None, h_thresh=None):
    hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS).astype(np.float32)
    channel_s = hls[:, :, 2]
    mask = (channel_s >= s_thresh[0]) & (channel_s <= s_thresh[1])
    return mask


def rgb_filter(rgb_image, r_thresh, g_thresh=None, b_thresh=None):
    channel_r = rgb_image[:, :, 0]
    mask = (channel_r >= r_thresh[0]) & (channel_r <= r_thresh[1])
    return mask


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
                              abs_x_thresh=cfg.thresh_config["abs_sobel_x_thresh"],
                              magnitude_thresh=cfg.thresh_config["gradient_magnitude_thresh"],
                              direction_thresh=cfg.thresh_config["gradient_direction_thresh"])
    # filter by channel S
    hls_mask = hls_filter(undistorted, s_thresh=cfg.thresh_config["s_channel_thresh"])

    # filter by channel R
    rgb_mask = rgb_filter(undistorted, r_thresh=cfg.thresh_config["r_channel_thresh"])

    # combine together
    threshed_binary = (sobel_mask | hls_mask | rgb_mask).astype(np.uint8)

    # warp the result
    warp_matrix_ = cv2.getPerspectiveTransform(cfg.warp_config["src_rect"], cfg.warp_config["dst_rect"])
    warped_binary = warp_image(threshed_binary, warp_matrix_)

    return warped_binary
