#!/usr/bin/env python
# coding=utf-8

import os.path as osp
import numpy as np

camera_calibration_img_dir = osp.join(osp.dirname(osp.abspath(__file__)), "camera_cal/")
camera_param_cache_path = osp.join(osp.dirname(osp.abspath(__file__)), "camera_params.p")

thresh_config = {
    # "abs_sobel_x_thresh": (30, 200),
    "abs_sobel_x_thresh": (20, 160),
    # "abs_sobel_y_thresh": (25, 180),
    "gradient_magnitude_thresh": (20, 200),
    "gradient_direction_thresh": (0.6, 1.3),
    # "s_channel_thresh": (135, 250),
    "s_channel_thresh": (160, 255),
    "r_channel_thresh": (205, 255),
}

warp_config = {
    # "src_rect": np.float32([[584, 457], [707, 457], [1102, 676], [299, 676]]),
    # "dst_rect": np.float32([[320, 0], [950, 0], [950, 720], [320, 720]])
    "src_rect": np.float32([[585, 460], [203, 720], [1127, 720], [705, 460]]),
    "dst_rect": np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
}

# confiure for the 1-D convolutional based sliding widow search
sliding_conv_config = {
    "width": 50,  # width of 1-D convolutional kernel
    "height": 80,  # divide the image into sub-image of this height
    "margin": 80,  # left and right distance from some center to search the maximum response of 1-D convolution
}

# configure for the simple sliding window search
sliding_window_config = {
    "height": 80,  # height of the search window
    "margin": 100,  # half the width of the search window
    "min_num_pixels": 50  # minimum number of pixels inside the search window to recalculate the search center
}

# the margin for searching along previous found lanes
prev_result_based_search_margin = 100

pixel_to_meter_config = {
    "x_meters_per_pixel": 3.7 / 700,
    "y_meters_per_pixel": 30 / 720
}
