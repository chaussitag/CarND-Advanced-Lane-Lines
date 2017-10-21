#!/usr/bin/env python
# coding=utf8

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob
from cv_based_lane_line_detection import load_cached_camera_parameters
from cv_based_lane_line_detection import thresh_config, warp_config, conv_det_config, window_det_config
from cv_based_lane_line_detection import undistort_image, sobel_filter, hls_filter, warp_image, get_warp_matrix
from cv_based_lane_line_detection import rgb_to_warped_binary, sliding_window_detect, sliding_window_conv_detect


# import cv_based_lane_line_detection as line_detection

def undistort_test(test_img_path):
    cam_intrinsic, cam_distortion = load_cached_camera_parameters("./camera_params.p")
    calib_test_image = cv2.imread(test_img_path)
    undistorted_calib_test_image = undistort_image(calib_test_image, cam_intrinsic, cam_distortion)
    # Visualize undistortion
    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    axes1.imshow(calib_test_image)
    axes1.set_title('Original', fontsize=15)
    axes2.imshow(undistorted_calib_test_image)
    axes2.set_title('Undistorted', fontsize=15)
    plt.show()


def sobel_filter_test(test_img_path):
    cam_intrinsic, cam_distortion = load_cached_camera_parameters("./camera_params.p")
    img = mpimg.imread(test_img_path)
    undistorted = undistort_image(img, cam_intrinsic, cam_distortion)
    gray_img = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    mask_img = sobel_filter(gray_img,
                            abs_x_thresh=thresh_config["abs_sobel_x_thresh"],
                            magnitude_thresh=thresh_config["gradient_magnitude_thresh"],
                            direction_thresh=thresh_config["gradient_direction_thresh"]).astype(np.uint8)
    # mask_img = sobel_filter(undistorted,
    #                         abs_x_thresh=(20, 100)).astype(np.uint8)
    # Visualize sobel based thresholding
    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    axes1.imshow(img)
    axes1.set_title('Original', fontsize=15)
    axes2.imshow(mask_img, cmap="gray")
    axes2.set_title('Filter By Sobel', fontsize=15)
    plt.show()


def hls_filter_test(test_img_path):
    cam_intrinsic, cam_distortion = load_cached_camera_parameters("./camera_params.p")
    img = mpimg.imread(test_img_path)
    undistorted = undistort_image(img, cam_intrinsic, cam_distortion)
    mask_img = hls_filter(undistorted, s_thresh=thresh_config["s_channel_tresh"]).astype(np.uint8)
    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    axes1.imshow(img)
    axes1.set_title('Original', fontsize=15)
    axes2.imshow(mask_img, cmap="gray")
    axes2.set_title('Filter By S-Channel', fontsize=15)
    plt.show()


def sobel_hsl_filter_test(test_img_path):
    cam_intrinsic, cam_distortion = load_cached_camera_parameters("./camera_params.p")
    img = mpimg.imread(test_img_path)
    undistorted = undistort_image(img, cam_intrinsic, cam_distortion)
    gray_img = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    sobel_mask_img = sobel_filter(gray_img,
                                  abs_x_thresh=thresh_config["abs_sobel_x_thresh"],
                                  magnitude_thresh=thresh_config["gradient_magnitude_thresh"],
                                  direction_thresh=thresh_config["gradient_direction_thresh"]).astype(np.uint8)
    hls_mask_img = hls_filter(undistorted, s_thresh=thresh_config["s_channel_tresh"]).astype(np.uint8)

    combined_result = np.dstack((np.zeros_like(gray_img), sobel_mask_img, hls_mask_img)) * 255
    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    axes1.imshow(img)
    axes1.set_title('Original', fontsize=15)
    axes2.imshow(combined_result)
    axes2.set_title('Filter By Sobel & S-Channel', fontsize=15)
    plt.show()


def sobel_hsl_warp_test(test_img_path):
    cam_intrinsic, cam_distortion = load_cached_camera_parameters("./camera_params.p")
    img = mpimg.imread(test_img_path)
    undistorted = undistort_image(img, cam_intrinsic, cam_distortion)
    gray_img = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    sobel_mask_img = sobel_filter(gray_img,
                                  abs_x_thresh=thresh_config["abs_sobel_x_thresh"],
                                  magnitude_thresh=thresh_config["gradient_magnitude_thresh"],
                                  direction_thresh=thresh_config["gradient_direction_thresh"]).astype(np.uint8)
    hls_mask_img = hls_filter(undistorted, s_thresh=thresh_config["s_channel_tresh"]).astype(np.uint8)

    threshed_result = np.dstack((np.zeros_like(gray_img), sobel_mask_img, hls_mask_img)) * 255
    warp_matrix = get_warp_matrix(warp_config["src_rect"], warp_config["dst_rect"])
    combined_result = warp_image(threshed_result, warp_matrix)
    f, (axes1, axes2, axes3) = plt.subplots(1, 3, figsize=(20, 10))
    axes1.imshow(img)
    axes1.set_title('Original', fontsize=15)
    axes2.imshow(threshed_result)
    axes2.set_title('Filter By Sobel & S-Channel', fontsize=15)
    axes3.imshow(combined_result)
    axes3.set_title("Soble & S-Channel & Warp", fontsize=15)
    plt.show()


def rgb_to_warped_test(test_img_path):
    cam_intrinsic, cam_distortion = load_cached_camera_parameters("./camera_params.p")
    img = mpimg.imread(test_img_path)
    warped_binary = rgb_to_warped_binary(img, cam_intrinsic, cam_distortion)

    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    axes1.imshow(img)
    axes1.set_title('Original', fontsize=15)
    axes2.imshow(warped_binary, cmap="gray")
    axes2.set_title('Warped Binary', fontsize=15)
    plt.show()


def sliding_window_conv_detect_test(test_img_path):
    cam_intrinsic, cam_distortion = load_cached_camera_parameters("./camera_params.p")
    img = mpimg.imread(test_img_path)
    warped_binary = rgb_to_warped_binary(img, cam_intrinsic, cam_distortion)
    left_fit, right_fit, debug_img = sliding_window_conv_detect(warped_binary,
                                                                conv_det_config["width"],
                                                                conv_det_config["height"],
                                                                conv_det_config["margin"],
                                                                return_debug_img=True)
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    f.suptitle(test_img_path.split("/")[-1], fontsize=30)
    axes1.imshow(img)
    axes1.set_title('Original', fontsize=15)

    axes2.imshow(debug_img, cmap="gray")
    axes2.set_title('Detected', fontsize=15)
    axes2.plot(left_fitx, ploty, color='yellow')
    axes2.plot(right_fitx, ploty, color='yellow')
    axes2.set_xlim(0, img.shape[1])
    axes2.set_ylim(img.shape[0], 0)
    plt.show()


def sliding_window_detect_test(test_img_path):
    cam_intrinsic, cam_distortion = load_cached_camera_parameters("./camera_params.p")
    img = mpimg.imread(test_img_path)
    warped_binary = rgb_to_warped_binary(img, cam_intrinsic, cam_distortion)
    left_fit, right_fit, debug_img = sliding_window_detect(warped_binary,
                                                           window_det_config["height"],
                                                           window_det_config["margin"],
                                                           window_det_config["min_num_pixels"],
                                                           return_debug_img=True)
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
    # ploty = np.linspace(warped_binary.shape[0]/4, warped_binary.shape[0] - 1, warped_binary.shape[0] * 3 / 4)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    f.suptitle(test_img_path.split("/")[-1], fontsize=30)
    axes1.imshow(img)
    axes1.set_title('Original', fontsize=15)

    axes2.imshow(debug_img)
    axes2.set_title('Detected', fontsize=15)
    axes2.plot(left_fitx, ploty, color='yellow')
    axes2.plot(right_fitx, ploty, color='yellow')
    axes2.set_xlim(0, img.shape[1])
    axes2.set_ylim(img.shape[0], 0)
    plt.show()


# undistort_test("camera_cal/calibration1.jpg")

test_image_list = glob.glob("test_images/*.jpg")
# test_image_list = ["test_images/test5.jpg"]
for test_image_path in test_image_list:
    # sobel_filter_test(test_image_path)
    # hls_filter_test(test_image_path)
    # sobel_hsl_filter_test(test_image_path)
    # sobel_hsl_warp_test(test_image_path)
    # rgb_to_warped_test(test_image_path)
    sliding_window_conv_detect_test(test_image_path)
    # sliding_window_detect_test(test_image_path)
