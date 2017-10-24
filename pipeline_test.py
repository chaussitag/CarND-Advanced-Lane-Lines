#!/usr/bin/env python
# coding=utf8

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Polygon
import numpy as np
import glob

from lane_detector import LaneDetector
from camera import load_cached_camera_parameters
from configure import thresh_config, warp_config, sliding_window_config, sliding_conv_config, roi_vertices
from image_filter import undistort_image, sobel_filter, hls_filter, warp_image, rgb_to_warped_binary, region_of_interest


def roi_test(test_img_path):
    img = mpimg.imread(test_img_path)
    roi_img = region_of_interest(img, roi_vertices)
    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    axes1.imshow(img)
    axes1.add_patch(Polygon(roi_vertices.reshape(-1, 2), closed=True, fill=False, color="green"))
    axes1.set_title('Original', fontsize=15)
    axes2.imshow(roi_img)
    axes2.set_title('ROI', fontsize=15)
    plt.show()


def undistort_test(test_img_path):
    cam_intrinsic, cam_distortion = load_cached_camera_parameters("./camera_params.p")
    calib_test_image = mpimg.imread(test_img_path)
    undistorted_calib_test_image = undistort_image(calib_test_image, cam_intrinsic, cam_distortion)
    # image_name = test_img_path.split("/")[-1]
    # name_without_suffix = image_name.split(".")[0]
    # mpimg.imsave(name_without_suffix + "_undistorted.png", undistorted_calib_test_image, format="png")
    # mpimg.imsave(name_without_suffix + ".png", calib_test_image, format="png")
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
    mask_img = hls_filter(undistorted, s_thresh=thresh_config["s_channel_thresh"]).astype(np.uint8)
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
    hls_mask_img = hls_filter(undistorted, s_thresh=thresh_config["s_channel_thresh"]).astype(np.uint8)

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
    hls_mask_img = hls_filter(undistorted, s_thresh=thresh_config["s_channel_thresh"]).astype(np.uint8)

    threshed_result = np.dstack((np.zeros_like(gray_img), sobel_mask_img, hls_mask_img)) * 255
    warp_matrix = cv2.getPerspectiveTransform(warp_config["src_rect"], warp_config["dst_rect"])
    combined_result = warp_image(threshed_result, warp_matrix)
    f, (axes1, axes2, axes3) = plt.subplots(1, 3, figsize=(20, 10))
    axes1.imshow(img)
    axes1.set_title('Original', fontsize=15)
    axes2.imshow(threshed_result)
    axes2.set_title('Filter By Sobel & S-Channel', fontsize=15)
    axes3.imshow(combined_result)
    axes3.set_title("Soble & S-Channel & Warp", fontsize=15)
    plt.show()

def warp_test(test_img_path):
    cam_intrinsic, cam_distortion = load_cached_camera_parameters("./camera_params.p")
    img = mpimg.imread(test_img_path)
    undistorted_img = undistort_image(img, cam_intrinsic, cam_distortion)
    warp_matrix = cv2.getPerspectiveTransform(warp_config["src_rect"], warp_config["dst_rect"])
    warped_img = warp_image(undistorted_img, warp_matrix)

    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    axes1.imshow(undistorted_img)
    axes1.add_patch(Polygon(warp_config["src_rect"], closed=True, fill=False, color="red"))
    axes1.set_title('undistorted image with source points drawn', fontsize=15)

    axes2.imshow(warped_img)
    axes2.add_patch(Polygon(warp_config["dst_rect"], closed=True, fill=False, color="red"))
    axes2.set_title('warped image with destination points drawn', fontsize=15)
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
    result = LaneDetector.sliding_window_conv_detect(warped_binary,
                                                     sliding_conv_config["width"],
                                                     sliding_conv_config["height"],
                                                     sliding_conv_config["margin"],
                                                     return_debug_img=True)

    left_fit = result.l_fit
    right_fit = result.r_fit
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    f.suptitle(test_img_path.split("/")[-1], fontsize=30)
    axes1.imshow(img)
    axes1.set_title('Original', fontsize=15)

    axes2.imshow(result.debug_image, cmap="gray")
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
    result = LaneDetector.sliding_window_detect(warped_binary,
                                                sliding_window_config["height"],
                                                sliding_window_config["margin"],
                                                sliding_window_config["min_num_pixels"],
                                                return_debug_img=True)
    left_fit = result.l_fit
    right_fit = result.r_fit
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw the detected region
    warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # unwarp the result
    unwarp_matrix_ = cv2.getPerspectiveTransform(warp_config["dst_rect"], warp_config["src_rect"])
    newwarp = warp_image(color_warp, unwarp_matrix_)
    # Combine the detected result with the original image
    result_img = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    f.suptitle(test_img_path.split("/")[-1], fontsize=30)
    axes1.imshow(result_img)
    axes1.set_title('Original with detected result', fontsize=15)
    left_curvature = result.l_curvature
    right_curvature = result.r_curvature
    center_offset = result.center_offset
    axes1.text(0.5, 0.95, 'left curvature : %.2fm' % (left_curvature,), color="yellow",
               horizontalalignment='center', transform=axes1.transAxes)
    axes1.text(0.5, 0.90, 'right curvature: %.2fm' % (right_curvature,), color="yellow",
               horizontalalignment='center', transform=axes1.transAxes)
    axes1.text(0.5, 0.85, 'center offset  : %2fm' % (center_offset,), color="yellow",
               horizontalalignment='center', transform=axes1.transAxes)

    axes2.imshow(result.debug_image)
    axes2.set_title('Detected', fontsize=15)
    axes2.plot(left_fitx, ploty, color='yellow')
    axes2.plot(right_fitx, ploty, color='yellow')
    axes2.set_xlim(0, img.shape[1])
    axes2.set_ylim(img.shape[0], 0)
    plt.show()


def pipe_test(test_img_path):
    detector = LaneDetector()
    img = mpimg.imread(test_img_path)
    result_img = detector.pipe_line(img)

    # combined = cv2.addWeighted(img, 1, result_img, 0.4, 0)

    f, (axes1, axes2) = plt.subplots(1, 2, figsize=(20, 10))
    f.suptitle(test_img_path.split("/")[-1], fontsize=30)
    axes1.imshow(img)
    axes1.set_xlim(0, img.shape[1])
    axes1.set_ylim(img.shape[0], 0)
    axes1.set_title('original frame', fontsize=15)

    axes2.imshow(result_img, cmap="gray")
    axes2.set_title('result frame with detected road', fontsize=15)
    axes2.set_xlim(0, img.shape[1])
    axes2.set_ylim(img.shape[0], 0)
    plt.show()

#undistort_test("camera_cal/calibration1.jpg")

#test_image_list = glob.glob("test_images/mytest*.jpg")
test_image_list = ["test_images/mytest4.jpg"]
for test_image_path in test_image_list:
    # undistort_test(test_image_path)
    # roi_test(test_image_path)
    # sobel_filter_test(test_image_path)
    # hls_filter_test(test_image_path)
    # sobel_hsl_filter_test(test_image_path)
    # sobel_hsl_warp_test(test_image_path)
    # warp_test(test_image_path)
    # rgb_to_warped_test(test_image_path)
    # sliding_window_conv_detect_test(test_image_path)
    #sliding_window_detect_test(test_image_path)
    pipe_test(test_image_path)
