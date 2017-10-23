#!/usr/bin/env python
# coding=utf-8

import configure as cfg
from camera import load_cached_camera_parameters, calibrate_camera
from image_filter import rgb_to_warped_binary, warp_image, region_of_interest
from utils import get_curvature, get_center_offset

import argparse
import cv2
import glob
import numpy as np
import os.path as osp
from moviepy.editor import VideoFileClip


class Result(object):
    def __init__(self):
        self.l_fit = None
        self.r_fit = None
        self.l_curvature = 0.0
        self.r_curvature = 0.0
        self.center_offset = 0.0
        self.debug_image = None


class LatestResults(object):
    def __init__(self, max_kept=4):
        self.l_fits = list()
        self.r_fits = list()
        self.l_curvatures = list()
        self.r_curvatures = list()
        self.center_offsets = list()
        self.debug_images = list()
        self._max_kept = max_kept

    def num_kept_results(self):
        return len(self.l_fits)

    def append_result(self, result):
        if self.num_kept_results() >= self._max_kept:
            self.pop_result(0)
        self.l_fits.append(result.l_fit)
        self.r_fits.append(result.r_fit)
        self.l_curvatures.append(result.l_curvature)
        self.r_curvatures.append(result.r_curvature)
        self.center_offsets.append(result.center_offset)
        self.debug_images.append(result.debug_image)

    def pop_result(self, index=0):
        self.l_fits.pop(index)
        self.r_fits.pop(index)
        self.l_curvatures.pop(index)
        self.r_curvatures.pop(index)
        self.center_offsets.pop(index)
        self.debug_images.pop(index)

    def average_with(self, result, n_used=2):

        l_fit_list = self.l_fits[-n_used:].copy()
        l_fit_list.append(result.l_fit)

        r_fit_list = self.r_fits[-n_used:].copy()
        r_fit_list.append(result.r_fit)

        l_curvature_list = self.l_curvatures[-n_used:].copy()
        l_curvature_list.append(result.l_curvature)

        r_curvature_list = self.r_curvatures[-n_used:].copy()
        r_curvature_list.append(result.r_curvature)

        center_offset_list = self.center_offsets[-n_used:].copy()
        center_offset_list.append(result.center_offset)

        averaged = Result()
        averaged.l_fit = np.mean(l_fit_list, axis=0)
        averaged.r_fit = np.mean(r_fit_list, axis=0)
        averaged.l_curvature = np.mean(l_curvature_list)
        averaged.r_curvature = np.mean(r_curvature_list)
        averaged.center_offset = np.mean(center_offset_list)
        averaged.debug_image = result.debug_image

        return averaged

    def get_average_result(self):
        average = Result()
        average.l_fit = np.mean(self.l_fits, axis=0)
        average.r_fit = np.mean(self.r_fits, axis=0)
        average.l_curvature = np.mean(self.l_curvatures)
        average.r_curvature = np.mean(self.r_curvatures)
        average.center_offset = np.mean(self.center_offsets)
        return average

    def get_last_result(self):
        if self.num_kept_results() < 1:
            return None
        latest = Result()
        latest.l_fit = self.l_fits[-1]
        latest.r_fit = self.r_fits[-1]
        latest.l_curvature = self.l_curvatures[-1]
        latest.r_curvature = self.r_curvatures[-1]
        latest.center_offset = self.center_offsets[-1]
        latest.debug_image = self.debug_images[-1]
        return latest


class LaneDetector(object):
    def __init__(self):
        # load the camera parameters
        camera_intrinsic, camera_distortion = load_cached_camera_parameters(cfg.camera_param_cache_path)
        # if load failed, calibrate the camera
        if camera_intrinsic is None or camera_distortion is None:
            # the list of calibration images
            calibration_image_list = glob.glob(cfg.camera_calibration_img_dir + 'calibration*.jpg')
            camera_intrinsic, camera_distortion = calibrate_camera(calibration_image_list, 9, 6,
                                                                   cfg.camera_param_cache_path)
        self._cam_intrinsic = camera_intrinsic
        self._cam_distortion = camera_distortion

        self._latest_results = LatestResults()

        self._consecutive_failed_cnt = 0
        self._max_allowed_failed = 5

        self._use_conv = False

        self._debug = True

    @staticmethod
    def sanity_check(detected_result):
        if detected_result is None:
            return False

        check_passed = True
        if abs(detected_result.center_offset) > 2:
            print("center offset too large: %d" % detected_result.center_offset)
            check_passed = False
        elif detected_result.l_curvature > (6 * detected_result.r_curvature) \
                or detected_result.r_curvature > (6 * detected_result.l_curvature):
            print("diff between left and right curvature too large:  l_curvature %.2f, r_curvature %.2f"
                  % (detected_result.l_curvature, detected_result.r_curvature))
            check_passed = False
        elif detected_result.l_curvature < 250 or detected_result.l_curvature > 10000 \
                or detected_result.r_curvature < 250 or detected_result.r_curvature > 10000:
            print("curvature invaid, l_curvature %.2f, r_curvature %.2f"
                  % (detected_result.l_curvature, detected_result.r_curvature))
            check_passed = False

        return check_passed

    def detect_from_scrach(self, warped_binary, need_debug_image=False):
        if self._use_conv:
            return self.sliding_window_conv_detect(warped_binary,
                                                   cfg.sliding_conv_config["width"],
                                                   cfg.sliding_conv_config["height"],
                                                   cfg.sliding_conv_config["margin"],
                                                   need_debug_image)
        else:
            return self.sliding_window_detect(warped_binary,
                                              cfg.sliding_window_config["height"],
                                              cfg.sliding_window_config["margin"],
                                              cfg.sliding_window_config["min_num_pixels"],
                                              need_debug_image)

    def pipe_line(self, rgb_image):
        #roi_image = region_of_interest(rgb_image, cfg.roi_vertices)
        roi_image = rgb_image
        warped_binary = rgb_to_warped_binary(roi_image, self._cam_intrinsic, self._cam_distortion)
        img_h, img_w = warped_binary.shape[0:2]

        detected = False
        if self._latest_results.num_kept_results() == 0: # the first frame
            cur_result = self.detect_from_scrach(warped_binary)
            # self.append_result(cur_result)
            detected = True
        else:
            if self._consecutive_failed_cnt >= self._max_allowed_failed:
                print("number of consecutive failed is %d, sliding window search from scrach"
                      % self._consecutive_failed_cnt)
                # too many failed, sliding window search from scrath
                cur_result = self.detect_from_scrach(warped_binary, self._debug)
                self._consecutive_failed_cnt = 0
            else:
                # use average of all previous results as the search base
                # prev_result = self._latest_results.get_last_result()
                prev_result = self._latest_results.get_average_result()
                cur_result = self.detect_based_prev_result(warped_binary, prev_result.l_fit,
                                                           prev_result.r_fit,
                                                           cfg.prev_result_based_search_margin,
                                                           self._debug)

            if self.sanity_check(cur_result):
                detected = True
                self._consecutive_failed_cnt = 0
            else:
                # sanity check failed, use previous result for current frame
                print("sanity check failed, used last valid result for current frame, self._consecutive_failed_cnt %d"
                      % self._consecutive_failed_cnt)
                cur_result = self._latest_results.get_last_result()
                #cur_result = self._latest_results.get_average_result()
                self._consecutive_failed_cnt += 1

        if detected:
            # average new result with the previous ones as the current result
            cur_result = self._latest_results.average_with(cur_result, 1)
            self._latest_results.append_result(cur_result)

        left_fit = cur_result.l_fit
        right_fit = cur_result.r_fit
        ploty = np.linspace(0, img_h - 1, img_h)
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
        unwarp_matrix_ = cv2.getPerspectiveTransform(cfg.warp_config["dst_rect"], cfg.warp_config["src_rect"])
        newwarp = warp_image(color_warp, unwarp_matrix_)

        # Combine the detected result with the original image
        result_img = cv2.addWeighted(rgb_image, 1, newwarp, 0.4, 0)

        # write the curvature and center offset
        left_curvature = cur_result.l_curvature
        right_curvature = cur_result.r_curvature
        center_offset = cur_result.center_offset
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 0, 0)
        thickness = 2
        cv2.putText(result_img, "left curvature : %.2fm" % left_curvature, (img_w // 2, 35),
                    font, font_scale, font_color, thickness)
        cv2.putText(result_img, "right curvature: %.2fm" % right_curvature, (img_w // 2, 65),
                    font, font_scale, font_color, thickness)
        cv2.putText(result_img, "center offset  : %.2fm" % center_offset, (img_w // 2, 95),
                    font, font_scale, font_color, thickness)

        if cur_result.debug_image is not None:
            debug_img_small = cv2.resize(cur_result.debug_image, None, fx=0.2, fy=0.2)
            small_h, small_w = debug_img_small.shape[0:2]
            result_img[0:small_h, 0:small_w] = debug_img_small

        return result_img

    @staticmethod
    def sliding_window_detect(warped_binary, window_height, margin, min_num_pixels, return_debug_img=False):
        if return_debug_img:
            debug_image = np.dstack((warped_binary, warped_binary, warped_binary)) * 255
        img_h, img_w = warped_binary.shape[0:2]
        # Take a histogram of the bottom 2/3 of the image
        histogram = np.sum(warped_binary[int(img_h * 2 / 3):, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        mid_point = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:mid_point])
        rightx_base = np.argmax(histogram[mid_point:]) + mid_point

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])
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
            win_y_low = warped_binary.shape[0] - (window + 1) * window_height
            win_y_high = warped_binary.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if return_debug_img:
                cv2.rectangle(debug_image, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(debug_image, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                              (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                               (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > min_num_pixels:
                leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > min_num_pixels:
                rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left lane pixel positions
        leftx = nonzero_x[left_lane_inds]
        lefty = nonzero_y[left_lane_inds]
        # Fit a second order polynomial to the left lane
        left_fit = np.polyfit(lefty, leftx, 2)

        # Extract left lane pixel positions
        rightx = nonzero_x[right_lane_inds]
        righty = nonzero_y[right_lane_inds]
        # Fit a second order polynomial to the right lane
        right_fit = np.polyfit(righty, rightx, 2)

        # calculate the radius of curvature is in meters
        y_in_pixels = np.linspace(0, img_h - 1, img_h // 5)
        left_x_in_pixels = left_fit[0] * y_in_pixels ** 2 + left_fit[1] * y_in_pixels + left_fit[2]
        right_x_in_pixels = right_fit[0] * y_in_pixels ** 2 + right_fit[1] * y_in_pixels + right_fit[2]
        left_curvature, right_curvature = get_curvature(y_in_pixels, left_x_in_pixels, right_x_in_pixels,
                                                        cfg.pixel_to_meter_config["x_meters_per_pixel"],
                                                        cfg.pixel_to_meter_config["y_meters_per_pixel"])

        # the center offset in meters
        lane_center_x_in_pixel = (left_x_in_pixels[-1] + right_x_in_pixels[-1]) / 2.0
        image_center_x = img_w / 2.0
        center_offset = get_center_offset(lane_center_x_in_pixel, image_center_x,
                                          cfg.pixel_to_meter_config["x_meters_per_pixel"])

        result = Result()
        result.l_fit = left_fit
        result.r_fit = right_fit
        result.l_curvature = left_curvature
        result.r_curvature = right_curvature
        result.center_offset = center_offset

        if return_debug_img:
            debug_image[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
            debug_image[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]
            result.debug_image = debug_image

        return result

    @staticmethod
    def detect_based_prev_result(warped_binary, prev_left_fit, prev_right_fit, margin, return_debug_img=False):
        nonzero = warped_binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        l_center_x = prev_left_fit[0] * (nonzero_y ** 2) + prev_left_fit[1] * nonzero_y + prev_left_fit[2]
        l_left_margin = l_center_x - margin
        l_right_margin = l_center_x + margin
        left_lane_inds = ((nonzero_x > l_left_margin) & (nonzero_x < l_right_margin))

        r_center_x = prev_right_fit[0] * (nonzero_y ** 2) + prev_right_fit[1] * nonzero_y + prev_right_fit[2]
        r_left_margin = r_center_x - margin
        r_right_margin = r_center_x + margin
        right_lane_inds = ((nonzero_x > r_left_margin) & (nonzero_x < r_right_margin))

        # min_pts = 10
        # if len(left_lane_inds) < min_pts or len(right_lane_inds) < min_pts:
        #     return None

        # extract left and right line pixel positions
        leftx = nonzero_x[left_lane_inds]
        lefty = nonzero_y[left_lane_inds]
        # Fit a second order polynomial to the left lane
        left_fit = np.polyfit(lefty, leftx, 2)

        # extract left and right line pixel positions
        rightx = nonzero_x[right_lane_inds]
        righty = nonzero_y[right_lane_inds]
        # Fit a second order polynomial to the right lane
        right_fit = np.polyfit(righty, rightx, 2)

        # calculate the radius of curvature is in meters
        y_in_pixels = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0] // 5)
        left_x_in_pixels = left_fit[0] * y_in_pixels ** 2 + left_fit[1] * y_in_pixels + left_fit[2]
        right_x_in_pixels = right_fit[0] * y_in_pixels ** 2 + right_fit[1] * y_in_pixels + right_fit[2]
        left_curvature, right_curvature = get_curvature(y_in_pixels, left_x_in_pixels, right_x_in_pixels,
                                                        cfg.pixel_to_meter_config["x_meters_per_pixel"],
                                                        cfg.pixel_to_meter_config["y_meters_per_pixel"])

        # the center offset in meters
        lane_center_x_in_pixel = (left_x_in_pixels[-1] + right_x_in_pixels[-1]) / 2.0
        image_center_x = warped_binary.shape[1] / 2.0
        center_offset = get_center_offset(lane_center_x_in_pixel, image_center_x,
                                          cfg.pixel_to_meter_config["x_meters_per_pixel"])

        result = Result()
        result.l_fit = left_fit
        result.r_fit = right_fit
        result.l_curvature = left_curvature
        result.r_curvature = right_curvature
        result.center_offset = center_offset
        # Create an image to draw on and an image to show the selection window
        if return_debug_img:
            focused_region_img = np.dstack((warped_binary, warped_binary, warped_binary)) * 255

            # Color in left and right line pixels
            focused_region_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
            focused_region_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

            # # Generate x and y values for plotting
            ploty = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0])
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            l_left_pts = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            l_right_pts = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((l_left_pts, l_right_pts))

            r_left_pts = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            r_right_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((r_left_pts, r_right_pts))

            search_area_img = np.zeros_like(focused_region_img)
            # Draw the lane onto the warped blank image
            cv2.fillPoly(search_area_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(search_area_img, np.int_([right_line_pts]), (0, 255, 0))
            debug_image = cv2.addWeighted(focused_region_img, 1, search_area_img, 0.3, 0)

            result.debug_image = debug_image

        return result

    @staticmethod
    def sliding_window_conv_detect(warped_binary, window_width, window_height, margin, return_debug_img=False):
        if return_debug_img:
            debug_image = np.dstack((warped_binary, warped_binary, warped_binary)) * 255

        nonzero = np.nonzero(warped_binary)
        nonzero_x = nonzero[1]
        nonzero_y = nonzero[0]

        img_h, img_w = warped_binary.shape[0:2]

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # the 1-D convolutional weights, [1, 1, ..., 1]
        conv_weight = np.ones(window_width)

        # First find the two starting positions for the left and right lane by using np.sum
        # to get the vertical image slice and then np.convolve the vertical image slice with the window template

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_histogram = np.sum(warped_binary[int(3 * img_h / 4):, :int(img_w / 2)], axis=0)
        l_center = np.argmax(np.convolve(conv_weight, l_histogram)) - window_width / 2
        win_top = img_h - window_height
        win_bottom = img_h
        left_win_left = max(0, int(l_center - window_width / 2))
        left_win_right = min(int(l_center + window_width / 2), img_w)
        good_left_inds = ((nonzero_y >= win_top) & (nonzero_y < win_bottom) &
                          (nonzero_x >= left_win_left) & (nonzero_x < left_win_right)).nonzero()[0]
        left_lane_inds.append(good_left_inds)

        r_histogram = np.sum(warped_binary[int(3 * img_h / 4):, int(img_w / 2):], axis=0)
        r_center = np.argmax(np.convolve(conv_weight, r_histogram)) - window_width / 2 + int(img_w / 2)
        right_win_left = max(0, int(r_center - window_width / 2))
        right_win_right = min(int(r_center + window_width / 2), img_w)
        good_right_inds = ((nonzero_y >= win_top) & (nonzero_y < win_bottom) &
                           (nonzero_x >= right_win_left) & (nonzero_x < right_win_right)).nonzero()[0]
        right_lane_inds.append(good_right_inds)

        if return_debug_img:
            cv2.rectangle(debug_image, (left_win_left, win_top), (left_win_right, win_bottom), (0, 255, 0), 2)
            cv2.rectangle(debug_image, (right_win_left, win_top), (right_win_right, win_bottom), (0, 255, 0), 2)

        win_area = window_width * window_height
        # the max distance between current and previous center
        max_dist = 0.8 * window_width
        # Go through each layer looking for max pixel locations
        for level in range(1, int(img_h / window_height)):
            # convolve the window into the vertical slice of the image
            win_top = int(img_h - (level + 1) * window_height)
            win_bottom = int(img_h - level * window_height)
            layer_histogram = np.sum(warped_binary[win_top:win_bottom, :], axis=0)
            layer_conv_signal = np.convolve(conv_weight, layer_histogram)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window,
            # not center of window
            offset = window_width / 2
            l_min_index = int(max(l_center + offset - margin, 0))
            l_max_index = int(min(l_center + offset + margin, warped_binary.shape[1]))
            l_argmax = np.argmax(layer_conv_signal[l_min_index:l_max_index])
            if layer_conv_signal[l_min_index + l_argmax] > (win_area / 50):
                l_center_candidate = l_min_index + l_argmax - offset
                # if the candidata center is far away from previous, adjust it to get close to previous one
                diff_with_prev = l_center_candidate - l_center
                if diff_with_prev > 0:
                    if diff_with_prev < max_dist:
                        l_center = l_center_candidate
                    else:
                        l_center = (l_center + min(l_center_candidate, l_center + max_dist * 2)) / 2
                elif diff_with_prev < 0:
                    if diff_with_prev > -max_dist:
                        l_center = l_center_candidate
                    else:
                        l_center = (l_center + max(l_center_candidate, l_center - max_dist * 2)) / 2

            left_win_left = max(0, int(l_center - offset))
            left_win_right = min(int(l_center + offset), img_w)
            good_left_inds = ((nonzero_y >= win_top) & (nonzero_y < win_bottom) &
                              (nonzero_x >= left_win_left) & (nonzero_x < left_win_right)).nonzero()[0]
            left_lane_inds.append(good_left_inds)

            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - margin, 0))
            r_max_index = int(min(r_center + offset + margin, warped_binary.shape[1]))
            r_argmax = np.argmax(layer_conv_signal[r_min_index:r_max_index])
            if layer_conv_signal[r_min_index + r_argmax] > (win_area / 50):
                r_center_candidate = r_min_index + r_argmax - offset
                # if the candidata center is far away from previous, adjust it to get close to previous one
                diff_with_prev = r_center_candidate - r_center
                if diff_with_prev > 0:
                    if diff_with_prev < max_dist:
                        r_center = r_center_candidate
                    else:
                        r_center = (r_center + min(r_center_candidate, r_center + max_dist * 2)) / 2
                elif diff_with_prev < 0:
                    if diff_with_prev > -max_dist:
                        r_center = r_center_candidate
                    else:
                        r_center = (r_center + max(r_center_candidate, r_center - max_dist * 2)) / 2
            right_win_left = max(0, int(r_center - offset))
            right_win_right = min(int(r_center + offset), img_w)
            good_right_inds = ((nonzero_y >= win_top) & (nonzero_y < win_bottom) &
                               (nonzero_x >= right_win_left) & (nonzero_x < right_win_right)).nonzero()[0]
            right_lane_inds.append(good_right_inds)

            if return_debug_img:
                cv2.rectangle(debug_image, (left_win_left, win_top), (left_win_right, win_bottom), (0, 255, 0), 2)
                cv2.rectangle(debug_image, (right_win_left, win_top), (right_win_right, win_bottom), (0, 255, 0), 2)

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

        # calculate the radius of curvature is in meters
        y_in_pixels = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0] // 5)
        left_x_in_pixels = left_fit[0] * y_in_pixels ** 2 + left_fit[1] * y_in_pixels + left_fit[2]
        right_x_in_pixels = right_fit[0] * y_in_pixels ** 2 + right_fit[1] * y_in_pixels + right_fit[2]
        left_curvature, right_curvature = get_curvature(y_in_pixels, left_x_in_pixels, right_x_in_pixels,
                                                        cfg.pixel_to_meter_config["x_meters_per_pixel"],
                                                        cfg.pixel_to_meter_config["y_meters_per_pixel"])

        # the center offset in meters
        lane_center_x_in_pixel = (left_x_in_pixels[-1] + right_x_in_pixels[-1]) / 2.0
        image_center_x = warped_binary.shape[1] / 2.0
        center_offset = get_center_offset(lane_center_x_in_pixel, image_center_x,
                                          cfg.pixel_to_meter_config["x_meters_per_pixel"])

        result = Result()
        result.l_fit = left_fit
        result.r_fit = right_fit
        result.l_curvature = left_curvature
        result.r_curvature = right_curvature
        result.center_offset = center_offset

        if return_debug_img:
            debug_image[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
            debug_image[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]
            result.debug_image = debug_image

        return result


def get_frame_process_func():
    detector = LaneDetector()

    def process_image(frame):
        return detector.pipe_line(frame)

    return process_image


if __name__ == "__main__":
    dir_of_this_file = osp.dirname(osp.abspath(__file__))
    default_video_file = osp.join(dir_of_this_file, "project_video.mp4")

    parser = argparse.ArgumentParser("lane detector")
    parser.add_argument("--input", "-i", help="path to the input video, default to project_video.mp4",
                        default=default_video_file)
    parser.add_argument("--output", "-o",
                        help="path to the output video, append '_output' to input name if not specified")

    args = parser.parse_args()

    if not osp.isfile(args.input):
        parser.error("the input %s is not a valid file" % args.input)

    if args.output is None:
        input_name = args.input.split("/")[-1]
        name_splits = input_name.split(".")
        args.output = osp.join(dir_of_this_file, name_splits[0] + "_output." + name_splits[-1])
    print("output is %s" % args.output)

    video_clip = VideoFileClip(args.input)
    process_frame_func = get_frame_process_func()
    white_clip = video_clip.fl_image(process_frame_func)  # NOTE: this function expects color images!!
    white_clip.write_videofile(args.output, audio=False)
