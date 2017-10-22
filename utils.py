#!/usr/bin/env python
# coding=utf-8

import numpy as np


def get_curvature(y_in_pixels, left_x_in_pixels, right_x_in_pixels, xm_per_pixel, ym_per_pixel):
    y_in_meters = y_in_pixels * ym_per_pixel
    left_x_in_meters = left_x_in_pixels * xm_per_pixel
    right_x_in_meters = right_x_in_pixels * xm_per_pixel

    # Fit polynomials to x,y in world space
    left_fit = np.polyfit(y_in_meters, left_x_in_meters, 2)
    right_fit = np.polyfit(y_in_meters, right_x_in_meters, 2)

    # Calculate the new radii of curvature
    y_eval = np.max(y_in_meters)
    left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curvature, right_curvature


def get_center_offset(lane_center_x_in_pixel, image_center_x, xm_per_pixel):
    return (image_center_x - lane_center_x_in_pixel) * xm_per_pixel
