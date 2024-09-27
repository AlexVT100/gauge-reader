"""
Read the value from an analog gauge

Synopsis:
    from gauge_reader import GaugeReader

    reader = GaugeReader(log, image)
    value = reader.read()
"""
import datetime
import math
import os
import os.path
import shutil
from datetime import datetime
from itertools import pairwise

import numpy as np

# import sys
# SITE_PACKAGES = '/usr/lib64/python3.12/site-packages'
# if SITE_PACKAGES not in sys.path:
#    sys.path.append(SITE_PACKAGES)

from .config import Config
from .util import *
from .debug import Debug
from .scale import Scale


class GaugeReader:
    """
    This class holds all the necessary stuff to read the value from an analog gauge

    Raises:
        RuntimeError
    """
    def __init__(self, log, img_file, no_im_show=False):
        self.log = log
        self.img_file = img_file
        self.config = Config

        # Debugging
        self.d = Debug('reader', log)

        # Load the gauge image
        img = cv.imread(img_file)
        if img is None or img.size == 0:
            # self.log.error(f'Error reading file {img_file}.')
            raise RuntimeError(f'File {img_file} does not exist or is brokren')

        # Store the image size and area (for convenience)
        self.img_w = img.shape[1]
        self.img_h = img.shape[0]

        # A copies of the original image
        self.img = np.copy(img)      # for processing
        self.img_debug = np.copy(img)     # for debug drawings

        # Calibration
        self.cal_seg_len = 0
        self.needle_area = 0

        self.scale = None

        #self.log.debug(f'opencv: {cv.__version__}')
        #self.log.debug(f'numpy: {np.__version__}')

    def read(self) -> str | None:
        """
        The main method that performs the recognition
        """

        #
        # Recognize the dial scale
        #

        self.scale = Scale(self.img, self.img_debug, self.log)
        if not self.scale.exec():
            return None

        scale_area = pi * self.scale.radius**2
        self.needle_area = self.config.needle_area * scale_area

        self.log.debug(f'scale area: {scale_area:.2f} px²')
        self.log.debug(f'mean needle area: {self.needle_area:.2f} px²')

        #
        # Find the needle
        #

        self._preproc_image()

        needle_contour = self._find_needle()
        if needle_contour is None:
            return None

        angle = self._measure_needle(needle_contour)
        if angle is None:
            return None

        value = self._calculate_value(angle)
        if value is None:
            return None

        value = f'{value:4.2f}'

        # Put the value on the image
        if img := self.d.img(self.img_debug, 'read/values'):
            pos = Point(rint(self.img_w * self.config.text_pos[0]), rint(self.img_h * self.config.text_pos[1]))
            img.text_h_centered(position=pos, text=f'{value}', font=FONT, scale=2, color=COLOR_GREEN, thickness=5)
            img.show()

        return value

    def _preproc_image(self) -> None:
        """
        Performs some image transforms to prepare it for the recognition
        """
        self.img = bright_contr(self.img, 50, 100)
        self.d.show(self.img, 'prep/brt_cont')

        self.img = cv.medianBlur(self.img, self.config.needle.blur)
        self.d.show(self.img, 'prep/blur')

        # Apply a threshold filter
        _, self.img = cv.threshold(self.img, self.config.needle.thresh, self.config.needle.thresh_maxval,
                                   cv.ADAPTIVE_THRESH_GAUSSIAN_C)
        # self.img_proc = cv.bitwise_not(self.img_proc)
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.d.show(self.img, 'prep/thresh')

        kernel = np.ones((7, 7), np.uint8)
        self.img = cv.erode(self.img, kernel, iterations=2)
        self.d.show(self.img, 'prep/erode')

        self.img = cv.dilate(self.img, kernel, iterations=2)
        self.d.show(self.img, 'prep/dilate')

    def _find_needle(self) -> np.ndarray | None:
        """
        Finds the needle, measures and returns its angle
        """

        # Find contours
        self.log.debug('finding contours...')
        contours, _ = cv.findContours(image=self.img, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
        self.log.debug(f'{len(contours)} contour(s) found')

        if img := self.d.img(self.img_debug, 'find_needle/all_contours'):
            img.drawContours(contours, -1, COLOR_BLUE, 2, 0)
            img.show()

        cal_contours = []

        # Evaluate each contour
        self.log.debug('evaluating contours (skipping those with area ≤ 500 px²)...')
        for i, contour in enumerate(contours):

            # Skip too small and too large areas
            cnt_area = cv.contourArea(contour)
            if cnt_area <= 500:
                continue

            if isclose(cnt_area, self.needle_area, rel_tol=self.config.needle_area_tol):
                # Accept the first contour that has the expected area
                needle_contour = contour
                break

            self.log.debug(f'contour {i}: {cnt_area} px² - skipped')

        else:
            self.log.error('No acceptable contours found')
            return None

        self.log.debug(f'contour {i}: {cnt_area} px² - accepted as needle')
        if img := self.d.img(self.img_debug, 'find_needle/accepted_contours'):
            img.drawContours(cal_contours + [needle_contour], -1, COLOR_BLUE, 1, cv.LINE_AA)
            img.show()

        return needle_contour

    def _measure_needle(self, contour: np.ndarray) -> (float, Line, np.ndarray, Point, Point):
        """
        Find the needle tip
        """
        # Find an enclosing triangle for the contour
        _, triangle = cv.minEnclosingTriangle(contour)

        # Fit the line to the triangle to determine its angle
        vx, vy, x, y = cv.fitLine(triangle, cv.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x, y = vx[0], vy[0], rint(x[0]), rint(y[0])

        # Find two extreme points on the line to draw line
        lefty = rint((-x * vy / vx) + y)
        righty = rint(((self.img_w - x) * vy / vx) + y)

        # Build the needle axis line
        axis = Line(Point(0, lefty), Point(self.img_w - 1, righty))

        # Find the needle tip point
        tip = self._find_needle_tip(triangle, axis)

        # The vector returned by cv.fitLine() is always in range (90°, -90°)
        # in the image's coordinate system ( 0 ≤ x ≤ 1>, -1 ≤ y ≤ 1)
        # so we need to take into account the tip position to determine the line direction
        angle = math.degrees(math.atan2(-vy, vx))
        if tip.x < x:
            angle += 180

        self.log.debug(f'measured needle angle: {angle:.6f}°')

        if img := self.d.img(self.img_debug, 'measure_needle/axis'):
            img.line(axis.p1, axis.p2, COLOR_RED, 1, cv.LINE_AA)
            img.circle((x, y), 3, COLOR_RED, 1, cv.LINE_AA)
            img.show()

        if img := self.d.img(self.img_debug, 'measure_needle/triangle'):
            img.polylines([np.int32(triangle)], True, COLOR_YELLOW, 1)
            img.circle(tip, 3, COLOR_YELLOW, 1)
            img.show()

        return angle

    @staticmethod
    def _find_needle_tip(contour: np.ndarray, line: Line) -> Point:
        """
        Find and return a contour's vertex closest to a line

        https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
        """
        lp1 = np.array(line.p1)
        lp2 = np.array(line.p2)

        dist = np.abs((np.cross(lp2 - lp1, lp1 - contour)) / np.linalg.norm(lp2 - lp1))
        min_idx = np.argmin(dist)

        return Point(rint(contour[min_idx][0][0]), rint(contour[min_idx][0][1]))

    def _calculate_value(self, abs_angle: float) -> (float | None):
        """
        Calculates the value the needle points to
        """

        rel_angle = self.scale.cart_angle_to_loc(abs_angle) - self.scale.zero_angle
        self.log.debug(f'needle angle: absolute {abs_angle:0.6f}°, relative {rel_angle:0.6f}°')

        if self.scale.scale[0] > rel_angle > self.scale.scale[-1]:
            self.log.warning(f'Detected angle {rel_angle} is out of range {self.scale.scale[0]}-{self.scale.scale[-1]}')
            return None

        value = 0.0
        for a1, a2 in pairwise(self.scale.scale):
            if a1 < rel_angle <= a2:
                v1 = self.scale.scale[a1]
                v2 = self.scale.scale[a2]

                # the angle falls between a1 and a2; approximate the value
                value = (rel_angle - a1) / (a2 - a1) * (v2 - v1) + v1
                break

        # if rel_angle <= GaugeParams.dead_min_angle:
        #     self.log.info(f'angle is in the dead zone (<={GaugeParams.dead_min_angle:.6f}); assuming value 0.00')
        #     return 0.0

        self.log.debug(f'calculated value: {value:.4f}')

        return value

    def save_debug_image(self, save_dir=None, suffix=''):
        """
        Save the debugging image (with possible drawings) to a file with a different name
        """
        path, name = os.path.split(self.img_file)
        name, ext = os.path.splitext(name)

        if save_dir is None:
            save_dir = path
        file_name = f'{save_dir}/{name}{suffix}{ext}'

        cv.imwrite(file_name, self.img_debug)

        stat = os.stat(self.img_file)
        os.utime(file_name, (stat.st_atime, stat.st_mtime))

        self.log.debug(f'Debug image saved as "{file_name}"')

    def copy_image(self):
        """
        Copy the original (unchanged) image file to another with the timestamp in the name
        """
        name, ext = os.path.splitext(self.img_file)
        dt = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        shutil.copy(self.img_file, f'{name}.{dt}{ext}')


def read_pressure(*args):
    """
    The wrapper to do the recognition from the HA's pyscript trigger
    """
    try:
        reader = GaugeReader(*args)
    except RuntimeError:
        return None

    value = reader.read()
    reader.save_debug_image(suffix='-debug')
    # if value is None:
    #     # copy the image if it couldn't be recognized
    #     reader.copy_image()
    return value
