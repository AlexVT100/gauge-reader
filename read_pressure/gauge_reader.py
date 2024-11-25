"""
Read the value from an analog gauge

Synopsis:
    from gauge_reader import GaugeReader

    reader = GaugeReader(log, image)
    value = reader.read()
"""
import datetime
import os
import os.path
import shutil
from datetime import datetime
from itertools import pairwise
from math import isclose

import cv2

from .config import Config, MarkSimilRange, MarkRatioRange, NeedleSimilRange, NeedleRatioRange
from .debug import Debug
from .util import *
from .version import VERSION

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
        self.d = Debug(log, no_im_show)

        # Load the gauge image
        img = cv2.imread(img_file)
        if img is None or img.size == 0:
            # self.log.error(f'Error reading file {img_file}.')
            raise RuntimeError(f'File {img_file} does not exist or is brokren')

        # Store the image size and area (for convenience)
        self.img_w = img.shape[1]
        self.img_h = img.shape[0]

        # A copies of the original image
        self.img = np.copy(img)      # for processing
        self.img_d = np.copy(img)     # for debug drawings

        self.scale = None
        self.center: tuple[float, float] = (0.0, 0.0)
        self.radius: float = 0.0
        self.zero_angle: float = 0.0

        self.version = f'Gauge Reader {VERSION}'

        #self.log.debug(f'opencv: {cv2.__version__}')
        #self.log.debug(f'numpy: {np.__version__}')

    def exec(self) -> str | None:
        """
        The main method that performs the recognition
        """
        self.log.info(self.version)
        cv2.putText(self.img_d, self.version, (10, self.img_h - 10), FONT, 0.75, COLOR_BLACK, 1, cv2.LINE_AA)
        self.log.info(f'Processing {self.img_file}')

        # Preprocess the gauge image and outline it
        self._preproc_image()
        marks, needle = self._outline_dial_()

        # Prepare the scale table
        marks = self._interpolate_missing_marks(marks)
        self.scale = self._generate_scale_table(marks)

        angle = self._measure_needle_angle(needle.contour)
        if angle is None:
            return None

        value = self._calculate_value(angle)
        if value is None:
            return None

        value = f'{value:4.2f}'

        # Put the value on the image
        with self.d.img(self.img_d, 'exec/values') as img:
            pos = Point(rint(self.img_w * self.config.text_pos[0]), rint(self.img_h * self.config.text_pos[1]))
            img.text_h_centered(position=pos, text=f'{value}', font=FONT, scale=2, color=COLOR_GREEN, thickness=5)

        return value

    def _preproc_image(self) -> None:
        """
        Performs some image transforms to prepare it for the recognition
        """
        self.d.show(self.img, 'prep/orig')

        if self.config.scale.blur is not None:
            self.img = cv2.medianBlur(self.img, self.config.scale.blur)
            self.d.show(self.img, 'prep/blur')

        if self.config.scale.brightness is not None:
            self.img = bright_contr(self.img, self.config.scale.brightness, self.config.scale.contrast)
            self.d.show(self.img, 'prep/brt_cont')

        # https://forum.openframeworks.cc/t/levels-with-opencv/1314/2
        # https://stackoverflow.com/questions/12023958/what-does-cvnormalize-src-dst-0-255-norm-minmax-cv-8uc1
        # https://stackoverflow.com/questions/42169247/apply-opencv-look-up-table-lut-to-an-image
        lut = np.zeros(256, np.dtype('uint8'))
        for i in range(0, 256):
            if i < self.config.scale.lut_min:
                v = 0
            elif i > self.config.scale.lut_max:
                v = 255
            else:
                v = rint(255 / (self.config.scale.lut_max - self.config.scale.lut_min) * (i - self.config.scale.lut_min))
            lut[i] = v

        self.img = cv2.LUT(self.img, lut)
        self.d.show(self.img, 'prep/lut')

        self.img = cv2.pyrMeanShiftFiltering(self.img, self.config.scale.mean_sp, self.config.scale.mean_sr)
        self.d.show(self.img, 'prep/mean')


        # Apply a threshold filter
        # https://docs.opencv2.org/4.x/d7/d4d/tutorial_py_thresholding.html
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img = cv2.adaptiveThreshold(self.img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 15, 4)
        self.d.show(self.img, 'prep/thresh')

        if self.config.scale.erode_iters is not None:
            kernel = np.ones((3, 3), np.uint8)
            self.img = cv2.dilate(self.img, kernel, iterations=self.config.scale.erode_iters)
            self.d.show(self.img, 'prep/erode')

    def _od_find_contours(self):
        """

        """
        contours, _ = cv2.findContours(self.img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # Debug: draw all found contours
        with self.d.img(self.img_d, 'outline_dial/all_contours') as img:
            img.drawContours(contours, -1, COLOR_BLUE, 1, cv2.LINE_AA)

        return contours

    def _od_select_contours(self, contours):
        """
        Analyzes all found contours and sorts them into the marks and the needle
        """
        marks = []
        needle: (Contour, None) = None
        rejected = []

        for c in contours:
            if c.size < 8:
                continue

            contour = Contour(cv2.convexHull(c))

            if contour.contour_area < 250:
                continue

            if MarkSimilRange.in_range(contour.similarity) and MarkRatioRange.in_range(contour.box_ratio):
                marks.append(contour)
                continue

            if NeedleSimilRange.in_range(contour.similarity) and NeedleRatioRange.in_range(contour.box_ratio):
                if needle is None or needle.contour_area > contour.contour_area:
                    needle = contour
                    continue

            # The contour is rejected; save for debugging
            rejected.append(contour)

        # Debug: Draw accepted contours
        with self.d.img(self.img_d, 'outline_dial/accepted_contours') as img:
            if needle is not None:
                img.drawContours([needle.contour], -1, COLOR_BLUE, 1, cv2.LINE_AA)
                img.text(f'{needle.similarity:.4f}|{needle.box_ratio:.4f}', needle.center, FONT, .5, COLOR_WHITE)
            for f in marks:
                img.drawContours([f.contour], -1, COLOR_BLUE, 1, cv2.LINE_AA)
                img.text(f'{f.similarity:.4f}|{f.box_ratio:.4f}', f.contour[0][0], FONT, .4, COLOR_WHITE)

        # Debug: Draw rejected contours
        with self.d.img(self.img_d, 'outline_dial/rejected_contours') as img:
            for c in rejected:
                img.drawContours([c.contour], -1, COLOR_BLUE, 1, cv2.LINE_AA)
                img.text(f'{c.similarity:.4f}|{c.box_ratio:.4f}', c.contour[0][0], FONT, .4, COLOR_WHITE)

        if needle is None:
            self.log.error('The needle is not found.')

        return marks, needle

    def _od_find_enclosing_circle(self, figures):
        """
        Find an enclosing circle for the center points of the selected rectangles
        """
        points = np.array([f.center for f in figures]).astype(int)
        center, radius = cv2.minEnclosingCircle(points)
        center = (center[0] + Config.shaft_displacement[0], center[1] + Config.shaft_displacement[1])

        # Debug: Draw the enclosing circle
        with self.d.img(self.img_d, 'outline_dial/circle') as img:
            img.circle(tuple(map(round, center)), rint(radius), COLOR_GREEN, 1, cv2.LINE_AA)
            img.circle(tuple(map(round, center)), 3, COLOR_GREEN, 1, cv2.LINE_AA)

        return center, radius

    def _od_remove_displaced_figs(self, figures):
        """
        Remove rectangles whose centers are not lying close to the circle
        """
        marks = []
        for i in range(len(figures)-1, -1, -1):
            f = figures[i]
            if not isclose(distance(self.center, f.center), self.radius, rel_tol=0.1):
                del figures[i]
                continue

            cart_angle = angle360(self.center, f.center)
            loc_angle = cart_angle_to_loc(cart_angle)
            marks.append(Mark(f.center, cart_angle, loc_angle, f.box))

        return marks

    def _outline_dial_(self):
        """
        Links
            https://theailearner.com/2020/11/03/opencv-minimum-area-rectangle/
            https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
            https://github.com/opencv/opencv/issues/19472
            https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
            https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18
        """
        # Find contours
        contours = self._od_find_contours()

        # Choose contours by their enclosing rectangle's proportions
        marks, needle = self._od_select_contours(contours)
        if marks is None or needle is None:
            return marks, needle

        # Find an enclosing circle for the center points of the selected rectangles
        self.center, self.radius = self._od_find_enclosing_circle(marks)

        # Remove figures whose centers are not lying close to the circle
        marks = self._od_remove_displaced_figs(marks)

        # Sort the mark list by the loc_angle in the ascending order
        marks.sort(key=lambda e: e.loc_angle)

        return marks, needle

    def _interpolate_missing_marks(self, marks):
        """
        Interpolate missing marks
        """
        avg_mark_delta = self.config.full_scale_angle / (rint(self.config.max_mark / self.config.mark_step) - 1)
        self.log.debug(f'Average angle between marks: {avg_mark_delta:.2f}')

        # Interpolate a zero mark
        self.zero_angle = marks[0].loc_angle - avg_mark_delta
        cart_angle = marks[0].cart_angle + avg_mark_delta
        mark = Mark(point=cross(self.center, self.radius, cart_angle), cart_angle=cart_angle, loc_angle=self.zero_angle)
        marks_new = [mark]

        self.log.debug(f'Added a zero mark at angle {self.zero_angle}')

        # One mark may be missed because it is covered with the needle, so we need to interpolate it
        for a1, a2 in pairwise(marks):
            marks_new.append(a1)
            delta = a2.loc_angle - a1.loc_angle
            if not isclose(delta, avg_mark_delta, rel_tol=0.1):
                d = delta / 2
                cart_angle = a1.cart_angle - d
                mark = Mark(point=cross(self.center, self.radius, cart_angle),
                            cart_angle=cart_angle, loc_angle=a1.loc_angle + d)
                marks_new.append(mark)
                self.log.debug(f'Added a missing mark at angle {mark.loc_angle:.2f}')
        else:
            marks_new.append(a2)

        # Debug: draw the accepted marks
        with self.d.img(self.img_d, 'outline_dial/marks') as img:
            for m in marks_new:
                img.circle(m.point, 3, COLOR_GREEN, 1, cv2.LINE_AA)

        # A basic check
        if (_l := len(marks_new)) != (_n := rint(self.config.max_mark / self.config.mark_step)):
            self.log.error(f'{_n} dial marks are expected but {_l} found')
            return None

        # Debug: draw the mark's axes
        with self.d.img(self.img_d, 'outline_dial/axes') as img:
            for mark in marks_new:
                color = COLOR_MAGENTA if mark.box is not None else COLOR_DARK_MAGENTA
                img.line_polar(self.radius + 25, mark.cart_angle, self.center, color, 1)
                img.text(text=f'{mark.loc_angle:.2f}', org=mark.point, fontFace=FONT, fontScale=0.5,
                         color=color, thickness=1, shift=(10, 5))

        return marks_new

    def _generate_scale_table(self, marks):
        """
        Generates a dict representing the scale:

            scale[angle: float] = value: float

        Each record represents a recognized mark. The angles are relative
        to the first mark thus angles start from zero.
        """
        scale = {0.0: self.config.mark_step}
        value = self.config.first_mark
        for mark in marks[1:]:
            scale[mark.loc_angle - self.zero_angle] = round(value, 1)
            value += self.config.mark_step

        return scale

    def _measure_needle_angle(self, contour: np.ndarray) -> float:
        """
        Find the needle tip
        """
        # Find an enclosing triangle for the contour
        _, triangle = cv2.minEnclosingTriangle(contour)

        # Find the needle tip, i.e. a triangle vertex with the minimum angle
        min_angle = 180.0
        tip = None
        for i1, ic, i2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            p1, c, p2 = triangle[i1][0], triangle[ic][0], triangle[i2][0]
            angle = angle3p(p1, c, p2)
            if min_angle > angle:
                min_angle = angle
                tip = Point(c[0], c[1])

        # The needle axis goes through the scale center and the needle tip
        axis = Line(tip, self.center)

        # Calculate the cartesian angle on the axis
        angle = angle360(self.center, tip)

        self.log.debug(f'Measured needle angle: {angle:.6f}° (cartesian)')

        with self.d.img(self.img_d, 'measure_needle/axis') as img:
            img.line(axis.p1, axis.p2, COLOR_RED, 1, cv2.LINE_AA)
            #img.circle(tip, 3, COLOR_RED, 1, cv2.LINE_AA)

        with self.d.img(self.img_d, 'measure_needle/triangle'):
            img.polylines([np.int32(triangle)], True, COLOR_YELLOW, 1)
            #img.circle(tip, 3, COLOR_YELLOW, 1)

        return angle

    def _calculate_value(self, abs_angle: float) -> (float | None):
        """
        Calculates the value the needle points to
        """

        rel_angle = cart_angle_to_loc(abs_angle) - self.zero_angle
        self.log.debug(f'needle angle: absolute {abs_angle:0.6f}°, relative {rel_angle:0.6f}°')

        if self.scale[0] > rel_angle > self.scale[-1]:
            self.log.warning(f'Detected angle {rel_angle} is out of range {self.scale[0]}-{self.scale[-1]}')
            return None

        value = 0.0
        for a1, a2 in pairwise(self.scale):
            if a1 < rel_angle <= a2:
                v1 = self.scale[a1]
                v2 = self.scale[a2]

                # the angle falls between a1 and a2; approximate the value
                value = (rel_angle - a1) / (a2 - a1) * (v2 - v1) + v1
                break

        # if rel_angle <= GaugeParams.dead_min_angle:
        #     self.log.info(f'angle is in the dead zone (<={GaugeParams.dead_min_angle:.6f}); assuming value 0.00')
        #     return 0.0

        self.log.debug(f'calculated value: {value:.4f}')

        # Debug: draw the approximated "microscale" between the two marks
        with self.d.img(self.img_d, 'calculate_value/microscale') as img:
            da = (a2 - a1) / (self.config.mark_step * 100)
            n = 0
            while a1 <= a2 * 1.001:
                point = cross(self.center, self.radius + 8, (a := 270 - self.zero_angle - a1))
                img.line_polar(10 if n % 5 == 0 else 5, a, point, COLOR_GREEN, 1, cv2.LINE_AA)
                a1 += da
                n += 1

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

        cv2.imwrite(file_name, self.img_d)

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

    value = reader.exec()
    reader.save_debug_image(suffix='-debug')
    # if value is None:
    #     # copy the image if it couldn't be recognized
    #     reader.copy_image()
    return value
