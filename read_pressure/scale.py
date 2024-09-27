from itertools import pairwise

import numpy as np

from .config import Config
from .debug import Debug
from .util import *

# Do not use dataclasses - they do not behave well in HA


class Mark:
    def __init__(self, point=None, box=None, contour=None, cart_angle=None, loc_angle=None):
        self.point: tuple = point
        self.box: [] = box
        self.contour: [] = contour
        self.cart_angle: float = cart_angle
        self.loc_angle: float = loc_angle


class Scale:
    def __init__(self, img, img_debug, log):
        self.img = np.copy(img)     # the copy of the main image to work with
        self.img_d = img_debug      # the image for the debug drawing
        self.log = log
        self.config = Config
        self.d = Debug('scale', log)

        self.center: tuple[float, float] = (0.0, 0.0)
        self.radius: float = None
        self.zero_angle: float = None
        self.scale: dict = {}

    def _preproc_image(self) -> None:
        """
        Performs some image transforms to prepare it for the recognition
        """
        self.img = bright_contr(self.img, self.config.scale.brightness, self.config.scale.contrast)
        self.d.show(self.img, 'prep/brt_cont')

        self.img = cv.medianBlur(self.img, self.config.scale.blur)
        self.d.show(self.img, 'prep/blur')

        #self.img = cv.pyrMeanShiftFiltering(self.img, self.config.scale.mean_sp, self.config.scale.mean_sr)
        #self.d.show(self.img, 'prep/mean')

        # Apply a threshold filter
        _, self.img = cv.threshold(self.img, self.config.scale.thresh, self.config.scale.thresh_maxval,
                                   cv.ADAPTIVE_THRESH_GAUSSIAN_C)
        # img = cv.bitwise_not(img)
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.d.show(self.img, 'prep/thresh')

        kernel = np.ones((3, 3), np.uint8)
        self.img = cv.erode(self.img, kernel, iterations=self.config.scale.erode_iters)
        self.d.show(self.img, 'prep/erode')

    @staticmethod
    def cart_angle_to_loc(angle):
        return 270 - angle if angle <= 270 else 270 - angle + 360

    def _find_scale(self):
        """
        Links
            https://theailearner.com/2020/11/03/opencv-minimum-area-rectangle/
            https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
            https://github.com/opencv/opencv/issues/19472
        """

        contours, _ = cv.findContours(self.img, mode=cv.RETR_LIST, method=cv.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        if img := self.d.img(self.img_d, 'find_scale/all_contours'):
            img.drawContours(contours, -1, COLOR_BLUE, 1, cv.LINE_AA)
            img.show()

        #
        # Stage 1: Filter contours by their enclosing rectangles' width-to-height proportion
        #
        marks = []
        for contour in contours:
            if contour.size < 8: continue

            rect = cv.minAreaRect(contour)
            ((cnt_x, cnt_y), (dim_x, dim_y), angle) = rect
            ratio = min(dim_x, dim_y) / max(dim_x, dim_y)

            # No erode
            # 1     0.22 - 0.25
            # 0.5   0.08 - 0.10 *
            # 0.1   0.10 - 0.17 *
            # 0     0.38858948326891457

            # erode_iters=1
            # 1     0.19 - 0.25
            # 0.5   0.05 - 0.07 *
            # 0.1   0.07 - 0.12 *

            if ratio < 0.05 or ratio > 0.25:    # erode_iters=1
                #self.log.debug(f'Ratio {ratio:.3f} - skipped')
                continue
            #self.log.debug(f'Ratio {ratio:.3f}')

            box = cv.boxPoints(rect).astype(int)
            marks.append(Mark(point=(cnt_x, cnt_y), box=box, contour=contour))

        # Debug: Draw accepted contours
        if img := self.d.img(self.img_d, 'find_scale/accepted_contours'):
            img.drawContours([m.contour for m in marks], -1, COLOR_BLUE, 1, cv.LINE_AA)
            img.show()

        #
        # Find an enclosing circle for the center points of the selected rectangles
        #
        points = np.array([m.point for m in marks]).astype(int)
        self.center, self.radius = cv.minEnclosingCircle(points)

        # Debug: Draw the enclosing circle
        if img := self.d.img(self.img_d, 'find_scale/circle'):
            img.circle(tuple(map(round, self.center)), rint(self.radius), COLOR_GREEN, 1, cv.LINE_AA)
            img.circle(tuple(map(round, self.center)), 3, COLOR_GREEN, 1, cv.LINE_AA)
            img.show()

        #
        # Stage 2: Remove rectangles whose centers are not lying close to the circle
        #
        for i in range(len(marks)-1, -1, -1):
            mark = marks[i]
            if not isclose(distance(self.center, mark.point), self.radius, rel_tol=0.1):
                del marks[i]
                continue

            mark.cart_angle = (cart_angle := angle360(self.center, mark.point))
            mark.loc_angle = self.cart_angle_to_loc(cart_angle)

        # Sort the mark list by the loc_angle in the ascending order
        marks.sort(key=lambda e: e.loc_angle)

        # Debug: draw the accepted marks
        if img := self.d.img(self.img_d, 'find_scale/marks'):
            for mark in marks:
                img.drawContours([mark.box], -1, COLOR_YELLOW, 1, cv.LINE_AA)
                img.circle(mark.point, 3, COLOR_YELLOW, 1, cv.LINE_AA)
            img.show()

        #
        # Interpolate skipped marks
        #
        avg_mark_delta = self.config.full_scale_angle / (rint(self.config.max_mark / self.config.mark_step) - 1)
        self.log.debug(f'Average angle between marks: {avg_mark_delta:.2f}')

        # Interpolate a zero mark
        self.zero_angle = marks[0].loc_angle - avg_mark_delta
        cart_angle = marks[0].cart_angle + avg_mark_delta
        marks.insert(0, Mark(point=cross(self.center, self.radius, cart_angle),
                             cart_angle=cart_angle, loc_angle=self.zero_angle))
        self.log.debug(f'Added a zero mark at angle {self.zero_angle}')

        # One mark may be missed because it is covered with the needle, so we need to interpolate it
        for i, (a1, a2) in enumerate(pairwise(marks[:])):
            delta = a2.loc_angle - a1.loc_angle
            if not isclose(delta, avg_mark_delta, rel_tol=0.1):
                d = delta / 2
                cart_angle = a1.cart_angle - d
                mark = Mark(point=cross(self.center, self.radius, cart_angle),
                            cart_angle=cart_angle, loc_angle=a1.loc_angle + d)
                marks.insert(i + 1, mark)
                self.log.debug(f'Added a missing mark at angle {mark.loc_angle:.2f}')

        # A basic check
        if (_l := len(marks)) != (_n := rint(self.config.max_mark / self.config.mark_step)):
            self.log.error(f'{_n} dial marks are expected but {_l} found')
            return None

        # Debug: draw the mark's axes
        if img := self.d.img(self.img_d, 'find_scale/axes'):
            for mark in marks:
                color = COLOR_MAGENTA if mark.box is not None else COLOR_DARK_MAGENTA
                img.line_polar(self.radius + 25, mark.cart_angle, self.center, color, 1)
                img.text(text=f'{mark.loc_angle:.2f}', org=mark.point, fontFace=FONT, fontScale=0.5,
                         color=color, thickness=1, shift=(10, 5))
            img.show()

        return marks

    def _calculate_scale(self, marks):
        """

        """
        self.scale[0.0] = self.config.mark_step
        value = self.config.first_mark
        for mark in marks[1:]:
            self.scale[mark.loc_angle - self.zero_angle] = round(value, 1)
            value += self.config.mark_step

    def exec(self):
        self._preproc_image()
        marks = self._find_scale()
        if not marks:
            return None
        self._calculate_scale(marks)
        return self.scale
