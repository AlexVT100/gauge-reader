import cv2

from .util import *

# 0 - skip
# 1 - draw
# 2 - draw and show

debug_conf = {
    'prep': {
        'orig':             0,
        'blur':             0,
        'lut':              0,
        'brt_cont':         0,
        'mean':             0,
        'thresh':           0,
        'erode':            0,
    },
    'outline_dial': {
        'all_contours':         0,
        'accepted_contours':    0,
        'rejected_contours':    0,
        'circle':               1,
        'marks':                1,
        'axes':                 0,
    },
    'measure_needle': {
        'axis':              1,
        'triangle':          1,
    },

    'exec': {
        'values':            1,
    },
}


class DebugImage:
    """
    Debugging image handling
    """
    def __init__(self, img, func: str, name: str, mode: int):
        self._img = img
        self._func = func
        self._name = name
        self.mode = mode

    def line(self, pt1, pt2, color, thickness=None, line_type=cv2.LINE_AA, shift=None) -> None:
        if self.mode >= 1:
            cv.line(self._img, pt1, pt2, color, thickness, line_type, shift)

    def circle(self, center, radius, color, thickness=None, line_type=cv2.LINE_AA, shift=None) -> None:
        if self.mode >= 1:
            cv.circle(self._img, (rint(center[0]), rint(center[1])), rint(radius), color, thickness, line_type, shift)

    # noinspection PyPep8Naming
    def drawContours(self, contours, contourIdx, color, thickness=None, lineType=cv2.LINE_AA,
                     hierarchy=None, maxLevel=None, offset=None) -> None:
        if self.mode >= 1:
            cv.drawContours(self._img, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset)

    def line_polar(self, length, angle, point: tuple, color: tuple, thickness: int, line_type: int = cv2.LINE_AA):
        # pylint: disable=too-many-arguments
        """
        Draws a line defined in polar coordinates
        """
        if self.mode < 1:
            return

        x, y = cv.polarToCart(length, angle=angle, x=point[0], y=point[1], angleInDegrees=True)

        p2 = (rint(x[0][0]), rint(y[0][0]))
        point = (rint(point[0]), rint(point[1]))
        p2 = p2[0] + point[0], point[1] - p2[1]

        cv.line(self._img, point, p2, color, thickness, line_type)

    # noinspection PyPep8Naming
    def text(self, text, org, fontFace, fontScale, color, thickness=1, lineType=cv2.LINE_AA,
             bottomLeftOrigin=None, shift=None):
        if self.mode < 1:
            return None

        if shift is not None:
            org = (org[0] + shift[0], org[1] + shift[1])
        return cv.putText(self._img, text, (rint(org[0]), rint(org[1])), fontFace, fontScale, color,
                          thickness, lineType, bottomLeftOrigin)

    def text_h_centered(self, position, text, font, scale, color, thickness, v_shift=0):
        # pylint: disable=too-many-arguments
        """
        Draws horizontally-centered text
        """
        if self.mode < 1:
            return

        size = cv.getTextSize(text, font, scale, thickness)[0]
        x = rint(position.x - size[0] / 2)
        y = rint(position.y + v_shift)

        cv.putText(img=self._img, text=text, org=(x, y), fontFace=font, fontScale=scale, color=color,
                   thickness=thickness)

    # noinspection PyPep8Naming
    def polylines(self, pts, isClosed, color, thickness=None, lineType=cv2.LINE_AA, shift=None):
        if self.mode < 1:
            return None
        return cv.polylines(self._img, pts, isClosed, color, thickness, lineType, shift)

    def show(self):
        """
        Shows <image> if enabled; does nothing oterwise
        """
        #if self._show:
        if self.mode >= 2:
            cv.imshow(self._func + '():' + self._name, self._img)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.show()


class Debug:
    def __init__(self, logger, no_im_show=False):
        self.conf = debug_conf
        self.log = logger
        self.no_im_show = no_im_show

    def img(self, img, key: str):
        """
        Check if the debugging drawing and viewing are enabled.

        Returns:
            - None if drawing for <name> is disabled (==0) or <name> does not exist in the debug configuration
            - DebugImage(name) instance otherwise
        """
        parts = key.split('/', 2)
        if len(parts) < 2:
            self.log.error(f'_d(): wrong debug key format: "{key}"')
            return None

        func, name = parts[0:2]

        if func not in self.conf:
            self.log.error(f'_d(): debug configuration for function "{func}" is not defined')
            return None

        if name not in self.conf[func]:
            self.log.error(f'_d(): name "{name}" in function "{func}" is not defined')
            return None

        mode = self.conf[func][name]
        if self.no_im_show and mode >= 2:
            mode = 1

        return DebugImage(img, func, name, mode)

    def show(self, img, name):
        """
        Shows the given image if showing is enabled or does nothing otherwise
        """
        if d := self.img(img, name):
            d.show()
