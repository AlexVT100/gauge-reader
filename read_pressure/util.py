from collections import namedtuple
from math import atan2, pi, sin, cos, radians

import cv2 as cv

# Colors are BGR
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_DARK_MAGENTA = (152, 0, 152)
COLOR_CYAN = (255, 255, 0)

FONT = cv.FONT_HERSHEY_SIMPLEX


class Figure:
    def __init__(self, contour):
        #self.point: tuple = point

        self.contour = contour
        self.contour_area = cv.contourArea(contour)

        rect = cv.minAreaRect(contour)
        (self.center, (dim_x, dim_y), angle) = rect

        self.box = cv.boxPoints(rect).astype(int)
        self.box_ratio = min(dim_x, dim_y) / max(dim_x, dim_y)
        self.box_area = dim_x * dim_y

        #self.cart_angle: float = cart_angle
        #self.loc_angle: float = loc_angle


class Mark:
    def __init__(self, point, cart_angle, loc_angle, box=None):
        self.point = point
        self.cart_angle: float = cart_angle
        self.loc_angle: float = loc_angle
        self.box = box


class Line:
    # pylint: disable=too-few-public-methods
    """
    The simple class representing a 2D line
    """
    def __init__(self, p1=None, p2=None):
        self.p1 = p1
        self.p2 = p2

Point = namedtuple('Point', 'x y')

def rint(number):
    """
    Converts a number to int with rounding
    """
    return int(round(number))


def distance(p1, p2):
    """ Calculates a distance between two points """
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def cross(center: tuple, radius: float, angle: float ):

    angle = radians(angle)
    x = radius * cos(angle) + center[0]
    y = center[1] - radius * sin(angle)

    return x, y


def angle360(p1, p2):
    """
    Calculates an angle in degrees of the line (p1, p2)) in Cartesian coordinates.
    Angles are measured counter-clockwise from the positive X-axis.
    """
    dy = p2[1] - p1[1]
    dx = p2[0] - p1[0]
    theta = atan2(-dy, dx)  # range (-PI, PI]
    theta *= 180 / pi      # rads to degrees, range (-180, 180]
    if theta < 0:
        theta = 360 + theta     # range [0, 360)
    return theta


def cart_angle_to_loc(angle):
    return 270 - angle if angle <= 270 else 270 - angle + 360


def bright_contr(img, brightness=0, contrast=0):
    """
    https://stackoverflow.com/a/50053219
    """

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv.addWeighted(img, alpha_b, img, 0, gamma_b)
    else:
        buf = img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
