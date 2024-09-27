import cv2 as cv
from math import atan2, pi, sqrt, sin, cos, radians, isclose
from collections import namedtuple

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


Point = namedtuple('Point', 'x y')


class Line:
    # pylint: disable=too-few-public-methods
    """
    The simple class representing a 2D line
    """
    def __init__(self, p1=None, p2=None):
        self.p1 = p1
        self.p2 = p2


def rint(number):
    """
    Converts a number to int with rounding
    """
    return int(round(number))


def midpoint(p1, p2):
    """ Calculates a miidle point of line (p1, p2) """
    return Point(x=(p1[0] + p2[0]) / 2, y=(p1[1] + p2[1]) / 2)


def distance(p1, p2):
    """ Calculates a distance between two points """
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

# def angle3(c: Point, p1: Point, p2: Point) -> float:
#     # https://pythonadventures.wordpress.com/2019/12/10/angle-between-two-lines/
#     ang = math.degrees(math.atan2(p2[1]-c[1], p2[0]-c[0]) - math.atan2(p1[1]-c[1], p1[0]-c[0]))
#     return ang + 360 if ang < 0 else ang


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


def draw_line_polar(img, length, angle, point: Point, color: tuple, thickness: int, line_type: int = 0):
    # pylint: disable=too-many-arguments
    """
    Draws a line defined in polar coordinates
    """
    x, y = cv.polarToCart(length, angle=angle, x=point.x, y=point.y, angleInDegrees=True)

    p2 = Point(x=int(round(x[0][0])), y=int(round(y[0][0])))
    point.to_int()
    p2 = p2.x + point.x, point.y - p2.y

    return cv.line(img, (point.x, point.y), p2, color, thickness, line_type)


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
