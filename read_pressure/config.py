"""
Settings for the particular setup of the camera/gauge pair
"""

from types import SimpleNamespace

from .util import Ranges

# The gauge and its position on the picture
Config = SimpleNamespace(
    # The gauge position
    # The gauge needle parameters
    dead_min_angle=1.165,       # the angle between the first tick and the needle in the "parked" position
    shaft_displacement=(4, 2),  # the displacement in pixels of the recognized scale center from the shaft center
                                # (the camera distortion or the gauge dial displacement?)

    # Scale
    full_scale_angle=270,
    mark_step=0.1,
    first_mark=0.2,
    max_mark=4.0,

    # Debugging
    text_pos=(0.5, 0.7),  # relatively to the image size

    scale=SimpleNamespace(
        blur=None,
        brightness=20,
        contrast=0,
        lut_min=64,
        lut_max=102,
        mean_sp=11,
        mean_sr=21,
        thresh=110,
        thresh_maxval=200,
        erode_iters=None,
    ),
)

#          Mark weight:  0.1           0.5             1.0
MarkSimilRange = Ranges((0.86, 0.99), (0.91, 0.97),   (0.74, 0.79))
MarkRatioRange = Ranges((0.10, 0.15), (0.072, 0.095), (0.22, 0.24))

NeedleSimilRange = Ranges((0.58, 0.62))
NeedleRatioRange = Ranges((0.19, 0.21))
