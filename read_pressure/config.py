"""
Settings for the particular setup of the camera/gauge pair
"""

from types import SimpleNamespace

# The gauge and its position on the picture
Config = SimpleNamespace(
    # The gauge position
    # The gauge needle parameters
    dead_min_angle=1.165,       # the angle between the first tick and the needle in the "parked" position
    needle_area=0.072,          # the proportion of the needle area of the scale circle area
    needle_area_tol=0.05,       # tolerance (percent of the needle area)

    # Scale
    full_scale_angle=270,
    mark_step=0.1,
    first_mark=0.2,
    max_mark=4.0,

    # Debugging
    text_pos=(0.5, 0.7),  # relatively to the image size

    scale=SimpleNamespace(
        blur=3,
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

MarkRatios = {
    None: {     # No erode
        '0.0': [0.05, 0.25],
        #'0.1': [0.13, 0.23],
        #'0.5': [0.09, 0.12],
        #'1.0': [0.25, 0.27],
    },
    1: {        # Erode 1 iteration
        '0.1': [0.12, 0.22],
        '0.5': [0.08, 0.11],
        '1.0': [0.25, 0.4],
    },
}