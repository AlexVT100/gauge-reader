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
        brightness=50,
        contrast=110,
        blur=7,
        mean_sp=11,
        mean_sr=21,
        thresh=80,
        thresh_maxval=255,
        erode_iters=1,
    ),
    needle=SimpleNamespace(
        blur=7,
        thresh=65,
        thresh_maxval=255
    ),
)
