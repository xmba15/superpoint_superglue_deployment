import numpy as np

__all__ = [
    "assert_single_channel",
]


def assert_single_channel(image: np.ndarray):
    assert len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)
