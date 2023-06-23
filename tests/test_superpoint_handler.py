import numpy as np
import pytest

from superpoint_superglue_deployment import SuperPointHandler


def test_initialization_success():
    SuperPointHandler(
        {
            "use_gpu": False,
            "input_shape": (-1, -1),
        }
    )


def test_initialization_failure():
    with pytest.raises(AssertionError):
        SuperPointHandler(
            {
                "use_gpu": False,
                "input_shape": (100, 100),
            }
        )

    with pytest.raises(AssertionError):
        SuperPointHandler(
            {
                "use_gpu": False,
                "input_shape": (3000, 100),
            }
        )


def test_inference():
    superpoint_handler = SuperPointHandler(
        {
            "use_gpu": True,
            "input_shape": (-1, -1),
            "keypoint_threshold": 0.001,
        }
    )
    image = np.random.rand(300, 300) * 255
    image = image.astype(np.uint8)
    keypoints, _ = superpoint_handler.detect_and_compute(image)
    assert len(keypoints) > 0


def test_inference_failure():
    superpoint_handler = SuperPointHandler(
        {
            "use_gpu": True,
            "input_shape": (-1, -1),
            "keypoint_threshold": 0.001,
        }
    )
    image = np.random.rand(300, 300, 3) * 255
    image = image.astype(np.uint8)
    with pytest.raises(AssertionError):
        superpoint_handler.detect_and_compute(image)
