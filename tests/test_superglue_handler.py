import numpy as np
import pytest

from superpoint_superglue_deployment import SuperGlueHandler, SuperPointHandler


def test_initialization_success():
    SuperGlueHandler()


def test_initialization_failure():
    with pytest.raises(AssertionError):
        SuperGlueHandler({"weights": "unknown"})


def test_inference():
    superpoint_handler = SuperPointHandler(
        {
            "use_gpu": True,
            "input_shape": (-1, -1),
            "keypoint_threshold": 0.001,
        }
    )
    query_image = np.random.rand(300, 300) * 255
    query_image = query_image.astype(np.uint8)
    ref_image = query_image.astype(np.uint8)
    ref_image = ref_image.astype(np.uint8)

    query_pred = superpoint_handler.run(query_image)
    ref_pred = superpoint_handler.run(ref_image)

    superglue_handler = SuperGlueHandler(
        {
            "use_gpu": False,
        }
    )
    superglue_handler.match(query_pred, ref_pred, query_image.shape[:2], ref_image.shape[:2])
