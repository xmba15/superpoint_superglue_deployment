import numpy as np

from superpoint_superglue_deployment import Matcher


def test_inference():
    query_image = np.random.rand(300, 300) * 255
    query_image = query_image.astype(np.uint8)
    ref_image = query_image.astype(np.uint8)
    ref_image = ref_image.astype(np.uint8)
    matcher = Matcher()
    matcher.match(query_image, ref_image)
