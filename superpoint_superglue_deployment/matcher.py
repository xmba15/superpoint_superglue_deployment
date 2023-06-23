from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from superpoint_superglue_deployment.superglue_handler import SuperGlueHandler
from superpoint_superglue_deployment.superpoint_handler import SuperPointHandler

__all__ = ["Matcher"]


class Matcher:
    __DEFAULT_CONFIG: Dict[str, Any] = {
        "superpoint": {
            "descriptor_dim": 256,
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": -1,
            "remove_borders": 4,
            "input_shape": (-1, -1),
        },
        "superglue": {
            "descriptor_dim": 256,
            "weights": "outdoor",
            "keypoint_encoder": [32, 64, 128, 256],
            "GNN_layers": ["self", "cross"] * 9,
            "sinkhorn_iterations": 100,
            "match_threshold": 0.2,
        },
        "use_gpu": False,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._config = self.__DEFAULT_CONFIG.copy()
        if config is not None:
            self._config.update(config)
        self._config["superpoint"].update({"use_gpu": self._config["use_gpu"]})
        self._config["superglue"].update({"use_gpu": self._config["use_gpu"]})
        self._superpoint_handler = SuperPointHandler(self._config["superpoint"])
        self._superglue_handler = SuperGlueHandler(self._config["superglue"])

    def match(
        self,
        query_image: np.ndarray,
        ref_image: np.ndarray,
    ) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], np.ndarray, np.ndarray, List[cv2.DMatch]]:
        """
        Parameters
        ----------
        query_image:
             Single channel 8bit image
        ref_image:
             Single channel 8bit image
        """
        query_pred = self._superpoint_handler.run(query_image)
        ref_pred = self._superpoint_handler.run(ref_image)
        query_kpts, query_descs = self._superpoint_handler.process_prediction(query_pred)
        ref_kpts, ref_descs = self._superpoint_handler.process_prediction(ref_pred)
        return (
            query_kpts,
            ref_kpts,
            query_descs,
            ref_descs,
            self._superglue_handler.match(
                query_pred,
                ref_pred,
                query_image.shape[:2],
                ref_image.shape[:2],
            ),
        )
