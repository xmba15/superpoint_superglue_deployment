import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from superpoint_superglue_deployment.core import assert_single_channel
from superpoint_superglue_deployment.superpoint import SuperPoint

__all__ = ["SuperPointHandler"]


class SuperPointHandler:
    __CACHED_DIR = os.path.join(os.path.expanduser("~"), ".cache/torch/hub/checkpoints")
    __MODEL_WEIGHTS_FILE_NAME = "superpoint_v1.pth"
    __MODEL_WEIGHTS_URL = (
        "https://github.com/xmba15/superpoint_superglue_deployment/releases/download/model_weights/superpoint_v1.pth"
    )

    __DEFAULT_CONFIG: Dict[str, Any] = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "input_shape": (-1, -1),
        "use_gpu": False,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._config = self.__DEFAULT_CONFIG.copy()
        if config is not None:
            self._config.update(config)

        os.makedirs(self.__CACHED_DIR, exist_ok=True)

        if all([e > 0 for e in self._config["input_shape"]]):
            self._validate_input_shape(self._config["input_shape"])

        if self._config["use_gpu"] and not torch.cuda.is_available():
            logger.info("gpu environment is not available, falling back to cpu")
            self._config["use_gpu"] = False
        self._device = torch.device("cuda" if self._config["use_gpu"] else "cpu")

        self._superpoint_engine = SuperPoint(self._config)

        if not os.path.isfile(os.path.join(self.__CACHED_DIR, self.__MODEL_WEIGHTS_FILE_NAME)):
            torch.hub.load_state_dict_from_url(self.__MODEL_WEIGHTS_URL, map_location=lambda storage, loc: storage)
        self._superpoint_engine.load_state_dict(
            torch.load(os.path.join(self.__CACHED_DIR, self.__MODEL_WEIGHTS_FILE_NAME))
        )
        self._superpoint_engine = self._superpoint_engine.eval().to(self._device)
        logger.info(f"loaded superpoint weights {self.__MODEL_WEIGHTS_FILE_NAME}")

    def _validate_input_shape(self, image_shape: Tuple[int, int]):
        assert (
            max(image_shape) >= 160 and max(image_shape) <= 2000
        ), f"input resolution {image_shape} is too small or too large"

    @property
    def device(self):
        return self._device

    def run(self, image: np.ndarray) -> Dict[str, Tuple[torch.Tensor]]:
        """
        Returns
        -------
        Dict[str, Tuple[torch.Tensor]]
            dict data in the following form:
            {
              "keypoints": List[torch.Tensor]  # tensor has shape: num keypoints x 2
              "scores": Tuple[torch.Tensor] # tensor has shape: num keypoints
              "descriptors": List[torch.Tensor] # tensor has shape: 256 x num keypoints
            }
        """
        assert_single_channel(image)
        self._validate_input_shape(image.shape[:2])
        with torch.no_grad():
            pred = self._superpoint_engine({"image": self._to_tensor(image)})
        if all([e > 0 for e in self._config["input_shape"]]):
            pred["keypoints"][0] = torch.mul(
                pred["keypoints"][0],
                torch.from_numpy(np.divide(image.shape[:2][::-1], self._config["input_shape"][::-1])).to(self._device),
            )
        return pred

    def process_prediction(self, pred: Dict[str, torch.Tensor]) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        keypoints_arr = pred["keypoints"][0].cpu().numpy()  # num keypoints x 2
        scores_arr = pred["scores"][0].cpu().numpy()  # num keypoints
        descriptors_arr = pred["descriptors"][0].cpu().numpy()  # 256 x num keypoints
        del pred

        num_keypoints = keypoints_arr.shape[0]
        if num_keypoints == 0:
            return [], np.array([])

        keypoints = []
        for idx in range(num_keypoints):
            keypoint = cv2.KeyPoint()
            keypoint.pt = keypoints_arr[idx]
            keypoint.response = scores_arr[idx]
            keypoints.append(keypoint)
        return keypoints, descriptors_arr.transpose(1, 0)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        pred = self.run(image)
        return self.process_prediction(pred)

    def detect(self, image) -> List[cv2.KeyPoint]:
        return self.detect_and_compute(image)[0]

    def _to_tensor(self, image: np.ndarray):
        if all([e > 0 for e in self._config["input_shape"]]):
            return (
                torch.from_numpy(cv2.resize(image, self._config["input_shape"][::-1]).astype(np.float32) / 255.0)
                .float()[None, None]
                .to(self._device)
            )
        else:
            return torch.from_numpy(image.astype(np.float32) / 255.0).float()[None, None].to(self._device)
