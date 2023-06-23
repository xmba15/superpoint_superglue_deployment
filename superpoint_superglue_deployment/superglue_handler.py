import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger

from superpoint_superglue_deployment.superglue import SuperGlue

__all__ = ["SuperGlueHandler"]


class SuperGlueHandler:
    __CACHED_DIR = os.path.join(os.path.expanduser("~"), ".cache/torch/hub/checkpoints")
    __MODEL_WEIGHTS_DICT: Dict[str, Any] = {
        "indoor": {
            "name": "superglue_indoor.pth",
            "url": "https://github.com/xmba15/superpoint_superglue_deployment/releases/download/model_weights/superglue_indoor.pth",  # noqa: E501
        },
        "outdoor": {
            "name": "superglue_outdoor.pth",
            "url": "https://github.com/xmba15/superpoint_superglue_deployment/releases/download/model_weights/superglue_outdoor.pth",  # noqa: E501
        },
    }
    __MODEL_WEIGHTS_OUTDOOR_FILE_NAME = "superglue_outdoor.pth"

    __DEFAULT_CONFIG: Dict[str, Any] = {
        "descriptor_dim": 256,
        "weights": "outdoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 100,
        "match_threshold": 0.2,
        "use_gpu": False,
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._config = self.__DEFAULT_CONFIG.copy()
        if config is not None:
            self._config.update(config)

        assert self._config["weights"] in self.__MODEL_WEIGHTS_DICT

        os.makedirs(self.__CACHED_DIR, exist_ok=True)

        if self._config["use_gpu"] and not torch.cuda.is_available():
            logger.info("gpu environment is not available, falling back to cpu")
            self._config["use_gpu"] = False
        self._device = torch.device("cuda" if self._config["use_gpu"] else "cpu")

        self._superglue_engine = SuperGlue(self._config)

        if not os.path.isfile(
            os.path.join(self.__CACHED_DIR, self.__MODEL_WEIGHTS_DICT[self._config["weights"]]["name"])
        ):
            torch.hub.load_state_dict_from_url(
                self.__MODEL_WEIGHTS_DICT[self._config["weights"]]["url"], map_location=lambda storage, loc: storage
            )
        self._superglue_engine.load_state_dict(
            torch.load(os.path.join(self.__CACHED_DIR, self.__MODEL_WEIGHTS_DICT[self._config["weights"]]["name"]))
        )
        self._superglue_engine = self._superglue_engine.eval().to(self._device)
        logger.info(f"loaded superglue weights {self.__MODEL_WEIGHTS_DICT[self._config['weights']]['name']}")

    @property
    def device(self):
        return self._device

    def run(
        self,
        query_pred: Dict[str, torch.Tensor],
        ref_pred: Dict[str, torch.Tensor],
        query_shape: Tuple[int, int],
        ref_shape: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        query_pred
             dict data in the following form
             {
              "keypoints": List[torch.Tensor]  # tensor has shape: num keypoints x 2
              "descriptors": List[torch.Tensor] # tensor has shape: 256 x num keypoints
              }
        ref_pred
             dict data in the same form as query_pred's
        """
        data_dict: Dict[str, Any] = dict()
        data_dict = {**data_dict, **{k + "0": v for k, v in query_pred.items()}}
        data_dict = {**data_dict, **{k + "1": v for k, v in ref_pred.items()}}
        for k in data_dict:
            if isinstance(data_dict[k], (list, tuple)):
                data_dict[k] = torch.stack(data_dict[k])
        del query_pred, ref_pred

        for k in data_dict:
            if isinstance(data_dict[k], torch.Tensor) and data_dict[k].device.type != self._device.type:
                data_dict[k] = data_dict[k].to(self._device)

        data_dict["image0_shape"] = query_shape
        data_dict["image1_shape"] = ref_shape

        with torch.no_grad():
            return self._superglue_engine(data_dict)

    def match(
        self,
        query_pred: Dict[str, torch.Tensor],
        ref_pred: Dict[str, torch.Tensor],
        query_shape: Tuple[int, int],
        ref_shape: Tuple[int, int],
    ) -> List[cv2.DMatch]:
        pred = self.run(
            query_pred,
            ref_pred,
            query_shape,
            ref_shape,
        )
        query_indices = pred["matches0"].cpu().numpy().squeeze(0)
        ref_indices = pred["matches1"].cpu().numpy().squeeze(0)
        query_matching_scores = pred["matching_scores0"].cpu().numpy().squeeze(0)

        del pred
        matched_query_indices = np.where(query_indices > -1)[0]
        matched_ref_indices = np.where(ref_indices > -1)[0]
        matches = [
            cv2.DMatch(
                _distance=1 - query_matching_scores[matched_query_idx],
                _queryIdx=matched_query_idx,
                _trainIdx=matched_ref_idx,
            )
            for matched_query_idx, matched_ref_idx in zip(matched_query_indices, matched_ref_indices)
        ]
        return matches
