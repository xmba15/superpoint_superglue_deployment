<p align="center">
<a href="https://github.com/xmba15/superpoint_superglue_deployment/actions/workflows/build.yml" target="_blank">
  <img src="https://github.com/xmba15/superpoint_superglue_deployment/actions/workflows/build.yml/badge.svg" alt="Build Status">
</a>
</p>

# 📝 simple library to make life easy when deploying superpoint, superglue models

---

## :gear: Installation

---

```bash
pip install superpoint_superglue_deployment
```

## :tada: TODO

---

- [x] interface to deploy superpoint, superglue
- [x] testing on real data

## :running: How to Run

---

### Basic usage

```python
import cv2
import numpy as np
from loguru import logger

from superpoint_superglue_deployment import Matcher


def main():
    query_image = cv2.imread("./data/images/one_pillar_pagoda_1.jpg")
    ref_image = cv2.imread("./data/images/one_pillar_pagoda_2.jpg")

    query_gray = cv2.imread("./data/images/one_pillar_pagoda_1.jpg", 0)
    ref_gray = cv2.imread("./data/images/one_pillar_pagoda_2.jpg", 0)

    superglue_matcher = Matcher(
        {
            "superpoint": {
                "input_shape": (-1, -1),
                "keypoint_threshold": 0.003,
            },
            "superglue": {
                "match_threshold": 0.5,
            },
            "use_gpu": True,
        }
    )
    query_kpts, ref_kpts, _, _, matches = superglue_matcher.match(query_gray, ref_gray)
    M, mask = cv2.findHomography(
        np.float64([query_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2),
        np.float64([ref_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2),
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=5.0,
        maxIters=10000,
        confidence=0.95,
    )
    logger.info(f"number of inliers: {mask.sum()}")
    matches = np.array(matches)[np.all(mask > 0, axis=1)]
    matches = sorted(matches, key=lambda match: match.distance)
    matched_image = cv2.drawMatches(
        query_image,
        query_kpts,
        ref_image,
        ref_kpts,
        matches[:50],
        None,
        flags=2,
    )
    cv2.imwrite("matched_image.jpg", matched_image)


if __name__ == "__main__":
    main()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/xmba15/superpoint_superglue_deployment/master/docs/images/matched_image.jpg" alt="matched image sample">
</p>

- [Notebook with detailed sample code for SuperPoint](notebooks/demo_superpoint.ipynb)
- [Notebook with detailed sample code for SuperGlue](notebooks/demo_superglue.ipynb)

## 🎛 Development environment

---

```bash
mamba env create --file environment.yml
mamba activate superpoint_superglue_deployment
```

## :gem: References

---

- [SuperPoint: Self-Supervised Interest Point Detection and Description.](https://github.com/rpautrat/SuperPoint)
- [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://github.com/magicleap/SuperGluePretrainedNetwork)
