import cv2
import numpy as np
from loguru import logger

from superpoint_superglue_deployment import Matcher


def get_args():
    import argparse

    parser = argparse.ArgumentParser("test matching two images")
    parser.add_argument("--query_path", "-q", type=str, required=True, help="path to query image")
    parser.add_argument("--ref_path", "-r", type=str, required=True, help="path to reference image")
    parser.add_argument("--use_gpu", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()

    query_image = cv2.imread(args.query_path)
    ref_image = cv2.imread(args.ref_path)

    query_gray = cv2.imread(args.query_path, 0)
    ref_gray = cv2.imread(args.ref_path, 0)

    superglue_matcher = Matcher(
        {
            "superpoint": {
                "input_shape": (-1, -1),
                "keypoint_threshold": 0.005,
            },
            "superglue": {
                "match_threshold": 0.2,
            },
            "use_gpu": args.use_gpu,
        }
    )
    query_kpts, ref_kpts, _, _, matches = superglue_matcher.match(query_gray, ref_gray)
    logger.info(f"number of matches by superpoint+superglue: {len(matches)}")
    _, mask = cv2.findHomography(
        np.array([query_kpts[m.queryIdx].pt for m in matches], dtype=np.float64).reshape(-1, 1, 2),
        np.array([ref_kpts[m.trainIdx].pt for m in matches], dtype=np.float64).reshape(-1, 1, 2),
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
        matches[:100],
        None,
        flags=2,
    )
    cv2.imwrite("matched_image.jpg", matched_image)
    cv2.imshow("matched_image", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
