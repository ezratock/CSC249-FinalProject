import cv2
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

RESULTS_FILE = Path("output/results.json")
CROPPED_DIR = Path("data/objects/hybrid_crop")
SCENE_DIR = Path("bop_dataset/test")
OUTPUT_FILE = Path("output/cv_results.json")
THRESHOLD_INLIERS = 10  # number of inliers needed to predict object presence
RANSAC_THRESHOLD = 5    # pixel threshold for inlier check
MAX_ITERATIONS = 1000   # max iterations for RANSAC

def computeHomographyMatrix(fourcorrs):
    A = np.zeros((8, 9))
    for i in range(4):
        x, y = fourcorrs[i][0]
        xp, yp = fourcorrs[i][1]
        A[i*2] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp]
        A[i*2 + 1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp]
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = H / H[2, 2]
    return H

def getInliers(corrList, h, threshold=5):
    inliers = []
    for corr in corrList:
        p = np.array([corr[0][0], corr[0][1], 1])
        p_prime = np.array([corr[1][0], corr[1][1], 1])
        transformed = h @ p
        transformed /= transformed[2]
        distance = np.linalg.norm(transformed[:2] - p_prime[:2])
        if distance < threshold:
            inliers.append(corr)
    return inliers

def ransac(corrList, thresh=0.5, max_iterations=1000):
    best_H = None
    best_inliers = []

    num_matches = len(corrList)
    if num_matches < 4:
        return None, []

    for _ in range(max_iterations):
        four_corrs = random.sample(corrList, 4)
        H = computeHomographyMatrix(four_corrs)
        inliers = getInliers(corrList, H, threshold=RANSAC_THRESHOLD)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    return best_H, best_inliers

# CV pipeline
with open(RESULTS_FILE, "r") as f:
    results = json.load(f)

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
print("Models created.")

def match_object_to_scene(crop_paths, scene_path):
    best_inliers_count = 0

    scene_img = cv2.imread(str(scene_path), cv2.IMREAD_GRAYSCALE)
    if scene_img is None:
        return 0

    kp_scene, des_scene = sift.detectAndCompute(scene_img, None)
    if des_scene is None:
        return 0

    for crop_path in crop_paths:
        crop_img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
        if crop_img is None:
            continue

        kp_crop, des_crop = sift.detectAndCompute(crop_img, None)
        if des_crop is None:
            continue

        matches = bf.knnMatch(des_crop, des_scene, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append((kp_crop[m.queryIdx].pt, kp_scene[m.trainIdx].pt))

        if len(good_matches) < 4:
            continue

        _, inliers = ransac(good_matches, thresh=0.5, max_iterations=MAX_ITERATIONS)
        inliers_count = len(inliers)

        best_inliers_count = max(best_inliers_count, inliers_count)

    return best_inliers_count

# Run predictions
for scene_id, frames in tqdm(results.items(), desc="Scenes"):
    for frame_id, objects in tqdm(frames.items(), desc=f"Scene {scene_id}", leave=False):
        scene_path = SCENE_DIR / scene_id / "rgb" / f"{frame_id}.png"

        for obj_entry in objects:
            obj_name = obj_entry["object"]

            crop_folder = CROPPED_DIR / obj_name
            if not crop_folder.exists():
                obj_entry["cv_prediction"] = 0
                continue

            crop_paths = sorted(crop_folder.glob(f"{scene_id}_*.png"))

            if not crop_paths:
                crop_paths = sorted(crop_folder.glob("*.png"))  # fallback: any crop

            if not crop_paths:
                obj_entry["cv_prediction"] = 0
                continue

            best_inliers = match_object_to_scene(crop_paths, scene_path)
            obj_entry["cv_prediction"] = 1 if best_inliers >= THRESHOLD_INLIERS else 0

# Save results
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"SIFT + RANSAC predictions saved to {OUTPUT_FILE}")
