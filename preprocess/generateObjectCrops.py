import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


BOP_ROOT = Path("bop_dataset")
TEST_SCENES = BOP_ROOT / "test"
DEST_ROOT = Path("data") / "cropped_objs"

# gray occlusion fill best for SIFT to avoid hallucinating features but retaining object shape?
GRAY = [128, 128, 128]

def crop_and_save(rgb_img, mask, save_path):
    x, y, w, h = cv2.boundingRect(mask)
    crop = rgb_img[y:y+h, x:x+w]
    mask_crop = mask[y:y+h, x:x+w]
    crop = cv2.bitwise_and(crop, crop, mask=mask_crop)
    cv2.imwrite(str(save_path), crop)

for scene_dir in sorted(TEST_SCENES.glob("000*")):
    scene_id = scene_dir.name
    rgb_dir = scene_dir / "rgb"
    mask_dir = scene_dir / "mask"
    visib_dir = scene_dir / "mask_visib"

    dest_full = DEST_ROOT / scene_id / "full_crop"
    dest_visib = DEST_ROOT / scene_id / "visib_crop"
    dest_hybrid = DEST_ROOT / scene_id / "hybrid_crop"
    dest_full.mkdir(parents=True, exist_ok=True)
    dest_visib.mkdir(parents=True, exist_ok=True)
    dest_hybrid.mkdir(parents=True, exist_ok=True)

    for mask_path in tqdm(sorted(mask_dir.glob("*.png")), desc=f"Scene {scene_id}"):
        fname = mask_path.name
        frame_id, obj_id = fname.replace(".png", "").split("_")
        rgb_path = rgb_dir / f"{frame_id}.png"
        visib_path = visib_dir / fname

        if not rgb_path.exists() or not visib_path.exists():
            continue

        # load data
        rgb = cv2.imread(str(rgb_path))
        mask_full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask_visib = cv2.imread(str(visib_path), cv2.IMREAD_GRAYSCALE)

        # full mask crop
        full_save = dest_full / fname
        crop_and_save(rgb, mask_full, full_save)

        # visible-only mask crop
        visib_save = dest_visib / fname
        crop_and_save(rgb, mask_visib, visib_save)

        # hybrid crop
        occluded_mask = cv2.bitwise_and(mask_full, cv2.bitwise_not(mask_visib))
        gray_fill = np.full(rgb.shape, GRAY, dtype=np.uint8)
        hybrid = np.where(occluded_mask[:, :, None] > 0, gray_fill, rgb)
        full_mask = cv2.bitwise_or(mask_visib, occluded_mask)
        x, y, w, h = cv2.boundingRect(full_mask)
        hybrid_crop = hybrid[y:y+h, x:x+w]
        final_mask = full_mask[y:y+h, x:x+w]
        hybrid_crop = cv2.bitwise_and(hybrid_crop, hybrid_crop, mask=final_mask)

        hybrid_save = dest_hybrid / fname
        cv2.imwrite(str(hybrid_save), hybrid_crop)
        # import pdb; pdb.set_trace()
