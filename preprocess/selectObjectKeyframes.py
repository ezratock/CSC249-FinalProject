import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

CROPPED_DIR = Path("data/cropped_objs")
DEST_DIR = Path("data/objects")
CROP_TYPES = ["visib_crop", "full_crop", "hybrid_crop"]
MAX_SCENES = 12
THUMB_H, THUMB_W = 200, 200
THUMB_COLS = 20

# step 1: load one visib_crop image per object from the first frame of each scene
images = []
image_paths = []
obj_ids = []
scene_ids = []
scene_dirs = sorted(CROPPED_DIR.glob("*/"))[:MAX_SCENES]

for scene_dir in scene_dirs:
    visib_dir = scene_dir / "visib_crop"
    frame_to_objs = defaultdict(list)

    for img_path in visib_dir.glob("*.png"):
        frame_id, obj_id = img_path.name.split("_")
        frame_to_objs[frame_id].append(img_path)

    if frame_to_objs:
        first_frame = sorted(frame_to_objs.keys())[0]
        for path in frame_to_objs[first_frame]:
            img = cv2.imread(str(path))
            images.append(img)
            image_paths.append(path)
            obj_ids.append(path.stem.split("_")[1])
            scene_ids.append(scene_dir.name)

# show all objects in a grid
rows = (len(images) + THUMB_COLS - 1) // THUMB_COLS

thumbs = []
for img in images:
    thumb = cv2.resize(img, (THUMB_W, THUMB_H))
    thumbs.append(thumb)

grid_rows = []
for r in range(rows):
    row_imgs = thumbs[r * THUMB_COLS:(r + 1) * THUMB_COLS]
    if len(row_imgs) < THUMB_COLS:
        blanks = [np.zeros_like(row_imgs[0]) for _ in range(THUMB_COLS - len(row_imgs))]
        row_imgs += blanks
    row = np.hstack(row_imgs)
    grid_rows.append(row)

grid = np.vstack(grid_rows)

cv2.imshow("All Object Crops (visib_crop)", grid)
cv2.waitKey(1)

# step 2: prompt user to name all objects
# names = input("Enter names for each object (space-separated, in order shown): ").strip().split()
names = "clip mug clip StarKist MasterChef spam bowl StarKist sugar drill banana mustard cheezits soup scissors soup SoftScrub sugar MasterChef pitcher StarKist mustard bowl spam soup clip SoftScrub drill sugar cheezits SoftScrub soup MasterChef sugar block mug pitcher drill banana MasterChef brick marker clip SoftScrub soup sugar chocolate pitcher jello drill marker StarKist spam soup cheezits"
names = names.strip().split()
cv2.destroyAllWindows()
assert len(names) == len(images), f"Expected {len(images)} names, got {len(names)}"

if DEST_DIR.exists(): # clear data/objects/ if it exists
    shutil.rmtree(DEST_DIR)
DEST_DIR.mkdir(parents=True, exist_ok=True)

# step 3: copy all crop images that match object IDs into folders
obj_name_map = { (scene_id, obj_id): name for scene_id, obj_id, name in zip(scene_ids, obj_ids, names) }
# loop through all crop types and copy matching obj_id files
for crop_type in CROP_TYPES:
    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        crop_dir = scene_dir / crop_type
        for img_path in crop_dir.glob("*.png"):
            frame_id, obj_id = img_path.stem.split("_")
            key = (scene_name, obj_id)
            if key in obj_name_map:
                name = obj_name_map[key]
                dest_dir = DEST_DIR / crop_type / name
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / f"{scene_name}_{img_path.name}"
                shutil.copy2(img_path, dest_path)

print(f"\nSelect keyframes for each object...")
# step 4: for each object, display all images labeled for selection
for obj_folder in sorted((DEST_DIR / "visib_crop").iterdir()):
    obj_name = obj_folder.name
    img_paths = sorted(obj_folder.glob("*.png"))

    if not img_paths:
        continue

    thumbs = []
    for idx, path in enumerate(img_paths):
        img = cv2.imread(str(path))
        img = cv2.resize(img, (THUMB_W, THUMB_H))
        cv2.putText(img, str(idx), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (203, 192, 255), 2)
        thumbs.append(img)

    # show all objects in a grid
    rows = (len(thumbs) + THUMB_COLS - 1) // THUMB_COLS
    grid_rows = []
    for r in range(rows):
        row_imgs = thumbs[r*THUMB_COLS:(r+1)*THUMB_COLS]
        if len(row_imgs) < THUMB_COLS:
            blanks = [np.zeros_like(row_imgs[0]) for _ in range(THUMB_COLS - len(row_imgs))]
            row_imgs += blanks
        grid_rows.append(np.hstack(row_imgs))
    grid = np.vstack(grid_rows)

    cv2.imshow(f"{obj_name} (select keyframes)", grid)
    cv2.waitKey(1)

# step 5: prompt user to select keyframes
    sel = input(f"Enter keyframe indices (space-separated) for '{obj_name}': ").strip().split()
    cv2.destroyAllWindows()

    key_indices = {int(s) for s in sel}
    key_filenames = {img_paths[i].name for i in key_indices if 0 <= i < len(img_paths)}

# step 6: delete non-keyframes for all crop types
    for crop_type in CROP_TYPES:
        crop_folder = DEST_DIR / crop_type / obj_name
        for img_path in crop_folder.glob("*.png"):
            if img_path.name not in key_filenames:
                img_path.unlink()

print(f"Complete. Cropped object keyframes in {DEST_DIR}.")