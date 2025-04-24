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
cols = 10
rows = (len(images) + cols - 1) // cols
thumb_h, thumb_w = 100, 100  # resize to small thumbnails

thumbs = []
for img in images:
    thumb = cv2.resize(img, (thumb_w, thumb_h))
    thumbs.append(thumb)

grid_rows = []
for r in range(rows):
    row_imgs = thumbs[r*cols:(r+1)*cols]
    if len(row_imgs) < cols:
        blanks = [np.zeros_like(row_imgs[0]) for _ in range(cols - len(row_imgs))]
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

# step 3: Copy all crop images that match object IDs into folders
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


# step 4: prompt user to select keyframes
# step 5: delete non-keyframes