import cv2
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BOP_ROOT = Path("bop_dataset/test")
LABELS_FILE = Path("data/labels.json")
MAX_FRAMES = 75
THUMB_W, THUMB_H = 320, 240
COLS = 10

def parse_indices(input_str, total):
    """Parses input like '1 3-5 7' -> set of frame indices"""
    result = set()
    tokens = input_str.strip().split()
    for token in tokens:
        if '-' in token:
            start, end = map(int, token.split('-'))
            result.update(range(start, end + 1))
        elif token.isdigit():
            result.add(int(token))
    return {i for i in result if 0 <= i < total}

# step 1: initialize structure
labels = defaultdict(lambda: defaultdict(list))

for scene_dir in sorted(BOP_ROOT.glob("000*")):
    scene_id = scene_dir.name
    rgb_dir = scene_dir / "rgb"
    frame_paths = sorted(rgb_dir.glob("*.png"))[:MAX_FRAMES]
    frame_ids = [path.stem for path in frame_paths]

# step 1: display frames in a grid numbered 0-74
    thumbs = []
    for idx, path in enumerate(frame_paths):
        img = cv2.imread(str(path))
        img = cv2.resize(img, (THUMB_W, THUMB_H))
        cv2.putText(img, str(idx), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (203, 192, 255), 2)
        thumbs.append(img)

    rows = (len(thumbs) + COLS - 1) // COLS
    grid_rows = []
    for r in range(rows):
        row_imgs = thumbs[r*COLS:(r+1)*COLS]
        if len(row_imgs) < COLS:
            row_imgs += [np.zeros_like(row_imgs[0]) for _ in range(COLS - len(row_imgs))]
        grid_rows.append(np.hstack(row_imgs))
    grid = np.vstack(grid_rows)

    cv2.imshow(f"Scene {scene_id} - Select Object Visibility", grid)
    cv2.waitKey(1)

# step 2: create empty entries
    for frame_id in frame_ids:
        labels[scene_id][frame_id] = []

# step 3: for each known object in scene, prompt user which frames it's NOT in
    known_objects = input(f"\nScene {scene_id}: Enter known objects in scene (space-separated): ").strip().split()

    for obj in known_objects:
        exclude_input = input(f"Which frames is '{obj}' NOT visible? (Press enter if all are visible): ")
        exclude_indices = parse_indices(exclude_input, len(frame_ids))
        for i, frame_id in enumerate(frame_ids):
            if i not in exclude_indices:
                labels[scene_id][frame_id].append(obj)

# step 4: prompt user if there are additional objects, and if so in which frames
    while True:
        extra_obj = input(f"Any extra object to add for scene {scene_id}? (Press enter to continue): ").strip()
        if not extra_obj:
            break
        frame_input = input(f"Which frames contain '{extra_obj}'? (space-separated or ranges): ")
        include_indices = parse_indices(frame_input, len(frame_ids))
        for i in include_indices:
            labels[scene_id][frame_ids[i]].append(extra_obj)

    cv2.destroyAllWindows()

# step 5: save labels
LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(LABELS_FILE, "w") as f:
    json.dump(labels, f, indent=2)

print(f"Complete. Labels saved to {LABELS_FILE}.")
