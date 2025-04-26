import json
import random
from pathlib import Path
from collections import defaultdict

LABELS_FILE = Path("data/labels.json")
OBJECTS_DIR = Path("data/objects/hybrid_crop")
OUTPUT_FILE = Path("output/results.json")
SEED = 42

with open(LABELS_FILE, "r") as f:
    labels = json.load(f)
all_objects = sorted([p.name for p in OBJECTS_DIR.iterdir() if p.is_dir()])

# initialize the structure
random.seed(SEED)
initial_results = defaultdict(lambda: defaultdict(list))

for scene_id, frames in labels.items():
    for frame_id, present_objects in frames.items():
        present_objects = sorted(set(present_objects)) # ensure no duplicates

        absent_objects = [obj for obj in all_objects if obj not in present_objects]
        sampled_absent = random.sample(absent_objects, k=len(present_objects))

        # add present objects
        for obj in present_objects:
            initial_results[scene_id][frame_id].append({
                "object": obj,
                "true_label": 1
            })

        # add absent objects
        for obj in sampled_absent:
            initial_results[scene_id][frame_id].append({
                "object": obj,
                "true_label": 0
            })

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

if OUTPUT_FILE.exists():
    print(f"{OUTPUT_FILE} already exists.")
else:
    with open(OUTPUT_FILE, "w") as f:
        json.dump(initial_results, f, indent=2)
    print(f"Initialized clean results saved to {OUTPUT_FILE}")
