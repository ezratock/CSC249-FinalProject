import json
from collections import defaultdict

# Step 1: Load the COCO annotations
with open("dataset/test/_annotations.coco.json", "r") as f:
    coco_data = json.load(f)

# Step 2: Map image_id -> filename
id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

# Step 3: Count occupied spaces per image
true_car_counts = defaultdict(int)

for ann in coco_data["annotations"]:
    # if ann["category_id"] == 2:  # 2 = space-occupied
        image_id = ann["image_id"]
        filename = id_to_filename[image_id]
        true_car_counts[filename] += 1

print(true_car_counts)