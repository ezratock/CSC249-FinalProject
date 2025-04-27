import json
from pathlib import Path
from collections import defaultdict

RESULTS_FILE = Path("output/cv_llm_results.json")

with open(RESULTS_FILE, "r") as f:
    results = json.load(f)

prediction_fields = set()

for scene_data in results.values():
    for frame_data in scene_data.values():
        for obj in frame_data:
            for key in obj.keys():
                if key.endswith("_prediction"):
                    prediction_fields.add(key)

prediction_fields = sorted(list(prediction_fields))  # e.g., ["cv_prediction", "llm_prediction"]
query_times = {}

print("\nEnter average query times (seconds per object) for each model:")

for model_field in prediction_fields:
    while True:
        try:
            time_input = float(input(f"  {model_field}: "))
            query_times[model_field] = time_input
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

stats = defaultdict(lambda: defaultdict(int))
scenes_tested = defaultdict(set)
frames_tested = defaultdict(lambda: defaultdict(set))  # model -> scene -> frames

for scene_id, frames in results.items():
    for frame_id, frame_objects in frames.items():
        for obj_entry in frame_objects:
            true_label = obj_entry["true_label"]

            for model_field in prediction_fields:
                prediction = obj_entry.get(model_field)
                if prediction is None:
                    continue  # skip if not predicted yet

                scenes_tested[model_field].add(scene_id)
                frames_tested[model_field][scene_id].add(frame_id)

                if prediction == 1 and true_label == 1:
                    stats[model_field]["TP"] += 1
                elif prediction == 0 and true_label == 0:
                    stats[model_field]["TN"] += 1
                elif prediction == 1 and true_label == 0:
                    stats[model_field]["FP"] += 1
                elif prediction == 0 and true_label == 1:
                    stats[model_field]["FN"] += 1


print("\n\nFINAL RESULTS SUMMARY\n")

for model_field in prediction_fields:
    tp = stats[model_field]["TP"]
    tn = stats[model_field]["TN"]
    fp = stats[model_field]["FP"]
    fn = stats[model_field]["FN"]

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0

    n_scenes = len(scenes_tested[model_field])
    n_frames = sum(len(frames) for frames in frames_tested[model_field].values())
    avg_frames_per_scene = n_frames / n_scenes if n_scenes else 0

    avg_query_time = query_times.get(model_field, 0)

    print(f"==============================")
    print(f" Model: {model_field}")
    print(f"==============================")
    print(f"  True Positives (TP): {tp}")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  Scenes tested: {n_scenes}")
    print(f"  Frames tested per scene: {avg_frames_per_scene}")
    print(f"  Estimated total query time: {avg_query_time:.2f} seconds")
    print("")

print("Finished analysis.")
