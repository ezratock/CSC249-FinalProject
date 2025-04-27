import os
import time
import json
import random
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL = "gemini-2.0-flash"
RPM = 15  # Rate per minute
FRAME_STRIDE = 5  # Only process every 4th frame
SCENE_DIR = Path("bop_dataset/test")
CROPPED_DIR = Path("data/objects/visib_crop")
RESULTS_INPUT_FILE = Path("output/cv_results.json")
RESULTS_OUTPUT_FILE = Path("output/cv_llm_results.json")
PROMPT_TEMPLATE = """Given the following scene image and several reference images of an object, determine if the object is present in the scene.
Respond with ONLY 'yes' or 'no'."""

# --- SET UP MODEL ---
with open("secrets/gemini-api.txt", "r") as f:
    api_key = f.read().strip()
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name=MODEL)

# --- LOAD RESULTS ---
with open(RESULTS_INPUT_FILE, "r") as f:
    results = json.load(f)
RESULTS_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# --- PROCESS ---

def ask_gemini(scene_path, crop_paths, retries=5):
    try:
        images = [Image.open(scene_path)] + [Image.open(crop) for crop in crop_paths]

        for attempt in range(retries):
            try:
                response = model.generate_content([PROMPT_TEMPLATE] + images)
                answer = response.text.strip().lower()
                if "yes" in answer:
                    return 1, None
                elif "no" in answer:
                    return 0, None
                else:
                    return None, f"Unexpected response: {response.text.strip()}"
            except Exception as e:
                if "429" in str(e):
                    print(f"Rate limit hit. Waiting 20 seconds before retrying (attempt {attempt+1}/{retries})...")
                    time.sleep(20)  # Wait extra if rate limited
                    continue  # Retry after sleeping
                else:
                    return None, str(e)

        return None, "Failed after retries."

    except Exception as e:
        return None, str(e)

# --- MAIN LOOP ---

overall_start = time.time()

for scene_id, frames in tqdm(results.items(), desc="Scenes"):
    frame_keys = sorted(frames.keys())
    selected_frame_keys = frame_keys[::FRAME_STRIDE]

    scene_start = time.time()

    for frame_id in tqdm(selected_frame_keys, desc=f"Scene {scene_id}", leave=False):
        scene_path = SCENE_DIR / scene_id / "rgb" / f"{frame_id}.png"

        objects = frames[frame_id]

        for obj_entry in objects:
            obj_name = obj_entry["object"]

            crop_folder = CROPPED_DIR / obj_name
            if not crop_folder.exists():
                obj_entry["llm_prediction"] = None
                continue

            crop_paths = sorted(crop_folder.glob(f"{scene_id}_*.png"))
            if not crop_paths:
                crop_paths = sorted(crop_folder.glob("*.png"))

            if not crop_paths:
                obj_entry["llm_prediction"] = None
                continue

            start_query = time.time()
            llm_pred, error = ask_gemini(scene_path, crop_paths[:3])
            end_query = time.time()

            obj_entry["llm_prediction"] = llm_pred

            if error:
                print(f"Error with {scene_id}-{frame_id}-{obj_name}: {error}")

            # Wait to respect RPM limit
            elapsed = end_query - start_query
            wait_time = max(0, (60 / RPM) - elapsed)
            time.sleep(wait_time)

    scene_end = time.time()
    scene_elapsed = scene_end - scene_start

    with open(RESULTS_OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Scene {scene_id} finished in {scene_elapsed:.2f} seconds (excluding sleep).")

overall_end = time.time()
print(f"\nAll selected frames processed. Total elapsed time (excluding sleep): {overall_end - overall_start:.2f} seconds.")

print(f"\nLLM predictions saved to {RESULTS_OUTPUT_FILE}")
