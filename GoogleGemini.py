import os
import csv
import time
import google.generativeai as genai
from PIL import Image
from pathlib import Path

MODEL = "gemini-2.0-flash"
RPM = 15
DATA_PATH = "dataset/test/"
OUTPUT_PATH = "output/gemini_results.csv"
PROMPT = "How many cars are in this image? Respond with ONLY the number as digits."


# === Step 1: Load API Key ===
with open("secrets/gemini-api.txt", "r") as f:
    api_key = f.read().strip()
genai.configure(api_key=api_key)

# === Step 2: Set up model ===
model = genai.GenerativeModel(model_name=MODEL)

# === Step 3: Loop through test images ===
image_dir = Path(DATA_PATH)
results = []

# === Step 4: Process each image with rate limiting (max 15 RPM) ===
image_files = sorted(image_dir.glob("*.jpg"))

for image_file in image_files[:5]:
    error = None
    gemini_response = ""
    parsed_car_count = None

    try:
        with Image.open(image_file) as img:
            response = model.generate_content([
                PROMPT,
                img
            ])
            gemini_response = response.text.strip()

            digits = [s for s in gemini_response.split() if s.isdigit()]
            if digits:
                parsed_car_count = int(digits[0])

    except Exception as e:
        error = str(e)

    results.append({
        "image": image_file.name,
        "gemini_response": gemini_response,
        "error": error,
        "parsed_car_count": parsed_car_count
    })

    if error is None:
        print(f"{image_file.name[:19]}: {parsed_car_count}")
    else:
        print(f"{image_file.name[:19]}: \n{error}")

    # Respect rate limit
    time.sleep(60/RPM)

# === Step 5: Save results to CSV ===
with open(OUTPUT_PATH, "w", newline="") as csvfile:
    fieldnames = ["image", "gemini_response", "error", "parsed_car_count"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Done. Results saved to {OUTPUT_PATH}")
