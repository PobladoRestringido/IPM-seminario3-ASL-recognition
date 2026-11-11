from pathlib import Path
import cv2
import numpy as np

from tools import extract_hand_landmarks

INPUT_DIR = "hand_references"
OUTPUT_IMG_DIR = "output/images"
OUTPUT_LM_DIR = "output/landmarks"

Path(OUTPUT_IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_LM_DIR).mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for image_path in Path(INPUT_DIR).iterdir():

    if image_path.suffix.lower() not in image_extensions:
        continue

    print(f"Processing {image_path.name}")

    annotated_img, landmarks_list = extract_hand_landmarks(
        str(image_path)
    )

    # Save annotated image
    out_img_path = Path(OUTPUT_IMG_DIR) / image_path.name
    cv2.imwrite(str(out_img_path), annotated_img)

    # Save landmarks
    out_lm_path = Path(OUTPUT_LM_DIR) / f"{image_path.stem}.npy"

    np.save(
        out_lm_path,
        np.array(landmarks_list, dtype=object),
        allow_pickle=True
    )

print("Done.")