from vision import (
    distance_to_percentage,
    classify_hand,
    normalize_landmarks,
    create_landmarker,
    initialize_camera,
)
from game import pick_word
from ui import (
    draw_title,
    draw_metrics,
    draw_feedback,
    draw_word_progress,
    draw_hand_guide,
    spawn_confetti,
    draw_word_complete_celebration,
    draw_letter_success,
)

import sys
from pathlib import Path
import random

import cv2
import mediapipe as mp
import numpy as np

WINDOW_NAME = "Sign Language Spelling Game"

# ============================================================
# Load reference signs
# ============================================================


def load_reference_signs(folder="output/landmarks"):
    references = {}

    folder = Path(folder)

    if not folder.exists():
        print(f"Folder not found: {folder}")
        return references

    for file in folder.glob("*.npy"):

        try:
            data = np.load(file, allow_pickle=True)

            if len(data) == 0:
                continue

            hand = np.asarray(data[0], dtype=np.float32)

            if hand.shape != (21, 3):
                print(f"Skipping {file.name}: " f"unexpected shape {hand.shape}")
                continue

            label = file.stem.upper()

            references[label] = normalize_landmarks(hand)

            print(f"Loaded sign: {label}")

        except Exception as e:
            print(f"Could not load {file.name}: {e}")

    return references


# ============================================================
# Main
# ============================================================

references = load_reference_signs("output/landmarks")

if len(references) == 0:
    print("No signs loaded.")
    sys.exit(1)

print()
print("Loaded signs:")
print(list(references.keys()))
print()

available_letters = sorted(
    {label.replace("_FLIPPED", "") for label in references.keys()}
)


target_word = pick_word(available_letters)
if target_word is None:
    print("No word in bank can be spelled with loaded signs.")
    sys.exit(1)

letter_index = 0  # which letter of target_word we're on
target_letter = target_word[letter_index]
correct_frames = 0
celebration_timer = 0
confetti = []
completed_word = ""
letter_feedback_timer = 0
last_completed_index = -1
gold_flash_timer = 0

guide_images = {}

for letter in available_letters:
    guide_path = Path("hand_guides") / f"{letter}_guide.png"

    if guide_path.exists():
        guide_images[letter] = cv2.imread(str(guide_path))

with create_landmarker() as landmarker:

    cap, frame_ms, timestamp = initialize_camera(window_name=WINDOW_NAME)

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        # ----------------------------------------------------
        # MediaPipe detection
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect_for_video(mp_image, timestamp)

        timestamp += frame_ms

        prediction = "---"
        distance = 0.0
        similarity = 0

        # ----------------------------------------------------
        # Hand processing
        # ----------------------------------------------------

        is_correct = False

        if result.hand_landmarks:

            hand_landmarks = result.hand_landmarks[0]

            # Draw skeleton
            for connection in mp.solutions.hands.HAND_CONNECTIONS:

                start_idx, end_idx = connection

                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]

                x1 = int(start.x * w)
                y1 = int(start.y * h)

                x2 = int(end.x * w)
                y2 = int(end.y * h)

                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw keypoints
            for lm in hand_landmarks:

                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            current_landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
            )

            prediction, distance = classify_hand(current_landmarks, references)

            similarity = distance_to_percentage(distance)

            prediction, distance = classify_hand(current_landmarks, references)

            is_correct = prediction == target_letter

            if is_correct:
                correct_frames += 1
            else:
                correct_frames = 0

            # Require ~0.5 seconds of stable recognition
            if correct_frames >= 15:

                letter_feedback_timer = 10
                last_completed_index = letter_index
                gold_flash_timer = 10
                letter_index += 1
                if letter_index >= len(target_word):
                    completed_word = target_word
                    celebration_timer = 45  # ~1.5 seconds

                    confetti = spawn_confetti(
                        w,
                        h,
                        count=250,
                    )

                    print(f"Word complete: {target_word}")

                    target_word = pick_word(available_letters)
                    letter_index = 0
                target_letter = target_word[letter_index]
                correct_frames = 0

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        if gold_flash_timer > 0:
            gold_flash_timer -= 1

        draw_title(frame)

        draw_word_progress(
            frame,
            target_word,
            letter_index,
            last_completed_index,
            gold_flash_timer,
        )

        draw_feedback(
            frame,
            prediction,
            is_correct,
        )

        draw_metrics(
            frame,
            similarity,
            distance,
        )

        draw_hand_guide(
            frame,
            guide_images,
            target_letter,
        )

        if celebration_timer > 0:

            celebration_timer -= 1

            frame, confetti = draw_word_complete_celebration(
                frame,
                confetti,
                completed_word,
                celebration_timer,
            )

        if letter_feedback_timer > 0:

            letter_feedback_timer -= 1

            draw_letter_success(
                frame,
                letter_feedback_timer,
            )

        if letter_feedback_timer > 0:

            letter_feedback_timer -= 1

            scale = 1.6 + 0.15 * letter_feedback_timer

            text = "Good!"

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 6)[0]

            x = (w - text_size[0]) // 2
            y = (h + text_size[1]) // 2

            # glow
            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                scale,
                (255, 255, 255),
                12,
            )

            # green foreground
            cv2.putText(
                frame,
                text,
                (x, y),
                cv2.FONT_HERSHEY_DUPLEX,
                scale,
                (0, 255, 0),
                4,
            )
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
