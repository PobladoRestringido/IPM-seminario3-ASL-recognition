import sys
from pathlib import Path
import random

import cv2
import mediapipe as mp
import numpy as np

# ============================================================
# MediaPipe setup
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ============================================================
# Landmark normalization
# ============================================================


def normalize_landmarks(landmarks):
    """
    Normalizes landmarks so comparison is
    less sensitive to position and scale.
    """

    landmarks = landmarks.copy()

    # Wrist becomes origin
    landmarks -= landmarks[0]

    # Use middle MCP joint as scale reference
    scale = np.linalg.norm(landmarks[9])

    if scale > 1e-6:
        landmarks /= scale

    return landmarks


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
# Classification
# ============================================================


def classify_hand(current_landmarks, references):
    current = normalize_landmarks(current_landmarks)

    best_label = None
    best_distance = float("inf")

    for label, reference in references.items():

        distance = np.linalg.norm(current - reference)

        if distance < best_distance:
            best_distance = distance
            best_label = label

    if best_label is not None:
        best_label = best_label.replace("_FLIPPED", "")

    return best_label, best_distance


def distance_to_similarity(distance):
    """
    Converts distance into an easy-to-read
    percentage for display.

    This is purely heuristic.
    """

    similarity = max(0, min(100, int(100 - distance * 20)))

    return similarity


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

target_letter = random.choice(available_letters)
correct_frames = 0

with HandLandmarker.create_from_options(options) as landmarker:

    # --------------------------------------------------------
    # WSL-friendly camera setup from your working project
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        fps = 30

    frame_ms = int(1000 / fps)

    timestamp = 0

    # Camera warmup
    for _ in range(10):
        cap.read()

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

            similarity = distance_to_similarity(distance)

            prediction, distance = classify_hand(
                current_landmarks,
                references
            )

            is_correct = prediction == target_letter

            if is_correct:
                correct_frames += 1
            else:
                correct_frames = 0

            # Require ~0.5 seconds of stable recognition
            if correct_frames >= 15:
                target_letter = random.choice(available_letters)
                correct_frames = 0

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Sign Language Prototype",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Show: {target_letter}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 165, 255),    # Orange
            2
        )

        feedback = (
            "Correct!"
            if is_correct
            else f"You are showing: {prediction}"
        )

        feedback_color = (
            (0, 255, 0)
            if is_correct
            else (0, 0, 255)
        )

        cv2.putText(
            frame,
            feedback,
            (10, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            feedback_color,
            2
        )

        cv2.putText(
            frame,
            f"Similarity: {similarity}%",
            (10, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Distance: {distance:.3f}",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
