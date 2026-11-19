import sys
from pathlib import Path
import random

import cv2
import mediapipe as mp
import numpy as np

WORD_BANK = [
    "APPLE",
    "BRIDGE",
    "CANDLE",
    "DRAGON",
    "EMBER",
    "FALCON",
    "GARDEN",
    "HOUSE",
    "ISLAND",
    "JUNGLE",
    "KITE",
    "LEMON",
    "MOUSE",
    "NIGHT",
    "OCEAN",
    "PIANO",
    "QUEEN",
    "RIVER",
    "STONE",
    "TIGER",
    "GINGER",
    "BIGGER",
    "DIGGER",
    "JEW",
    "NETANYAHU",
]

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


def pick_word():
    # Keep only words whose letters are all recognizable
    eligible = [w for w in WORD_BANK if all(c in available_letters for c in w)]
    return random.choice(eligible) if eligible else None


target_word = pick_word()
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

    cv2.namedWindow("Sign Language Recognition", cv2.WINDOW_NORMAL)

    cv2.setWindowProperty(
        "Sign Language Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

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

                    for _ in range(250):
                        confetti.append(
                            [
                                random.randint(0, w),
                                random.randint(0, h),
                                random.randint(-15, 15),
                                random.randint(-25, -5),
                                (
                                    random.randint(0, 255),
                                    random.randint(0, 255),
                                    random.randint(0, 255),
                                ),
                            ]
                        )

                    print(f"Word complete: {target_word}")

                    target_word = pick_word()
                    letter_index = 0
                target_letter = target_word[letter_index]
                correct_frames = 0

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        if gold_flash_timer > 0:
            gold_flash_timer -= 1

        cv2.putText(
            frame,
            "Sign Language Prototype",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Show the whole word with the current letter highlighted in brackets
        x = 10
        y = 80

        for i, letter in enumerate(target_word):

            if i == last_completed_index and gold_flash_timer > 0:
                color = (0, 215, 255)  # gold
                display = letter

            elif i < letter_index:
                color = (0, 255, 0)  # green
                display = letter

            elif i == letter_index:
                color = (0, 165, 255)  # current = orange
                display = f"[{letter}]"

            else:
                color = (180, 180, 180)  # future = grey
                display = letter

            cv2.putText(
                frame,
                display,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
            )

            text_size = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]

            x += text_size[0] + 5

        feedback = "Correct!" if is_correct else f"You are showing: {prediction}"

        feedback_color = (0, 255, 0) if is_correct else (0, 0, 255)

        cv2.putText(
            frame, feedback, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, feedback_color, 2
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

        guide = guide_images.get(target_letter)

        if guide is not None:

            guide_height = 180

            aspect_ratio = guide.shape[1] / guide.shape[0]

            guide_width = int(guide_height * aspect_ratio)

            guide_resized = cv2.resize(guide, (guide_width, guide_height))

            y1 = 10
            y2 = y1 + guide_height

            x2 = w - 10
            x1 = x2 - guide_width

            frame[y1:y2, x1:x2] = guide_resized

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # =====================================================
        # EXTREME SUCCESS CELEBRATION
        # =====================================================

        if celebration_timer > 0:

            celebration_timer -= 1

            # rainbow flash overlay
            flash_color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
            )

            overlay = frame.copy()
            overlay[:] = flash_color

            alpha = 0.25 + 0.15 * np.sin(celebration_timer)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # update confetti
            new_confetti = []

            for p in confetti:

                p[0] += p[2]
                p[1] += p[3]

                p[3] += 1  # gravity

                if p[1] < h + 50:

                    cv2.circle(frame, (int(p[0]), int(p[1])), 4, p[4], -1)

                    new_confetti.append(p)

            confetti = new_confetti

            # screen shake
            shake_x = random.randint(-10, 10)
            shake_y = random.randint(-10, 10)

            M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])

            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # gigantic glowing success text
            text = f"{completed_word}!"

            size = 2.5 + 0.5 * np.sin(celebration_timer)

            cv2.putText(
                frame,
                text,
                (40, h // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                size,
                (255, 255, 255),
                12,
            )

            cv2.putText(
                frame,
                "LEGENDARY!",
                (60, h // 2 + 80),
                cv2.FONT_HERSHEY_DUPLEX,
                1.5,
                (0, 255, 255),
                4,
            )

            cv2.putText(
                frame,
                text,
                (40, h // 2),
                cv2.FONT_HERSHEY_DUPLEX,
                size,
                (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
                4,
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

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
