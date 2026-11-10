import cv2
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def extract_hand_landmarks(
    image_path,
    model_path="models/hand_landmarker.task",
    num_hands=2
):
    """
    Returns:
        output_image : image with landmarks drawn
        landmarks    : list of (21,3) numpy arrays, one per hand
    """

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=num_hands
    )

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    output_image = image_bgr.copy()

    landmarks_list = []

    h, w = output_image.shape[:2]

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # Store landmarks numerically
            landmarks = np.array([
                [lm.x, lm.y, lm.z]
                for lm in hand_landmarks
            ])

            landmarks_list.append(landmarks)

            # Draw skeleton
            for connection in mp.solutions.hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection

                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]

                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)

                cv2.line(
                    output_image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

            # Draw points
            for lm in hand_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)

                cv2.circle(
                    output_image,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )

    return output_image, landmarks_list