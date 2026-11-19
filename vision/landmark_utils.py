import numpy as np


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
