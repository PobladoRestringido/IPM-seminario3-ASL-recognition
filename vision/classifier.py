from .landmark_utils import normalize_landmarks
import numpy as np


def distance_to_percentage(distance):
    """
    Converts distance into an easy-to-read
    percentage for display.

    This is purely heuristic.
    """

    similarity = max(0, min(100, int(100 - distance * 20)))

    return similarity


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
