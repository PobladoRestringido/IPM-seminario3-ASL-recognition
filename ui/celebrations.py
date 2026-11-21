import cv2
import numpy as np
import random


def spawn_confetti(
    width,
    height,
    count=250,
):
    confetti = []

    for _ in range(count):
        confetti.append(
            [
                random.randint(0, width),
                random.randint(0, height),
                random.randint(-15, 15),
                random.randint(-25, -5),
                (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                ),
            ]
        )

    return confetti


def draw_word_complete_celebration(
    frame,
    confetti,
    completed_word,
    celebration_timer,
):
    h, w, _ = frame.shape

    flash_color = (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255),
    )

    overlay = frame.copy()
    overlay[:] = flash_color

    alpha = 0.25 + 0.15 * np.sin(
        celebration_timer
    )

    frame = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
    )

    new_confetti = []

    for p in confetti:

        p[0] += p[2]
        p[1] += p[3]

        p[3] += 1

        if p[1] < h + 50:
            cv2.circle(
                frame,
                (int(p[0]), int(p[1])),
                4,
                p[4],
                -1,
            )
            new_confetti.append(p)

    confetti = new_confetti

    shake_x = random.randint(-10, 10)
    shake_y = random.randint(-10, 10)

    M = np.float32(
        [
            [1, 0, shake_x],
            [0, 1, shake_y],
        ]
    )

    frame = cv2.warpAffine(
        frame,
        M,
        (w, h),
        borderMode=cv2.BORDER_REFLECT,
    )

    text = f"{completed_word}!"
    size = 2.5 + 0.5 * np.sin(
        celebration_timer
    )

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

    return frame, confetti


def draw_letter_success(
    frame,
    timer,
):
    h, w, _ = frame.shape

    scale = 1.6 + 0.15 * timer

    text = "Good!"

    text_size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        6,
    )[0]

    x = (w - text_size[0]) // 2
    y = (h + text_size[1]) // 2

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        (255, 255, 255),
        12,
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        (0, 255, 0),
        4,
    )