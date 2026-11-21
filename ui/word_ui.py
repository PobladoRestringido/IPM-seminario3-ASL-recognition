import cv2


def draw_title(frame):
    cv2.putText(
        frame,
        "Sign Language Prototype",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def draw_word_progress(
    frame,
    target_word,
    letter_index,
    last_completed_index,
    gold_flash_timer,
):
    x = 10
    y = 80

    for i, letter in enumerate(target_word):

        if i == last_completed_index and gold_flash_timer > 0:
            color = (0, 215, 255)
            display = letter

        elif i < letter_index:
            color = (0, 255, 0)
            display = letter

        elif i == letter_index:
            color = (0, 165, 255)
            display = f"[{letter}]"

        else:
            color = (180, 180, 180)
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

        text_size = cv2.getTextSize(
            display,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2,
        )[0]

        x += text_size[0] + 5


def draw_feedback(
    frame,
    prediction,
    is_correct,
):
    feedback = "Correct!" if is_correct else f"You are showing: {prediction}"

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
        2,
    )


def draw_metrics(
    frame,
    similarity,
    distance,
):
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