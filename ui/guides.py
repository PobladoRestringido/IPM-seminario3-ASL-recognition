import cv2


def draw_hand_guide(
    frame,
    guide_images,
    target_letter,
):
    guide = guide_images.get(target_letter)

    if guide is None:
        return

    h, w, _ = frame.shape

    guide_height = 180

    aspect_ratio = (
        guide.shape[1] / guide.shape[0]
    )

    guide_width = int(
        guide_height * aspect_ratio
    )

    guide_resized = cv2.resize(
        guide,
        (guide_width, guide_height),
    )

    y1 = 10
    y2 = y1 + guide_height

    x2 = w - 10
    x1 = x2 - guide_width

    frame[y1:y2, x1:x2] = guide_resized

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        2,
    )