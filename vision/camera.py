import cv2


def initialize_camera(
    camera_index=0,
    width=640,
    height=480,
    warmup_frames=10,
    window_name="Sign Language Spelling Game",
):
    """
    Create and configure a webcam for low-latency MediaPipe usage.
    Returns:
        cap
        frame_ms
        timestamp
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_ms = int(1000 / fps)

    # Camera warm-up
    for _ in range(warmup_frames):
        cap.read()
    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )
    return cap, frame_ms, 0
