import sys
import cv2
import mediapipe as mp
import numpy as np
import time
from config import config

# Configuración de la tarea
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=config.model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)

def calculate_distance(point1, point2) -> float:
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance

def new_circle(padding) -> None:
    global random_circle_position
    # We no longer need circle_layer. We just need the position.
    # Note: Using hardcoded 640x480 for safety, but ideally use cap properties
    random_circle_position = (np.random.randint(padding, 640-padding), np.random.randint(padding, 480-padding))

with PoseLandmarker.create_from_options(options) as landmarker:
    # CHANGE 1: Force V4L2, MJPG, and set resolution
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        fps = 30

    mp_pose = mp.solutions.pose
    frame_ms = int(1000 / fps)
    timestamp = start_time = end_time = sum_t = counter = final_counter = 0

    # Warm up the camera to avoid black/truncated frames in WSL
    for _ in range(10):
        cap.read()

    new_circle(config.padding)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # CHANGE 2: Draw circle directly on the frame instead of using np.where
        cv2.circle(frame, random_circle_position, 20, (255, 0, 0), -1) # Blue circle

        t = end_time - start_time
        sum_t += t
        start_time = time.time()

        # CHANGE 3: Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += frame_ms

        if result.pose_landmarks:
            for person_landmarks in result.pose_landmarks:
                h, w, _ = frame.shape

                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = person_landmarks[start_idx]
                    end = person_landmarks[end_idx]

                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                for idx, landmark in enumerate(person_landmarks):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                landmarks_points = [(landmark.x, landmark.y) for landmark in [person_landmarks[19], person_landmarks[20]]]
                for landmark in landmarks_points:
                    if calculate_distance(landmark, (random_circle_position[0]/w, random_circle_position[1]/h)) < 0.05:
                        counter += 1
                        sum_t = 0
                        new_circle(config.padding)
                        break

        if sum_t > config.circle_time:
            sum_t = 0
            new_circle(config.padding)
        else:
            radius = int((20 + config.circle_time) - (config.circle_time_radius * (sum_t - config.circle_time)) / config.circle_time)
            cv2.circle(frame, random_circle_position, radius, (255, 255, 255), 2)

        end_time = time.time()

        cv2.putText(frame, f'Time: {config.game_time:.2f} - Points: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        config.game_time -= t
        
        if config.game_time <= 0:
            frame = np.zeros((480, 640, 3), dtype=np.uint8) # Simplified shape
            cv2.putText(frame, f'Score: {final_counter} points', (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Press Enter to continue ...', (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Press ESC to exit', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)

            if cv2.waitKey(1) & 0xFF == 13:
                config.game_time = 20
                counter = 0
            
            if cv2.waitKey(1) & 0xFF == 27:
                sys.exit(0)
        else:
            final_counter = counter

        cv2.imshow("Pose Landmarker", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()