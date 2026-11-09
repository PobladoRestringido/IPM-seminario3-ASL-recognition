import sys
import cv2
import mediapipe as mp
import numpy as np
import time
from config import config

# ---------- Initialization Section ----------
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='models/hand_landmarker.task'),  # Ensure this path is correct
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2, 
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_distance(point1, point2) -> float:
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance

def new_circle(padding) -> None:
    global random_circle_position
    random_circle_position = (np.random.randint(padding, 640-padding), np.random.randint(padding, 480-padding))

with HandLandmarker.create_from_options(options) as landmarker:
    # Camera Initialization
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        fps = 30

    frame_ms = int(1000 / fps)
    timestamp = start_time = end_time = sum_t = counter = final_counter = 0

    # Warm up the camera
    for _ in range(10):
        cap.read()

    new_circle(config.padding)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Draw target circle
        cv2.circle(frame, random_circle_position, 20, (255, 0, 0), -1) 

        t = end_time - start_time
        sum_t += t
        start_time = time.time()

        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = landmarker.detect_for_video(mp_image, timestamp)
        timestamp += frame_ms    

        # --- HAND DETECTION & DRAWING ---
        if result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in result.hand_landmarks:
                # Draw hand skeleton connections
                for connection in mp.solutions.hands.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Draw keypoints
                for landmark in hand_landmarks:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # Check collision with index (8) and middle (12) fingertips
                fingertips = [hand_landmarks[8], hand_landmarks[12]]
                for tip in fingertips:
                    if calculate_distance(
                        (tip.x, tip.y),
                        (random_circle_position[0]/w, random_circle_position[1]/h)
                    ) < 0.05:
                        counter += 1
                        sum_t = 0
                        new_circle(config.padding)
                        break # Breaks out of the fingertip loop so you don't double-count
                else:
                    continue
                break # Breaks out of the hand_landmarks loop if a collision happened

        # --- GAME TIMER & UI --- (Note: this is now properly OUTSIDE the hand detection if-statement)
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
            frame = np.zeros((480, 640, 3), dtype=np.uint8) 
            cv2.putText(frame, f'Score: {final_counter} points', (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Press Enter to continue ...', (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Press ESC to exit', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)

            if cv2.waitKey(1) & 0xFF == 13: # Enter key
                config.game_time = 20
                counter = 0
            
            if cv2.waitKey(1) & 0xFF == 27: # ESC key
                sys.exit(0)
        else:
            final_counter = counter

        cv2.imshow("Hand Landmarker Game", frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()