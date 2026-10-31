"""
Hand Gesture Swara Player
Computer Vision Club, BITS Goa | Week 4

Show 1-8 fingers (both hands total) to play Sa Re Ga Ma Pa Dha Ni Sa.
Keys: q = quit, m = mute/unmute.

Run from the project folder:  python main.py
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ---------------------------------------------------------------- config
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Milestone 6: finger count -> sound file
sounds_mapping = {
    1: "sounds_swaras/1_Sa.mp3",
    2: "sounds_swaras/2_Re.mp3",
    3: "sounds_swaras/3_Ga.mp3",
    4: "sounds_swaras/4_Ma.mp3",
    5: "sounds_swaras/5_Pa.mp3",
    6: "sounds_swaras/6_Dha.mp3",
    7: "sounds_swaras/7_Ni.mp3",
    8: "sounds_swaras/8_Sa_upper.mp3",
}
swara_names = {1: "Sa", 2: "Re", 3: "Ga", 4: "Ma",
               5: "Pa", 6: "Dha", 7: "Ni", 8: "Sa (upper)"}

STABLE_FRAMES = 3   # count must be identical this many frames in a row
MIN_GAP = 0.4       # minimum seconds between two sounds

# 21-landmark skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
FINGER_TIPS = [8, 12, 16, 20]  # index, middle, ring, pinky


# ---------------------------------------------------------------- helpers
def draw_landmarks(frame, hand_landmarks):
    """Landmarks are fractions (0-1); convert to pixels before drawing."""
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, points[s], points[e], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def count_fingers(hand_landmarks, label):
    """Count raised fingers on one hand (frame is already mirrored)."""
    count = 0

    # Thumb moves sideways, so compare x. Direction depends on handedness.
    # If the thumb counts backwards on your setup, swap < and >.
    tip_x, joint_x = hand_landmarks[4].x, hand_landmarks[3].x
    if label == "Right" and tip_x < joint_x:
        count += 1
    elif label == "Left" and tip_x > joint_x:
        count += 1

    # Other fingers: y grows downward, so "up" = tip y smaller than PIP y (tip - 2)
    for tip in FINGER_TIPS:
        if hand_landmarks[tip].y < hand_landmarks[tip - 2].y:
            count += 1
    return count


def main():
    # ---- model file (downloaded once) ----
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    # ---- Milestone 7: audio, loaded once so notes can overlap ----
    pygame.mixer.init()
    sounds = {n: pygame.mixer.Sound(path) for n, path in sounds_mapping.items()}

    # ---- Milestone 2: detector, created ONCE before the loop ----
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    # ---- Milestone 1: webcam ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Close Zoom/Teams/Camera or try VideoCapture(1).")
        landmarker.close()
        return

    start_time = time.time()
    last_ts = -1

    prev_raw = -1          # count in previous frame
    stable_run = 0         # frames in a row with the same count
    last_played = 0        # last count that triggered a sound (0 = re-armed)
    last_play_time = 0.0
    flash_until = 0.0
    current_swara = ""
    muted = False

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)  # mirror BEFORE detection

            # ---- Milestone 3: detect ----
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            if timestamp_ms <= last_ts:  # timestamps must strictly increase
                timestamp_ms = last_ts + 1
            last_ts = timestamp_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # ---- Milestone 4 + 5: draw and count (count resets every frame) ----
            count = 0
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].category_name
                count += count_fingers(hand_landmarks, label)
                draw_landmarks(frame, hand_landmarks)

            # ---- Milestone 7: trigger rules ----
            # Play only when the count is stable AND different from the last note.
            # This avoids wrong notes while raising fingers one by one.
            stable_run = stable_run + 1 if count == prev_raw else 1
            prev_raw = count
            now = time.time()

            if count == 0:
                last_played = 0  # a fist / no hand re-arms the trigger
            elif stable_run >= STABLE_FRAMES and count != last_played:
                if count in sounds:
                    if now - last_play_time >= MIN_GAP:
                        if not muted:
                            sounds[count].play()
                        current_swara = swara_names[count]
                        last_played = count
                        last_play_time = now
                        flash_until = now + 0.6
                else:
                    last_played = count  # 9 or 10: silent

            # ---- on-screen display ----
            flashing = now < flash_until
            cv2.putText(frame, f"Fingers: {count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            if current_swara:
                color = (0, 255, 255) if flashing else (200, 200, 200)
                size = 2.0 if flashing else 1.0
                cv2.putText(frame, current_swara, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, size, color, 3)
            if flashing:
                cv2.rectangle(frame, (0, 0),
                              (frame.shape[1] - 1, frame.shape[0] - 1),
                              (0, 255, 255), 8)
            if muted:
                cv2.putText(frame, "MUTED", (10, frame.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow("Hand Gesture Swara Player", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                muted = not muted
                if muted:
                    pygame.mixer.stop()
    finally:
        # ---- cleanup ----
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
