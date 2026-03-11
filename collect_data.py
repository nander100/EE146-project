#!/usr/bin/env python3
"""
Hand Position Data Collector
Collects 500 valid and 500 invalid hand position samples using MediaPipe.

Controls:
  V  → record current frame as VALID
  I  → record current frame as INVALID
  Q  → quit (saves progress)

Output: hand_data.npz  (X: [N, 42], y: [N])
  - X contains flattened (x, y) coords of 21 landmarks, normalized to wrist
  - y: 1 = valid, 0 = invalid
"""

import cv2
import mediapipe as mp
import numpy as np
import time

TARGET = 500
OUT_FILE = "hand_data.npz"

hands_detector = mp.solutions.hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6,
)
draw = mp.solutions.drawing_utils
connections = mp.solutions.hands.HAND_CONNECTIONS


def extract_features(lm):
    """Flatten 21 landmarks to 42 floats, normalized relative to wrist (lm[0])."""
    wrist_x, wrist_y = lm[0].x, lm[0].y
    feats = []
    for p in lm:
        feats.append(p.x - wrist_x)
        feats.append(p.y - wrist_y)
    return np.array(feats, dtype=np.float32)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    valid_samples = []
    invalid_samples = []

    flash_msg = ""
    flash_until = 0.0

    print(f"Collecting {TARGET} valid and {TARGET} invalid samples.")
    print("Press V → valid  |  I → invalid  |  Q → quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(rgb)

        lm = None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark
            draw.draw_landmarks(frame, hand, connections)

        h, w, _ = frame.shape

        # progress bars
        v_count = len(valid_samples)
        i_count = len(invalid_samples)

        cv2.rectangle(frame, (10, 10), (10 + int(200 * v_count / TARGET), 30), (0, 220, 0), -1)
        cv2.rectangle(frame, (10, 10), (210, 30), (0, 220, 0), 2)
        cv2.putText(frame, f"Valid:   {v_count}/{TARGET}", (220, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

        cv2.rectangle(frame, (10, 40), (10 + int(200 * i_count / TARGET), 60), (0, 60, 255), -1)
        cv2.rectangle(frame, (10, 40), (210, 60), (0, 60, 255), 2)
        cv2.putText(frame, f"Invalid: {i_count}/{TARGET}", (220, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 255), 2)

        # flash feedback
        if time.time() < flash_until:
            cv2.putText(frame, flash_msg, (w // 2 - 120, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 200), 3)

        if lm is None:
            cv2.putText(frame, "No hand detected", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

        cv2.imshow("Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if lm is None:
            continue

        feats = extract_features(lm)

        if key == ord("v") and v_count < TARGET:
            valid_samples.append(feats)
            flash_msg = f"✓ Valid saved ({v_count + 1})"
            flash_until = time.time() + 0.4

        elif key == ord("i") and i_count < TARGET:
            invalid_samples.append(feats)
            flash_msg = f"✗ Invalid saved ({i_count + 1})"
            flash_until = time.time() + 0.4

        # auto-quit when both targets reached
        if len(valid_samples) >= TARGET and len(invalid_samples) >= TARGET:
            print("Both targets reached — saving and exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if valid_samples or invalid_samples:
        X_valid = np.array(valid_samples) if valid_samples else np.empty((0, 42), dtype=np.float32)
        X_invalid = np.array(invalid_samples) if invalid_samples else np.empty((0, 42), dtype=np.float32)

        X = np.vstack([X_valid, X_invalid])
        y = np.concatenate([
            np.ones(len(valid_samples), dtype=np.float32),
            np.zeros(len(invalid_samples), dtype=np.float32),
        ])

        np.savez(OUT_FILE, X=X, y=y)
        print(f"Saved {len(valid_samples)} valid + {len(invalid_samples)} invalid → {OUT_FILE}")
    else:
        print("No samples collected.")


if __name__ == "__main__":
    main()