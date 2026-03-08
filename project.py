#!/usr/bin/env python3
"""
Gesture Media Controller
Double pinch  → Play/Pause
Swipe LEFT    → Prev track
Swipe RIGHT   → Next track
Q             → Quit
"""

import cv2
import mediapipe as mp
import time
from collections import deque
from pynput.keyboard import Controller, Key

SWIPE_THRESHOLD = 0.12
PINCH_THRESHOLD = 0.06
DOUBLE_PINCH_WINDOW = 1.0
COOLDOWN = 0.6
TRAIL_LEN = 12

kb = Controller()

def media_key(k):
    kb.press(k)
    kb.release(k)

class GestureController:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.6,
        )
        self.draw = mp.solutions.drawing_utils
        self.connections = mp.solutions.hands.HAND_CONNECTIONS
        self.trail = deque(maxlen=TRAIL_LEN)
        self.last_trigger = 0.0
        self.last_pinch_t = 0.0
        self.pinch_count = 0
        self.gesture_text = ""

    def _tip(self, lm, i):
        return lm[i].x, lm[i].y

    def _dist(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    def _is_pinch(self, lm):
        return self._dist(self._tip(lm, 4), self._tip(lm, 8)) < PINCH_THRESHOLD

    def _check_swipe(self):
        if len(self.trail) < TRAIL_LEN:
            return None
        dx = self.trail[-1][0] - self.trail[0][0]
        dy = self.trail[-1][1] - self.trail[0][1]
        if abs(dx) > abs(dy) and abs(dx) > SWIPE_THRESHOLD:
            return "LEFT" if dx < 0 else "RIGHT"
        return None

    def process(self, frame):
        now = time.time()
        on_cd = (now - self.last_trigger) < COOLDOWN
        h, w, _ = frame.shape

        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark
            self.draw.draw_landmarks(frame, hand, self.connections)

            ix, iy = self._tip(lm, 8)
            self.trail.append((ix, iy))
            pinched = self._is_pinch(lm)

            # Draw trail
            pts = [(int(p[0]*w), int(p[1]*h)) for p in self.trail]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (0, int(255*i/len(pts)), 255), 2)

            if not on_cd:
                # Double pinch → play/pause
                if pinched:
                    if (now - self.last_pinch_t) < DOUBLE_PINCH_WINDOW:
                        self.pinch_count += 1
                    else:
                        self.pinch_count = 1
                    self.last_pinch_t = now
                    if self.pinch_count >= 2:
                        media_key(Key.media_play_pause)
                        self.gesture_text = "Play / Pause"
                        self.last_trigger = now
                        self.pinch_count = 0
                        self.trail.clear()

                # Swipe left/right → prev/next
                elif lm[8].y < lm[6].y:  # index finger up
                    swipe = self._check_swipe()
                    if swipe == "LEFT":
                        media_key(Key.media_previous)
                        self.gesture_text = "Prev Track"
                        self.last_trigger = now
                        self.trail.clear()
                    elif swipe == "RIGHT":
                        media_key(Key.media_next)
                        self.gesture_text = "Next Track"
                        self.last_trigger = now
                        self.trail.clear()
        else:
            self.trail.clear()

        if self.gesture_text:
            cv2.putText(frame, self.gesture_text, (w//2-100, h-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 120), 2)
        return frame


def main():
    print("Gesture Media Controller")
    print("  Double pinch → Play/Pause | Swipe L/R → Prev/Next | Q → Quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ctrl = GestureController()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = ctrl.process(cv2.flip(frame, 1))
        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
