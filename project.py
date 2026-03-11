#!/usr/bin/env python3
"""
Gesture Media Controller  —  with HandValidator

Double pinch           → Play/Pause
Swipe LEFT (1 finger)  → Prev track
Swipe RIGHT (1 finger) → Next track
Two-finger slide UP    → Volume Up
Two-finger slide DOWN  → Volume Down
Closed hand            → Pause input
Q                      → Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from pynput.keyboard import Controller, Key

SWIPE_THRESHOLD = 0.12
VERTICAL_THRESHOLD = 0.20
PINCH_THRESHOLD = 0.06
STOP_COMMANDS_THRESHOLD = 0.20
DOUBLE_PINCH_WINDOW = 1.0
COOLDOWN = 0.6
TRAIL_LEN = 12

VALIDATOR_MODEL = "hand_validator.npz"   # produced by train_validator.py
VALIDATOR_THRESHOLD = 0.5               # sigmoid output ≥ this → valid

kb = Controller()

def media_key(k):
    kb.press(k)
    kb.release(k)


# ── HandValidator ─────────────────────────────────────────────────────────────

class HandValidator:
    """
    Tiny 42→32→16→1 MLP loaded from a .npz file.
    Returns True when the hand pose looks like a valid control gesture.

    Usage:
        validator = HandValidator("hand_validator.npz")
        is_valid  = validator.is_valid(lm)   # lm = mediapipe hand.landmark
    """

    def __init__(self, model_path: str, threshold: float = VALIDATOR_THRESHOLD):
        data = np.load(model_path)
        self.mu   = data["mu"]
        self.std  = data["std"]
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.W3, self.b3 = data["W3"], data["b3"]
        self.threshold = threshold
        print(f"[HandValidator] loaded from {model_path}")

    # ── feature extraction (must match collect_data.py) ──────────────────────
    @staticmethod
    def _extract(lm):
        wrist_x, wrist_y = lm[0].x, lm[0].y
        feats = []
        for p in lm:
            feats.append(p.x - wrist_x)
            feats.append(p.y - wrist_y)
        return np.array(feats, dtype=np.float32)

    # ── forward pass ─────────────────────────────────────────────────────────
    def _predict(self, x: np.ndarray) -> float:
        x = (x - self.mu) / self.std
        a = np.maximum(0, x @ self.W1 + self.b1)   # ReLU
        a = np.maximum(0, a @ self.W2 + self.b2)   # ReLU
        logit = a @ self.W3 + self.b3
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30))))

    def is_valid(self, lm) -> bool:
        """Return True if hand position passes the learned validity check."""
        score = self._predict(self._extract(lm))
        return score >= self.threshold

    def score(self, lm) -> float:
        """Return raw confidence score in [0, 1]."""
        return self._predict(self._extract(lm))


# ── GestureController ────────────────────────────────────────────────────────

class GestureController:

    def __init__(self, validator: HandValidator | None = None):

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

        self.volume_level = 50
        self.show_volume_until = 0

        self.validator = validator   # None → validation disabled

    def _tip(self, lm, i):
        return lm[i].x, lm[i].y

    def _dist(self, a, b):
        return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

    def _is_pinch(self, lm):
        return self._dist(self._tip(lm, 4), self._tip(lm, 8)) < PINCH_THRESHOLD

    def _is_hand_closed(self, lm):
        palm = (lm[0].x, lm[0].y)
        total = sum(self._dist(palm, (p.x, p.y)) for p in lm)
        return total / len(lm) < STOP_COMMANDS_THRESHOLD

    def _two_fingers(self, lm):
        index_up  = lm[8].y < lm[6].y
        middle_up = lm[12].y < lm[10].y
        ring_down = lm[16].y > lm[14].y
        return index_up and middle_up and ring_down

    def _check_swipe(self):
        if len(self.trail) < TRAIL_LEN:
            return None
        dx = self.trail[-1][0] - self.trail[0][0]
        dy = self.trail[-1][1] - self.trail[0][1]
        if abs(dx) > abs(dy) and abs(dx) > SWIPE_THRESHOLD:
            return "LEFT" if dx < 0 else "RIGHT"
        if abs(dy) > abs(dx) and abs(dy) > VERTICAL_THRESHOLD:
            return "UP" if dy < 0 else "DOWN"
        return None

    def draw_volume_bar(self, frame):
        if time.time() > self.show_volume_until:
            return
        h, w, _ = frame.shape
        bar_x, bar_y, bar_w, bar_h = 40, 100, 30, 250
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
        fill_h = int((self.volume_level / 100) * bar_h)
        cv2.rectangle(frame,
                      (bar_x, bar_y + bar_h - fill_h),
                      (bar_x + bar_w, bar_y + bar_h),
                      (0, 255, 0), -1)
        cv2.putText(frame, f"{self.volume_level}%",
                    (bar_x - 10, bar_y + bar_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ── draw validator confidence bar (top-right) ─────────────────────────────
    def _draw_validator_bar(self, frame, score: float):
        h, w, _ = frame.shape
        bar_h, bar_w = 150, 18
        bx, by = w - 55, 80
        colour = (0, int(255 * score), int(255 * (1 - score)))
        cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (180, 180, 180), 1)
        fill = int(score * bar_h)
        cv2.rectangle(frame, (bx, by + bar_h - fill), (bx + bar_w, by + bar_h), colour, -1)
        cv2.putText(frame, "VAL", (bx - 2, by - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame, f"{score:.2f}", (bx - 5, by + bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

    def process(self, frame):
        now = time.time()
        on_cd = (now - self.last_trigger) < COOLDOWN

        h, w, _ = frame.shape
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark

            self.draw.draw_landmarks(frame, hand, self.connections)

            # ── validator check ───────────────────────────────────────────────
            val_score = 1.0          # default: pass (when no model loaded)
            hand_valid = True
            if self.validator is not None:
                val_score  = self.validator.score(lm)
                hand_valid = val_score >= VALIDATOR_THRESHOLD
                self._draw_validator_bar(frame, val_score)

            if not hand_valid:
                self.gesture_text = "Invalid pose"
                self.trail.clear()
                if self.gesture_text:
                    cv2.putText(frame, self.gesture_text,
                                (w//2 - 120, h - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 80, 255), 2)
                self.draw_volume_bar(frame)
                return frame
            # ─────────────────────────────────────────────────────────────────

            ix, iy = self._tip(lm, 8)
            self.trail.append((ix, iy))

            pinched = self._is_pinch(lm)
            hand_closed = self._is_hand_closed(lm)
            two_fingers = self._two_fingers(lm)

            pts = [(int(p[0]*w), int(p[1]*h)) for p in self.trail]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i],
                         (0, int(255*i/len(pts)), 255), 2)

            if hand_closed:
                self.gesture_text = "Input Paused"
                self.trail.clear()

            elif not on_cd:

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

                elif two_fingers:
                    swipe = self._check_swipe()
                    if swipe == "UP":
                        media_key(Key.media_volume_up)
                        self.volume_level = min(100, self.volume_level + 2)
                        self.show_volume_until = time.time() + 2
                        self.gesture_text = "Volume Up"
                        self.last_trigger = now
                        self.trail.clear()
                    elif swipe == "DOWN":
                        media_key(Key.media_volume_down)
                        self.volume_level = max(0, self.volume_level - 2)
                        self.show_volume_until = time.time() + 2
                        self.gesture_text = "Volume Down"
                        self.last_trigger = now
                        self.trail.clear()

                elif lm[8].y < lm[6].y:
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
            cv2.putText(frame, self.gesture_text,
                        (w//2 - 120, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 120), 2)

        self.draw_volume_bar(frame)
        return frame


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    import os

    print("Gesture Media Controller")
    print("Double pinch → Play/Pause")
    print("Swipe LEFT → Prev Track")
    print("Swipe RIGHT → Next Track")
    print("Two Finger Slide UP → Volume Up")
    print("Two Finger Slide DOWN → Volume Down")
    print("Closed hand → Pause Input")
    print("Press Q to quit\n")

    # load validator if model file exists
    validator = None
    if os.path.exists(VALIDATOR_MODEL):
        validator = HandValidator(VALIDATOR_MODEL, threshold=VALIDATOR_THRESHOLD)
    else:
        print(f"[info] {VALIDATOR_MODEL} not found — running without pose validation.")
        print("[info] Run collect_data.py then train_validator.py to enable it.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ctrl = GestureController(validator=validator)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        frame = ctrl.process(frame)
        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()