import os
import sys
import tempfile
from collections import deque
from math import ceil
from time import monotonic

import cv2
import mediapipe as mp

from game import (
    choose_computer_move,
    determine_winner,
    get_result_message,
    recognize_gesture,
    update_score,
)

COUNTDOWN_SECONDS = 3
STABLE_FRAMES = 5

COLOR_TEXT = (245, 248, 250)
COLOR_MUTED = (165, 181, 195)
COLOR_PANEL = (22, 30, 42)
COLOR_CARD = (38, 49, 64)
COLOR_ACCENT = (75, 210, 255)
COLOR_WIN = (95, 220, 135)
COLOR_LOSE = (95, 105, 245)
COLOR_TIE = (225, 200, 95)

MEDIAPIPE_STARTUP_LOGS = (
    "All log messages before absl::InitializeLog()",
    "gl_context.cc",
    "Created TensorFlow Lite XNNPACK delegate",
    "inference_feedback_manager.cc",
    "landmark_projection_calculator.cc",
)


def run_with_filtered_mediapipe_logs(operation, *args, **kwargs):
    """Hide known MediaPipe startup chatter without swallowing other errors."""
    with tempfile.TemporaryFile() as captured_logs:
        sys.stderr.flush()
        original_stderr = os.dup(2)
        try:
            os.dup2(captured_logs.fileno(), 2)
            result = operation(*args, **kwargs)
        finally:
            sys.stderr.flush()
            os.dup2(original_stderr, 2)
            os.close(original_stderr)
            captured_logs.seek(0)
            for line in captured_logs:
                message = line.decode("utf-8", errors="replace")
                if not any(fragment in message for fragment in MEDIAPIPE_STARTUP_LOGS):
                    sys.stderr.write(message)
            sys.stderr.flush()
    return result


def draw_interface(
    frame,
    player_score,
    computer_score,
    detected_gesture,
    round_message,
    countdown=None,
):
    height, width = frame.shape[:2]
    overlay = frame.copy()
    header_bottom = min(142, height // 3)
    footer_top = max(header_bottom + 20, height - 82)
    cv2.rectangle(overlay, (0, 0), (width, header_bottom), COLOR_PANEL, -1)
    cv2.rectangle(overlay, (0, footer_top), (width, height), COLOR_PANEL, -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    cv2.rectangle(frame, (0, header_bottom), (width, header_bottom + 2), COLOR_ACCENT, -1)
    cv2.putText(
        frame,
        "RISE",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        COLOR_TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "ROCK  /  PAPER  /  SCISSORS",
        (26, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        COLOR_MUTED,
        2,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (31, 112), 6, COLOR_ACCENT, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"HAND  {detected_gesture.upper()}",
        (47, 119),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        COLOR_TEXT,
        2,
        cv2.LINE_AA,
    )

    card_width = min(148, max(105, (width - 360) // 2))
    card_height = 90
    card_gap = 12
    cards_width = card_width * 2 + card_gap
    cards_left = max(24, width - cards_width - 24)
    card_top = max(12, (header_bottom - card_height) // 2)
    for label, score, left in (
        ("YOU", player_score, cards_left),
        ("CPU", computer_score, cards_left + card_width + card_gap),
    ):
        cv2.rectangle(
            frame,
            (left, card_top),
            (left + card_width, card_top + card_height),
            COLOR_CARD,
            -1,
        )
        cv2.rectangle(
            frame,
            (left, card_top),
            (left + 3, card_top + card_height),
            COLOR_ACCENT if label == "YOU" else COLOR_MUTED,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (left + 14, card_top + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            COLOR_MUTED,
            2,
            cv2.LINE_AA,
        )
        score_text = str(score)
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        cv2.putText(
            frame,
            score_text,
            (left + card_width - score_size[0] - 12, card_top + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )

    message_color = COLOR_TEXT
    if "win" in round_message.lower():
        message_color = COLOR_WIN
    elif "lose" in round_message.lower():
        message_color = COLOR_LOSE
    elif "tie" in round_message.lower():
        message_color = COLOR_TIE

    cv2.putText(
        frame,
        round_message,
        (24, footer_top + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        min(
            0.72,
            0.72
            * (width - 48)
            / max(
                1,
                cv2.getTextSize(
                    round_message, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
                )[0][0],
            ),
        ),
        message_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "SPACE  PLAY ROUND     R  RESET SCORE     Q  QUIT",
        (24, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        COLOR_MUTED,
        2,
        cv2.LINE_AA,
    )

    if countdown is not None:
        text = str(countdown)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)[0]
        center = (width // 2, (header_bottom + footer_top) // 2)
        cv2.circle(frame, center, 62, COLOR_PANEL, -1, cv2.LINE_AA)
        cv2.circle(frame, center, 62, COLOR_ACCENT, 2, cv2.LINE_AA)
        position = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            COLOR_ACCENT,
            6,
            cv2.LINE_AA,
        )

    return frame


def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Could not open the webcam. Check its connection and permissions.")
        camera.release()
        return

    player_score = 0
    computer_score = 0
    gesture_samples = deque(maxlen=STABLE_FRAMES)
    stable_gesture = None
    countdown_end = None
    round_message = "Press SPACE to start a round."
    hands = None
    first_frame = True

    try:
        hands = run_with_filtered_mediapipe_logs(
            mp.solutions.hands.Hands,
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a webcam frame. Check that the camera is available.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if first_frame:
                results = run_with_filtered_mediapipe_logs(hands.process, rgb_frame)
                first_frame = False
            else:
                results = hands.process(rgb_frame)

            hand_detected = bool(results.multi_hand_landmarks)
            candidate = None
            if hand_detected:
                hand = results.multi_hand_landmarks[0]
                candidate = recognize_gesture(hand)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(105, 235, 160), thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(255, 195, 90), thickness=2
                    ),
                )
            gesture_samples.append(candidate)

            if (
                len(gesture_samples) == STABLE_FRAMES
                and gesture_samples[0] is not None
                and all(sample == gesture_samples[0] for sample in gesture_samples)
            ):
                stable_gesture = gesture_samples[0]
            else:
                stable_gesture = None

            if stable_gesture is not None:
                detected_gesture = stable_gesture.title()
            elif hand_detected:
                detected_gesture = "Hold steady"
            else:
                detected_gesture = "No hand"

            now = monotonic()
            countdown = None
            if countdown_end is not None:
                seconds_left = countdown_end - now
                if seconds_left <= 0:
                    countdown_end = None
                    if stable_gesture is None:
                        round_message = "No clear move. Press SPACE to try again."
                    else:
                        computer_move = choose_computer_move()
                        outcome = determine_winner(stable_gesture, computer_move)
                        player_score, computer_score = update_score(
                            outcome, player_score, computer_score
                        )
                        round_message = (
                            f"{get_result_message(outcome)} "
                            f"You: {stable_gesture} | CPU: {computer_move} (random)"
                        )
                else:
                    countdown = ceil(seconds_left)

            draw_interface(
                frame,
                player_score,
                computer_score,
                detected_gesture,
                round_message,
                countdown,
            )
            cv2.imshow("RISE Rock, Paper, Scissors", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                player_score = 0
                computer_score = 0
                round_message = "Scores reset. Press SPACE to play."
            if key == ord(" ") and countdown_end is None:
                gesture_samples.clear()
                stable_gesture = None
                countdown_end = monotonic() + COUNTDOWN_SECONDS
                round_message = "Show your move and hold it steady."
    finally:
        if hands is not None:
            hands.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
