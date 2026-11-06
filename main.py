import os
import sys
import tempfile
from collections import deque
from math import ceil, dist, sin
from time import monotonic

import cv2
import mediapipe as mp
import numpy as np

from game import (
    choose_computer_move,
    determine_winner,
    get_result_message,
    recognize_gesture,
    update_score,
)

COUNTDOWN_SECONDS = 5
STABLE_FRAMES = 5
REVEAL_SECONDS = 2.5

COLOR_TEXT = (245, 248, 250)
COLOR_MUTED = (165, 181, 195)
COLOR_PANEL = (22, 30, 42)
COLOR_CARD = (38, 49, 64)
COLOR_ACCENT = (75, 210, 255)
COLOR_WIN = (95, 220, 135)
COLOR_LOSE = (95, 105, 245)
COLOR_TIE = (225, 200, 95)
COLOR_FIRE = (40, 140, 255)

# Glove styles: colors are (blue, green, red). Press G in the game to cycle through them.
GLOVE_STYLES = (
    {"name": "Classic Red", "kind": "glove", "main": (45, 45, 215), "trim": (245, 245, 245)},
    {"name": "Ocean Blue", "kind": "glove", "main": (205, 125, 35), "trim": (255, 235, 170)},
    {"name": "Neon Pink", "kind": "glove", "main": (190, 70, 255), "trim": (255, 255, 120)},
    {"name": "Cat Paw", "kind": "paw", "main": (70, 150, 240), "trim": (180, 160, 255)},
    {"name": "Midnight Cat", "kind": "paw", "main": (50, 45, 45), "trim": (190, 150, 255)},
    {"name": "Cartoon", "kind": "cartoon", "main": (250, 250, 250), "trim": (225, 225, 225)},
    {"name": "Robot", "kind": "robot", "main": (175, 170, 165), "trim": (255, 220, 60)},
    {"name": "Monster", "kind": "monster", "main": (80, 190, 90), "trim": (215, 235, 240)},
    {"name": "Boxing", "kind": "boxing", "main": (40, 30, 200), "trim": (250, 250, 250)},
    {"name": "Skeleton", "kind": "skeleton", "main": None, "trim": None},
)
FINGER_CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
PALM_POINTS = (0, 1, 2, 5, 9, 13, 17)
KNUCKLES = (5, 9, 13, 17)

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


def shade(color, amount):
    return tuple(int(channel * amount) for channel in color)


def hand_shape(frame, hand):
    """Turn MediaPipe's 21 hand points into pixel positions and sizes."""
    height, width = frame.shape[:2]
    points = [(int(mark.x * width), int(mark.y * height)) for mark in hand.landmark]

    # Scale everything to the hand size so it fits near and far from the camera.
    size = max(1, int(dist(points[0], points[9])))
    wrist, middle = np.array(points[0]), np.array(points[9])
    direction = (middle - wrist) / size
    across = np.array([-direction[1], direction[0]]) * size * 0.5
    wrist_sides = [tuple(int(v) for v in wrist + sign * across * 0.75) for sign in (-1, 1)]
    palm = cv2.convexHull(
        np.array([points[i] for i in PALM_POINTS] + wrist_sides, dtype=np.int32)
    )
    palm_center = np.mean([points[i] for i in (0, 5, 9, 13, 17)], axis=0) + direction * size * 0.1
    return {
        "points": points,
        "size": size,
        "finger": max(8, size // 4),
        "wrist": wrist,
        "direction": direction,
        "across": across,
        "palm": palm,
        "palm_center": tuple(int(v) for v in palm_center),
    }


def fill_hand(frame, shape, color, finger_width, palm_border=0):
    """Fill the palm and draw every finger as a thick rounded line."""
    points = shape["points"]
    if palm_border:
        cv2.polylines(frame, [shape["palm"]], True, color, palm_border, cv2.LINE_AA)
    else:
        cv2.fillConvexPoly(frame, shape["palm"], color, cv2.LINE_AA)
    for chain in FINGER_CHAINS:
        for start, end in zip(chain, chain[1:]):
            cv2.line(frame, points[start], points[end], color, finger_width, cv2.LINE_AA)
        cv2.circle(frame, points[chain[-1]], finger_width // 2, color, -1, cv2.LINE_AA)


def draw_body(frame, shape, main, outline, finger_scale=1.0):
    """Outline first, then the main color on top, for a cartoon look."""
    width = int(shape["finger"] * finger_scale)
    fill_hand(frame, shape, outline, width + 8, palm_border=8)
    fill_hand(frame, shape, main, width)
    return width


def draw_cuff(frame, shape, color, outline, thickness_scale=0.33, width_scale=1.0, back=0.05):
    center = shape["wrist"] - shape["direction"] * shape["size"] * back
    across = shape["across"] * width_scale
    ends = [tuple(int(v) for v in center + sign * across) for sign in (-1, 1)]
    thickness = max(6, int(shape["size"] * thickness_scale))
    cv2.line(frame, ends[0], ends[1], outline, thickness + 8, cv2.LINE_AA)
    cv2.line(frame, ends[0], ends[1], color, thickness, cv2.LINE_AA)
    return center, ends, thickness


def fingertip_direction(points, chain):
    tip, before = np.array(points[chain[-1]]), np.array(points[chain[-2]])
    length = max(1.0, float(np.linalg.norm(tip - before)))
    return tip, (tip - before) / length


def draw_claws(frame, shape, color, outline, length_scale):
    points, finger = shape["points"], shape["finger"]
    for chain in FINGER_CHAINS:
        tip, forward = fingertip_direction(points, chain)
        side = np.array([-forward[1], forward[0]])
        base = tip + forward * finger * 0.25
        claw = np.array(
            [
                base + side * finger * 0.22,
                base + forward * finger * length_scale,
                base - side * finger * 0.22,
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, claw, color, cv2.LINE_AA)
        cv2.polylines(frame, [claw], True, outline, 2, cv2.LINE_AA)


def draw_glove_style(frame, shape, main, trim):
    outline = shade(main, 0.45)
    shine = tuple(min(255, int(channel * 1.35) + 25) for channel in main)
    draw_cuff(frame, shape, trim, outline)
    finger = draw_body(frame, shape, main, outline)
    for chain in FINGER_CHAINS:
        cv2.circle(frame, shape["points"][chain[-1]], max(2, finger // 6), shine, -1, cv2.LINE_AA)
    for knuckle in KNUCKLES:
        cv2.circle(frame, shape["points"][knuckle], max(3, finger // 5), trim, -1, cv2.LINE_AA)


def draw_paw_style(frame, shape, fur, beans):
    outline = shade(fur, 0.5) if sum(fur) > 300 else (20, 20, 20)
    draw_claws(frame, shape, (235, 240, 245), outline, 0.8)
    finger = draw_body(frame, shape, fur, outline, finger_scale=1.3)
    points = shape["points"]
    # Tabby stripes across each finger.
    stripe = shade(fur, 0.65) if sum(fur) > 300 else (75, 70, 70)
    for chain in FINGER_CHAINS[1:]:
        middle = np.array(points[chain[1]])
        _, forward = fingertip_direction(points, chain[:3])
        side = np.array([-forward[1], forward[0]]) * finger * 0.35
        cv2.line(
            frame,
            tuple(int(v) for v in middle - side),
            tuple(int(v) for v in middle + side),
            stripe,
            max(2, finger // 7),
            cv2.LINE_AA,
        )
    # Toe beans on every fingertip.
    for chain in FINGER_CHAINS:
        cv2.circle(frame, points[chain[-1]], max(4, int(finger * 0.3)), beans, -1, cv2.LINE_AA)
    # Big palm pad: three overlapping circles.
    center = np.array(shape["palm_center"])
    across = shape["across"] / max(1.0, float(np.linalg.norm(shape["across"])))
    pad = max(6, int(shape["size"] * 0.17))
    toward_wrist = -shape["direction"] * pad * 0.6
    for offset in (-across * pad * 0.8, across * pad * 0.8, toward_wrist):
        cv2.circle(frame, tuple(int(v) for v in center + offset), pad, beans, -1, cv2.LINE_AA)


def draw_cartoon_style(frame, shape, main, trim):
    outline = (25, 25, 25)
    center, ends, thickness = draw_cuff(
        frame, shape, main, outline, thickness_scale=0.45, width_scale=1.25, back=0.15
    )
    # A line along the cuff makes it look rolled over.
    roll = shape["direction"] * thickness * 0.15
    cv2.line(
        frame,
        tuple(int(v) for v in np.array(ends[0]) + roll),
        tuple(int(v) for v in np.array(ends[1]) + roll),
        trim,
        max(2, thickness // 6),
        cv2.LINE_AA,
    )
    draw_body(frame, shape, main, outline, finger_scale=1.4)


def draw_robot_style(frame, shape, metal, glow):
    outline = (45, 45, 50)
    draw_cuff(frame, shape, (90, 90, 95), outline, thickness_scale=0.4)
    finger = draw_body(frame, shape, metal, outline, finger_scale=1.1)
    points = shape["points"]
    for chain in FINGER_CHAINS:
        for joint in chain[:-1]:
            cv2.circle(frame, points[joint], max(3, finger // 3), outline, -1, cv2.LINE_AA)
            cv2.circle(frame, points[joint], max(2, finger // 5), (215, 215, 220), -1, cv2.LINE_AA)
        cv2.circle(frame, points[chain[-1]], finger // 2, (70, 70, 75), -1, cv2.LINE_AA)
    # Glowing power core in the palm, gently pulsing.
    pulse = (sin(monotonic() * 6) + 1) / 2
    core = max(6, int(shape["size"] * (0.16 + 0.04 * pulse)))
    cv2.circle(frame, shape["palm_center"], core + 5, outline, -1, cv2.LINE_AA)
    cv2.circle(frame, shape["palm_center"], core, glow, -1, cv2.LINE_AA)
    cv2.circle(frame, shape["palm_center"], max(2, core // 2), (255, 255, 255), -1, cv2.LINE_AA)


def draw_monster_style(frame, shape, skin, claw_color):
    outline = (90, 30, 70)
    draw_claws(frame, shape, claw_color, outline, 1.5)
    finger = draw_body(frame, shape, skin, outline, finger_scale=1.15)
    # A few bumps on the palm.
    center = np.array(shape["palm_center"])
    for dx, dy, scale in ((-0.25, 0.05, 0.08), (0.2, -0.1, 0.06), (0.05, 0.25, 0.07)):
        spot = center + np.array([dx, dy]) * shape["size"]
        cv2.circle(
            frame,
            tuple(int(v) for v in spot),
            max(3, int(shape["size"] * scale)),
            shade(skin, 0.7),
            -1,
            cv2.LINE_AA,
        )


def draw_boxing_style(frame, shape, main, laces):
    outline = shade(main, 0.4)
    center, ends, thickness = draw_cuff(
        frame, shape, laces, outline, thickness_scale=0.55, width_scale=1.1, back=0.2
    )
    # Very thick fingers merge into one big mitt.
    draw_body(frame, shape, main, outline, finger_scale=2.4)
    shine = tuple(min(255, int(channel * 1.3) + 40) for channel in main)
    cv2.circle(frame, shape["points"][9], max(4, shape["size"] // 7), shine, -1, cv2.LINE_AA)
    # Criss-cross laces on the cuff.
    along = np.array(ends[1]) - np.array(ends[0])
    up = shape["direction"] * thickness * 0.3
    for step in (0.3, 0.5, 0.7):
        point = np.array(ends[0]) + along * step
        for sign in (-1, 1):
            cv2.line(
                frame,
                tuple(int(v) for v in point - up + along * 0.06 * sign),
                tuple(int(v) for v in point + up - along * 0.06 * sign),
                outline,
                2,
                cv2.LINE_AA,
            )


STYLE_DRAWERS = {
    "glove": draw_glove_style,
    "paw": draw_paw_style,
    "cartoon": draw_cartoon_style,
    "robot": draw_robot_style,
    "monster": draw_monster_style,
    "boxing": draw_boxing_style,
}


def draw_glove(frame, hand, style, on_fire=False):
    """Draw the chosen glove style on top of the tracked hand."""
    shape = hand_shape(frame, hand)
    if on_fire:
        # Flickering orange glow behind the glove during a 3+ win streak.
        flicker = 14 + round((sin(monotonic() * 10) + 1) * 5)
        fill_hand(frame, shape, COLOR_FIRE, int(shape["finger"] * 1.4) + flicker * 2, palm_border=flicker * 2)
    STYLE_DRAWERS[style["kind"]](frame, shape, style["main"], style["trim"])


def draw_move_icon(frame, move, center, size):
    """Draw a simple rock, paper, or scissors picture centered at `center`."""
    cx, cy = center
    outline = (30, 30, 35)
    if move == "rock":
        bumps = (1.0, 0.82, 0.95, 0.78, 1.0, 0.86, 0.92, 0.8, 0.97, 0.84)
        rock = np.array(
            [
                (
                    cx + size * 0.5 * r * np.cos(2 * np.pi * i / len(bumps)),
                    cy + size * 0.42 * r * np.sin(2 * np.pi * i / len(bumps)),
                )
                for i, r in enumerate(bumps)
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [rock], (125, 130, 140), cv2.LINE_AA)
        cv2.polylines(frame, [rock], True, outline, 3, cv2.LINE_AA)
        cv2.ellipse(
            frame,
            (cx - size // 8, cy - size // 8),
            (size // 6, size // 12),
            -20,
            0,
            360,
            (175, 180, 190),
            -1,
            cv2.LINE_AA,
        )
        cv2.polylines(
            frame,
            [np.array([(cx + size // 10, cy - size // 20), (cx + size // 5, cy + size // 10), (cx + size // 8, cy + size // 5)], dtype=np.int32)],
            False,
            outline,
            2,
            cv2.LINE_AA,
        )
    elif move == "paper":
        half_w, half_h, fold = int(size * 0.36), int(size * 0.46), int(size * 0.16)
        left, top, right, bottom = cx - half_w, cy - half_h, cx + half_w, cy + half_h
        sheet = np.array(
            [(left, top), (right - fold, top), (right, top + fold), (right, bottom), (left, bottom)],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [sheet], (250, 250, 245), cv2.LINE_AA)
        cv2.polylines(frame, [sheet], True, outline, 3, cv2.LINE_AA)
        corner = np.array([(right - fold, top), (right - fold, top + fold), (right, top + fold)], dtype=np.int32)
        cv2.fillPoly(frame, [corner], (215, 215, 210), cv2.LINE_AA)
        cv2.polylines(frame, [corner], True, outline, 2, cv2.LINE_AA)
        for row in range(4):
            y = top + fold + 12 + row * (half_h * 2 - fold - 24) // 4
            cv2.line(frame, (left + 12, y), (right - 12, y), (210, 170, 120), 2, cv2.LINE_AA)
    elif move == "scissors":
        blade = (205, 210, 220)
        handle = (60, 70, 230)
        pivot = np.array([cx, cy - size * 0.02])
        for sign in (-1, 1):
            tip = np.array([cx + sign * size * 0.26, cy - size * 0.5])
            ring = (int(cx - sign * size * 0.2), int(cy + size * 0.3))
            # Each blade runs from just past the pivot up to its tip, crossing the other blade.
            forward = (tip - pivot) / np.linalg.norm(tip - pivot)
            side = np.array([-forward[1], forward[0]]) * size * 0.08
            start = pivot - forward * size * 0.08
            blade_shape = np.array(
                [start + side, tip, start - side * 0.4], dtype=np.int32
            )
            cv2.fillConvexPoly(frame, blade_shape, blade, cv2.LINE_AA)
            cv2.polylines(frame, [blade_shape], True, outline, 2, cv2.LINE_AA)
            arm = max(6, size // 12)
            cv2.line(frame, tuple(int(v) for v in pivot), ring, outline, arm + 4, cv2.LINE_AA)
            cv2.line(frame, tuple(int(v) for v in pivot), ring, handle, arm, cv2.LINE_AA)
            radius = max(8, size // 9)
            cv2.circle(frame, ring, radius, outline, max(6, size // 14) + 4, cv2.LINE_AA)
            cv2.circle(frame, ring, radius, handle, max(6, size // 14), cv2.LINE_AA)
        cv2.circle(frame, tuple(int(v) for v in pivot), max(4, size // 18), outline, -1, cv2.LINE_AA)


def draw_reveal(frame, player_move, computer_move, outcome):
    """Show both moves as big pictures after a round: YOU vs CPU."""
    height, width = frame.shape[:2]
    card = min(220, height // 3)
    gap = card // 2
    top = height // 2 - card // 2 - 10
    left_you = width // 2 - gap // 2 - card
    left_cpu = width // 2 + gap // 2
    you_color = {"win": COLOR_WIN, "lose": COLOR_LOSE, "tie": COLOR_TIE}[outcome]
    cpu_color = {"win": COLOR_LOSE, "lose": COLOR_WIN, "tie": COLOR_TIE}[outcome]

    for label, move, left, color in (
        ("YOU", player_move, left_you, you_color),
        ("CPU", computer_move, left_cpu, cpu_color),
    ):
        cv2.rectangle(frame, (left, top), (left + card, top + card), COLOR_PANEL, -1)
        cv2.rectangle(frame, (left, top), (left + card, top + card), color, 4, cv2.LINE_AA)
        draw_move_icon(frame, move, (left + card // 2, top + card // 2 - 8), int(card * 0.6))
        caption = f"{label}: {move.upper()}"
        text_w = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
        cv2.putText(
            frame,
            caption,
            (left + (card - text_w) // 2, top + card - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_TEXT,
            2,
            cv2.LINE_AA,
        )

    vs_size = cv2.getTextSize("VS", cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    cv2.putText(
        frame,
        "VS",
        (width // 2 - vs_size[0] // 2, top + card // 2 + vs_size[1] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        COLOR_TEXT,
        3,
        cv2.LINE_AA,
    )


def draw_interface(
    frame,
    player_score,
    computer_score,
    player_streak,
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

    if player_streak > 0:
        on_fire = player_streak >= 3
        badge_color = COLOR_FIRE if on_fire else COLOR_ACCENT
        badge_text = f"ON FIRE x{player_streak}" if on_fire else f"STREAK x{player_streak}"
        (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        left, top = 24, header_bottom + 18
        right, bottom = left + text_w + 36, top + text_h + 28
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_PANEL, -1)
        border = 2
        if on_fire:
            pulse = (sin(monotonic() * 8) + 1) / 2
            border = 2 + round(pulse * 4)
        cv2.rectangle(frame, (left, top), (right, bottom), badge_color, border, cv2.LINE_AA)
        cv2.putText(
            frame,
            badge_text,
            (left + 18, bottom - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            badge_color,
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
        "SPACE  PLAY ROUND     R  RESET SCORE     G  GLOVES     Q  QUIT",
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
    player_streak = 0
    glove_index = 0
    reveal = None
    reveal_until = 0
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
                glove = GLOVE_STYLES[glove_index]
                if glove["kind"] == "skeleton":
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
                else:
                    draw_glove(frame, hand, glove, on_fire=player_streak >= 3)
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
                        reveal = (stable_gesture, computer_move, outcome)
                        reveal_until = now + REVEAL_SECONDS
                        player_score, computer_score = update_score(
                            outcome, player_score, computer_score
                        )
                        if outcome =="win":
                            player_streak += 1
                        elif outcome == 'lose': 
                            player_streak = 0
                        round_message = (
                            f"{get_result_message(outcome)} "
                            f"You: {stable_gesture} | CPU: {computer_move} (random) | Streak: {player_streak}"
                        )
                else:
                    countdown = ceil(seconds_left)

            draw_interface(
                frame,
                player_score,
                computer_score,
                player_streak,
                detected_gesture,
                round_message,
                countdown,
            )
            if reveal is not None and now < reveal_until:
                draw_reveal(frame, *reveal)
            cv2.imshow("RISE Rock, Paper, Scissors", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("g"):
                glove_index = (glove_index + 1) % len(GLOVE_STYLES)
                round_message = f"Glove: {GLOVE_STYLES[glove_index]['name']}"
            if key == ord("r"):
                player_score = 0
                computer_score = 0
                player_streak = 0
                reveal = None
                round_message = "Scores reset. Press SPACE to play."
            if key == ord(" ") and countdown_end is None:
                gesture_samples.clear()
                stable_gesture = None
                countdown_end = monotonic() + COUNTDOWN_SECONDS
                reveal = None
                round_message = "Show your move and hold it steady."
    finally:
        if hands is not None:
            hands.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
