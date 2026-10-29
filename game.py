import random


MOVES = ("rock", "paper", "scissors")
RESULT_MESSAGES = {
    "win": "You win!",
    "lose": "You lose!",
    "tie": "It's a tie!",
}


def recognize_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    finger_tips = (8, 12, 16, 20)
    finger_pips = (6, 10, 14, 18)
    extended = [
        landmarks[tip].y < landmarks[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    ]

    if not any(extended):
        return "rock"
    if all(extended):
        return "paper"
    if extended == [True, True, False, False]:
        return "scissors"
    return None


def choose_computer_move():
    return random.choice(MOVES)


def determine_winner(player_move, computer_move):
    if player_move not in MOVES or computer_move not in MOVES:
        raise ValueError("Moves must be rock, paper, or scissors.")
    if player_move == computer_move:
        return "tie"
    if (player_move, computer_move) in (
        ("rock", "scissors"),
        ("paper", "rock"),
        ("scissors", "paper"),
    ):
        return "win"
    return "lose"


def get_result_message(result):
    return RESULT_MESSAGES[result]


def update_score(result, player_score, computer_score):
    if result == "win":
        player_score += 1
    elif result == "lose":
        computer_score += 1
    elif result != "tie":
        raise ValueError("Result must be win, lose, or tie.")
    return player_score, computer_score
