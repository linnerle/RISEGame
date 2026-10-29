# RISE SCHOLARS: Building on Rock, Paper, Scissors

Extend the working webcam game with a feature of your choice. Teams can continue a STARS project or start from the same starter files; no previous workshop is required.

## Learning Goals

- [ ] Trace how a round result changes the game.
- [ ] Add and test a small feature in existing code.
- [ ] Explain how you handled an edge case.

## Find Your Way Around

- `game.py` contains the game rules: gesture recognition, computer move selection, winner checks, and score updates.
- `main.py` connects the webcam, game rules, keyboard controls, and on-screen display.
- Start with `game.py` for a rule change, or the `draw_interface()` function in `main.py` for a visual change. You do not need to rewrite the webcam or MediaPipe setup to add a game feature.
- The computer currently chooses randomly. It does not learn from your moves.

You can explore the starter code in small pieces. Some webcam setup and third-party log filtering in `main.py` is intentionally more advanced; you can leave that code alone.

## 60-Minute Workshop

- **0–10 min:** Get an introduction and orient to the project.
- **10–20 min:** Run the game and explore its code.
- **20–40 min:** Implement the recommended feature or a challenge.
- **40–50 min:** Test and debug your change.
- **50–60 min:** Demo your feature and give a one-minute pitch.

## One Possible Feature: Winning Streak

**Goal:** Count your consecutive wins and reset the streak after a loss.

**Work in:** `main.py` — the `player_score` and `computer_score` variables in `main()`, and the round-result block that calls `update_score()`.

- [ ] Add a `player_streak` variable beside the score variables.
- [ ] Increase your streak when `outcome` is `"win"`.
- [ ] Reset the streak when `outcome` is `"lose"`.
- [ ] Decide whether a tie should keep or reset the streak.
- [ ] Show the streak in `round_message` or in `draw_interface()`.
- [ ] Test a win followed by another win, then a loss.

## Choose an Optional Challenge

The winning streak above is one possible project, not a requirement. Choose one of these ideas, adapt it, combine ideas with your team, or keep the starter game as it is.

### Teach the computer to notice patterns

**Goal:** Keep a short record of your moves and let the computer choose a counter to the move you use most often.

**Pseudocode:**

```text
when a round finishes:
    add your move to recent_moves
    find the most common move in recent_moves
    choose the move that beats it
    show you which move the computer noticed
```

Decide what the computer should do before it has seen any moves (for example, choose randomly). This is a simple pattern-based strategy, not machine learning.

**Optional class example:** This helper is not part of the running game. Copy it into `game.py` only if you want to experiment, then decide where to call `remember()` and `most_common_move()`.

```python
from collections import Counter, deque


class MoveHistory:
    """Optional helper for experimenting with your recent moves."""

    def __init__(self, max_rounds=10):
        # Keep only the most recent moves so the history stays small.
        self.moves = deque(maxlen=max_rounds)

    def remember(self, move):
        # Call this after a round has a valid move from you.
        self.moves.append(move)

    def most_common_move(self):
        # There is no pattern to report until a move has been recorded.
        if not self.moves:
            return None
        return Counter(self.moves).most_common(1)[0][0]
```

The existing `choose_computer_move()` function in `game.py` always picks randomly. Changing it to use history is optional; if you try, keep the random choice for an empty history and test both cases.

### Best-of-five match

**Goal:** Announce a match winner when one side reaches three round wins.

**Work in:** `main.py` — `player_score`, `computer_score`, and the round-result block in `main()`.

- [ ] Check the scores after each round.
- [ ] Show a match-winner message when either score reaches three.
- [ ] Play until one side reaches three wins.

### Recent round history

**Goal:** Show the last few player and computer moves.

**Work in:** `main.py` — the round-result block in `main()` and `draw_interface()`.

- [ ] Choose a small fixed number of rounds to remember.
- [ ] Add each completed round's moves to a history list.
- [ ] Display the recent moves in `draw_interface()`.
- [ ] Play several rounds and check that only recent moves remain visible.

### Configurable countdown

**Goal:** Let players change how many seconds they have before a move is captured.

**Work in:** `main.py` — `COUNTDOWN_SECONDS` and the code that sets `countdown_end`.

- [ ] Change the countdown duration using `COUNTDOWN_SECONDS`.
- [ ] Run a round and check that the displayed countdown matches.

### Make a winning-streak tracker class

**Goal:** Group streak behavior into a small class instead of keeping several related values in `main()`.

**Pseudocode:**

```text
create a streak tracker when the game starts
after each round:
    if you won: increase the streak
    if the computer won: reset the streak
    if it was a tie: choose whether to keep or reset the streak
when scores reset: reset the streak too
show the current streak on screen
```

This is a design sketch, not required starter code. A class could keep the streak count and provide `record_result(outcome)` and `reset()` methods. Decide how ties behave, then test each result and the score-reset key.

## Test Your Feature

- [ ] Run `python main.py` and play multiple rounds.
- [ ] Test wins, losses, and ties where your feature uses them.
- [ ] Press R and check how your feature behaves after a score reset.
- [ ] Confirm the feature still works after restarting the game.

## One-Minute Team Pitch

Demo the feature, describe one design choice, and share a bug or edge case you tested.
