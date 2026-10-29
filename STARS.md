# RISE STARS: Rock, Paper, Scissors with Python

Make a small, visible change to a webcam game that already works. Work in teams of 2–3; you do not need to understand hand tracking to personalize the game.

There is no required coding experience. Start with the activity that feels comfortable: follow the guided first edit if this is your first time coding, or pick a more challenging idea if you have coded before. You can finish after one change.

## Learning Goals

- [ ] Run a Python project and try its controls.
- [ ] Find and change a value or message in existing code, with help if needed.
- [ ] Test a change and tell others what it does.

## Find Your Way Around

- `game.py` has the game rules and text: the gesture names, random computer move, win/loss/tie messages, and score updates.
- `main.py` runs the webcam game and draws the on-screen interface.
- For a first change, start with the messages in `game.py`. For a visual change, try the `COLOR_` values near the top of `main.py`.
- The computer picks a move randomly. It does not learn from your moves.

You do not need to understand or change the webcam and hand-tracking setup to personalize the game. It is okay to leave unfamiliar code alone.

## 60-Minute Workshop

- **0–10 min:** Meet your team and get an introduction.
- **10–20 min:** Set up the project and play a few rounds.
- **20–30 min:** Explore the two Python files together.
- **30–45 min:** Complete the main activity.
- **45–55 min:** Test and personalize your change.
- **55–60 min:** Give a one-minute team pitch.

## Start Here: Change the Round Messages

**Goal:** Make the win, loss, and tie messages sound like your team.

**Look in:** `game.py` — the `RESULT_MESSAGES` dictionary and `get_result_message()` function.

- [ ] Open `game.py` and find `RESULT_MESSAGES`.
- [ ] Choose one message to change first. Keep the quotation marks and the comma at the end of the line.
- [ ] Save the file, run `python main.py`, and play until you see your message.
- [ ] If that works, change the other two messages too.

**First time editing code?** Change only the words between the quotation marks. For example, change `"You win!"` to `"Team victory!"`, but leave the quotes and punctuation in place. Ask a teammate or instructor to help you find the line if you get stuck.

## Optional Challenges

The message activity is enough for a complete project. If you want another step, choose any idea below that fits your experience, ask a teammate to pair with you, or make up your own.

### New to coding: Personalize one more message

**Look in:** `main.py` — the `round_message` text in `main()`.

- [ ] Find the message shown when a round starts or scores reset.
- [ ] Change only the words inside the quotation marks.
- [ ] Save and press SPACE or R in the game to see your change.

### New to coding: Pick a color theme

**Look in:** `main.py` — the `COLOR_` values near the top and `draw_interface()`.

- [ ] Choose new color values for the interface.
- [ ] Run the game and check that the text is still easy to read.

Color values are three numbers in **blue, green, red** order, each from 0 to 255. If you are not sure where to start, change just one number and ask your instructor or teammate to help choose a color.

### Have coded a little: Change the countdown

**Look in:** `main.py` — `COUNTDOWN_SECONDS` near the top.

- [ ] Change the number of seconds.
- [ ] Start a round and check that the new countdown feels comfortable.

### Have coded a little: Give the game a new name

**Look in:** `main.py` — the `"RISE"` and `"ROCK / PAPER / SCISSORS"` text in `draw_interface()`.

- [ ] Choose a short name or subtitle for your version.
- [ ] Change the displayed text and run the game to see how it looks.

### Ready for a challenge: Add a match goal

**Look in:** `main.py` — the round-result block in `main()`, where scores are updated.

- [ ] Decide how many wins should end a match.
- [ ] After a round, show a message when either side reaches that score.
- [ ] Try a short match and check that it ends at the right time.

### Ready for a challenge: Try a different scoring rule

**Look in:** `game.py` — `update_score()`.

- [ ] Decide how you want wins and losses to change the score.
- [ ] Change the rule and play a round to check it.
- [ ] Ask a teammate what happens on a tie.

## Test Your Game

- [ ] Start the game with `python main.py`.
- [ ] Check that the webcam image appears mirrored.
- [ ] Press SPACE and hold a clear rock, paper, or scissors gesture steady.
- [ ] Check that the result and scores appear.
- [ ] Press R to reset the scores.
- [ ] Press Q to quit.

## One-Minute Team Pitch

Show your change, explain what you edited, and share one thing you learned.
