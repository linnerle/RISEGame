# RISE: Webcam Rock, Paper, Scissors

Play Rock, Paper, Scissors against the computer using hand gestures and your webcam. The computer picks a move at random each round; it does not learn from or adapt to your moves.

## Get started

You need **Python 3.11** and a webcam.

1. Get the project from the GitHub link your instructor shared: select **Code → Download ZIP** and extract it, or clone it with `git clone <repository-url> RISEGame`.
2. Open a terminal in the extracted or cloned `RISEGame` folder. In VS Code, use **File → Open Folder** to open that same folder.
3. Follow the setup steps below.

### 1. Create a virtual environment

Choose the command for your operating system:

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use **Command Prompt** and run `.venv\Scripts\activate.bat`.

**macOS**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

**Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

If `python3.11` is not found, install Python 3.11 first. The pinned MediaPipe version in this project supports Python 3.11.

### 2. Install the packages

Run this with the virtual environment activated:

```bash
python -m pip install -r requirements.txt
```

### 3. Start the game

```bash
python main.py
```

Show one hand to the webcam, press **SPACE**, and hold a clear gesture steady through the countdown.

## Controls

| Key | Action |
| --- | --- |
| **SPACE** | Start a round |
| **R** | Reset both scores |
| **Q** | Quit |

## Using VS Code

Open the project folder in VS Code. If VS Code asks, choose **Yes** to use the `.venv` environment. To select it manually, open the Command Palette (`Ctrl+Shift+P` on Windows/Linux or `Cmd+Shift+P` on macOS), choose **Python: Select Interpreter**, then select the interpreter inside `.venv`.

## Troubleshooting

- **Python command not found:** Install Python 3.11. On Windows, use `py -3.11`; on macOS/Linux, use `python3.11`.
- **`No module named ...`:** Activate `.venv` and run `python -m pip install -r requirements.txt` again.
- **Webcam does not open:** Allow camera access for your terminal or VS Code, connect the webcam, and close other apps that may be using it.
- **No hand is detected:** Improve the lighting, keep your hand in view, and hold a clear rock, paper, or scissors pose steady.

## Workshop guides

- [STARS beginner workshop](STARS.md)
- [SCHOLARS intermediate workshop](SCHOLARS.md)

## Project files

- `main.py` — Webcam, game loop, keyboard controls, and on-screen interface.
- `game.py` — Gesture recognition and Rock, Paper, Scissors game logic.
- `requirements.txt` — Python package versions.
