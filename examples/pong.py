"""A small local Pong game built with tkinter.

Run with:
    python examples/pong.py
"""

from __future__ import annotations

import random
import tkinter as tk


WIDTH = 960
HEIGHT = 600
PADDLE_WIDTH = 18
PADDLE_HEIGHT = 110
BALL_SIZE = 18
PADDLE_SPEED = 8
BALL_START_SPEED = 5.0
BALL_SPEEDUP = 1.04
WIN_SCORE = 7
FRAME_MS = 16
BG = "#08111b"
FG = "#f4f1de"
ACCENT = "#ff6b35"
ACCENT_2 = "#4cc9f0"


class PongGame:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pong")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            root,
            width=WIDTH,
            height=HEIGHT,
            bg=BG,
            highlightthickness=0,
        )
        self.canvas.pack()

        self.left_score = 0
        self.right_score = 0
        self.running = True
        self.message = ""

        self.left_up = False
        self.left_down = False
        self.right_up = False
        self.right_down = False

        paddle_gap = 36
        self.left_paddle = self.canvas.create_rectangle(
            paddle_gap,
            HEIGHT / 2 - PADDLE_HEIGHT / 2,
            paddle_gap + PADDLE_WIDTH,
            HEIGHT / 2 + PADDLE_HEIGHT / 2,
            fill=ACCENT,
            width=0,
        )
        self.right_paddle = self.canvas.create_rectangle(
            WIDTH - paddle_gap - PADDLE_WIDTH,
            HEIGHT / 2 - PADDLE_HEIGHT / 2,
            WIDTH - paddle_gap,
            HEIGHT / 2 + PADDLE_HEIGHT / 2,
            fill=ACCENT_2,
            width=0,
        )
        self.ball = self.canvas.create_oval(0, 0, BALL_SIZE, BALL_SIZE, fill=FG, width=0)

        self.score_text = self.canvas.create_text(
            WIDTH / 2,
            60,
            text="0   0",
            fill=FG,
            font=("Menlo", 38, "bold"),
        )
        self.help_text = self.canvas.create_text(
            WIDTH / 2,
            HEIGHT - 30,
            text="W/S vs Up/Down  •  Space to pause  •  R to restart",
            fill="#94a3b8",
            font=("Menlo", 14),
        )
        self.center_text = self.canvas.create_text(
            WIDTH / 2,
            HEIGHT / 2,
            text="",
            fill=FG,
            font=("Menlo", 24, "bold"),
        )

        self.ball_dx = BALL_START_SPEED
        self.ball_dy = 0.0
        self.reset_ball(direction=random.choice((-1, 1)))
        self.draw_court()
        self.update_score()
        self.show_message("Ready?")

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.tick()

    def draw_court(self) -> None:
        for y in range(15, HEIGHT, 38):
            self.canvas.create_rectangle(
                WIDTH / 2 - 4,
                y,
                WIDTH / 2 + 4,
                y + 22,
                fill="#243447",
                width=0,
            )

    def on_key_press(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "w":
            self.left_up = True
        elif key == "s":
            self.left_down = True
        elif key == "up":
            self.right_up = True
        elif key == "down":
            self.right_down = True
        elif key == "space":
            self.running = not self.running
            self.show_message("Paused" if not self.running else "")
        elif key == "r":
            self.restart()

    def on_key_release(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "w":
            self.left_up = False
        elif key == "s":
            self.left_down = False
        elif key == "up":
            self.right_up = False
        elif key == "down":
            self.right_down = False

    def restart(self) -> None:
        self.left_score = 0
        self.right_score = 0
        self.running = True
        self.update_score()
        self.center_paddles()
        self.reset_ball(direction=random.choice((-1, 1)))
        self.show_message("New game")

    def center_paddles(self) -> None:
        self.set_paddle_y(self.left_paddle, HEIGHT / 2 - PADDLE_HEIGHT / 2)
        self.set_paddle_y(self.right_paddle, HEIGHT / 2 - PADDLE_HEIGHT / 2)

    def set_paddle_y(self, paddle: int, top: float) -> None:
        x1, _, x2, _ = self.canvas.coords(paddle)
        bottom = top + PADDLE_HEIGHT
        self.canvas.coords(paddle, x1, top, x2, bottom)

    def move_paddle(self, paddle: int, dy: float) -> None:
        x1, y1, x2, y2 = self.canvas.coords(paddle)
        if y1 + dy < 0:
            dy = -y1
        elif y2 + dy > HEIGHT:
            dy = HEIGHT - y2
        self.canvas.move(paddle, 0, dy)

    def reset_ball(self, direction: int) -> None:
        x = WIDTH / 2 - BALL_SIZE / 2
        y = HEIGHT / 2 - BALL_SIZE / 2
        self.canvas.coords(self.ball, x, y, x + BALL_SIZE, y + BALL_SIZE)
        angle = random.uniform(-2.5, 2.5)
        self.ball_dx = BALL_START_SPEED * direction
        self.ball_dy = angle

    def update_score(self) -> None:
        self.canvas.itemconfigure(self.score_text, text=f"{self.left_score}   {self.right_score}")

    def show_message(self, text: str) -> None:
        self.message = text
        self.canvas.itemconfigure(self.center_text, text=text)

    def tick(self) -> None:
        if self.running:
            self.step()
        self.root.after(FRAME_MS, self.tick)

    def step(self) -> None:
        if self.left_up:
            self.move_paddle(self.left_paddle, -PADDLE_SPEED)
        if self.left_down:
            self.move_paddle(self.left_paddle, PADDLE_SPEED)
        if self.right_up:
            self.move_paddle(self.right_paddle, -PADDLE_SPEED)
        if self.right_down:
            self.move_paddle(self.right_paddle, PADDLE_SPEED)

        self.canvas.move(self.ball, self.ball_dx, self.ball_dy)
        bx1, by1, bx2, by2 = self.canvas.coords(self.ball)

        if by1 <= 0 or by2 >= HEIGHT:
            self.ball_dy *= -1

        if self.intersects(self.left_paddle) and self.ball_dx < 0:
            self.bounce_from_paddle(self.left_paddle, 1)
        elif self.intersects(self.right_paddle) and self.ball_dx > 0:
            self.bounce_from_paddle(self.right_paddle, -1)

        if bx2 < 0:
            self.right_score += 1
            self.after_point(direction=-1)
        elif bx1 > WIDTH:
            self.left_score += 1
            self.after_point(direction=1)

    def intersects(self, paddle: int) -> bool:
        px1, py1, px2, py2 = self.canvas.coords(paddle)
        bx1, by1, bx2, by2 = self.canvas.coords(self.ball)
        return bx2 >= px1 and bx1 <= px2 and by2 >= py1 and by1 <= py2

    def bounce_from_paddle(self, paddle: int, direction: int) -> None:
        px1, py1, px2, py2 = self.canvas.coords(paddle)
        bx1, by1, bx2, by2 = self.canvas.coords(self.ball)
        paddle_center = (py1 + py2) / 2
        ball_center = (by1 + by2) / 2
        offset = (ball_center - paddle_center) / (PADDLE_HEIGHT / 2)

        speed = (self.ball_dx ** 2 + self.ball_dy ** 2) ** 0.5 * BALL_SPEEDUP
        self.ball_dx = abs(speed * direction)
        self.ball_dy = speed * offset

        if direction > 0:
            self.canvas.coords(self.ball, px2, by1, px2 + BALL_SIZE, by2)
        else:
            self.canvas.coords(self.ball, px1 - BALL_SIZE, by1, px1, by2)

    def after_point(self, direction: int) -> None:
        self.update_score()
        if self.left_score >= WIN_SCORE or self.right_score >= WIN_SCORE:
            winner = "Left player wins" if self.left_score > self.right_score else "Right player wins"
            self.running = False
            self.show_message(f"{winner}  •  Press R to restart")
            return

        self.center_paddles()
        self.reset_ball(direction=direction)
        self.show_message("Point scored")
        self.root.after(700, lambda: self.show_message(""))


def main() -> None:
    root = tk.Tk()
    PongGame(root)
    root.mainloop()


if __name__ == "__main__":
    main()
