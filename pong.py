import random
import sys

import pygame


WIDTH, HEIGHT = 960, 540
FPS = 60

PADDLE_WIDTH, PADDLE_HEIGHT = 16, 96
PADDLE_SPEED = 7
AI_SPEED = 6
BALL_SIZE = 16
BALL_SPEED = 6
WIN_SCORE = 7

WHITE = (240, 240, 240)
BLACK = (15, 18, 24)
BLUE = (80, 170, 255)
RED = (255, 110, 110)
GRAY = (90, 95, 110)


def clamp(value, low, high):
    return max(low, min(high, value))


class Paddle:
    def __init__(self, x, y, color):
        self.rect = pygame.Rect(x, y, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.color = color

    def move(self, direction):
        self.rect.y += direction * PADDLE_SPEED
        self.rect.y = clamp(self.rect.y, 0, HEIGHT - PADDLE_HEIGHT)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=8)

    def track_ball(self, ball):
        if ball.rect.centery < self.rect.centery - 8:
            self.rect.y -= AI_SPEED
        elif ball.rect.centery > self.rect.centery + 8:
            self.rect.y += AI_SPEED
        self.rect.y = clamp(self.rect.y, 0, HEIGHT - PADDLE_HEIGHT)


class Ball:
    def __init__(self):
        self.rect = pygame.Rect(0, 0, BALL_SIZE, BALL_SIZE)
        self.reset(random.choice((-1, 1)))

    def reset(self, direction):
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.speed_x = direction * BALL_SPEED
        self.speed_y = random.choice((-1, 1)) * random.uniform(2.5, 4.5)

    def update(self, left_paddle, right_paddle):
        self.rect.x += int(self.speed_x)
        self.rect.y += int(self.speed_y)

        if self.rect.top <= 0:
            self.rect.top = 0
            self.speed_y *= -1
        elif self.rect.bottom >= HEIGHT:
            self.rect.bottom = HEIGHT
            self.speed_y *= -1

        if self.rect.colliderect(left_paddle.rect) and self.speed_x < 0:
            self._bounce(left_paddle)
        elif self.rect.colliderect(right_paddle.rect) and self.speed_x > 0:
            self._bounce(right_paddle)

        if self.rect.left <= 0:
            return "right"
        if self.rect.right >= WIDTH:
            return "left"
        return None

    def _bounce(self, paddle):
        offset = (self.rect.centery - paddle.rect.centery) / (PADDLE_HEIGHT / 2)
        self.speed_x *= -1
        self.speed_x = (abs(self.speed_x) + 0.35) * (1 if self.speed_x > 0 else -1)
        self.speed_y = offset * 5.5

        if self.speed_x > 0:
            self.rect.left = paddle.rect.right
        else:
            self.rect.right = paddle.rect.left

    def draw(self, screen):
        pygame.draw.ellipse(screen, WHITE, self.rect)


def draw_court(screen):
    screen.fill(BLACK)
    pygame.draw.line(screen, GRAY, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 3)
    for y in range(20, HEIGHT, 32):
        pygame.draw.rect(screen, GRAY, (WIDTH // 2 - 3, y, 6, 18), border_radius=3)


def draw_ui(screen, score_left, score_right, font, small_font, message):
    left_text = font.render(str(score_left), True, BLUE)
    right_text = font.render(str(score_right), True, RED)
    title = small_font.render("W/S vs CPU", True, WHITE)

    screen.blit(left_text, (WIDTH // 2 - 110, 30))
    screen.blit(right_text, (WIDTH // 2 + 70, 30))
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 16))

    if message:
        overlay = small_font.render(message, True, WHITE)
        screen.blit(overlay, (WIDTH // 2 - overlay.get_width() // 2, HEIGHT - 40))


def restart_round(left_paddle, right_paddle, ball, serve_direction):
    left_paddle.rect.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
    right_paddle.rect.y = HEIGHT // 2 - PADDLE_HEIGHT // 2
    ball.reset(serve_direction)


def main():
    pygame.init()
    pygame.display.set_caption("Pong")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont("arial", 54, bold=True)
    info_font = pygame.font.SysFont("arial", 28)

    left_paddle = Paddle(36, HEIGHT // 2 - PADDLE_HEIGHT // 2, BLUE)
    right_paddle = Paddle(WIDTH - 36 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, RED)
    ball = Ball()

    score_left = 0
    score_right = 0
    paused = False
    winner = None
    message = "W/S moves. Space pauses. R restarts after a win. Esc quits."

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_SPACE and winner is None:
                    paused = not paused
                    message = "Paused" if paused else "W/S moves. Space pauses. R restarts after a win. Esc quits."
                if event.key == pygame.K_r and winner is not None:
                    score_left = 0
                    score_right = 0
                    winner = None
                    paused = False
                    restart_round(left_paddle, right_paddle, ball, random.choice((-1, 1)))
                    message = "New game started."

        keys = pygame.key.get_pressed()
        if not paused and winner is None:
            if keys[pygame.K_w]:
                left_paddle.move(-1)
            if keys[pygame.K_s]:
                left_paddle.move(1)
            right_paddle.track_ball(ball)

            scorer = ball.update(left_paddle, right_paddle)
            if scorer == "left":
                score_left += 1
                restart_round(left_paddle, right_paddle, ball, 1)
            elif scorer == "right":
                score_right += 1
                restart_round(left_paddle, right_paddle, ball, -1)

            if score_left >= WIN_SCORE:
                winner = "Left player wins! Press R to restart."
                message = winner
            elif score_right >= WIN_SCORE:
                winner = "Right player wins! Press R to restart."
                message = winner

        draw_court(screen)
        left_paddle.draw(screen)
        right_paddle.draw(screen)
        ball.draw(screen)
        draw_ui(screen, score_left, score_right, score_font, info_font, message)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
