import pygame
import sys

# Constants
WIDTH, HEIGHT = 800, 600
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
BALL_SIZE = 15
FPS = 60

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pong")
    clock = pygame.time.Clock()

    # Game state
    paddle_a = pygame.Rect(20, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
    paddle_b = pygame.Rect(WIDTH - 20 - PADDLE_WIDTH, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
    ball = pygame.Rect(WIDTH//2, HEIGHT//2, BALL_SIZE, BALL_SIZE)
    ball_speed = [5, 5]
    score_a = 0
    score_b = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Movement
        keys = pygame.key.get_pressed()
        # Player A (Left) - Controlled by User
        if keys[pygame.K_w] and paddle_a.top > 0:
            paddle_a.y -= 7
        if keys[pygame.K_s] and paddle_a.bottom < HEIGHT:
            paddle_a.y += 7

        # Player B (Right) - Controlled by User
        if keys[pygame.K_UP] and paddle_b.top > 0:
            paddle_b.y -= 7
        if keys[pygame.K_DOWN] and paddle_b.bottom < HEIGHT:
            paddle_b.y += 7

        # AI for Paddle A (Left) to track the ball
        if ball.y < paddle_a.centery:
            paddle_a.y -= 5
        elif ball.y > paddle_a.centery:
            paddle_a.y += 5

        # Ball movement
        ball.x += ball_speed[0]
        ball.y += ball_speed[1]

        # Collisions
        if ball.colliderect(paddle_a) or ball.colliderect(paddle_b):
            ball_speed[0] *= -1

        # Wall bounce
        if ball.top <= 0 or ball.bottom >= HEIGHT:
            ball_speed[1] *= -1

        # Scoring
        if ball.left <= 0:
            score_b += 1
            ball.x, ball.y = WIDTH//2, HEIGHT//2
            ball_speed = [5, 5]
        elif ball.right >= WIDTH:
            score_a += 1
            ball.x, ball.y = WIDTH//2, HEIGHT//2
            ball_speed = [-5, -5]

        # Drawing
        screen.fill((0, 0, 0))
        pygame.draw.rect(screen, (255, 255, 255), paddle_a)
        pygame.draw.rect(screen, (255, 255, 255), paddle_b)
        pygame.draw.ellipse(screen, (255, 255, 255), ball)
        pygame.draw.aaline(screen, (255, 255, 255), (WIDTH//2, 0), (WIDTH//2, HEIGHT))

        pygame.display.flip()

        clock.tick(FPS)

if __name__ == "__main__":
    main()