import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pong")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# --- Class Definitions ---

class Paddle(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, color):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.speed = 7

    def update(self, keys):
        # Player 1 movement (W/S)
        if self.y > 0:
            if keys[pygame.K_w]:
                self.y -= self.speed
        if self.y < SCREEN_HEIGHT - self.rect.height:
            if keys[pygame.K_s]:
                self.y += self.speed

class Ball(pygame.sprite.Sprite):
    def __init__(self, x, y, size):
        super().__init__()
        self.size = size
        self.image = pygame.Surface([size, size])
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        
        # Initial random direction
        self.speed_x = 5 * random.choice([-1, 1])
        self.speed_y = 5 * random.choice([-1, 1])
        
        # Speed cap
        self.MAX_SPEED = 10

    def update(self):
        # Apply speed cap
        if abs(self.speed_x) > self.MAX_SPEED:
            self.speed_x = self.speed_x * (self.MAX_SPEED / abs(self.speed_x))
        if abs(self.speed_y) > self.MAX_SPEED:
            self.speed_y = self.speed_y * (self.MAX_SPEED / abs(self.speed_y))

        self.x += self.speed_x
        self.y += self.speed_y

        # Wall collision (Top and Bottom)
        if self.y <= 0 or self.y + self.size >= SCREEN_HEIGHT:
            self.speed_y *= -1

class Game:
    def __init__(self):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Setup Paddles
        self.player1 = Paddle(0, SCREEN_HEIGHT // 2 - 50, 10, 100, WHITE)
        self.player2 = Paddle(SCREEN_WIDTH - 10, SCREEN_HEIGHT // 2 - 50, 10, 100, WHITE)
        
        # Setup Ball
        ball_size = 10
        ball_x = SCREEN_WIDTH // 2 - ball_size // 2
        ball_y = SCREEN_HEIGHT // 2 - ball_size // 2
        self.ball = Ball(ball_x, ball_y, ball_size)
        
        # Speed cap for the ball
        self.ball.MAX_SPEED = 10 
        
        # Scoring
        self.score1 = 0
        self.score2 = 0
        
        # Font setup
        self.font = pygame.font.Font(None, 74)

    def handle_input(self):
        keys = pygame.key.get_pressed()
        self.player1.update(keys)
        self.player2.update(keys)

    def check_collisions(self):
        ball_rect = self.ball.rect

        # Paddle Collision (Player 1 - Left)
        if ball_rect.colliderect(self.player1.rect):
            self.ball.speed_x *= -1.05
            # Adjust ball position to prevent sticking
            if self.ball.speed_x < 0:
                self.ball.rect.right = self.player1.rect.right
            else:
                self.ball.rect.left = self.player1.rect.left

        # Paddle Collision (Player 2 - Right)
        if ball_rect.colliderect(self.player2.rect):
            self.ball.speed_x *= -1.05
            # Adjust ball position to prevent sticking
            if self.ball.speed_x < 0:
                self.ball.rect.right = self.player2.rect.right
            else:
                self.ball.rect.left = self.player2.rect.left

        # Scoring / Reset Ball (Left and Right walls)
        if self.ball.rect.left < 0:
            self.score2 += 1
            self.reset_ball()
        elif self.ball.rect.right > SCREEN_WIDTH:
            self.score1 += 1
            self.reset_ball()

    def reset_ball(self):
        self.ball.x = SCREEN_WIDTH // 2 - self.ball.size // 2
        self.ball.y = SCREEN_HEIGHT // 2 - self.ball.size 

    def draw(self):
        self.screen.fill(BLACK)
        
        # Draw Center Line (optional, but good for Pong)
        pygame.draw.line(self.screen, WHITE, SCREEN_WIDTH // 2, 0, SCREEN_WIDTH // 2, SCREEN_HEIGHT)

        # Draw Paddles
        self.player1.image.fill(WHITE)
        self.player1.rect.topleft = (self.player1.x, self.player1.y)
        self.screen.blit(self.player1.image, self.player1.rect)

        self.player2.image.fill(WHITE)
        self.player2.rect.topleft = (self.player2.x, self.player2.y)
        self.screen.blit(self.player2.image, self.player2.rect)

        # Draw Ball
        self.screen.blit(self.ball.image, self.ball.rect)

        # Draw Score
        score_text = self.font.render(f"{self.score1}  {self.score2}", True, WHITE)
        self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 50))

        pygame.display.flip()

    def run(self):
        while self.running:
            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            # Game Logic Updates
            self.handle_input()
            self.ball.update()
            self.check_collisions()

            # Drawing
            self.draw()

            # Frame Rate Capping
            self.clock.tick(60)
            
        pygame.quit()

if __name__ == '__main__':
    game = Game()
    game.run()
