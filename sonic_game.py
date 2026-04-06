import pygame
import sys

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
SKY_BLUE = (135, 206, 235)
SONIC_BLUE = (0, 0, 255)
GREEN = (34, 139, 34)
BROWN = (139, 69, 19)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40, 40))
        self.image.fill(SONIC_BLUE)
        self.rect = self.image.get_rect()
        self.rect.center = (100, SCREEN_HEIGHT - 100)
        
        # Physics variables
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 7
        self.jump_power = -16
        self.gravity = 0.8
        self.is_grounded = False

    def update(self):
        # Input handling
        keys = pygame.key.get_pressed()
        self.vel_x = 0
        if keys[pygame.K_LEFT]:
            self.vel_x = -self.speed
        if keys[pygame.K_RIGHT]:
            self.vel_x = self.speed
        if keys[pygame.K_SPACE] and self.is_grounded:
            self.vel_y = self.jump_power
            self.is_grounded = False

        # Apply gravity
        self.vel_y += self.gravity

        # Apply movement
        self.rect.x += self.vel_x
        self.rect.y += self.vel_y

        # Floor collision (simple)
        if self.rect.bottom > SCREEN_HEIGHT - 50:
            self.rect.bottom = SCREEN_HEIGHT - 50
            self.vel_y = 0
            self.is_grounded = True

        # Screen boundaries
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.right > SCREEN_WIDTH:
            self.rect.right = SCREEN_WIDTH

class Ring(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((20, 10))
        pygame.draw.ellipse(self.image, (255, 215, 0), [0, 0, 20, 10])
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Sonic Clone - Sega Genesis Style")
        self.clock = pygame.time.Clock()
        self.running = True
        self.score = 0

        self.all_sprites = pygame.sprite.Group()
        self.rings = pygame.sprite.Group()

        self.player = Player()
        self.all_sprites.add(self.player)

        # Spawn some rings
        for i in range(5):
            ring = Ring(200 + (i * 100), SCREEN_HEIGHT - 80)
            self.all_sprites.add(ring)
            self.rings.add(ring)

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update()
            self.draw()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        self.all_sprites.update()

        # Ring collection
        hits = pygame.sprite.spritecollide(self.player, self.rings, True)
        for hit in hits:
            self.score += 1

    def draw(self):
        self.screen.fill(SKY_BLUE)
        
        # Draw ground
        pygame.draw.rect(self.screen, GREEN, [0, SCREEN_HEIGHT - 50, SCREEN_WIDTH, 50])
        pygame.draw.rect(self.screen, BROWN, [0, SCREEN_HEIGHT - 30, SCREEN_WIDTH, 30])

        self.all_sprites.draw(self.screen)

        # Draw score
        font = pygame.font.SysFont("Arial", 32)
        score_text = font.render(f"RINGS: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 20))

        pygame.display.flip()

if __name__ == "__main__":
    game = Game()
    game.run()
    pygame.quit()
    sys.exit()
