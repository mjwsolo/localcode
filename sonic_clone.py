import pygame
import sys

# Configuration
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
SKY_BLUE = (135, 206, 235)
BROWN = (139, 69, 19)

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((40, 40))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.x = 100
        self.rect.y = SCREEN_HEIGHT - 100
        
        self.vel_x = 0
        self.vel_y = 0
        self.speed = 7
        self.jump_power = -15
        self.gravity = 0.8
        self.is_jumping = False

    def update(self, platforms):
        # Input handling
        keys = pygame.key.get_pressed()
        self.vel_x = 0
        if keys[pygame.K_LEFT]:
            self.vel_x = -self.speed
        if keys[pygame.K_RIGHT]:
            self.vel_x = self.speed

        # Apply gravity
        self.vel_y += self.gravity

        # Move horizontally
        self.rect.x += self.vel_x
        self.check_collision(platforms, 'horizontal')

        # Move vertically
        self.rect.y += self.vel_y
        self.check_collision(platforms, 'vertical')

        # Jump logic
        if keys[pygame.K_SPACE] and not self.is_jumping:
            self.vel_y = self.jump_power
            self.is_jumping = True

    def check_collision(self, platforms, direction):
        hits = pygame.sprite.spritecollide(self, platforms, False)
        if hits:
            if direction == 'horizontal':
                if self.vel_x > 0:
                    self.rect.right = hits[0].rect.left
                elif self.vel_x < 0:
                    self.rect.left = hits[0].rect.right
            else:
                if self.vel_y > 0:
                    self.rect.bottom = hits[0].rect.top
                    self.vel_y = 0
                    self.is_jumping = False
                elif self.vel_y < 0:
                    self.rect.top = hits[0].rect.bottom
                    self.vel_y = 0

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h):
        super().__init__()
        self.image = pygame.Surface((w, h))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Sonic Style Platformer")
    clock = pygame.time.Clock()

    all_sprites = pygame.sprite.Group()
    platforms = pygame.sprite.Group()

    # Create Player
    player = Player()
    all_sprites.add(player)

    # Create Level (Floor and some platforms)
    floor = Platform(0, SCREEN_HEIGHT - 40, SCREEN_WIDTH, 40)
    floor.image.fill(BROWN)
    all_sprites.add(floor)
    platforms.add(floor)

    p1 = Platform(200, 450, 200, 20)
    p2 = Platform(450, 350, 200, 20)
    p3 = Platform(150, 250, 150, 20)
    
    for p in [p1, p2, p3]:
        all_sprites.add(p)
        platforms.add(p)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update
        player.update(platforms)

        # Draw
        screen.fill(SKY_BLUE)
        all_sprites.draw(screen)
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()