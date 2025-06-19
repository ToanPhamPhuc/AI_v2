import pygame
import json
from config.config import *
import sys
import os
import random
# Initialize Pygame
pygame.init()

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird")
clock = pygame.time.Clock()

PIPE_HEATMAP_FILE = 'data/pipe_heatmap.json'
ADAPTIVE_GAP_FILE = 'data/adaptive_gap_offset.json'

# Load assets
ASSET_DIR = os.path.join(os.path.dirname(__file__), '../flappy-bird/assets')
background_img = pygame.image.load(os.path.join(ASSET_DIR, 'background.png')).convert()
ground_img = pygame.image.load(os.path.join(ASSET_DIR, 'ground.png')).convert_alpha()
pipe_top_img = pygame.image.load(os.path.join(ASSET_DIR, 'pipe_top.png')).convert_alpha()
pipe_bottom_img = pygame.image.load(os.path.join(ASSET_DIR, 'pipe_bottom.png')).convert_alpha()
bird_up_img = pygame.image.load(os.path.join(ASSET_DIR, 'bird_up.png')).convert_alpha()
bird_mid_img = pygame.image.load(os.path.join(ASSET_DIR, 'bird_mid.png')).convert_alpha()
bird_down_img = pygame.image.load(os.path.join(ASSET_DIR, 'bird_down.png')).convert_alpha()

# Global persistent heatmap
if os.path.exists(PIPE_HEATMAP_FILE):
    with open(PIPE_HEATMAP_FILE, 'r') as f:
        _heatmap_data = json.load(f)
        GLOBAL_PIPE_HEATMAP = {
            'top': _heatmap_data.get('top', [0]*SCREEN_HEIGHT),
            'bottom': _heatmap_data.get('bottom', [0]*SCREEN_HEIGHT)
        }
else:
    GLOBAL_PIPE_HEATMAP = {'top': [0]*SCREEN_HEIGHT, 'bottom': [0]*SCREEN_HEIGHT}

if os.path.exists(ADAPTIVE_GAP_FILE):
    with open(ADAPTIVE_GAP_FILE, 'r') as f:
        ADAPTIVE_GAP_OFFSET = json.load(f).get('offset', 0)
else:
    ADAPTIVE_GAP_OFFSET = 0

def save_pipe_heatmap():
    with open(PIPE_HEATMAP_FILE, 'w') as f:
        json.dump(GLOBAL_PIPE_HEATMAP, f)

def load_pipe_heatmap():
    global GLOBAL_PIPE_HEATMAP
    if os.path.exists(PIPE_HEATMAP_FILE):
        with open(PIPE_HEATMAP_FILE, 'r') as f:
            _heatmap_data = json.load(f)
            GLOBAL_PIPE_HEATMAP = {
                'top': _heatmap_data.get('top', [0]*SCREEN_HEIGHT),
                'bottom': _heatmap_data.get('bottom', [0]*SCREEN_HEIGHT)
            }

def load_high_score():
    """Load high score from file"""
    try:
        if os.path.exists(HIGH_SCORE_FILE):
            with open(HIGH_SCORE_FILE, 'r') as file:
                return int(file.read().strip())
    except (ValueError, IOError):
        pass
    return 0

def save_high_score(score):
    """Save high score to file"""
    try:
        with open(HIGH_SCORE_FILE, 'w') as file:
            file.write(str(score))
    except IOError:
        pass

# High score tracking
high_score = load_high_score()

# Bird class
class Bird:
    def __init__(self):
        self.x = 50
        self.y = SCREEN_HEIGHT // 2
        self.radius = 15
        self.velocity = 0
        self.hit_ceiling = False  # Track ceiling collision
        self.width = 34  # Approximate sprite width
        self.height = 24  # Approximate sprite height

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def move(self):
        self.velocity += GRAVITY
        self.y += self.velocity
        # Handle ceiling collision (treat as wall, not death)
        if self.y - self.height // 2 < 0:
            self.y = self.height // 2  # Stop at ceiling
            self.velocity = 0  # Stop upward movement
            self.hit_ceiling = True  # Mark ceiling collision
        else:
            self.hit_ceiling = False  # Reset ceiling collision flag

    def get_frame(self):
        # Choose frame based on velocity
        if self.velocity < -2:
            return bird_up_img
        elif self.velocity > 2:
            return bird_down_img
        else:
            return bird_mid_img

    def get_rect(self):
        img = self.get_frame()
        return img.get_rect(center=(self.x, int(self.y)))

    def draw(self):
        img = self.get_frame()
        rect = img.get_rect(center=(self.x, int(self.y)))
        screen.blit(img, rect)

    def reset(self):
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.hit_ceiling = False

# Pipe class
class Pipe:
    def __init__(self, x):
        self.x = x
        self.top_height = random.randint(50, SCREEN_HEIGHT - PIPE_GAP - GROUND_HEIGHT - 50)
        self.bottom_height = SCREEN_HEIGHT - self.top_height - PIPE_GAP - GROUND_HEIGHT
        self.width = pipe_top_img.get_width()
        self.collision_points = []

    def move(self):
        self.x -= PIPE_SPEED

    def draw(self):
        # Draw top pipe
        top_rect = pipe_top_img.get_rect(bottomleft=(self.x, self.top_height))
        screen.blit(pipe_top_img, top_rect)
        # Draw bottom pipe
        bottom_rect = pipe_bottom_img.get_rect(topleft=(self.x, SCREEN_HEIGHT - self.bottom_height - GROUND_HEIGHT))
        screen.blit(pipe_bottom_img, bottom_rect)
        # Draw heatmap for top pipe
        for y in range(self.top_height):
            hits = GLOBAL_PIPE_HEATMAP['top'][y]
            if hits > 0:
                intensity = min(255, hits * 30)
                color = (0, 0, 128 + intensity//2)
                pygame.draw.line(screen, color, (self.x, y), (self.x + self.width, y))
        # Draw heatmap for bottom pipe
        for y in range(self.bottom_height):
            y_screen = SCREEN_HEIGHT - self.bottom_height - GROUND_HEIGHT + y
            hits = GLOBAL_PIPE_HEATMAP['bottom'][y_screen]
            if hits > 0:
                intensity = min(255, hits * 30)
                color = (0, 0, 128 + intensity//2)
                pygame.draw.line(screen, color, (self.x, y_screen), (self.x + self.width, y_screen))
        # Draw collision points (optional)
        for point in self.collision_points:
            x, y = point
            pygame.draw.circle(screen, (0, 0, 255), (int(x), int(y)), 3)

    def is_off_screen(self):
        return self.x + self.width < 0

    def collides_with(self, bird):
        # Use an ellipse collider for the bird (like Unity's 2D circle collider, but matching the sprite)
        bird_rect = bird.get_rect()
        # Top pipe collision
        top_pipe_rect = pygame.Rect(self.x, 0, self.width, self.top_height)
        if bird_rect.colliderect(top_pipe_rect):
            self.collision_points.append((bird.x, bird.y))
            for y in range(max(0, int(bird.y - bird.height // 2)), min(self.top_height, int(bird.y + bird.height // 2))):
                if 0 <= y < SCREEN_HEIGHT:
                    GLOBAL_PIPE_HEATMAP['top'][y] += 1
            return True
        # Bottom pipe collision
        bottom_pipe_rect = pygame.Rect(self.x, SCREEN_HEIGHT - self.bottom_height - GROUND_HEIGHT, self.width, self.bottom_height)
        if bird_rect.colliderect(bottom_pipe_rect):
            self.collision_points.append((bird.x, bird.y))
            for y in range(max(0, int(bird.y - bird.height // 2)), min(self.bottom_height, int(bird.y + bird.height // 2))):
                y_screen = SCREEN_HEIGHT - self.bottom_height - GROUND_HEIGHT + y
                if 0 <= y_screen < SCREEN_HEIGHT:
                    GLOBAL_PIPE_HEATMAP['bottom'][y_screen] += 1
            return True
        return False

ground_offset = 0  # Global variable for ground movement

def draw_ground():
    """Draw the ground layer using the ground sprite, aligned with the collider at the bottom, and scroll it horizontally."""
    global ground_offset
    ground_y = SCREEN_HEIGHT - GROUND_HEIGHT
    x = -ground_offset
    while x < SCREEN_WIDTH:
        screen.blit(ground_img, (x, ground_y))
        x += ground_img.get_width()

def show_game_over_screen(score):
    global high_score
    if score > high_score:
        high_score = score
        save_high_score(high_score)  # Save new high score to file
    
    screen.fill(WHITE)
    
    # Game Over text
    font_large = pygame.font.Font(None, 72)
    game_over_text = font_large.render("GAME OVER", True, RED)
    game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 100))
    screen.blit(game_over_text, game_over_rect)
    
    # Score text
    font_medium = pygame.font.Font(None, 48)
    score_text = font_medium.render(f"Score: {score}", True, BLACK)
    score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 30))
    screen.blit(score_text, score_rect)
    
    # High Score text
    high_score_text = font_medium.render(f"High Score: {high_score}", True, BLACK)
    high_score_rect = high_score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 20))
    screen.blit(high_score_text, high_score_rect)
    
    # Restart instruction
    font_small = pygame.font.Font(None, 36)
    restart_text = font_small.render("Press SPACE to restart", True, BLACK)
    restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 80))
    screen.blit(restart_text, restart_rect)
    
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

def save_adaptive_gap_offset():
    with open(ADAPTIVE_GAP_FILE, 'w') as f:
        json.dump({'offset': ADAPTIVE_GAP_OFFSET}, f)

def get_adaptive_gap_center(top_height):
    return top_height + PIPE_GAP // 2 + ADAPTIVE_GAP_OFFSET

def adjust_adaptive_gap_offset(direction, step=5):
    global ADAPTIVE_GAP_OFFSET
    if direction == 'down':
        ADAPTIVE_GAP_OFFSET = min(PIPE_GAP//2-10, ADAPTIVE_GAP_OFFSET + step)
    elif direction == 'up':
        ADAPTIVE_GAP_OFFSET = max(-PIPE_GAP//2+10, ADAPTIVE_GAP_OFFSET - step)
    save_adaptive_gap_offset()

# Main game loop
def main():
    global high_score, ground_offset
    bird = Bird()
    pipes = [Pipe(SCREEN_WIDTH + 200)]
    score = 0
    running = True

    while running:
        # Clear screen with white background
        screen.fill(WHITE)
        # Move ground
        ground_offset = (ground_offset - PIPE_SPEED) % ground_img.get_width()
        # Draw ground layer FIRST (behind everything)
        draw_ground()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                bird.flap()

        bird.move()
        bird.draw()

        # Move and draw pipes
        for pipe in pipes:
            pipe.move()
            pipe.draw()
            if pipe.collides_with(bird):
                show_game_over_screen(score)
                # Reset game
                bird.reset()
                pipes = [Pipe(SCREEN_WIDTH + 200)]
                score = 0
                continue

        # Remove off-screen pipes and add new ones
        if pipes[0].is_off_screen():
            pipes.pop(0)
            pipes.append(Pipe(SCREEN_WIDTH + 200))
            score += 1

        # Check if bird hits the ground (removed ceiling check)
        if bird.get_rect().bottom > SCREEN_HEIGHT - GROUND_HEIGHT:
            show_game_over_screen(score)
            # Reset game
            bird.reset()
            pipes = [Pipe(SCREEN_WIDTH + 200)]
            score = 0
            continue

        # Display score (on top layer)
        font = pygame.font.Font(None, 36)
        text = font.render("Score: {}".format(score), True, BLACK)
        screen.blit(text, (10, 10))
        
        # Display high score (on top layer)
        high_score_text = font.render("High Score: {}".format(high_score), True, BLACK)
        screen.blit(high_score_text, (10, 50))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main() 