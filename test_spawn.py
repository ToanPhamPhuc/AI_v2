import pygame
import sys
from main import Bird, Pipe, screen, clock, draw_ground
from config import *

def test_spawn_position():
    """Test to verify bird spawn position and show visual improvements"""
    pygame.init()    
    print("Testing bird spawn position and visual improvements...")
    print("Press SPACE to flap, ESC to quit")
    print("A: Toggle Axes | H: Toggle Hitboxes | C: Toggle Collision Zones")
    
    # Visual toggles
    show_axes = False
    show_hitboxes = False
    show_collision_zones = False
    
    bird = Bird()
    pipes = [Pipe(SCREEN_WIDTH + 200)]
    running = True
    
    while running:
        screen.fill(WHITE)
        draw_ground()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    bird.flap()
                elif event.key == pygame.K_a:
                    show_axes = not show_axes
                    print("Axes: " + ("ON" if show_axes else "OFF"))
                elif event.key == pygame.K_h:
                    show_hitboxes = not show_hitboxes
                    print("Hitboxes: " + ("ON" if show_hitboxes else "OFF"))
                elif event.key == pygame.K_c:
                    show_collision_zones = not show_collision_zones
                    print("Collision Zones: " + ("ON" if show_collision_zones else "OFF"))
        
        # Update bird
        bird.move()
        
        # Draw axes if enabled
        if show_axes:
            # Vertical center line
            pygame.draw.line(screen, (200, 200, 200), (SCREEN_WIDTH//2, 0), (SCREEN_WIDTH//2, SCREEN_HEIGHT), 1)
            # Horizontal center line
            pygame.draw.line(screen, (200, 200, 200), (0, SCREEN_HEIGHT//2), (SCREEN_WIDTH, SCREEN_HEIGHT//2), 1)
            # Gap center line
            if pipes:
                gap_center_y = pipes[0].top_height + PIPE_GAP // 2
                pygame.draw.line(screen, (255, 0, 255), (0, gap_center_y), (SCREEN_WIDTH, gap_center_y), 2)
        
        # Draw pipes
        for pipe in pipes:
            pipe.move()
            pipe.draw()
            if show_hitboxes:
                pygame.draw.rect(screen, RED, (pipe.x, 0, pipe.width, pipe.top_height), 3)
                pygame.draw.rect(screen, RED, (pipe.x, SCREEN_HEIGHT - pipe.bottom_height - GROUND_HEIGHT, pipe.width, pipe.bottom_height), 3)
        
        # Draw bird
        pygame.draw.circle(screen, BLUE, (bird.x, int(bird.y)), bird.radius)
        if show_hitboxes:
            pygame.draw.circle(screen, RED, (bird.x, int(bird.y)), bird.radius, 3)
        
        # Display info
        font = pygame.font.Font(None, 24)
        bird_y_text = font.render("Bird Y: {}".format(int(bird.y)), True, BLACK)
        screen_center_text = font.render("Screen Center: {}".format(SCREEN_HEIGHT//2), True, BLACK)
        gap_center_text = font.render("Gap Center: {}".format(int(pipes[0].top_height + PIPE_GAP//2) if pipes else 'N/A'), True, BLACK)
        
        screen.blit(bird_y_text, (10, 10))
        screen.blit(screen_center_text, (10, 35))
        screen.blit(gap_center_text, (10, 60))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    test_spawn_position() 