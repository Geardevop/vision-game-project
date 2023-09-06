import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Jumping Character")

# Colors
white = (255, 255, 255)

# Character attributes
character_width = 50
character_height = 50
character_x = width // 2 - character_width // 2
character_y = height - character_height
character_y_velocity = 0
jump_power = -20  # Negative value to move character upwards

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and character_y == height - character_height:
                character_y_velocity = jump_power

    # Update character position
    character_y_velocity += 0.5  # Gravity
    character_y += character_y_velocity

    # Keep character within the screen bounds
    if character_y > height - character_height:
        character_y = height - character_height
        character_y_velocity = 0

    # Clear the screen
    screen.fill(white)

    # Draw character
    pygame.draw.rect(screen, (0, 0, 255), (character_x, character_y, character_width, character_height))

    # Update display
    pygame.display.flip()

# Clean up
pygame.quit()
sys.exit()
