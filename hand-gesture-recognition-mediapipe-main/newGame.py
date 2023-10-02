from typing import Any
import pygame
import random
import sys
from multiprocessing import Queue
SCORE = 0

def run_pygame(queue):
    global SCORE
    # Define colors
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    # Initialize Pygame
    pygame.init()

    #hand picture
    hand = pygame.image.load("C:\\Users\\wallr\\Pictures\\Saved Pictures\\image.png")
    hand = pygame.transform.scale(hand, (80, 80))


    # Define screen dimensions
    screen_width = 900
    screen_height = 700
    x=0
    y=0
    hand_name = "";


    falling_rectangles_text = ["Open", "Close","OK"]
    pygame.font.init()
    font = pygame.font.Font(pygame.font.get_default_font(), 36) 

    # Create a Pygame window
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pygame Rectangle")

    class FallingRectangle(pygame.sprite.Sprite):
        def __init__(self):

            super().__init__()
            self.image = pygame.Surface((50, 50))
            self.image.fill(RED)
            self.rect = self.image.get_rect()
            self.rect.x = random.randint(0, screen_width - self.rect.width)
            self.rect.y = 0
            self.speed = 1
            self.name = random.choice(falling_rectangles_text) 

        def update(self):
            self.rect.y += self.speed
            if self.rect.y > screen_height:
                self.kill()  # Remove the sprite when it goes off-screen
        #check hand of user is overlapping of FallingRectangle
        def checkIsOverlaping(self, rect_of_hand_user_x, rect_of_hand_user_y, hand_rect, hand_result_name):
            topRightHand = rect_of_hand_user_x+80;
            global SCORE
            topRightRect = self.rect.x+80
            if rect_of_hand_user_x < topRightRect and self.rect.y < topRightHand and self.rect.colliderect(hand_rect) and self.name == hand_result_name:
                SCORE +=1
                self.kill()
        def draw_name(self):
            text_surface = font.render(self.name, True, (255, 255, 255))  # Render the text
            text_rect = text_surface.get_rect(center=(self.rect.x + self.rect.width // 2, self.rect.y + self.rect.height // 2))
            screen.blit(text_surface, text_rect)


    falling_rectangles = pygame.sprite.Group()
 
    clock = pygame.time.Clock()
    target_fps = 120
    running = True
    while running:
        # หยุดเกม
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Clear the screen
        screen.fill(WHITE)
        # สร้างกล่องที่ตกลงมา
        if len(falling_rectangles) < 2 and random.randint(1, 100) < 5:
            falling_rectangles.add(FallingRectangle())

        # Update and draw falling rectangles
        falling_rectangles.update()
        falling_rectangles.draw(screen)    
        if not queue.empty():
            start_point = queue.get()
            info_text = queue.get()
            text = info_text.split(":")
            hand_name = text[1].strip()
            x, y = start_point
            # ตัวจับ event
            hand_rect = pygame.Rect(x, y, 80, 80)
            #check overlapping rectangles
            for rect in falling_rectangles:
                rect.draw_name()
                rect.checkIsOverlaping(x, y, hand_rect, hand_name)

        #เช็คว่ามือโดนวัตถุไหม
        screen.blit(hand, (x, y))
        #วาด Scores
        score_text = font.render(f"Score: {SCORE}", True, (255, 0,0))
        screen.blit(score_text, (10, 10))  # Adjust the position as needed
        # Draw มือจาก position
        pygame.draw.rect(screen, GREEN, (0, screen_height - 450, screen_width, 400), 2)
        # Update the display
        pygame.display.flip()
        # Cap the frame rate to the target FPS
        clock.tick(target_fps)

if __name__ == '__main__':
    # Create a Queue for communication
    queue = Queue()

    # Start the Pygame process
    run_pygame(queue)
