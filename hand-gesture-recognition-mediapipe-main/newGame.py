from typing import Any
import pygame
import random
import sys
from multiprocessing import Queue
SCORE = 0



#===========================================================================Image loading===================================================
#hand Green
close_left_green = pygame.image.load("assets/Hand_green/close_left.png")
close_right_green = pygame.image.load("assets/Hand_green/close_right.png")
love_left_green = pygame.image.load("assets/Hand_green/love_left.png")
love_right_green = pygame.image.load("assets/Hand_green/love_right.png")
ok_left_green = pygame.image.load("assets/Hand_green/ok_left.png")
ok_right_green = pygame.image.load("assets/Hand_green/ok_right.png")
one_left_green = pygame.image.load("assets/Hand_green/one_left.png")
one_right_green = pygame.image.load("assets/Hand_green/one_right.png")
open_left_green = pygame.image.load("assets/Hand_green/open_left.png")
open_right_green = pygame.image.load("assets/Hand_green/open_right.png")

#hand Red
close_left_red = pygame.image.load("assets/Hand_red/close_left.png")
close_right_red = pygame.image.load("assets/Hand_red/close_right.png")
love_left_red = pygame.image.load("assets/Hand_red/love_left.png")
love_right_red = pygame.image.load("assets/Hand_red/love_right.png")
ok_left_red = pygame.image.load("assets/Hand_red/ok_left.png")
ok_right_red = pygame.image.load("assets/Hand_red/ok_right.png")
one_left_red = pygame.image.load("assets/Hand_red/one_left.png")
one_right_red = pygame.image.load("assets/Hand_red/one_right.png")
open_left_red = pygame.image.load("assets/Hand_red/open_left.png")
open_right_red = pygame.image.load("assets/Hand_red/open_right.png")


#hand white
close_left_white = pygame.image.load("assets/Hand_white/close_left.png")
close_right_white = pygame.image.load("assets/Hand_white/close_right.png")
love_left_white = pygame.image.load("assets/Hand_white/love_left.png")
love_right_white = pygame.image.load("assets/Hand_white/love_right.png")
ok_left_white = pygame.image.load("assets/Hand_white/ok_left.png")
ok_right_white = pygame.image.load("assets/Hand_white/ok_right.png")
one_left_white = pygame.image.load("assets/Hand_white/one_left.png")
one_right_white = pygame.image.load("assets/Hand_white/one_right.png")
open_left_white = pygame.image.load("assets/Hand_white/open_left.png")
open_right_white = pygame.image.load("assets/Hand_white/open_right.png")
#map  and menuBG
map1 = pygame.image.load("assets/Map/map1.png")
map2 = pygame.image.load("assets/Map/map2.png")
map3 = pygame.image.load("assets/Map/map3.png")
menuBg = pygame.image.load("assets/Map/MenuBG.png")

#healtbar
zeroToFour = pygame.image.load("assets/Health_bar/0-4.png")
oneToFour = pygame.image.load("assets/Health_bar/1-4.png")
twoToFour = pygame.image.load("assets/Health_bar/2-4.png")
threeToFour = pygame.image.load("assets/Health_bar/3-4.png")
fourToFour = pygame.image.load("assets/Health_bar/4-4.png")

#=============================================================================================

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
    #Hp
    hpImage = fourToFour
    countHp = 0
    hp_changed = False

    # Define screen dimensions
    screen_width = 900
    screen_height = 700
    x=0
    y=0
    hand_name = "";
    # Create a Pygame window
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pygame Rectangle")
    # Load the background image
    map1 = pygame.image.load("assets/Map/map1.png")


    falling_rectangles_hand = ["Open", "Close","OK"]
    falling_rectangles_side = ["Left", "Right"]
    pygame.font.init()
    font = pygame.font.Font(pygame.font.get_default_font(), 36) 

    # Create a Pygame window
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Pygame Rectangle")

    class FallingRectangle(pygame.sprite.Sprite):
        def __init__(self):
            super().__init__()
            self.rect = pygame.Rect(0, 0, 50, 50)
            self.rect.x = random.randint(0, screen_width - self.rect.width)
            self.rect.y = 0
            self.speed = 1
            self.name = random.choice(falling_rectangles_hand) 
            self.side = random.choice(falling_rectangles_side) 
            self.image = None

        def update(self):
            self.rect.y += self.speed
            if self.rect.y > screen_height:
                self.kill()  # Remove the sprite when it goes off-screen
            if self.name == "Open" and self.side == "Left":
                self.image = pygame.transform.scale(open_left_green, (80,80))
            if self.name == "Open" and self.side == "Right":
                self.image = pygame.transform.scale(open_right_green, (80,80))
            if self.name == "Close" and self.side == "Left":
                self.image = pygame.transform.scale(close_left_green, (80,80))
            if self.name == "Close" and self.side == "Right":
                self.image = pygame.transform.scale(close_right_green, (80,80))
            if self.name == "OK" and self.side == "Left":
                self.image =  pygame.transform.scale(ok_left_green, (80,80))
            if self.name == "OK" and self.side == "Right":
                self.image = pygame.transform.scale(ok_right_green, (80,80))
        #check hand of user is overlapping of FallingRectangle
        def checkIsOverlaping(self, rect_of_hand_user_x, rect_of_hand_user_y, hand_rect, hand_result_name, hand_result_side):
            topRightHand = rect_of_hand_user_x+80;
            global SCORE
            global hpImage
            global countHp 
            topRightRect = self.rect.x+80
            if rect_of_hand_user_x < topRightRect and self.rect.y < topRightHand and self.rect.colliderect(hand_rect) and self.name == hand_result_name and self.side == hand_result_side:
                SCORE +=1
                self.kill()
            # else:
            #     hpLIst = [threeToFour, twoToFour, oneToFour,zeroToFour]
            #     hpImage = hpLIst[countHp]


    falling_rectangles = pygame.sprite.Group()
 
    clock = pygame.time.Clock()
    target_fps = 120
    running = True
    while running:
        if SCORE >= 10 and SCORE <= 20:
            map = map2
        elif SCORE >= 20 and SCORE <= 30:
            map = map3
        else:
            map = map1
        # หยุดเกม
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        # Clear the screen
        screen.fill(WHITE)
        screen.blit(map, (0, 0))
        # สร้างกล่องที่ตกลงมา
        if len(falling_rectangles) < 2 and random.randint(1, 100) < 5:
            falling_rectangles.add(FallingRectangle())

        screen.blit(map, (0, 0))    
        # Update and draw falling rectangles
        falling_rectangles.update()
        falling_rectangles.draw(screen) 

        if not queue.empty():
            start_point = queue.get()
            info_text = queue.get()
            text = info_text.split(":")
            side_hand = text[0]
            hand_name = text[1].strip()
            if(hand_name == "OK"and side_hand=="Left"):
                hand = pygame.transform.scale(ok_left_white, (80,80))
            if(hand_name == "OK"and side_hand=="Right"):
                hand = pygame.transform.scale(ok_right_white, (80,80))
            if(hand_name == "Open" and side_hand=="Left"):
                hand = pygame.transform.scale(open_left_white, (80,80))
            if(hand_name == "Open" and side_hand=="Right"):
                hand = pygame.transform.scale(open_right_white, (80,80))
            if(hand_name == "Close" and side_hand=="Left"):
               hand = pygame.transform.scale(close_left_white, (80,80))
            if(hand_name == "Close" and side_hand=="Right"):
               hand = pygame.transform.scale(close_right_white, (80,80))
            x, y = start_point
            # ตัวจับ event
            hand_rect = pygame.Rect(x, y, 80, 80)
            #check overlapping rectangles
            for rect in falling_rectangles:
                rect.checkIsOverlaping(x, y, hand_rect, hand_name, side_hand)

        # Update the background image      
        #เช็คว่ามือโดนวัตถุไหม
        screen.blit(hand, (x, y))
        #วาด Scores
        score_text = font.render(f"Score: {SCORE}", True, (255, 0,0))
    
        screen.blit(score_text, (10, 10))  # Adjust the position as needed
        screen.blit( pygame.transform.scale(hpImage, (200,100)),(10,20))
        # Draw มือจาก position
        # Update the display
        pygame.display.flip()
        # Cap the frame rate to the target FPS
        clock.tick(target_fps)

if __name__ == '__main__':
    # Create a Queue for communication
    queue = Queue()
    # Start the Pygame process
    run_pygame(queue)
