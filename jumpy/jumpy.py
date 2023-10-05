#import libraries
import pygame
import random
import os


#===========================================================================Image loading===================================================
#hand Green
close_left_green = pygame.image.load("asset/Hand_green/close_left.png")
close_right_green = pygame.image.load("asset/Hand_green/close_right.png")
love_left_green = pygame.image.load("asset/Hand_green/love_left.png")
love_right_green = pygame.image.load("asset/Hand_green/love_right.png")
ok_left_green = pygame.image.load("asset/Hand_green/ok_left.png")
ok_right_green = pygame.image.load("asset/Hand_green/ok_right.png")
one_left_green = pygame.image.load("asset/Hand_green/one_left.png")
one_right_green = pygame.image.load("asset/Hand_green/one_right.png")
open_left_green = pygame.image.load("asset/Hand_green/open_left.png")
open_right_green = pygame.image.load("asset/Hand_green/open_right.png")

#hand Red
close_left_red = pygame.image.load("asset/Hand_red/close_left.png")
close_right_red = pygame.image.load("asset/Hand_red/close_right.png")
love_left_red = pygame.image.load("asset/Hand_red/love_left.png")
love_right_red = pygame.image.load("asset/Hand_red/love_right.png")
ok_left_red = pygame.image.load("asset/Hand_red/ok_left.png")
ok_right_red = pygame.image.load("asset/Hand_red/ok_right.png")
one_left_red = pygame.image.load("asset/Hand_red/one_left.png")
one_right_red = pygame.image.load("asset/Hand_red/one_right.png")
open_left_red = pygame.image.load("asset/Hand_red/open_left.png")


#hand white
close_left_white = pygame.image.load("asset/Hand_white/close_left.png")
close_right_white = pygame.image.load("asset/Hand_white/close_right.png")
love_left_white = pygame.image.load("asset/Hand_white/love_left.png")
love_right_white = pygame.image.load("asset/Hand_white/love_right.png")
ok_left_white = pygame.image.load("asset/Hand_white/ok_left.png")
ok_right_white = pygame.image.load("asset/Hand_white/ok_right.png")
one_left_white = pygame.image.load("asset/Hand_white/one_left.png")
one_right_white = pygame.image.load("asset/Hand_white/one_right.png")
open_left_white = pygame.image.load("asset/Hand_white/open_left.png")

#map  and menuBG
map1 = pygame.image.load("asset/Map/map1.png")
map2 = pygame.image.load("asset/Map/map2.png")
map3 = pygame.image.load("asset/Map/map3.png")
menuBg = pygame.image.load("asset/Map/MenuBG.png")

#healtbar
zeroToFour = pygame.image.load("asset/Healt_bar/0-4.png")
oneToFour = pygame.image.load("asset/Healt_bar/1-4.png")
twoToFour = pygame.image.load("asset/Healt_bar/2-4.png")
threeToFour = pygame.image.load("asset/Healt_bar/3-4.png")
fourToFour = pygame.image.load("asset/Healt_bar/4-4.png")




#initialise pygame
pygame.init()

#game window dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600

#WAIT SPACEBAR ACTION
clock = pygame.time.Clock()
JUMP_INTERVAL = 200

#create game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Jumpy')

#set frame rate
clock = pygame.time.Clock()
FPS = 60

#game variables
SCROLL_THRESH = 200
GRAVITY = 1
MAX_PLATFORMS = 10
scroll = 0
bg_scroll = 0
game_over = False
score = 0
fade_counter = 0

if os.path.exists('score/score.txt'):
	with open('score/score.txt', 'r') as file:
		high_score = int(file.read())
else:
	high_score = 0

#define colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PANEL = (153, 217, 234)

#define font
font_small = pygame.font.SysFont('Lucida Sans', 20)
font_big = pygame.font.SysFont('Lucida Sans', 24)

#load images
jumpy_image = pygame.image.load('assets/jump.png').convert_alpha()
bg_image = pygame.image.load('assets/bg.png').convert_alpha()
platform_image = pygame.image.load('assets/wood.png').convert_alpha()

#function for outputting text onto the screen
def draw_text(text, font, text_col, x, y):
	img = font.render(text, True, text_col)
	screen.blit(img, (x, y))

#function for drawing info panel
def draw_panel():
	pygame.draw.rect(screen, PANEL, (0, 0, SCREEN_WIDTH, 30))
	pygame.draw.line(screen, WHITE, (0, 30), (SCREEN_WIDTH, 30), 2)
	draw_text('SCORE: ' + str(score), font_small, WHITE, 0, 0)


#function for drawing the background
def draw_bg(bg_scroll):
	screen.blit(bg_image, (0, 0 + bg_scroll))
	screen.blit(bg_image, (0, -600 + bg_scroll))

#player class
class Player():
	def __init__(self, x, y):
		self.image = pygame.transform.scale(jumpy_image, (45, 45))
		self.width = 25
		self.height = 40
		self.rect = pygame.Rect(0, 0, self.width, self.height)
		self.rect.center = (x, y)
		self.vel_y = 0
		self.flip = False
		self.last_jump_time = pygame.time.get_ticks() 

	def move(self):
		#reset variables
		scroll = 0
		dx = 0
		dy = 0
		clock = pygame.time.Clock()
		JUMP_INTERVAL = 200
		#process keypresses
		for event in pygame.event.get():
			if event.type == pygame.KEYUP:
				if event.key == pygame.K_SPACE:
					current_time = pygame.time.get_ticks()
					if current_time - self.last_jump_time >= JUMP_INTERVAL:
						self.vel_y = -20
						self.last_jump_time = current_time
		key = pygame.key.get_pressed()
		if key[pygame.K_a]:
			dx = -10
			self.flip = True
		if key[pygame.K_d]:
			dx = 10
			self.flip = False
		# elif key[pygame.K_SPACE]:
		# 	# if event.key == pygame.K_SPACE:
		# 		self.vel_y = -5
		#gravity
		self.vel_y += GRAVITY
		dy += self.vel_y
		#ensure player doesn't go off the edge of the screen
		if self.rect.left + dx < 0:
			dx = -self.rect.left
		if self.rect.right + dx > SCREEN_WIDTH:
			dx = SCREEN_WIDTH - self.rect.right


		#check collision with platforms
		for platform in platform_group:
			#collision in the y direction
			if platform.rect.colliderect(self.rect.x, self.rect.y + dy, self.width, self.height):
				#check if above the platform
				if self.rect.bottom < platform.rect.centery:
					if self.vel_y > 0:
						self.rect.bottom = platform.rect.top
						dy = 0
						self.vel_y = -1
						print(platform.rect.right)

		#check if the player has bounced to the top of the screen
		if self.rect.top <= SCROLL_THRESH:
			#if player is jumping
			if self.vel_y < 0:
				scroll = -dy

		#update rectangle position
		self.rect.x += dx
		self.rect.y += dy + scroll

		return scroll

	def draw(self):
		screen.blit(pygame.transform.flip(self.image, self.flip, False), (self.rect.x - 12, self.rect.y - 5))
		pygame.draw.rect(screen, WHITE, self.rect, 2)



#platform class
class Platform(pygame.sprite.Sprite):
	def __init__(self, x, y, width, moving):
		pygame.sprite.Sprite.__init__(self)
		self.image = pygame.transform.scale(platform_image, (width, 10))
		self.moving = moving
		self.move_counter = random.randint(0, 100)
		self.direction = random.choice([-1, 1])
		self.speed = random.randint(1, 3)
		self.rect = self.image.get_rect()
		self.rect.x = x
		self.rect.y = y

	def update(self, scroll):
		#moving platform side to side if it is a moving platform
		if self.moving == True:
			self.move_counter += 1
			self.rect.x += self.direction * self.speed

		#change platform direction if it has moved fully or hit a wall
		if self.move_counter >= 100 or self.rect.left < 0 or self.rect.right > SCREEN_WIDTH:
			self.direction *= -1
			self.move_counter = 0

		#update platform's vertical position
		self.rect.y += scroll

		#check if platform has gone off the screen
		if self.rect.top > SCREEN_HEIGHT:
			self.kill()

#player instance
jumpy = Player(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150)

#create sprite groups
platform_group = pygame.sprite.Group()

#create starting platform
platform = Platform(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT - 50, 100, False)
platform_group.add(platform)

#game loop
run = True
while run:

	clock.tick(FPS)

	if game_over == False:
		scroll = jumpy.move()

		#draw background
		bg_scroll += scroll
		if bg_scroll >= 600:
			bg_scroll = 0
		draw_bg(bg_scroll)

		#generate platforms
		if len(platform_group) < MAX_PLATFORMS:
			p_w = random.randint(90, 120)
			p_x = random.randint(0, SCREEN_WIDTH - p_w)
			p_y = platform.rect.y - random.randint(120, 150)
			p_type = 1
			if p_type == 1:
				p_moving = True
			else:
				p_moving = False
			platform = Platform(p_x, p_y, p_w, p_moving)
			platform_group.add(platform)

		#update platforms
		platform_group.update(scroll)

		#update score
		if scroll > 0:
			score += scroll

		#draw line at previous high score
		pygame.draw.line(screen, WHITE, (0, score - high_score + SCROLL_THRESH), (SCREEN_WIDTH, score - high_score + SCROLL_THRESH), 3)
		draw_text('HIGH SCORE', font_small, WHITE, SCREEN_WIDTH - 130, score - high_score + SCROLL_THRESH)

		#draw sprites
		platform_group.draw(screen)
		jumpy.draw()

		#draw panel
		draw_panel()

		#check game over
		if jumpy.rect.top > SCREEN_HEIGHT:
			game_over = True
	else:
		if fade_counter < SCREEN_WIDTH:
			fade_counter += 5
			for y in range(0, 6, 2):
				pygame.draw.rect(screen, BLACK, (0, y * 100, fade_counter, 100))
				pygame.draw.rect(screen, BLACK, (SCREEN_WIDTH - fade_counter, (y + 1) * 100, SCREEN_WIDTH, 100))
		else:
			draw_text('GAME OVER!', font_big, WHITE, 130, 200)
			draw_text('SCORE: ' + str(score), font_big, WHITE, 130, 250)
			draw_text('PRESS SPACE TO PLAY AGAIN', font_big, WHITE, 40, 300)
			#update high score
			if score > high_score:
				high_score = score
				with open('score.txt', 'w') as file:
					file.write(str(high_score))
			key = pygame.key.get_pressed()
			if key[pygame.K_SPACE]:
				#reset variables
				game_over = False
				score = 0
				scroll = 0
				fade_counter = 0
				#reposition jumpy
				jumpy.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT - 150)
				#reset platforms
				platform_group.empty()
				#create starting platform
				platform = Platform(SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT - 50, 100, False)
				platform_group.add(platform)


	#event handler
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			#update high score
			if score > high_score:
				high_score = score
				with open('score.txt', 'w') as file:
					file.write(str(high_score))
			run = False


	#update display window
	pygame.display.update()



pygame.quit()
