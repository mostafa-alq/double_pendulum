import pygame as pg
import math

from physics import step, Params

pg.init()

# Clock
clock = pg.time.Clock()
FPS = 60

# Consts
FULLSCREEN = False
DEFAULT_WIDTH, DEFAULT_HEIGHT = (800, 450)
WIDTH = pg.display.Info().current_w if FULLSCREEN == True else DEFAULT_WIDTH
HEIGHT = pg.display.Info().current_h if FULLSCREEN == True else DEFAULT_HEIGHT
BG_COLOUR = (0,0,0)
screen = pg.display.set_mode((WIDTH, HEIGHT))
running = True
RAIL_COLOUR = (255,255,255)
PENDULUM_COLOUR = (255,0,0)
pixels_per_meter = WIDTH / 16

last_print_time = pg.time.get_ticks()
dt = 1 / FPS

params = Params(g=8.0665, M=1.0, m1=1.0, m2=1.0, l1=1.0, l2=1.5)
state = [0, 0, 1, -3, -1, 5]

# Screen
pg.display.set_caption('Tutorial 1')


# Functions
def translate_coordinates(x, y):
    pixel_x = WIDTH // 2 + x * pixels_per_meter
    pixel_y = HEIGHT // 2 - y * pixels_per_meter
    return int(pixel_x), int(pixel_y)


def get_bob_positions(state, params):
    x, v, the1, z1, the2, z2 = state
    x1 = x + params.l1 * math.sin(the1)
    y1 = -params.l1 * math.cos(the1)
    x2 = x1 + params.l2 * math.sin(the2)
    y2 = y1 - params.l2 * math.cos(the2)
    return x, x1, y1, x2, y2


# Main Loop
while running:
    for event in pg.event.get():
        mouse_pos_x, mouse_pos_y = pg.mouse.get_pos()
        if event.type == pg.QUIT:
            running = False
    now = pg.time.get_ticks()

    # FPS counter
    if now - last_print_time >= 1000:
        print(clock.get_fps())
        last_print_time = now
    clock.tick(FPS)

    state = step(state, 0.0, dt, params)


    screen.fill(BG_COLOUR)

    # Draw rail
    pg.draw.line(screen, RAIL_COLOUR, translate_coordinates(-4, 0), translate_coordinates(4, 0), 2)

    # Draw pendulum
    cart_x, x1, y1, x2, y2 = get_bob_positions(state, params)
    cart_pos = translate_coordinates(cart_x, 0)
    bob1_pos = translate_coordinates(x1, y1)
    bob2_pos = translate_coordinates(x2, y2)

    pg.draw.line(screen, PENDULUM_COLOUR, cart_pos, bob1_pos, 2)
    pg.draw.line(screen, PENDULUM_COLOUR, bob1_pos, bob2_pos, 2)
    pg.draw.circle(screen, PENDULUM_COLOUR, bob1_pos, 8)
    pg.draw.circle(screen, PENDULUM_COLOUR, bob2_pos, 8)

    pg.display.flip()