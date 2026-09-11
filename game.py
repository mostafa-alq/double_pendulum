import pygame as pg
import math
from collections import deque

from physics import step, Params

pg.init()

# Clock
clock = pg.time.Clock()
FPS = 60

# Consts
FULLSCREEN = False
DEFAULT_WIDTH, DEFAULT_HEIGHT = (1400, 700)
WIDTH = pg.display.Info().current_w if FULLSCREEN == True else DEFAULT_WIDTH
HEIGHT = pg.display.Info().current_h if FULLSCREEN == True else DEFAULT_HEIGHT
GRAPH_PANEL_WIDTH = 220
MAIN_WIDTH = WIDTH - GRAPH_PANEL_WIDTH  # pendulum viewport
BG_COLOUR = (0,0,0)
screen = pg.display.set_mode((WIDTH, HEIGHT))
running = True
RAIL_COLOUR = (255,255,255)
ROD_COLOUR = (255,255,255)
BOB_COLOUR = (150,255,150)
CART_COLOUR = (150,150,150)
WHEEL_COLOUR = (50,50,50)
CART_WIDTH, CART_HEIGHT = 50, 12
WHEEL_RADIUS = 6
TRAIL_LENGTH = 30
RAIL_TICK_COLOUR = (100,100,100)
GAUGE_COLOUR = (0,200,255)
GAUGE_BG_COLOUR = (60,60,60)
GAUGE_WIDTH, GAUGE_HEIGHT = 200, 16
pixels_per_meter = MAIN_WIDTH / 16
GAUGE_X, GAUGE_Y = MAIN_WIDTH // 2 - GAUGE_WIDTH // 2, 20

GRAPH_BG_COLOUR = (30,30,35)
GRAPH_HISTORY = 200
ANGLE_BASE_COLOUR = (100,200,100)
ANGLE_MID_COLOUR = (230,150,60)
PHASE_COLOUR = (230,90,60)
GRID_COLOUR = (70,70,75)

GRAPH_GAP = 8
PANEL_HEIGHT = (HEIGHT - GRAPH_GAP * 2) // 3
GRAPH1_RECT = pg.Rect(MAIN_WIDTH, 0, GRAPH_PANEL_WIDTH, PANEL_HEIGHT)
GRAPH2_RECT = pg.Rect(MAIN_WIDTH, PANEL_HEIGHT + GRAPH_GAP, GRAPH_PANEL_WIDTH, PANEL_HEIGHT)
GRAPH3_RECT = pg.Rect(MAIN_WIDTH, (PANEL_HEIGHT + GRAPH_GAP) * 2, GRAPH_PANEL_WIDTH,
                       HEIGHT - (PANEL_HEIGHT + GRAPH_GAP) * 2)

last_print_time = pg.time.get_ticks()
dt = 1 / FPS
MAX_FORCE = 30.0
X_MAX = 4.0
DAMPING = 15.0

params = Params(g=8.0665, M=1.0, m1=1.0, m2=1.0, l1=1.0, l2=1.5)
state = [0, 0, 1, -3, -1, 5]
trail = deque(maxlen=TRAIL_LENGTH)
theta1_history = deque(maxlen=GRAPH_HISTORY)
theta2_history = deque(maxlen=GRAPH_HISTORY)

# Screen
pg.display.set_caption('Double Pendulum')
font = pg.font.SysFont(None, 24)
small_font = pg.font.SysFont(None, 16)


# Functions
def translate_coordinates(x, y):
    pixel_x = MAIN_WIDTH // 2 + x * pixels_per_meter
    pixel_y = HEIGHT // 2 - y * pixels_per_meter
    return int(pixel_x), int(pixel_y)

def draw_radial_gauge(rect, angle_deg, colour, title, tick_step=45):
    pg.draw.rect(screen, GRAPH_BG_COLOUR, rect)

    cx, cy = rect.centerx, rect.centery + 10
    radius = min(rect.width, rect.height) // 2 - 22

    pg.draw.circle(screen, GRID_COLOUR, (cx, cy), radius, 2)
    for tick_deg in range(0, 360, tick_step):
        tick_rad = math.radians(tick_deg)
        outer = (cx + radius * math.sin(tick_rad), cy - radius * math.cos(tick_rad))
        inner = (cx + (radius - 8) * math.sin(tick_rad), cy - (radius - 8) * math.cos(tick_rad))
        pg.draw.line(screen, GRID_COLOUR, inner, outer, 2)

    theta = math.radians(angle_deg)
    needle_end = (cx + radius * 0.85 * math.sin(theta), cy - radius * 0.85 * math.cos(theta))
    pg.draw.line(screen, colour, (cx, cy), needle_end, 3)
    pg.draw.circle(screen, colour, (cx, cy), 5)

    wrapped_deg = angle_deg % 360
    label = font.render(f"{title}  {wrapped_deg:.1f} deg", True, RAIL_COLOUR)
    screen.blit(label, (rect.left + 4, rect.top + 2))

def draw_phase_graph(rect, history_x, history_y, colour, title, x_range=None, y_range=None):
    pg.draw.rect(screen, GRAPH_BG_COLOUR, rect)
    if len(history_x) >= 2:
        lo_x, hi_x = x_range if x_range is not None else (min(history_x), max(history_x))
        lo_y, hi_y = y_range if y_range is not None else (min(history_y), max(history_y))
        span_x = (hi_x - lo_x) or 1
        span_y = (hi_y - lo_y) or 1
        points = []
        for vx, vy in zip(history_x, history_y):
            px = rect.left + (vx - lo_x) / span_x * rect.width
            py = rect.bottom - (vy - lo_y) / span_y * rect.height
            px = max(rect.left, min(rect.right, px))
            py = max(rect.top, min(rect.bottom, py))
            points.append((px, py))
        pg.draw.lines(screen, colour, False, points, 2)
    label = font.render(title, True, RAIL_COLOUR)
    screen.blit(label, (rect.left + 4, rect.top + 2))

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

    keys = pg.key.get_pressed()
    if keys[pg.K_LEFT]:
        force = -MAX_FORCE
    elif keys[pg.K_RIGHT]:
        force = MAX_FORCE
    else:
        # Brake when the user presses nothing
        force = -DAMPING * state[1]

    state = step(state, force, dt, params)

    if abs(state[0]) > X_MAX:
        state[0] = X_MAX if state[0] > 0 else -X_MAX
        state[1] = 0

    screen.fill(BG_COLOUR)

    # Draw rail
    pg.draw.line(screen, RAIL_COLOUR, translate_coordinates(-4, 0), translate_coordinates(4, 0), 6)
    for meter in range(-4, 5):
        tick_top = translate_coordinates(meter, 0.15)
        tick_bottom = translate_coordinates(meter, -0.15)
        pg.draw.line(screen, RAIL_TICK_COLOUR, tick_top, tick_bottom, 2)

    # Draw pendulum
    cart_x, x1, y1, x2, y2 = get_bob_positions(state, params)
    cart_pos = translate_coordinates(cart_x, 0)
    bob1_pos = translate_coordinates(x1, y1)
    bob2_pos = translate_coordinates(x2, y2)
    trail.append(bob2_pos)
    theta1_history.append(math.degrees(state[2]))
    theta2_history.append(math.degrees(state[4]))

    # Draw fading trail
    for i in range(1, len(trail)):
        frac = i / (len(trail) - 1) if len(trail) > 1 else 1
        faded_colour = tuple(int(c * frac) for c in BOB_COLOUR)
        pg.draw.line(screen, faded_colour, trail[i - 1], trail[i], 3)

    # Draw cart 
    cart_rect = pg.Rect(0, 0, CART_WIDTH, CART_HEIGHT)
    cart_rect.center = cart_pos
    pg.draw.rect(screen, CART_COLOUR, cart_rect)
    pg.draw.circle(screen, WHEEL_COLOUR, (cart_rect.left + 10, cart_rect.bottom), WHEEL_RADIUS)
    pg.draw.circle(screen, WHEEL_COLOUR, (cart_rect.right - 10, cart_rect.bottom), WHEEL_RADIUS)

    pg.draw.line(screen, ROD_COLOUR, cart_pos, bob1_pos, 2)
    pg.draw.line(screen, ROD_COLOUR, bob1_pos, bob2_pos, 2)
    pg.draw.circle(screen, BOB_COLOUR, bob1_pos, 8)
    pg.draw.circle(screen, BOB_COLOUR, bob2_pos, 8)

    # Draw force gauge
    gauge_rect = pg.Rect(GAUGE_X, GAUGE_Y, GAUGE_WIDTH, GAUGE_HEIGHT)
    pg.draw.rect(screen, GAUGE_BG_COLOUR, gauge_rect)
    pg.draw.rect(screen, RAIL_COLOUR, gauge_rect, 2)

    gauge_centre_x = GAUGE_X + GAUGE_WIDTH // 2
    pg.draw.line(screen, RAIL_COLOUR, (gauge_centre_x, GAUGE_Y), (gauge_centre_x, GAUGE_Y + GAUGE_HEIGHT), 2)

    force_frac = max(-1.0, min(1.0, force / MAX_FORCE))
    bar_width = int(force_frac * (GAUGE_WIDTH // 2))
    if bar_width >= 0:
        bar_rect = pg.Rect(gauge_centre_x, GAUGE_Y, bar_width, GAUGE_HEIGHT)
    else:
        bar_rect = pg.Rect(gauge_centre_x + bar_width, GAUGE_Y, -bar_width, GAUGE_HEIGHT)
    pg.draw.rect(screen, GAUGE_COLOUR, bar_rect)

    force_text = font.render(f"Force: {force:+.1f} N", True, RAIL_COLOUR)
    screen.blit(force_text, (GAUGE_X, GAUGE_Y + GAUGE_HEIGHT + 5))

    # Draw the three graphs
    theta1_now = theta1_history[-1] if theta1_history else 0
    theta2_now = theta2_history[-1] if theta2_history else 0
    draw_radial_gauge(GRAPH1_RECT, theta1_now, ANGLE_BASE_COLOUR, "Angle base")
    draw_radial_gauge(GRAPH2_RECT, theta2_now, ANGLE_MID_COLOUR, "Angle mid")
    draw_phase_graph(GRAPH3_RECT, theta1_history, theta2_history, PHASE_COLOUR, "Angle base vs mid",
                      x_range=(theta1_now - 180, theta1_now + 180),
                      y_range=(theta2_now - 180, theta2_now + 180))

    pg.display.flip()