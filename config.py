# Game Configuration
# All constants and settings for the Flappy Bird game and AI

# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BROWN = (139, 69, 19)  # Ground color

# Game variables
GRAVITY = 0.5
FLAP_STRENGTH = -10
PIPE_SPEED = 5
PIPE_GAP = 150
GROUND_HEIGHT = 50  # Height of the ground layer

# File paths
HIGH_SCORE_FILE = "high_score.txt"
Q_TABLE_FILE = "q_table.json"

# AI Learning parameters
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.95
EPSILON = 0.05

# AI Training parameters
DEFAULT_EPISODES = 1000
RENDER_EVERY = 100

# AI State discretization
BIRD_Y_DIVISOR = 50
BIRD_VELOCITY_DIVISOR = 2
PIPE_X_DIVISOR = 50
PIPE_GAP_Y_DIVISOR = 50

# Reward system parameters
SURVIVAL_REWARD = 1.0
SCORE_REWARD = 10.0
DEATH_PENALTY = -100.0
HEIGHT_PENALTY = -0.1
PIPE_PROXIMITY_REWARD = 2.0
PIPE_DISTANCE_PENALTY = -1.0 