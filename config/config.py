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
PIPE_GAP = 180  # Increased from 150 for a wider gap
GROUND_HEIGHT = 100  # Height of the ground layer (increased for lower ground)

# File paths
HIGH_SCORE_FILE = "high_score.txt"
Q_TABLE_FILE = "q_table.json"

# AI Learning parameters - IMPROVED
LEARNING_RATE = 0.15  # Increased for faster learning
DISCOUNT_FACTOR = 0.99  # High for long-term planning
EPSILON = 0.2  # Higher initial exploration
EPSILON_DECAY = 0.9999  # Slower decay to maintain exploration longer
EPSILON_MIN = 0.05  # Higher minimum to maintain some exploration

# AI Training parameters
DEFAULT_EPISODES = 1000
RENDER_EVERY = 100

# AI State discretization - IMPROVED
BIRD_Y_DIVISOR = 25  # Finer state representation
BIRD_VELOCITY_DIVISOR = 1  # Fine velocity states
PIPE_X_DIVISOR = 25  # Finer distance representation
PIPE_GAP_Y_DIVISOR = 25  # Finer gap position

# Reward system parameters - IMPROVED
SURVIVAL_REWARD = 0.05  # Reduced to avoid over-rewarding survival
SCORE_REWARD = 100.0  # Increased for stronger score motivation
DEATH_PENALTY = -30.0  # Reduced to avoid too harsh penalties
HEIGHT_PENALTY = -0.1  # Penalty for being too high
PIPE_PROXIMITY_REWARD = 2.0  # Reward for being near pipes
PIPE_DISTANCE_PENALTY = -1.0  # Penalty for being far from pipes
FLAP_PENALTY = -0.2  # Penalty for unnecessary flapping
CEILING_COLLISION_PENALTY = -5.0  # Penalty for hitting ceiling 