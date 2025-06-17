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

# AI Learning parameters - IMPROVED
LEARNING_RATE = 0.1  # Increased from 0.01 for faster learning
DISCOUNT_FACTOR = 0.99  # Increased from 0.95 for better long-term planning
EPSILON = 0.1  # Increased from 0.05 for more exploration
EPSILON_DECAY = 0.9995  # New: decay epsilon over time
EPSILON_MIN = 0.01  # New: minimum epsilon value

# AI Training parameters
DEFAULT_EPISODES = 1000
RENDER_EVERY = 100

# AI State discretization - IMPROVED
BIRD_Y_DIVISOR = 30  # Reduced from 50 for finer state representation
BIRD_VELOCITY_DIVISOR = 1  # Reduced from 2 for finer velocity states
PIPE_X_DIVISOR = 30  # Reduced from 50 for finer distance representation
PIPE_GAP_Y_DIVISOR = 30  # Reduced from 50 for finer gap position

# Reward system parameters - IMPROVED
SURVIVAL_REWARD = 0.1  # Reduced from 1.0 to avoid over-rewarding survival
SCORE_REWARD = 50.0  # Increased from 10.0 for stronger score motivation
DEATH_PENALTY = -50.0  # Reduced from -100.0 to avoid too harsh penalties
HEIGHT_PENALTY = -0.05  # Reduced from -0.1
PIPE_PROXIMITY_REWARD = 1.0  # Reduced from 2.0
PIPE_DISTANCE_PENALTY = -0.5  # Reduced from -1.0
FLAP_PENALTY = -0.1  # New: small penalty for unnecessary flapping 