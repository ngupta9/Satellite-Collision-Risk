# config.py
"""
Configuration constants for the collision risk project.
"""

# Space-Track API (move credentials to environment variables in production)
SPACETRACK_USERNAME = "your_email@example.com"
SPACETRACK_PASSWORD = "your_password"

# Data collection parameters
SATELLITE_LIMIT = 500
DAYS_BACK = 7
MAX_PAIRS = 30000

# Risk thresholds (NASA/ESA operational criteria)
HIGH_RISK_THRESHOLD = 1e-4
MEDIUM_RISK_THRESHOLD = 1e-6

# Dataset parameters
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0005
NUM_EPOCHS = 150

# File paths
CLOSEST_APPROACH_CACHE = f'cache/closest_approach_{MAX_PAIRS//1000}k.pkl'
COLLISION_DATA_CACHE = f'cache/collision_data_{MAX_PAIRS//1000}k.pkl'
PLOT_DATA_FILE = f'pics/collision_data_{MAX_PAIRS//1000}k.png'
PLOT_TRAINING_FILE = f'pics/training_metrics_{MAX_PAIRS//1000}k.png'
PLOT_TEST_FILE = f'pics/test_eval_{MAX_PAIRS//1000}k.png'
MODEL_PATH = f'models/best_collision_model_{MAX_PAIRS//1000}k.pth'