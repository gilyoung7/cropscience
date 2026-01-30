from pathlib import Path

PATH_DAILY = Path("/home/gpu4080/ygdata/apple/joined_SAMPLE_GDD_2013_2024_APPLE_복숭아순나방.csv")
PATH_OBS   = Path("/home/gpu4080/ygdata/apple/peach_moth_obs_with_fuzzy_features_LONG2.csv")
GDD_DIR    = Path("/home/gpu4080/ygdata/apple/GDD_since_db")

COUNT_COL = "(트랩)복숭아순나방-마리수"

THRESHOLD = 1
SEASON_START_DOY = 1
SEASON_END_DOY   = 365

DOY_START = 60
DOY_END   = 300
MAX_GAP   = 30

SEEDS = [0, 1, 2]
SPLIT_SEED = 42
PATIENCE = 6
MAX_EPOCHS = 60
MIN_DELTA = 1e-3

LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0

W_INTERVAL = 1.0
W_RIGHT    = 0.5
W_LEFT     = 0.5

BATCH_TRAIN = 64
BATCH_EVAL  = 128
NUM_WORKERS = 4
PIN_MEMORY  = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

D_MODEL  = 64
N_HEAD   = 4
N_LAYERS = 3
DROPOUT  = 0.2
MAX_LEN  = 400
