# will be used for the following:
# - storing hyperparameters for the neural network
#   - ex: learning rate, batch size, number of epochs, number of layers, etc.
# - storing some path directories
# - NN input/output dimensions

import torch
import os
import math

class PathsConfig:

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    

    RAW_TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'raw-games')
    PROCESSED_TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'processed-games')
    RAW_VALIDATION_DATA_DIR = os.path.join(DATA_DIR, 'raw-validation')
    PROCESSED_VALIDATION_DATA_DIR = os.path.join(DATA_DIR, 'processed-validation')

    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    CHECKPOINT_DIR = os.path.join(MODELS_DIR, 'checkpoints')

    LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
    STEPS_LOG_DIR = os.path.join(DATA_DIR, 'steps')
    POST_VALIDATION_DATA_DIR = os.path.join(DATA_DIR, 'post-validation-data')


class HardwareBasedConfig:

    BATCH_SIZE = 1024

    NUM_WORKERS = 8
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

class TrainingConfig:

    # ~~ Training hyperparameters ~~

    # Peak learning rate for OneCycleLR scheduler.
    # With batch_size=1024, scaled up from the base 0.001 @ batch 64 via linear scaling rule,
    # then tuned conservatively.  OneCycleLR will warm up to this and anneal back down.
    LEARNING_RATE = 0.006

    TOTAL_SAMPLES_TO_TRAIN = 1_000_000_000  # total number of samples to train on

    TOTAL_TARGET_TRAINING_STEPS = TOTAL_SAMPLES_TO_TRAIN // HardwareBasedConfig.BATCH_SIZE + 1

    WEIGHT_DECAY = 0.0001  # L2 regularization (decoupled via AdamW)

    OPTIMIZER = 'adamw'  # AdamW decouples weight decay from adaptive LR

    VALIDATION_SPLIT = 0.02  # 2% of the data will be used for validation

    # Gradient clipping max norm — stabilizes early training when loss is high
    MAX_GRAD_NORM = 1.0

    # ~~ Logging/Checkpointing ~~
    SAVE_CHECKPOINTS_EVERY_N_STEPS = 25_000  # save a checkpoint every N training steps
    VALIDATE_EVERY_N_STEPS = 250_000_000_000_000 # TODO: this currently does not work, so its a super high number to prevent use
    LOG_EVERY_N_STEPS = 100  # log training progress every N training steps (was 1 — disk I/O bottleneck)

    # ~~ Learning Rate Schedule ~~
    # OneCycleLR: warmup for the first 2% of steps, then cosine anneal
    LR_WARMUP_PCT = 0.02  # fraction of total steps used for LR warmup

    # ~~ Loss Weighting ~~
    # Policy CE loss starts at ~ln(4672)≈8.45 while value MSE starts at ~0.5-1.0.
    # Scale value loss so the value head gets meaningful gradient signal.
    VALUE_LOSS_WEIGHT = 1.0  # adjust to 2.0-4.0 if value head is still under-learning


class ModelConfig:

    NUM_INPUT_PLANES = 20  # number of input planes (see fen_parser.py for details)
    
    INPUT_SHAPE = (NUM_INPUT_PLANES, 8, 8)  # input shape for the model

    # The move encoder defines 73 move planes:
    #   56 queen-like (8 directions × 7 distances)
    #   + 8 knight
    #   + 9 underpromotion (3 pieces × 3 directions)
    TOTAL_MOVE_PLANES = 73
    POLICY_HEAD_SIZE = 64 * TOTAL_MOVE_PLANES  # 64 squares × 73 move types = 4672

    # ~~ Model Architecture ~~
    # size table: b = residual block (depth), f = filters (width)
    #
    # |-----------|-----------|------------|------------|------------|
    # | 20b x 64f | 20b x 96f | 20b x 128f | 20b x 196f |*20b x 256f*|  ← recommended
    # |-----------|-----------|------------|------------|------------|
    # | 16b x 64f | 16b x 96f | 16b x 128f | 16b x 196f | 16b x 256f |
    # |-----------|-----------|------------|------------|------------|
    # | 12b x 64f | 12b x 96f | 12b x 128f | 12b x 196f | 12b x 256f |
    # |-----------|-----------|------------|------------|------------|
    # |  8b x 64f |  8b x 96f |  8b x 128f |  8b x 196f | 8b x 256f  |
    # |-----------|-----------|------------|------------|------------|
    # |  4b x 64f |  4b x 96f |  4b x 128f |  4b x 196f | 4b x 256f  |
    # |-----------|-----------|------------|------------|------------|
    #
    # 20b-256f (~24M params) is the sweet spot for RTX 4060 + ~110M positions.
    # Fits comfortably in 8GB VRAM with AMP.  Meaningful step up from 16b-196f (~11.5M).

    NUM_RESIDUAL_BLOCKS = 20
    NUM_FILTERS = 256


# ! IMPORTANT: this is ARBITRARY and again i have NO CLUE what this means, or if we are even going to use it
class SelfPlayConfig:
    """
    Configuration for the self-play data generation process.
    """
    # TODO: Finalize what this will be
    # Path to the compiled C++ engine executable
    ENGINE_EXECUTABLE_PATH = os.path.join(PathsConfig.ROOT_DIR, "build", "HyperionEngine") # Example name

    # ^ IDEK if we are gonna use this, it initially seem likely not
    # Number of MCTS simulations to run for each move during self-play
    MCTS_SIMULATIONS_PER_MOVE = 800

    # Number of games to generate in each self-play iteration
    GAMES_PER_ITERATION = 5000

    # ! IMPORTANT: I have NO CLUE what this means
    # Dirichlet noise alpha value for root node exploration
    DIRICHLET_ALPHA = 0.3

    # ! IMPORTANT: this is ARBITRARY and should be researched more, as it seems it is very important and useful
    # Temperature for move selection during the opening phase of self-play
    # Higher temperature = more exploration.
    OPENING_TEMPERATURE = 1.0
    TEMPERATURE_CUTOFF_MOVE = 30 # After this move, temperature becomes ~0 (play greedily)
