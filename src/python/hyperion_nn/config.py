# will be used for the following:
# - storing hyperparameters for the neural network
#   - ex: learning rate, batch size, number of epochs, number of layers, etc.
# - storing some path directories
# - NN input/output dimensions

import torch
import os



class PathsConfig:

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    RAW_TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'processed')

    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    CHECKPOINT_DIR = os.path.join(MODELS_DIR, 'checkpoints')

    LOGS_DIR = os.path.join(ROOT_DIR, 'logs')


class TrainingDataConfig:

    # ! IMPORTANT: this is ARBITRARY and should be changed to match the actual hardware capabilities (vram, gpu, etc.)
    BATCH_SIZE = 512 

    # ! IMPORTANT: this is ARBITRARY and should be changed to match the actual hardware capabilities (cores, threads, clock speed, etc.)
    NUM_WORKERS = 4

    VALIDATION_SPLIT = 0.02  # 2% of the data will be used for validation

class TrainingConfig:

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ! IMPORTANT: this is ARBITRARY and should be researched more
    LEARNING_RATE = 0.001  # initial learning rate for the optimizer

    # ^ ! IMPORTANT: this is KINDA ARBITRARY and should be discussed a bit more
    NUM_EPOCHS = 2  # number of epochs to train the model

    # ! IMPORTANT: this is ARBITRARY and i have NO CLUE what this means
    OPTIMIZER = 'adam'  # optimizer to use for training (e.g., 'adam', 'sgd', etc.)
    
    # & This might or might not be used in the future, but it is here for now
    # MOMENTUM = 0.9  # momentum for the optimizer (if applicable, e.g., for SGD)

    # ! IMPORTANT: this is ARBITRARY and should be researched more
    WEIGHT_DECAY = 1e-4  # weight decay for the optimizer (L2 regularization)

    # ^ IMPORTANT: this is ARBITRARY and should be researched more, though it seems to be not that complicated
    POLICY_LOSS_WEIGHT = 1.0  # weight for the policy loss in the total loss calculation
    VALUE_LOSS_WEIGHT = 1.0  # weight for the value loss in the total loss calculation

class ModelConfig:

    NUM_INPUT_PLANES = 20  # number of input planes (see fen_parser.py for details)
    INPUT_SHAPE = (NUM_INPUT_PLANES, 8, 8)  # input shape for the model

    # ! IMPORTANT: these are ARBITRARY and should be changed ;ater with finalized NN Arch
    NUM_RESIDUAL_BLOCKS = -1
    NUM_FILTERS = -2

    POLICY_HEAD_SIZE = 64 * 73  # 64 squares * 73 possible moves (including underpromotions)


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
    
