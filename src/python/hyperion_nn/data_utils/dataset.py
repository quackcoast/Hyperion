# will be used for the following:
# - the Dataset class that will be used to format the data for the neural network
# - will read the csv and use fen_parser/move_encoder to return a tuple of (nn_input, policy_target, value_target)

import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

from hyperion_nn.data_utils.fen_parser import fen_to_nn_input
# from hyperion_nn.config import TrainingConfig



class ChessDataset(Dataset):
    """
    Creates a PyTorch Dataset from a CSV file containing chess game data.
    Each row in the CSV file includes:
    - A FEN string
    - A UCI move string
    - The game outcome

    The dataset will then convert it into the following:
    - nn_input_planes: A tensor of shape (num_planes, 8, 8) representing the chessboard state.
    - policy_target (one-hot encoded): A tensor of shape (64, 73) representing the policy target.
    """

    def __init__(self, csv_file_path, config):

        self.config = config
        self.csv_file_path = csv_file_path

        