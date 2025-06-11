# will be used for the following:
# - the Dataset class that will be used to format the data for the neural network
# - will read the csv and use fen_parser/move_encoder to return a tuple of (nn_input, policy_target, value_target)

import pickle
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import logging
import hyperion_nn.data_utils.fen_parser as fen_parser
import hyperion_nn.data_utils.move_encoder as move_encoder
import hyperion_nn.config as config
import hyperion_nn.utils.constants as constants


logger = logging.getLogger(__name__)

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

    def __init__(self, force_recalculate_offset=False):

        self.csv_file_path = os.path.join(config.PathsConfig.RAW_TRAINING_DATA_DIR, 'games-2024.csv')
        self.offset_file_path = os.path.join(config.PathsConfig.PROCESSED_TRAINING_DATA_DIR, 'games-2024.csv-offsets.pkl')
        self.offsets = [] # List to store byte offsets for each row in the CSV file

        if (force_recalculate_offset == False) and os.path.exists(self.offset_file_path):
            logger.info(f"Loading offsets from {self.offset_file_path}...")
            try:
                with open(self.offset_file_path, 'rb') as f_offset:
                    self.offsets = pickle.load(f_offset)
                logger.info(f"Loaded {len(self.offsets)} total offsets successfully.")
            
            except Exception as e:
                logger.error(f"Failed to load offsets from {self.offset_file_path}: {e}")
                self.offsets = None
        else:
            if (force_recalculate_offset == True):
                logger.info("Forcing recalculation of offsets...")
            else:
                logger.info(f"Offsets file {self.offset_file_path} does not exist. Recalculating offsets...")
            self._calculate_and_save_offsets()

    def _calculate_and_save_offsets(self):
        """
        Calculate the offsets for each row in the CSV file.
        This is used to quickly access the data without re-reading the entire file.
        """
        logger.info("Calculating offsets...")
        self.offsets = []

        try:
            
            current_byte_offset = 0
            
            with open(self.csv_file_path, 'rb') as f:
                for line_bytes in f:
                    self.offsets.append(current_byte_offset)
                    current_byte_offset += len(line_bytes)
            logger.info(f"Calculated {len(self.offsets)} offsets successfully.")

            # Save the offsets to a file for future use
            with open(self.offset_file_path, 'wb') as f_offset:
                pickle.dump(self.offsets, f_offset)
            logger.info(f"Offsets saved to {self.offset_file_path} successfully.")

        except FileNotFoundError:
            logger.error(f"CSV file {self.csv_file_path} not found. Please check the path.")
            self.offsets = []
        
        except Exception as e:
            logger.error(f"Failed to calculate and save offsets: {e}")
            self.offsets = []

        
    def __len__(self):
        """
        Returns the number of rows in the dataset.
        """
        return len(self.offsets)
    
    # ^ This method may be changed in the future to better handle multithreading, including a init_worker() func
    def __getitem__(self, idx):
        """
        Returns the data for a specific index in the dataset.
        Args:
            idx (int): The index of the item to retrieve.
        Returns:
            tuple: A tuple containing:
                - nn_input_planes (torch.Tensor): The input planes for the neural network.
                - policy_target (torch.Tensor): The one-hot encoded policy target.
                - value_target (torch.Tensor): The value target.
            """


        if not self.offsets:
            logger.error("Offsets not calculated. Please run _calculate_ans_save_offsets() first.")
            raise RuntimeError("Offsets not calculated. Please run _calculate_ans_save_offsets() first.")
        
        if idx < 0 or idx >= len(self.offsets):
            logger.error(f"Index {idx} out of bounds for dataset of length {len(self.offsets)}.")
            raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self.offsets)}.")
        
        targer_offset = self.offsets[idx]

        try:
            with open(self.csv_file_path, 'rb') as f:
                f.seek(targer_offset)
                line_str = f.readline().decode('utf-8').strip()

        except FileNotFoundError:
            logger.error(f"CSV file {self.csv_file_path} not found. Please check the path.")
            raise RuntimeError(f"CSV file {self.csv_file_path} not found. Please check the path.")
        except Exception as e:
            logger.error(f"Failed to read line at offset {targer_offset}: {e}")
            raise RuntimeError(f"Failed to read line at offset {targer_offset}: {e}")
        
        line_items = line_str.split(',')
        if len(line_items) != 3:
            logger.error(f"Invalid line format at offset {targer_offset}: {line_str}")
            raise ValueError(f"Invalid line format at offset {targer_offset}: {line_str}")
        
        fen_str, uci_move_str, game_outcome_str = line_items
        game_outcome = int(game_outcome_str)

        nn_input_planes_np = fen_parser.fen_to_nn_input(fen_str)
        nn_input_planes = torch.tensor(nn_input_planes_np, dtype=torch.float32)
 
        policy_idx = move_encoder.uci_to_policy_index(uci_move_str, fen_parser.get_piece_at_square(fen_str, uci_move_str[:2]), fen_parser.get_turn(fen_str))
        policy_target = torch.zeros(64 * constants.TOTAL_OUTPUT_PLANES, dtype=torch.float32)
        
        if 0 <= policy_idx < 64 * constants.TOTAL_OUTPUT_PLANES:
            policy_target[policy_idx] = 1.0
        else:
            logger.error(f"Policy index {policy_idx} out of bounds for UCI move {uci_move_str} in FEN {fen_str} at sample {idx}.")
            raise ValueError(f"Policy index {policy_idx} out of bounds for UCI move {uci_move_str} in FEN {fen_str} at sample {idx}.")

        value_target = torch.tensor(float(game_outcome), dtype=torch.float32)

        return nn_input_planes, policy_target, value_target

                
