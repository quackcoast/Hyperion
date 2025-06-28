# will be used for the following:
# - the Dataset class that will be used to format the data for the neural network
# - will read the csv and use fen_parser/move_encoder to return a tuple of (nn_input, policy_target, value_target)

import math
import os
import lmdb
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, get_worker_info
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

    def __init__(self, lmdb_file_path=None, csv_file_path=None, force_create_lmdb=False):

        if (lmdb_file_path is None) and (csv_file_path is None):
            raise ValueError("Either lmdb_file_path or csv_file_path must be provided.")
        
        if lmdb_file_path:
            self.lmdb_file_path = lmdb_file_path
        elif csv_file_path:
            self.csv_file_path = csv_file_path
            # Derive the LMDB file path from the CSV file path
            if os.path.dirname(csv_file_path) == config.PathsConfig.RAW_TRAINING_DATA_DIR:
                self.lmdb_file_path = os.path.join(config.PathsConfig.PROCESSED_TRAINING_DATA_DIR, os.path.basename(csv_file_path).replace('.csv', '.lmdb'))
            elif os.path.dirname(csv_file_path) == config.PathsConfig.RAW_VALIDATION_DATA_DIR:
                self.lmdb_file_path = os.path.join(config.PathsConfig.PROCESSED_VALIDATION_DATA_DIR, os.path.basename(csv_file_path).replace('.csv', '.lmdb'))
            else:
                raise ValueError("CSV file must be in a valid raw training or validation data directory.")

       
        if (force_create_lmdb == True) or (not os.path.exists(self.lmdb_file_path)):
            if (force_create_lmdb == True):
                logger.info("Forcing creating LMDBs...")
            else:
                logger.info(f"LMDB file {self.lmdb_file_path} does not exist. Creating LMDB shard...")
            self._create_lmdb_shard()
        

        logger.info(f"Loading LMDB shard from {self.lmdb_file_path}...")

        # start a read-only transaction to ONLY get the length of the dataset
        lmdb_env = lmdb.open(self.lmdb_file_path,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             subdir=False,
                             max_readers=max(config.HardwareBasedConfig.NUM_WORKERS, 1))

        with lmdb_env.begin(buffers=True) as txn:
            raw_len = txn.get(b"__len__")
            if raw_len is None:
                logger.error(f"LMDB shard at {self.lmdb_file_path} is missing the '__len__' entry.")
                raise RuntimeError(f"LMDB shard at {self.lmdb_file_path} is missing the '__len__' entry.")
            self._length = int.from_bytes(raw_len, "little")
        lmdb_env.close()

        self.lmdb_env = None  # Will be initialized in the init_worker() method

    def init_worker(self):
        """
        Initializes the LMDB environment for each worker process.
        This method should be called in the worker_init_fn of the DataLoader.
        """
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading, no need to initialize
            return
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(self.lmdb_file_path,
                                    subdir=False,
                                    readonly=True,
                                    lock=False,
                                    readahead=False,
                                    max_readers=max(config.HardwareBasedConfig.NUM_WORKERS, 1))
            # logger.debug(f"Worker {os.getpid()} opened LMDB {self.lmdb_file_path}")

    def _create_lmdb_shard(self):

        # create the processed data dir if it doesn't exist
        os.makedirs(config.PathsConfig.PROCESSED_TRAINING_DATA_DIR, exist_ok=True)

        csv_file_size = os.path.getsize(self.csv_file_path)
        mmap_target_size = math.ceil(csv_file_size * 1.4)  # 25% larger than the CSV file size
        
        try:
            lmdb_env = lmdb.open(self.lmdb_file_path,
                                map_size=mmap_target_size,
                                subdir=False,       # treat the path as a file, not a directory
                                sync=True,          # force data to be written to disk right after a commit
                                metasync=True,      # force metadata/bookkeeping pages to be written to disk right after a commit
                                map_async=False,    # use asynchronous I/O for writing data to disk while the data is being processed
                                writemap=False)      # write directly to the memory map on the disk
        except Exception as e:
            logger.error(f"Failed to create LMDB environment at {self.lmdb_file_path}: {e}")
            logger.error(f"Target map size was {mmap_target_size} bytes. ({mmap_target_size / (1024*1024):.2f} MB)")
            raise e
        txn = lmdb_env.begin(write=True) # the obj that will handle the transactions (txn means transaction)
        idx = 0


        with open(self.csv_file_path, 'rb') as f:
            progress_bar = tqdm(total=csv_file_size, unit='B', unit_scale=True, desc='Creating LMDB shard')
            for raw_byte_line in f:
                key = f"{idx:010d}".encode('ascii')  # makes a unique key for each line, zero-padded to 10 digits (ex: 0000314271)
                txn.put(key, raw_byte_line)  # store the raw byte line in the lmdb with the key
                progress_bar.update(len(raw_byte_line))
                idx += 1


                # final commit of any remaining
        txn.put(b"__len__", (idx).to_bytes(8, "little")) # metadata entry to store the length of the dataset
        txn.commit()
        lmdb_env.sync()
        lmdb_env.close()
                
        logger.info(f"LMDB shard created successfully at {self.lmdb_file_path} with {idx} entries.")

    def __len__(self):
        """
        Returns the number of rows in the dataset.
        """
        return self._length
    
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
      
        if idx < 0 or idx >= self._length:
            logger.error(f"Index {idx} out of bounds for dataset of length {self._length}.")
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self._length}.")
        
        if self.lmdb_env is None:
            self.init_worker()

        with self.lmdb_env.begin(buffers=False) as txn:
            key = f"{idx:010d}".encode('ascii')
            raw_line_bytes = txn.get(key)
            if raw_line_bytes is None:
                logger.error(f"No data found for index {idx} in LMDB shard.")
                raise RuntimeError(f"No data found for index {idx} in LMDB shard.")
        
        try: 
            line_str = raw_line_bytes.decode('utf-8').strip()
            line_items = line_str.split(',')
            fen_str, uci_move_str, game_outcome_str = line_items
            game_outcome = int(game_outcome_str)

            nn_input_planes_np = fen_parser.fen_to_nn_input(fen_str)
            nn_input_planes = torch.from_numpy(nn_input_planes_np).float()
    
            policy_idx = move_encoder.uci_to_policy_index(uci_move_str, fen_parser.get_piece_at_square(fen_str, uci_move_str[:2]), fen_parser.get_turn(fen_str))
            policy_target = torch.zeros(move_encoder.POLICY_HEAD_SIZE, dtype=torch.float32)
            
            if 0 <= policy_idx < move_encoder.POLICY_HEAD_SIZE:
                policy_target[policy_idx] = 1.0
            else:
                logger.error(f"Policy index {policy_idx} out of bounds for UCI move {uci_move_str} in FEN {fen_str} at sample {idx}.")
                raise ValueError(f"Policy index {policy_idx} out of bounds for UCI move {uci_move_str} in FEN {fen_str} at sample {idx}.")

            value_target = torch.tensor([float(game_outcome)], dtype=torch.float32)
            
            return nn_input_planes, policy_target, value_target
        except ValueError:
            # When int(game_outcome_str) fails
            logger.warning(f"Skipping malformed row at index {idx} due to ValueError. Line content: '{line_str}'")
            return None

                