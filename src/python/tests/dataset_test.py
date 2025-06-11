import os
import pickle
import pytest
import torch
import numpy as np

# Corrected import path for the class under test
from hyperion_nn.data_utils.dataset import ChessDataset

# --- Test Fixtures: Setting up a controlled environment ---

# Save the original os.path.join function BEFORE any tests run
original_os_path_join = os.path.join

@pytest.fixture
def mock_config(mocker, tmp_path):
    """
    Mocks the config module and redirects path joining to a temporary directory.
    This fixture now also handles setting up the dummy CSV file.
    """
    # --- Setup temporary directories ---
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    
    # --- Mock the config object ---
    mock_cfg = mocker.MagicMock()
    mock_cfg.PathsConfig.RAW_TRAINING_DATA_DIR = str(raw_dir) # Use string paths
    mock_cfg.PathsConfig.PROCESSED_TRAINING_DATA_DIR = str(processed_dir)
    
    mock_constants = mocker.MagicMock()
    mock_constants.TOTAL_OUTPUT_PLANES = 73
    
    # --- Patch the modules where they are looked up ---
    mocker.patch('hyperion_nn.data_utils.dataset.config', mock_cfg)
    mocker.patch('hyperion_nn.data_utils.dataset.constants', mock_constants)
    
    # --- THE FIX: Patch os.path.join using the saved original ---
    # This patch is now part of the fixture setup.
    def side_effect_join(*args):
        # This custom side effect calls the REAL os.path.join
        return original_os_path_join(*args)

    mocker.patch('hyperion_nn.data_utils.dataset.os.path.join', side_effect=side_effect_join)
    
    return mock_cfg, mock_constants

@pytest.fixture
def mock_parsers(mocker):
    """A single fixture to mock both fen_parser and move_encoder."""
    # Mock fen_parser functions
    mocker.patch(
        'hyperion_nn.data_utils.dataset.fen_parser.fen_to_nn_input',
        return_value=np.zeros((20, 8, 8), dtype=np.float32)
    )
    mocker.patch('hyperion_nn.data_utils.dataset.fen_parser.get_piece_at_square', return_value='P')
    mocker.patch('hyperion_nn.data_utils.dataset.fen_parser.get_turn', return_value='w')

    # Mock move_encoder function
    def mock_uci_to_policy_index(uci_str, piece, turn):
        if uci_str == "e2e4": return 512
        elif uci_str == "e7e8q": return 1900
        elif uci_str == "bad_move": return -1
        elif uci_str == "out_of_bounds": return 99999
        else: return 100
    
    mocker.patch(
        'hyperion_nn.data_utils.dataset.move_encoder.uci_to_policy_index',
        side_effect=mock_uci_to_policy_index
    )

@pytest.fixture
def dummy_csv_path(mock_config):
    """Creates a dummy CSV file and returns its path."""
    mock_cfg, _ = mock_config
    csv_path = os.path.join(mock_cfg.PathsConfig.RAW_TRAINING_DATA_DIR, "games-2024.csv")
    
    csv_content = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,e2e4,1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1,e7e5,-1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2,g1f3,0",
        "malformed,line",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,d7d5,not_an_int",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,bad_move,1",
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,out_of_bounds,1",
    ]
    
    with open(csv_path, 'w') as f:
        f.write("\n".join(csv_content))
    
    return csv_path


# --- Test Class for ChessDataset ---

class TestChessDataset:

    # Test Initialization and Offset Handling
    def test_init_calculates_offsets_if_pkl_not_exist(self, mock_config, mock_parsers, dummy_csv_path, mocker):
        spy = mocker.spy(ChessDataset, '_calculate_and_save_offsets')
        dataset = ChessDataset()
        spy.assert_called_once()
        assert len(dataset.offsets) == 7
        assert os.path.exists(dataset.offset_file_path)

    def test_init_loads_offsets_if_pkl_exists(self, mock_config, mock_parsers, dummy_csv_path, mocker):
        mock_cfg, _ = mock_config
        offset_path = os.path.join(mock_cfg.PathsConfig.PROCESSED_TRAINING_DATA_DIR, 'games-2024.csv-offsets.pkl')
        dummy_offsets = [0, 10, 20, 30]
        with open(offset_path, 'wb') as f:
            pickle.dump(dummy_offsets, f)
        spy = mocker.spy(ChessDataset, '_calculate_and_save_offsets')
        dataset = ChessDataset()
        spy.assert_not_called()
        assert dataset.offsets == dummy_offsets
        assert len(dataset) == 4

    def test_init_force_recalculates_offsets(self, mock_config, mock_parsers, dummy_csv_path, mocker):
        mock_cfg, _ = mock_config
        offset_path = os.path.join(mock_cfg.PathsConfig.PROCESSED_TRAINING_DATA_DIR, 'games-2024.csv-offsets.pkl')
        with open(offset_path, 'wb') as f:
            pickle.dump([0, 10], f)
        spy = mocker.spy(ChessDataset, '_calculate_and_save_offsets')
        dataset = ChessDataset(force_recalculate_offset=True)
        spy.assert_called_once()
        assert len(dataset.offsets) == 7

    # Test __getitem__ method
    def test_getitem_happy_path(self, mock_config, mock_parsers, dummy_csv_path):
        mock_cfg, mock_const = mock_config
        dataset = ChessDataset()
        nn_input, policy_target, value_target = dataset[0]
        assert isinstance(nn_input, torch.Tensor)
        assert nn_input.shape == (20, 8, 8)
        assert policy_target.shape == (64 * mock_const.TOTAL_OUTPUT_PLANES,)
        assert value_target.item() == 1.0
        assert torch.argmax(policy_target).item() == 512

    def test_getitem_out_of_bounds(self, mock_config, mock_parsers, dummy_csv_path):
        dataset = ChessDataset()
        with pytest.raises(IndexError):
            _ = dataset[99]
        with pytest.raises(IndexError):
            _ = dataset[-1]

    def test_getitem_malformed_line(self, mock_config, mock_parsers, dummy_csv_path):
        dataset = ChessDataset()
        with pytest.raises(ValueError, match="Invalid line format"):
            _ = dataset[3]

    def test_getitem_invalid_outcome_format(self, mock_config, mock_parsers, dummy_csv_path):
        dataset = ChessDataset()
        with pytest.raises(ValueError, match="invalid literal for int()"):
            _ = dataset[4]

    def test_getitem_invalid_policy_index(self, mock_config, mock_parsers, dummy_csv_path):
        dataset = ChessDataset()
        with pytest.raises(ValueError, match="Policy index .* out of bounds"):
            _ = dataset[6]