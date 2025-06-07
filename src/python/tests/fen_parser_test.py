import hyperion_nn.data_utils.fen_parser as fp
import numpy as np
import pytest

# helper function to check the en passant plane
def check_ep_plane(plane_array, expected_coords_rc=None):
    """
    Checks if the en passant plane is correctly set.
    If expected_coords_rc is None, expects all zeros.
    Otherwise, expects 1.0 at expected_coords_rc (row, col) and 0.0 elsewhere.
    """
    if expected_coords_rc is None:
        assert np.all(plane_array == 0.0), "EP plane should be all zeros"
    else:
        row, col = expected_coords_rc
        expected_plane = np.zeros((8, 8), dtype=np.float32)
        expected_plane[row, col] = 1.0
        assert np.array_equal(plane_array, expected_plane), \
            f"EP plane mismatch. Expected 1.0 at ({row},{col})"

# ! --- Pytest Test Functions ---

def test_output_shape_and_dtype():
    """Test the basic output shape and data type."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    nn_input = fp.fen_to_nn_input(fen)
    assert nn_input.shape == (fp.TOTAL_PLANES, 8, 8)
    assert nn_input.dtype == np.float32

# Parameterize tests for different FEN components
# (fen_string, expected_white_to_move_val)
SIDE_TO_MOVE_CASES = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1.0), # White to move
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", 0.0), # Black to move
    ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", 1.0),
]
@pytest.mark.parametrize("fen, expected_stm_val", SIDE_TO_MOVE_CASES)
def test_side_to_move(fen, expected_stm_val):
    nn_input = fp.fen_to_nn_input(fen)
    assert np.all(nn_input[fp.SIDE_TO_MOVE_PLANE, :, :] == expected_stm_val)

# (fen_string, expected_wk, expected_wq, expected_bk, expected_bq)
CASTLING_CASES = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1.0, 1.0, 1.0, 1.0),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kk - 0 1", 1.0, 0.0, 1.0, 0.0),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Qq - 0 1", 0.0, 1.0, 0.0, 1.0),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1", 0.0, 0.0, 0.0, 0.0),
    ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", 1.0, 1.0, 0.0, 0.0), # White can castle, Black cannot (no rooks/king moved)
    ("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1", 0.0, 0.0, 1.0, 1.0), # Black can castle, White cannot
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1.0, 1.0, 1.0, 1.0), # Kiwipete
]
@pytest.mark.parametrize("fen, wk, wq, bk, bq", CASTLING_CASES)
def test_castling_rights(fen, wk, wq, bk, bq):
    nn_input = fp.fen_to_nn_input(fen)
    assert np.all(nn_input[fp.WK_CASTLE_PLANE, :, :] == wk)
    assert np.all(nn_input[fp.WQ_CASTLE_PLANE, :, :] == wq)
    assert np.all(nn_input[fp.BK_CASTLE_PLANE, :, :] == bk)
    assert np.all(nn_input[fp.BQ_CASTLE_PLANE, :, :] == bq)

# (fen_string, expected_ep_coords_rc) where coords are (row, col) for a1=[0,0] or None
EN_PASSANT_CASES = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", None), # No EP
    ("rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2", (5, 4)), # White can capture on e6 (row 5, col 4)
    ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq d3 0 2", (2, 3)), # Black can capture on d3 (row 2, col 3)
    ("8/4k3/8/8/3pP3/8/4K3/8 w - d3 0 1", (2,3)), # White can capture on d3 (pawn on e2 moved to e4, black pawn on d4)
    ("8/4k3/8/3Pp3/8/8/4K3/8 b - e6 0 1", (5,4)), # Black can capture on e6
]
@pytest.mark.parametrize("fen, expected_ep_coords", EN_PASSANT_CASES)
def test_en_passant(fen, expected_ep_coords):
    nn_input = fp.fen_to_nn_input(fen)
    check_ep_plane(nn_input[fp.EN_PASSANT_PLANE], expected_ep_coords)

# (fen_string, expected_fifty_move_val_normalized, expected_fullmove_val_normalized)
COUNTERS_CASES = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.0/100.0, 1.0/200.0),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 25 50", 25.0/100.0, 50.0/200.0),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 1", 100.0/100.0, 1.0/200.0), # Max fifty move
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 150 1", 100.0/100.0, 1.0/200.0),# Capped fifty move
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 200", 0.0/100.0, 200.0/200.0),# Max fullmove
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 300", 0.0/100.0, 200.0/200.0),# Capped fullmove
]
@pytest.mark.parametrize("fen, expected_fifty, expected_full", COUNTERS_CASES)
def test_counters(fen, expected_fifty, expected_full):
    nn_input = fp.fen_to_nn_input(fen)
    assert np.allclose(nn_input[fp.FIFTY_MOVE_PLANE, :, :], expected_fifty)
    assert np.allclose(nn_input[fp.FULLMOVE_PLANE, :, :], expected_full)

def test_initial_position_piece_planes():
    """Detailed check of piece planes for the initial position."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    nn_input = fp.fen_to_nn_input(fen)

    # White pieces (a1 is [0,0], so white back rank is row 0, white pawns row 1)
    assert np.all(nn_input[fp.P_W_PLANE, 1, :] == 1.0)
    assert nn_input[fp.R_W_PLANE, 0, 0] == 1.0 and nn_input[fp.R_W_PLANE, 0, 7] == 1.0
    assert nn_input[fp.N_W_PLANE, 0, 1] == 1.0 and nn_input[fp.N_W_PLANE, 0, 6] == 1.0
    assert nn_input[fp.B_W_PLANE, 0, 2] == 1.0 and nn_input[fp.B_W_PLANE, 0, 5] == 1.0
    assert nn_input[fp.Q_W_PLANE, 0, 3] == 1.0
    assert nn_input[fp.K_W_PLANE, 0, 4] == 1.0

    # Black pieces (black back rank is row 7, black pawns row 6)
    assert np.all(nn_input[fp.P_B_PLANE, 6, :] == 1.0)
    assert nn_input[fp.R_B_PLANE, 7, 0] == 1.0 and nn_input[fp.R_B_PLANE, 7, 7] == 1.0
    assert nn_input[fp.N_B_PLANE, 7, 1] == 1.0 and nn_input[fp.N_B_PLANE, 7, 6] == 1.0
    assert nn_input[fp.B_B_PLANE, 7, 2] == 1.0 and nn_input[fp.B_B_PLANE, 7, 5] == 1.0
    assert nn_input[fp.Q_B_PLANE, 7, 3] == 1.0
    assert nn_input[fp.K_B_PLANE, 7, 4] == 1.0

    # Ensure other piece planes are empty for squares where these pieces are not
    # Example: White pawn plane should be 0 on row 0
    assert np.all(nn_input[fp.P_W_PLANE, 0, :] == 0.0)


def test_empty_board_with_kings(): # FEN requires kings
    fen = "8/4k3/8/8/8/8/3K4/8 w - - 0 1" # White king on d2, Black king on e7
    nn_input = fp.fen_to_nn_input(fen)
    # White king on d2 (row 1, col 3)
    assert nn_input[fp.K_W_PLANE, 1, 3] == 1.0
    # Black king on e7 (row 6, col 4)
    assert nn_input[fp.K_B_PLANE, 6, 4] == 1.0

    # Sum of all piece planes should be 2 (one for each king)
    total_pieces_on_board = 0
    for i in range(12): # Iterate through all 12 piece planes
        total_pieces_on_board += np.sum(nn_input[i,:,:])
    assert total_pieces_on_board == 2.0

