import pytest
from hyperion_nn.data_utils import move_encoder
from hyperion_nn.utils import constants

# Helper to make test cases more readable
# We pass the piece and turn context needed by the functions
def encode_decode_symmetrical(uci_str, piece, turn):
    """Encodes a UCI string and decodes it back, asserting they are identical."""
    # Encode
    policy_index = move_encoder.uci_to_policy_index(uci_str, piece, turn)
    
    # Decode
    # NOTE: The decoder needs the piece and turn context as well to resolve ambiguities
    decoded_uci = move_encoder.policy_index_to_uci(policy_index, piece, turn)
    
    print(f"Testing: {uci_str} -> Idx: {policy_index} -> Decoded: {decoded_uci}")
    assert uci_str == decoded_uci

# --- Test Cases ---
# We use @pytest.mark.parametrize to run the same test function with many different inputs.
# This keeps the test code clean and organized.

# NOTE: hello

# Test Case 1: Standard Pawn Moves
@pytest.mark.parametrize("uci,piece,turn", [
    ("e2e4", constants.Piece.PAWN, constants.WHITE),
    ("d7d5", constants.Piece.PAWN, constants.BLACK),
    ("e4d5", constants.Piece.PAWN, constants.WHITE), # Pawn capture
    ("a2a3", constants.Piece.PAWN, constants.WHITE),
    ("h7h6", constants.Piece.PAWN, constants.BLACK),
])
def test_pawn_moves(uci, piece, turn):
    encode_decode_symmetrical(uci, piece, turn)

# Test Case 2: Knight Moves
@pytest.mark.parametrize("uci,piece,turn", [
    ("g1f3", constants.Piece.KNIGHT, constants.WHITE),
    ("b8c6", constants.Piece.KNIGHT, constants.BLACK),
    ("f3g1", constants.Piece.KNIGHT, constants.WHITE), # Move back
    ("h8g6", constants.Piece.KNIGHT, constants.BLACK),
    ("a1b3", constants.Piece.KNIGHT, constants.WHITE),
])
def test_knight_moves(uci, piece, turn):
    encode_decode_symmetrical(uci, piece, turn)

# Test Case 3: Sliding Moves (Bishop, Rook, Queen)
@pytest.mark.parametrize("uci,piece,turn", [
    # Bishops
    ("f1d3", constants.Piece.BISHOP, constants.WHITE),
    ("c8g4", constants.Piece.BISHOP, constants.BLACK),
    # Rooks
    ("a1a7", constants.Piece.ROOK, constants.WHITE),
    ("h8f8", constants.Piece.ROOK, constants.BLACK),
    # Queens
    ("d1h5", constants.Piece.QUEEN, constants.WHITE),
    ("d8a5", constants.Piece.QUEEN, constants.BLACK),
    ("e4e8", constants.Piece.QUEEN, constants.WHITE),
    ("a1a2", constants.Piece.ROOK, constants.WHITE), 
])
def test_sliding_moves(uci, piece, turn):
    encode_decode_symmetrical(uci, piece, turn)

# Test Case 4: King Moves (including Castling)
# Castling is encoded as a 2-square king move.
@pytest.mark.parametrize("uci,piece,turn", [
    ("e1e2", constants.Piece.KING, constants.WHITE),
    ("e8d7", constants.Piece.KING, constants.BLACK),
    ("e1g1", constants.Piece.KING, constants.WHITE), # White Kingside Castle
    ("e1c1", constants.Piece.KING, constants.WHITE), # White Queenside Castle
    ("e8g8", constants.Piece.KING, constants.BLACK), # Black Kingside Castle
    ("e8c8", constants.Piece.KING, constants.BLACK), # Black Queenside Castle
])
def test_king_moves(uci, piece, turn):
    encode_decode_symmetrical(uci, piece, turn)

# Test Case 5: Queen Promotions (a special type of sliding move)
@pytest.mark.parametrize("uci,piece,turn", [
    ("e7e8q", constants.Piece.PAWN, constants.WHITE), # Forward
    ("a7b8q", constants.Piece.PAWN, constants.WHITE), # Capture
    ("b2a1q", constants.Piece.PAWN, constants.BLACK), # Capture
    ("h2h1q", constants.Piece.PAWN, constants.BLACK), # Forward
])
def test_queen_promotions(uci, piece, turn):
    encode_decode_symmetrical(uci, piece, turn)

# Test Case 6: Underpromotions (the most complex case)
@pytest.mark.parametrize("uci,piece,turn", [
    # To Knight
    ("e7e8n", constants.Piece.PAWN, constants.WHITE),
    ("a7b8n", constants.Piece.PAWN, constants.WHITE),
    ("g2f1n", constants.Piece.PAWN, constants.BLACK),
    # To Bishop
    ("e7d8b", constants.Piece.PAWN, constants.WHITE),
    ("h2g1b", constants.Piece.PAWN, constants.BLACK),
    # To Rook
    ("a7a8r", constants.Piece.PAWN, constants.WHITE),
    ("h2h1r", constants.Piece.PAWN, constants.BLACK),
])
def test_underpromotions(uci, piece, turn):
    encode_decode_symmetrical(uci, piece, turn)

# Test Case 7: Specific Bug Checks
# These test specific bugs found during review of your code.
def test_bug_fixes():
    # Bug 1: Incorrect distance calculation for vertical moves
    # Your original code used abs(end_file - start_file) which is 0 for vertical moves.
    # This test will fail until that is fixed.
    encode_decode_symmetrical("e2e4", constants.Piece.PAWN, constants.WHITE)
    
    # Bug 2: Incorrect indexing for underpromotions in uci_to_policy_index
    # Your original code had `uci_str[5]`, which is out of bounds for a 5-char string.
    # It should be `uci_str[4]`.
    encode_decode_symmetrical("e7e8n", constants.Piece.PAWN, constants.WHITE)

    # Bug 3: Queen promotion check in decoder needs rank variables defined
    # Your original decoder code for queen promotion used from_rank_idx and to_rank_idx
    # before they were defined. This test will fail until they are defined earlier.
    encode_decode_symmetrical("e7e8q", constants.Piece.PAWN, constants.WHITE)

    # Bug 4: Decoding a sliding move in policy_index_to_uci was incorrect.
    # It used the move_plane_idx directly instead of decomposing it into direction and distance.
    encode_decode_symmetrical("a1a7", constants.Piece.ROOK, constants.WHITE)