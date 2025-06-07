# will be used for the following:
# - converting FEN strings into the 8x8x20 NN input format/tensor

import numpy as np
import hyperion_nn.utils.constants as constants

# & Constants:

# core piece planes
P_W_PLANE = 0  # white pawns
N_W_PLANE = 1  # white knights
B_W_PLANE = 2  # white bishops
R_W_PLANE = 3  # white rooks
Q_W_PLANE = 4  # white queens
K_W_PLANE = 5  # white kings
P_B_PLANE = 6  # black pawns
N_B_PLANE = 7  # black knights
B_B_PLANE = 8  # black bishops
R_B_PLANE = 9  # black rooks
Q_B_PLANE = 10 # black queens
K_B_PLANE = 11 # black kings

# auciliary planes
SIDE_TO_MOVE_PLANE = 12 # 1.0 for White, 0.0 for Black
WK_CASTLE_PLANE = 13    # 1.0 if white kingside castling available
WQ_CASTLE_PLANE = 14    # white queenside castling available
BK_CASTLE_PLANE = 15    # black kingside Castling available
BQ_CASTLE_PLANE = 16    # black queenside Castling available
EN_PASSANT_PLANE = 17   # marks the en passant target square
FIFTY_MOVE_PLANE = 18   # normalized halfmove clock (for 50-move rule)
FULLMOVE_PLANE = 19     # normalized fullmove number

# other constants

TOTAL_PLANES = 20 # total layers in the NN input

PIECE_TO_PLANE_MAP = {
    'P': P_W_PLANE,
    'N': N_W_PLANE,
    'B': B_W_PLANE,
    'R': R_W_PLANE,
    'Q': Q_W_PLANE,
    'K': K_W_PLANE,
    'p': P_B_PLANE,
    'n': N_B_PLANE,
    'b': B_B_PLANE,
    'r': R_B_PLANE,
    'q': Q_B_PLANE,
    'k': K_B_PLANE
}

MAX_EXP_MOVE = 200  # max number of expected moves for full move clock normalization

def fen_to_nn_input(fen: str) -> np.ndarray: 
    """
    Convert a FEN string to a tensor representation.
    
    Args:
        fen (str): The FEN string representing the chess position.
        
    Returns:
        np.ndarray: A 3D numpy array (8x8x20) representing the chessboard state.
    """

    nn_input = np.zeros((TOTAL_PLANES, 8, 8), dtype=np.float32)

    fen_parts = fen.split(" ")
    pieces = fen_parts[0]
    color_to_move = fen_parts[1]
    castling = fen_parts[2]
    en_passant = fen_parts[3]
    halfmove_clock = fen_parts[4]
    fullmove_number = fen_parts[5]

    # ! process pieces

    row_idx = 7
    col_idx = 0

    for char in pieces:
        if char == '/':
            row_idx -= 1
            col_idx = 0
        elif char.isdigit():
            col_idx += int(char)
        else:
            plane = PIECE_TO_PLANE_MAP[char]
            nn_input[plane, row_idx, col_idx] = 1.0
            col_idx += 1

    # ! process side to move
    if color_to_move == 'w':
        nn_input[SIDE_TO_MOVE_PLANE, :, :] = 1.0

    # ! process castling rights
    if castling != '-':
        if 'K' in castling:
            nn_input[WK_CASTLE_PLANE, :, :] = 1.0
        if 'Q' in castling:
            nn_input[WQ_CASTLE_PLANE, :, :] = 1.0
        if 'k' in castling:
            nn_input[BK_CASTLE_PLANE, :, :] = 1.0
        if 'q' in castling:
            nn_input[BQ_CASTLE_PLANE, :, :] = 1.0
    
    # ! process en passant target square
    if en_passant != '-':
        print(f"En passant square: {en_passant}")
        file_idx = ord(en_passant[0]) - ord('a')  # convert file letter to index (0-7)
        rank_idx = int(en_passant[1]) - 1  # convert rank number to index (0-7)
        nn_input[EN_PASSANT_PLANE, rank_idx, file_idx] = 1.0

    # ! process halfmove clock
    halfmove_clock = int(halfmove_clock)
    nn_input[FIFTY_MOVE_PLANE, :, :] = min(1, halfmove_clock / 100)

    # ! process fullmove number
    fullmove_number = int(fullmove_number)
    nn_input[FULLMOVE_PLANE, :, :] = min(1, fullmove_number / MAX_EXP_MOVE)

    return nn_input

def get_piece_at_square(fen_str: str, square: str) -> constants.Piece | None:
    """
    Get the piece at a specific square from a FEN string.
    
    Args:
        fen_str (str): The FEN string representing the chess position.
        square (str): The square in algebraic notation (e.g., 'e4').
        
    Returns:
        constants.Piece | None: The piece at the square, or None if empty.
    """
    # Convert square to row and column indices
    col = ord(square[0]) - ord('a')  # 'a' -> 0, 'b' -> 1, ..., 'h' -> 7
    row = int(square[1]) - 1  # '1' -> 0, '2' -> 1, ..., '8' -> 7

    # Parse the FEN string
    pieces = fen_str.split(" ")[0]
    rank_strs = pieces.split("/")
    target_rank = rank_strs[7 - row]  # FEN ranks are from 8 to 1, so we reverse the order

    # find the piece at the specified column
    col_count = 0
    for char in target_rank:
        if char.isdigit():
            col_count += int(char)
        else:
            if col_count == col:
                return constants.PIECE_CHAR_MAP[char]
            
    return None  # If we reach here, the square is empty or invalid

def get_turn(fen_str: str) -> constants.Piece:
    """
    Get the color to move from a FEN string.
    
    Args:
        fen_str (str): The FEN string representing the chess position.
        
    Returns:
        constants.Piece: The color to move (WHITE or BLACK).
    """
    color_to_move = fen_str.split(" ")[1]
    return constants.WHITE if color_to_move == 'w' else constants.BLACK


# # * Test code, to delete later *

# if __name__ == "__main__":
#     start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#     # With a1 = (0,0):
#     # White pawns should be on NumPy row 1.
#     # Black pawns should be on NumPy row 6.

#     print("--- Testing Start Position (a1 is [0,0]) ---")
#     nn_input_start = fen_to_nn_input(start_fen)
#     print(f"Shape: {nn_input_start.shape}")

#     # Verify a few specific plane values for start_fen
#     # White pawns on FEN rank 2 (NumPy row 1)
#     assert np.all(nn_input_start[P_W_PLANE, 1, :] == 1.0), "White pawns not on row 1"
#     # White rooks on FEN rank 1 (NumPy row 0), files a (0) and h (7)
#     assert nn_input_start[R_W_PLANE, 0, 0] == 1.0, "White rook not on a1 (0,0)"
#     assert nn_input_start[R_W_PLANE, 0, 7] == 1.0, "White rook not on h1 (0,7)"

#     # Black pawns on FEN rank 7 (NumPy row 6)
#     assert np.all(nn_input_start[P_B_PLANE, 6, :] == 1.0), "Black pawns not on row 6"
#     # Black rooks on FEN rank 8 (NumPy row 7), files a (0) and h (7)
#     assert nn_input_start[R_B_PLANE, 7, 0] == 1.0, "Black rook not on a8 (7,0)"
#     assert nn_input_start[R_B_PLANE, 7, 7] == 1.0, "Black rook not on h8 (7,7)"

#     # Side to move is white
#     assert np.all(nn_input_start[SIDE_TO_MOVE_PLANE, :, :] == 1.0)
#     # All castling rights available
#     assert np.all(nn_input_start[WK_CASTLE_PLANE, :, :] == 1.0)
#     # ... (other assertions)

#     print("\nBasic assertions passed for start_fen (a1 is [0,0])!")

#     # Test en passant
#     # FEN: "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2"
#     # White to move, en passant possible on e6 (after Black played ...e7-e5)
#     # e6: file 'e' (col 4), rank '6' (NumPy row 5)
#     ep_fen_white_capture = "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 2"
#     nn_input_ep_w = fen_to_nn_input(ep_fen_white_capture)
#     assert nn_input_ep_w[EN_PASSANT_PLANE, 5, 4] == 1.0, "EP square e6 not set correctly"
#     print("EP test for White capture passed!")

#     # FEN: "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq d3 0 2"
#     # Black to move, en passant possible on d3 (after White played d2-d4)
#     # d3: file 'd' (col 3), rank '3' (NumPy row 2)
#     ep_fen_black_capture = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq d3 0 2"
#     nn_input_ep_b = fen_to_nn_input(ep_fen_black_capture)
#     assert nn_input_ep_b[EN_PASSANT_PLANE, 2, 3] == 1.0, "EP square d3 not set correctly"
#     assert np.all(nn_input_ep_b[SIDE_TO_MOVE_PLANE, :, :] == 0.0), "Side to move should be Black"
#     print("EP test for Black capture passed!")