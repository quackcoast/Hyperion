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

    # ! process pieces (layers 0-11; 1.0 for presence of piece, 0.0 for absence)
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

    # ! process side to move (layer 12; 1.0 for white, 0.0 for black)
    if color_to_move == 'w':
        nn_input[SIDE_TO_MOVE_PLANE, :, :] = 1.0

    # ! process castling rights (layers 13-16; 1.0 for available castling rights, 0.0 for unavailable)
    if castling != '-':
        if 'K' in castling:
            nn_input[WK_CASTLE_PLANE, :, :] = 1.0
        if 'Q' in castling:
            nn_input[WQ_CASTLE_PLANE, :, :] = 1.0
        if 'k' in castling:
            nn_input[BK_CASTLE_PLANE, :, :] = 1.0
        if 'q' in castling:
            nn_input[BQ_CASTLE_PLANE, :, :] = 1.0
    
    # ! process en passant target square (layer 17; 1.0 for the square, 0.0 for others)
    if en_passant != '-':
        print(f"En passant square: {en_passant}")
        file_idx = ord(en_passant[0]) - ord('a')  # convert file letter to index (0-7)
        rank_idx = int(en_passant[1]) - 1  # convert rank number to index (0-7)
        nn_input[EN_PASSANT_PLANE, rank_idx, file_idx] = 1.0

    # ! process halfmove clock (layer 18; normalized to [0, 1])
    halfmove_clock = int(halfmove_clock)
    nn_input[FIFTY_MOVE_PLANE, :, :] = min(1, halfmove_clock / 100)

    # ! process fullmove number (layer 19; normalized to [0, 1])
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