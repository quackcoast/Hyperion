# will be used for the following:
# - convert UCI strings into a index on the 64x73 NN policy output
# - possibly convert the indices into UCI strings for the policy target

import hyperion_nn.utils.constants as constants

# ~ Constants for move plane representation ~
QUEEN_MOVE_PLANES = 8 * 7 # 8 directions (N, NE, E, SE, S, SW, W, NW) * 7 squares in each direction
KNIGHT_MOVE_PLANES = 8 # 8 possible knight moves
UNDERPROMOTION_MOVE_PLANES = 3 * 3 # 3 underpromotion options (N, E, W) * 3 piece types (N, B, R)
TOTAL_MOVE_PLANES = QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES + UNDERPROMOTION_MOVE_PLANES
POLICY_HEAD_SIZE = 64 * TOTAL_MOVE_PLANES

# ~ Constants for direction delta/delta idx mapping ~
QUEEN_DIRECTION_MAP = {
    8: 0, # a north move (delta index 8) is represented as the direction index 0
    9: 1, # a northeast move (delta index 9) is represented as the direction index 1
    1: 2, # an east move (delta index 1) is represented as the direction index 2
    -7: 3, # a southeast move (delta index -7) is represented as the direction index 3
    -8: 4, # south move; use your brain to figure out the rest
    -9: 5, # southwest move
    -1: 6, # west move
    7: 7  # northwest move
}

KNIGHT_DIRECTION_MAP = {
    17: 0, # NNE move
    10: 1, # ENE move
    -6: 2, # ESE move
    -15: 3, # SSE move
    -17: 4, # SSW move
    -10: 5, # WSW move
    6: 6, # WNW move
    15: 7  # NNW move
}

UNDERPROMOTION_MAP = {
    constants.Piece.KNIGHT: 0,  # Knight underpromotion
    constants.Piece.BISHOP: 1,  # Bishop underpromotion
    constants.Piece.ROOK: 2     # Rook underpromotion
}




def _get_queen_move_dir_and_distance(move: str) -> tuple[int, int]:
    """
    Get the direction and distance of a move.
    
    Args:
        move (str): The UCI move string (e.g., "e2e4").
        
    Returns:
        tuple: (direction delta, distance) where direction is an index for the move direction
               and distance is the number of squares moved.
    """

    start_idx = constants.SQUARE_INDICIES[move[:2]]
    end_idx = constants.SQUARE_INDICIES[move[2:4]]

    delta_idx = end_idx - start_idx

    start_file = start_idx % 8
    start_rank = start_idx // 8
    end_file = end_idx % 8
    end_rank = end_idx // 8

    if (start_file == end_file): # check if N or S
        distance = abs(end_file - start_file)
    else: # (start_rank == end_rank): # check if E or W, and diagonal (since diagonal distance is calculated the same way)
        distance = abs(end_rank - start_rank)
    # elif (abs(start_file - end_file) == abs(start_rank - end_rank)): # check for diagonals 
    #     distance = abs(end_rank - start_rank)

    return (delta_idx // distance, distance)

def _get_knight_move_dir_and_distance(move: str) -> tuple[int, int]:
    """
    Get the direction and distance of a knight move.
    
    Args:
        move (str): The UCI move string (e.g., "g1f3").
        
    Returns:
        tuple: (direction delta, distance) where direction is an index for the knight move direction
               and distance is always 1 for knight moves.
    """
    
    start_idx = constants.SQUARE_INDICIES[move[:2]]
    end_idx = constants.SQUARE_INDICIES[move[2:4]]

    delta_idx = end_idx - start_idx

    return (delta_idx, 1)  # Knight moves always have a distance of 1

def uci_to_policy_index(uci_str: str, piece: constants.Piece, turn: bool) -> int:
    """
    Convert a UCI move string into an index for the policy output.
    
    Args:
        uci_str (str): The UCI move string (e.g., "e2e4").
        piece (constants.Piece): The piece being moved.
        turn (bool): True if it's white's turn, False if it's black's turn.
    """
    
    if piece == constants.Piece.KNIGHT:
        direction_delta, distance = _get_knight_move_dir_and_distance(uci_str)
    elif len(uci_str) == 5 and uci_str[5] != 'q':
        promotion_piece = constants.PIECE_CHAR_MAP[uci_str[5]]

        

