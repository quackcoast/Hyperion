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
    constants.Piece.KNIGHT: 0,  # knight underpromotion
    constants.Piece.BISHOP: 1,  # bishop underpromotion
    constants.Piece.ROOK: 2     # rook underpromotion
}

def _get_delta_idx(move: str) -> int:

    start_idx = constants.SQUARE_INDICES[move[:2]]
    end_idx = constants.SQUARE_INDICES[move[2:4]]

    return end_idx - start_idx


def _get_queen_move_dir_and_distance(move: str) -> tuple[int, int]:
    """
    Get the direction and distance of a move.
    
    Args:
        move (str): The UCI move string (e.g., "e2e4").
        
    Returns:
        tuple: (direction delta, distance) where direction is an index for the move direction
               and distance is the number of squares moved.
    """

    start_idx = constants.SQUARE_INDICES[move[:2]]
    end_idx = constants.SQUARE_INDICES[move[2:4]]

    delta_idx = end_idx - start_idx

    start_file = start_idx % 8
    start_rank = start_idx // 8
    end_file = end_idx % 8
    end_rank = end_idx // 8

    if (start_file == end_file): # check if N or S
        distance = abs(end_rank - start_rank)
    else: # (start_rank == end_rank): # check if E or W, and diagonal (since diagonal distance is calculated the same way)
        distance = abs(end_file - start_file)
    # elif (abs(start_file - end_file) == abs(start_rank - end_rank)): # check for diagonals
    #     distance = abs(end_rank - start_rank)

    return (delta_idx // distance, distance)

def _get_underpromotion_dir_offset(move: str, turn: bool) -> int:
    """
    Get the underpromotion offset for a pawn move.
    
    Args:
        move (str): The UCI move string (e.g., "e7e8q").
        
    Returns:
        int: The underpromotion offset.
    """

    delta_idx = _get_delta_idx(move=move)

    if (turn == constants.WHITE):
        if (delta_idx == 8):
            return 0
        elif (delta_idx == 7):
            return 1
        else:
            return 2
    else:
        if (delta_idx == -8):
            return 0
        elif (delta_idx == -9):
            return 1
        else:
            return 2



def uci_to_policy_index(uci_str: str, piece: constants.Piece, turn: bool) -> int:
    """
    Convert a UCI move string into an index for the policy output.
    
    Args:
        uci_str (str): The UCI move string (e.g., "e2e4").
        piece (constants.Piece): The piece being moved.
        turn (bool): True if it's white's turn, False if it's black's turn.

    Returns:
        int: The index for the policy output.
    """

    if (piece == constants.Piece.KNIGHT):
        delta_idx = _get_delta_idx(uci_str)
        move_type_idx = QUEEN_MOVE_PLANES + KNIGHT_DIRECTION_MAP[delta_idx]
    elif (len(uci_str) == 5 and uci_str[4] != 'q'):
        promotion_piece = constants.PIECE_CHAR_MAP[uci_str[4]]
        direction_offset = _get_underpromotion_dir_offset(move=uci_str, turn=turn)
        move_type_idx = QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES + (UNDERPROMOTION_MAP[promotion_piece] * 3 + direction_offset) # 3 here represents the amt of direction offset layers per underpromotion
    else:
        dir_delta, distance = _get_queen_move_dir_and_distance(uci_str)
        move_type_idx = QUEEN_DIRECTION_MAP[dir_delta] * 7 + (distance - 1) # 7 represents the amt of possible distances for a given move dir, -1 is to normalize the distance to an index
        
    return constants.SQUARE_INDICES[uci_str[:2]] * TOTAL_MOVE_PLANES + move_type_idx


def policy_index_to_uci(policy_index: int, piece: constants.Piece, turn: bool) -> str:
    """
    Convert a policy index back to a UCI move string.
    Args:
        policy_index (int): The index in the policy output.
        piece (constants.Piece): The piece being moved.
        turn (bool): True if it's white's turn, False if it's black's turn (using constants library).

    Returns:
        str: The UCI move string.
    """


    from_square_idx = policy_index // TOTAL_MOVE_PLANES
    move_plane_idx = policy_index % TOTAL_MOVE_PLANES

    promotion_char = ""

    if (move_plane_idx > QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES - 1):
        standardized_plane_idx = move_plane_idx - (QUEEN_MOVE_PLANES + KNIGHT_MOVE_PLANES)
        promotion_piece_idx = standardized_plane_idx // 3
        direction_offset = standardized_plane_idx % 3 

        inv_underpromotion_map = {value: key for key, value in UNDERPROMOTION_MAP.items()} # switch the keys and values in the UNDERPROMOTION_MAP dict
    
        promotion_piece = inv_underpromotion_map[promotion_piece_idx]

        if (promotion_piece == constants.Piece.KNIGHT):
            promotion_char = "n"
        elif (promotion_piece == constants.Piece.BISHOP):
            promotion_char = "b"
        else:
            promotion_char = "r"

        if (turn == constants.WHITE):
            if (direction_offset == 0):
                delta_idx = 8
            elif (direction_offset == 1):
                delta_idx = 7
            else:
                delta_idx = 9
        else:
            if (direction_offset == 0):
                delta_idx = -8
            elif (direction_offset == 1):
                delta_idx = -9
            else:
                delta_idx = -7

    elif (move_plane_idx > QUEEN_MOVE_PLANES - 1):
        standardized_plane_idx = move_plane_idx - QUEEN_MOVE_PLANES

        inv_knight_map = {value: key for key, value in KNIGHT_DIRECTION_MAP.items()}

        delta_idx = inv_knight_map[standardized_plane_idx]

    else:
        direction_idx = move_plane_idx // 7
        distance = (move_plane_idx % 7) + 1

        inv_queen_map = {value: key for key, value in QUEEN_DIRECTION_MAP.items()}

        direction_delta = inv_queen_map[direction_idx]

        delta_idx = direction_delta * distance

        if (piece == constants.Piece.PAWN):
            from_rank_idx = from_square_idx // 8
            to_rank_idx = (from_square_idx + delta_idx) // 8

            if (((turn == constants.WHITE) and (from_rank_idx == 6) and (to_rank_idx == 7))
            or ((turn == constants.BLACK) and (from_rank_idx == 1) and (to_rank_idx == 0))):

                promotion_char = "q"

    to_square_idx = from_square_idx + delta_idx

    from_file_char = chr(from_square_idx % 8 + ord('a'))
    from_rank_char = chr(from_square_idx // 8 + ord('1'))
    to_file_char = chr(to_square_idx % 8 + ord('a'))
    to_rank_char = chr(to_square_idx // 8 + ord('1'))

    return f"{from_file_char}{from_rank_char}{to_file_char}{to_rank_char}{promotion_char}"
    



        



        

        

