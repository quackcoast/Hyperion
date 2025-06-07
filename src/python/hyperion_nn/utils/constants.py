from enum import IntEnum

class Piece(IntEnum):
    """
    Enum for the pieces in a chess game.
    """
    EMPTY = 0
    PAWN = 1
    KNIGHT = 2
    BISHOP = 3
    ROOK = 4
    QUEEN = 5
    KING = 6

WHITE = True
BLACK = False

_square_names = [f"{file}{rank}" for rank in "12345678" for file in "abcdefgh"]

SQUARE_INDICIES = IntEnum('SQUARES', _square_names)

PIECE_CHAR_MAP = {
    'P': Piece.PAWN,
    'N': Piece.KNIGHT,
    'B': Piece.BISHOP,
    'R': Piece.ROOK,
    'Q': Piece.QUEEN,
    'K': Piece.KING,
    'p': Piece.PAWN,
    'n': Piece.KNIGHT,
    'b': Piece.BISHOP,
    'r': Piece.ROOK,
    'q': Piece.QUEEN,
    'k': Piece.KING,
    '.': Piece.EMPTY
}