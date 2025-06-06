#ifndef HYPERION_CORE_POSITION_HPP
#define HYPERION_CORE_POSITION_HPP

#include "bitboard.hpp"
#include "constants.hpp"
#include "zobrist.hpp"
#include <string>
#include <array>
#include <vector>

namespace hyperion {
namespace core {

struct Move;

class Position {
public:
    // --- Bitboards ---
    // One bitboard for each piece type and color
    // piece_bbs[piece_type_e][color_e (WHITE/BLACK)]
    std::array<std::array<bitboard_t, 2>, NUM_PIECE_TYPES> piece_bbs;

    // Combined bitboards for faster access
    std::array<bitboard_t, 2> color_bbs; // color_bbs[WHITE] = all white pieces, color_bbs[BLACK] = all black pieces
    bitboard_t occupied_bb;              // All occupied squares (color_bbs[WHITE] | color_bbs[BLACK])

    // --- Game State ---
    int side_to_move;
    int castling_rights;       // Use constants like WK_CASTLE_FLAG, etc.
    square_e en_passant_square;  // NO_SQ if no EP square
    int halfmove_clock;        // For 50-move rule
    int fullmove_number;       // Incremented after Black's move

    zobrist_key_t current_hash; // Current position's Zobrist hash

    // A "mailbox" representation: board[square_index] = piece_char or piece_enum
    std::array<int, NUM_SQUARES> board_mailbox; // Stores combined piece type and color, or EMPTY_SQUARE

public:
    Position(); // Default constructor: sets up starting position
    void set_from_fen(const std::string& fen_string);
    std::string to_fen() const;

    // --- Accessors ---
    int get_side_to_move() const { return side_to_move; }
    bitboard_t get_pieces(piece_type_e p_type, int p_color) const;
    bitboard_t get_pieces_by_type(piece_type_e p_type) const; // All pieces of a type (e.g. all pawns)
    bitboard_t get_pieces_by_color(int p_color) const;
    bitboard_t get_occupied_squares() const;
    square_e get_king_square(int king_color) const;

    //mailbox mutators
    int make_mailbox_entry(piece_type_e type, int color) const;
    piece_type_e  get_piece_type_from_mailbox_val(int mb_val) const;
    int get_color_from_mailbox_val(int mb_val) const;

    // Piece on a specific square
    int get_piece_on_square(square_e sq) const; // Returns combined piece_type & color, or EMPTY_SQUARE

    // --- Move Execution ---
    // Returns true if the move was legal and made, false otherwise 
    void make_move(const Move& m);
    void unmake_move(const Move& m); // Needs the move that was made

    // --- Legality & Game State Checks ---
    // Checks if a square is attacked by the given color
    bool is_square_attacked(square_e sq, int attacker_color) const;
    // Checks if the king of the current side_to_move is in check
    bool is_in_check() const;
    // Checks if the king of the specified color is in check
    bool is_king_in_check(int king_color_to_check) const;

    // Generates a bitboard of all pieces of 'attacker_color' attacking 'sq'
    // bitboard_t attackers_to(square_e sq, int attacker_color) const; 

private:
    void clear_board_state();
    void update_derived_bitboards_and_mailbox(); // From piece_bbs to color_bbs, occupied_bb, board_mailbox
    void compute_initial_hash(); // Calculates hash from scratch for the current state
    // Store state for unmake_move
    struct StateInfo {
        int castling_rights;
        square_e en_passant_square;
        int halfmove_clock;
        zobrist_key_t hash;
        piece_type_e captured_piece_type; // Type of piece captured, P_NONE if no capture
        // square_e captured_piece_square; // Not strictly needed if move implies it
    };
    std::vector<StateInfo> history_stack;
};

} // namespace core
} // namespace hyperion

#endif // HYPERION_CORE_POSITION_HPP