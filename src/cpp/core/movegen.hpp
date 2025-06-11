#ifndef HYPERION_CORE_MOVEGEN_HPP
#define HYPERION_CORE_MOVEGEN_HPP

#include "position.hpp" 
#include "move.hpp"    
#include <vector>      

namespace hyperion {
namespace core {

class MoveGenerator {
public:
    MoveGenerator();

    // --- Primary Move Generation Function ---
    // Generates all legal moves for the side to move in the given position.
    // The generated moves are added to the 'move_list'.
    
    void generate_legal_moves(const Position& pos, std::vector<Move>& move_list);
     // --- Pseudo-Legal Move Generation ---
    void generate_pseudo_legal_moves(const Position& pos, std::vector<Move>& pseudo_legal_move_list);

private:
    // --- Helper functions for generating moves for specific piece types ---
    // These functions would generate pseudo-legal moves, which are then
    // checked for legality (e.g., not leaving the king in check) by the main function
    // or by themselves.

    void add_pawn_moves(const Position& pos, int color, std::vector<Move>& move_list);
    void add_knight_moves(const Position& pos, int color, square_e from_sq, std::vector<Move>& move_list);
    void add_bishop_moves(const Position& pos, int color, square_e from_sq, std::vector<Move>& move_list);
    void add_rook_moves(const Position& pos, int color, square_e from_sq, std::vector<Move>& move_list);
    void add_queen_moves(const Position& pos, int color, square_e from_sq, std::vector<Move>& move_list);
    void add_king_moves(const Position& pos, int color, square_e from_sq, std::vector<Move>& move_list);

    // Helper for castling moves
    void add_castling_moves(const Position& pos, int color, std::vector<Move>& move_list);
    void add_move(const Position& pos, square_e from_sq, square_e to_sq, piece_type_e moved_piece,
                  int color, std::vector<Move>& move_list, MoveFlag extra_flags = NORMAL_MOVE);

    void add_pawn_promotion_moves(square_e from_sq, square_e to_sq,
                                  piece_type_e captured_piece, bool is_capture,
                                  std::vector<Move>& move_list);

};

} // namespace core
} // namespace hyperion

#endif // HYPERION_CORE_MOVEGEN_HPP