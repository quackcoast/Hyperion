#include "movegen.hpp"
#include "constants.hpp" 
#include "bitboard.hpp"  
#include "position.hpp" 
#include "move.hpp"
#include <memory>

namespace hyperion {
namespace core {

MoveGenerator::MoveGenerator() {
    return;
}


void MoveGenerator::generate_legal_puzzle_moves(const Position& pos, std::vector<Move>& legal_move_list) {
    // 1. Generate all pseudo-legal moves into a TEMPORARY list.
    std::vector<Move> pseudo_legal_moves;
    generate_pseudo_legal_moves(pos, pseudo_legal_moves);

    // 2. Clear the final output list.
    legal_move_list.clear();

    // Allocate the temporary position on the HEAP, not the stack.
    // This creates a copy of 'pos' in heap memory.
    std::unique_ptr<Position> temp_pos = std::make_unique<Position>(pos);
    
    int player_making_move = pos.get_side_to_move();

    // 4. Iterate over the pseudo-legal list.
    for (const Move& move : pseudo_legal_moves) {
        // Use the '->' operator to access members of an object managed by a pointer.
        temp_pos->make_move(move);

        if (!temp_pos->is_king_in_check(player_making_move)) {
            legal_move_list.push_back(move);
        }

        temp_pos->unmake_move(move);
    }
}
// --- Primary Move Generation Function ---
// NEW GENERATE LEGAL MOVES, calls puseodo legal moves which should be faster ( i think)

void MoveGenerator::generate_legal_moves(const Position& pos, std::vector<Move>& legal_move_list) {
    // 1. Generate all pseudo-legal moves into a temporary list.
    std::vector<Move> pseudo_legal_moves;
    generate_pseudo_legal_moves(pos, pseudo_legal_moves);

    // 2. Clear the output list before we fill it with truly legal moves.
    legal_move_list.clear();

    // 3. For each pseudo-legal move, check if it's legal.
    int player_making_move = pos.get_side_to_move();
    Position temp_pos = pos; // Create one temporary position to reuse.

    for (const Move& move : pseudo_legal_moves) {
        // Apply the move to the temporary position.
        temp_pos.make_move(move);

        // Check if the king of the player who made the move is now in check.
        // If the king is NOT in check, the move is legal.
        if (!temp_pos.is_king_in_check(player_making_move)) {
            legal_move_list.push_back(move);
        }

        // Unmake the move to restore the temp_pos for the next iteration.
        temp_pos.unmake_move(move);
    }
}

/*
//OLD GENERATE_LEGAL_MOVES
void MoveGenerator::generate_legal_moves(const Position& pos, std::vector<Move>& legal_move_list) {
    legal_move_list.clear(); // Ensuring the list is empty before filling

    int side_to_move = pos.get_side_to_move();
    // int opponent_color = (side_to_move == WHITE) ? BLACK : WHITE; // Useful for captures

    // 1. Generate pawn moves (including promotions and en passant)
    add_pawn_moves(pos, side_to_move, legal_move_list);

    // 2. Generate knight moves
    bitboard_t knights = pos.get_pieces(P_KNIGHT, side_to_move);
    bitboard_t temp_knights = knights;
    while (temp_knights) {
        square_e from_sq = static_cast<square_e>(pop_lsb(temp_knights));
        add_knight_moves(pos, side_to_move, from_sq, legal_move_list);
    }

    // 3. Generate bishop moves
    bitboard_t bishops = pos.get_pieces(P_BISHOP, side_to_move);
    bitboard_t temp_bishops = bishops;
    while (temp_bishops) {
        square_e from_sq = static_cast<square_e>(pop_lsb(temp_bishops));
        add_bishop_moves(pos, side_to_move, from_sq, legal_move_list);
    }

    // 4. Generate rook moves
    bitboard_t rooks = pos.get_pieces(P_ROOK, side_to_move);
    bitboard_t temp_rooks = rooks;
    while (temp_rooks) {
        square_e from_sq = static_cast<square_e>(pop_lsb(temp_rooks));
        add_rook_moves(pos, side_to_move, from_sq, legal_move_list);
    }

    // 5. Generate queen moves
    bitboard_t queens = pos.get_pieces(P_QUEEN, side_to_move);
    bitboard_t temp_queens = queens;
    while (temp_queens) {
        square_e from_sq = static_cast<square_e>(pop_lsb(temp_queens));
        add_queen_moves(pos, side_to_move, from_sq, legal_move_list);
    }

    // 6. Generate king moves (excluding castling)
    square_e king_sq = pos.get_king_square(side_to_move);
    if (king_sq != square_e::NO_SQ) {
        add_king_moves(pos, side_to_move, king_sq, legal_move_list);
    }
    
    // 7. Generate castling moves
    add_castling_moves(pos, side_to_move, legal_move_list);

    
    int player_making_move = pos.get_side_to_move();
    Position temp_checking_pos = pos;

    // --- Filter pseudo-legal moves for legality ---
    // side_to_move was determined from the original 'pos'.
    // So, let's say: int side_to_move = WHITE; (captured before the loop)
    for (const Move& pseudo_move : legal_move_list) {
        temp_checking_pos.make_move(pseudo_move); // Modifies temp_checking_pos

        // Check if the king of the player who made the move is now in check
        if (!temp_checking_pos.is_king_in_check(player_making_move)) {
            legal_move_list.push_back(pseudo_move);
        }

        // Unmake the move on the temporary copy to restore it for the next pseudo_move
        temp_checking_pos.unmake_move(pseudo_move);
        // After unmake_move, temp_checking_pos should be identical to the original `pos`.
    }
}
*/
//--- pusedo move generation ---
void MoveGenerator::generate_pseudo_legal_moves(const Position& pos, std::vector<Move>& pseudo_legal_move_list) {
    pseudo_legal_move_list.clear();
    pseudo_legal_move_list.reserve(256);

    int side_to_move = pos.get_side_to_move();

    // 1. Generate pawn moves (including promotions and en passant)
    add_pawn_moves(pos, side_to_move, pseudo_legal_move_list);

    // 2. Generate knight moves
    bitboard_t knights = pos.get_pieces(P_KNIGHT, side_to_move);
    bitboard_t temp_knights = knights;
    while (temp_knights) {
        square_e from_sq = static_cast<square_e>(pop_lsb(temp_knights));
        add_knight_moves(pos, side_to_move, from_sq, pseudo_legal_move_list);
    }

    // 3. Generate bishop moves
    bitboard_t bishops = pos.get_pieces(P_BISHOP, side_to_move);
    bitboard_t temp_bishops = bishops;
    while (temp_bishops) {
        square_e from_sq = static_cast<square_e>(pop_lsb(temp_bishops));
        add_bishop_moves(pos, side_to_move, from_sq, pseudo_legal_move_list);
    }

    // 4. Generate rook moves
    bitboard_t rooks = pos.get_pieces(P_ROOK, side_to_move);
    bitboard_t temp_rooks = rooks;
    while (temp_rooks) {
        square_e from_sq = static_cast<square_e>(pop_lsb(temp_rooks));
        add_rook_moves(pos, side_to_move, from_sq, pseudo_legal_move_list);
    }

    // 5. Generate queen moves
    bitboard_t queens = pos.get_pieces(P_QUEEN, side_to_move);
    bitboard_t temp_queens = queens;
    while (temp_queens) {
        square_e from_sq = static_cast<square_e>(pop_lsb(temp_queens));
        add_queen_moves(pos, side_to_move, from_sq, pseudo_legal_move_list);
    }

    // 6. Generate king moves (non-castling)
    square_e king_sq = pos.get_king_square(side_to_move);
    if (king_sq != square_e::NO_SQ) { // Should always find a king in a valid position
        add_king_moves(pos, side_to_move, king_sq, pseudo_legal_move_list);
    }

    // 7. Generate castling moves
    add_castling_moves(pos, side_to_move, pseudo_legal_move_list);
}

// --- Helper functions implementation ---

void MoveGenerator::add_pawn_moves(const Position& pos, int color, std::vector<Move>& move_list) {
    bitboard_t pawns = pos.get_pieces(P_PAWN, color);
    bitboard_t occupied_squares = pos.get_occupied_squares();
    bitboard_t empty_squares = ~occupied_squares;
    int opponent_color = (color == WHITE) ? BLACK : WHITE;
    bitboard_t opponent_pieces = pos.get_pieces_by_color(opponent_color);

    bitboard_t current_pawns_bb; // To iterate with pop_lsb

    if (color == WHITE) {
        // --- 1. Single Pawn Pushes (White) ---
        bitboard_t single_push_targets = (pawns << 8) & empty_squares;

        // Promotions by single push
        bitboard_t promo_single_pushes = single_push_targets & RANK_8_BB;
        current_pawns_bb = promo_single_pushes;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) - 8);
            add_pawn_promotion_moves(from_sq, to_sq, P_NONE, false, move_list);
        }

        // Normal single pushes (not promotions)
        bitboard_t normal_single_pushes = single_push_targets & ~RANK_8_BB; // Exclude 8th rank
        current_pawns_bb = normal_single_pushes;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) - 8);
            move_list.push_back(Move::make_normal(from_sq, to_sq, P_PAWN));
        }

        // --- 2. Double Pawn Pushes (White) ---
        // Pawns must be on rank 2, and their single push target must be empty,
        // and their double push target must be empty.
        // `single_push_targets` already ensures the first step is to an empty square.
        // We consider only those pawns on rank 2 that could make a single push.
        bitboard_t double_push_candidates = (pawns & RANK_2_BB); // Pawns on starting rank
        bitboard_t double_push_targets = ((double_push_candidates << 8) & empty_squares) << 8 & empty_squares;
        // A more direct way:
        // bitboard_t double_push_targets = ((pawns & RANK_2_BB) << 16) & empty_squares & (empty_squares << 8); // Ensures intermediate square is empty

        current_pawns_bb = double_push_targets;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) - 16);
            move_list.push_back(Move(from_sq, to_sq, P_PAWN, P_NONE, DOUBLE_PAWN_PUSH));
        }

        // --- 3. Pawn Captures (White) ---
        // Capture Right (pawns moving from SW to NE, attack to their "right" visually)
        bitboard_t capture_right_targets = ((pawns & NOT_FILE_H_BB) << 9) & opponent_pieces; // Shift NE, avoid H-file wrap

        // Promotions by capture right
        bitboard_t promo_capture_right = capture_right_targets & RANK_8_BB;
        current_pawns_bb = promo_capture_right;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) - 9);
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(pos.get_piece_on_square(to_sq));
            add_pawn_promotion_moves(from_sq, to_sq, captured_piece, true, move_list);
        }
        // Normal captures right
        bitboard_t normal_capture_right = capture_right_targets & ~RANK_8_BB;
        current_pawns_bb = normal_capture_right;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) - 9);
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(pos.get_piece_on_square(to_sq));
            move_list.push_back(Move::make_capture(from_sq, to_sq, P_PAWN, captured_piece));
        }

        // Capture Left (pawns moving from SE to NW, attack to their "left" visually)
        bitboard_t capture_left_targets = ((pawns & NOT_FILE_A_BB) << 7) & opponent_pieces; // Shift NW, avoid A-file wrap

        // Promotions by capture left
        bitboard_t promo_capture_left = capture_left_targets & RANK_8_BB;
        current_pawns_bb = promo_capture_left;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) - 7);
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(pos.get_piece_on_square(to_sq));
            add_pawn_promotion_moves(from_sq, to_sq, captured_piece, true, move_list);
        }
        // Normal captures left
        bitboard_t normal_capture_left = capture_left_targets & ~RANK_8_BB;
        current_pawns_bb = normal_capture_left;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) - 7);
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(pos.get_piece_on_square(to_sq));
            move_list.push_back(Move::make_capture(from_sq, to_sq, P_PAWN, captured_piece));
        }

        // --- 4. En Passant (White) ---
        if (pos.en_passant_square != square_e::NO_SQ) {
            // White can only EP if the target EP square is on rank 6 (meaning black pawn just moved to rank 5 from rank 7)
            if (get_rank_idx(pos.en_passant_square) == RANK_6_IDX) { // Rank 6 is index 5
                // Find white pawns that attack the en passant square
                // `pawn_attacks[BLACK][ep_sq]` gives squares a black pawn on ep_sq would attack.
                // If any of these are our white pawns, they can EP.
                bitboard_t ep_attackers = pawn_attacks[BLACK][static_cast<int>(pos.en_passant_square)] & pawns;
                current_pawns_bb = ep_attackers;
                while (current_pawns_bb) {
                    square_e from_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
                    move_list.push_back(Move(from_sq, pos.en_passant_square, P_PAWN, P_PAWN, EN_PASSANT_CAPTURE | CAPTURE));
                }
            }
        }

    } else { // color == BLACK (Symmetrical logic)
        // --- 1. Single Pawn Pushes (Black) ---
        bitboard_t single_push_targets = (pawns >> 8) & empty_squares;

        // Promotions by single push
        bitboard_t promo_single_pushes = single_push_targets & RANK_1_BB;
        current_pawns_bb = promo_single_pushes;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) + 8);
            add_pawn_promotion_moves(from_sq, to_sq, P_NONE, false, move_list);
        }

        // Normal single pushes
        bitboard_t normal_single_pushes = single_push_targets & ~RANK_1_BB;
        current_pawns_bb = normal_single_pushes;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) + 8);
            move_list.push_back(Move::make_normal(from_sq, to_sq, P_PAWN));
        }

        // --- 2. Double Pawn Pushes (Black) ---
        bitboard_t double_push_candidates = (pawns & RANK_7_BB);
        bitboard_t double_push_targets = ((double_push_candidates >> 8) & empty_squares) >> 8 & empty_squares;
        // bitboard_t double_push_targets = ((pawns & RANK_7_BB) >> 16) & empty_squares & (empty_squares >> 8);


        current_pawns_bb = double_push_targets;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) + 16);
            move_list.push_back(Move(from_sq, to_sq, P_PAWN, P_NONE, DOUBLE_PAWN_PUSH));
        }

        // --- 3. Pawn Captures (Black) ---
        // Capture Right (pawns moving from NE to SW, attack to their "right" visually)
        bitboard_t capture_right_targets = ((pawns & NOT_FILE_H_BB) >> 7) & opponent_pieces; // Shift SE, avoid H-file wrap

        // Promotions by capture right
        bitboard_t promo_capture_right = capture_right_targets & RANK_1_BB;
        current_pawns_bb = promo_capture_right;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) + 7);
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(pos.get_piece_on_square(to_sq));
            add_pawn_promotion_moves(from_sq, to_sq, captured_piece, true, move_list);
        }
        // Normal captures right
        bitboard_t normal_capture_right = capture_right_targets & ~RANK_1_BB;
        current_pawns_bb = normal_capture_right;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) + 7);
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(pos.get_piece_on_square(to_sq));
            move_list.push_back(Move::make_capture(from_sq, to_sq, P_PAWN, captured_piece));
        }

        // Capture Left (pawns moving from NW to SE, attack to their "left" visually)
        bitboard_t capture_left_targets = ((pawns & NOT_FILE_A_BB) >> 9) & opponent_pieces; // Shift SW, avoid A-file wrap

        // Promotions by capture left
        bitboard_t promo_capture_left = capture_left_targets & RANK_1_BB;
        current_pawns_bb = promo_capture_left;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) + 9);
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(pos.get_piece_on_square(to_sq));
            add_pawn_promotion_moves(from_sq, to_sq, captured_piece, true, move_list);
        }
        // Normal captures left
        bitboard_t normal_capture_left = capture_left_targets & ~RANK_1_BB;
        current_pawns_bb = normal_capture_left;
        while (current_pawns_bb) {
            square_e to_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
            square_e from_sq = static_cast<square_e>(static_cast<int>(to_sq) + 9);
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(pos.get_piece_on_square(to_sq));
            move_list.push_back(Move::make_capture(from_sq, to_sq, P_PAWN, captured_piece));
        }

        // --- 4. En Passant (Black) ---
        if (pos.en_passant_square != square_e::NO_SQ) {
            // Black can only EP if the target EP square is on rank 3 (meaning white pawn just moved to rank 4 from rank 2)
            if (get_rank_idx(pos.en_passant_square) == RANK_3_IDX) { // Rank 3 is index 2
                bitboard_t ep_attackers = pawn_attacks[WHITE][static_cast<int>(pos.en_passant_square)] & pawns;
                current_pawns_bb = ep_attackers;
                while (current_pawns_bb) {
                    square_e from_sq = static_cast<square_e>(pop_lsb(current_pawns_bb));
                    move_list.push_back(Move(from_sq, pos.en_passant_square, P_PAWN, P_PAWN, EN_PASSANT_CAPTURE | CAPTURE));
                }
            }
        }
    }
}

void MoveGenerator::add_knight_moves(const Position& pos, int color, square_e from_sq, std::vector<Move>& move_list) {
    bitboard_t knight_attack_squares = hyperion::core::knight_attacks[static_cast<int>(from_sq)];
    bitboard_t friendly_pieces = pos.get_pieces_by_color(color);

    // Knights can only move to squares not occupied by their own pieces
    bitboard_t valid_landing_squares = knight_attack_squares & ~friendly_pieces;
    
    bitboard_t temp_valid_landings = valid_landing_squares;
    while (temp_valid_landings) {
        square_e to_sq = static_cast<square_e>(pop_lsb(temp_valid_landings));
        int piece_on_to_sq_val = pos.get_piece_on_square(to_sq);

        if (piece_on_to_sq_val == EMPTY_MAILBOX_VAL) {
            move_list.push_back(Move::make_normal(from_sq, to_sq, P_KNIGHT));
        } else { // Opponent piece (since friendly_pieces were excluded)
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(piece_on_to_sq_val);
            move_list.push_back(Move::make_capture(from_sq, to_sq, P_KNIGHT, captured_piece));
        }
    }
}

void MoveGenerator::add_bishop_moves(const Position& pos, int color, square_e from_sq, std::vector<Move>& move_list) {
    bitboard_t bishop_attack_squares = get_bishop_slider_attacks(from_sq, pos.get_occupied_squares());
    // --- RAY CASTING ---
    /*
    bitboard_t bishop_attack_squares = hyperion::core::generate_attacks_slow_internal(
        static_cast<int>(from_sq),      // The square of the bishop
        pos.get_occupied_squares(),     // All pieces on the board act as blockers
        false                           // is_rook = false, so it generates bishop attacks
    );
    */
    bitboard_t friendly_pieces = pos.get_pieces_by_color(color);

    bitboard_t valid_landing_squares = bishop_attack_squares & ~friendly_pieces;

    bitboard_t temp_valid_landings = valid_landing_squares;
    while (temp_valid_landings) {
        square_e to_sq = static_cast<square_e>(pop_lsb(temp_valid_landings));
        int piece_on_to_sq_val = pos.get_piece_on_square(to_sq);

        if (piece_on_to_sq_val == EMPTY_MAILBOX_VAL) {
            move_list.push_back(Move::make_normal(from_sq, to_sq, P_BISHOP));
        } else {
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(piece_on_to_sq_val);
            move_list.push_back(Move::make_capture(from_sq, to_sq, P_BISHOP, captured_piece));
        }
    }
}

void MoveGenerator::add_rook_moves(const Position& pos, int color, square_e from_sq, std::vector<Move>& move_list) {
    bitboard_t rook_attack_squares = get_rook_slider_attacks(from_sq, pos.get_occupied_squares());
    // --- RAY CASTING ---
    /*
    bitboard_t rook_attack_squares = hyperion::core::generate_attacks_slow_internal(
        static_cast<int>(from_sq),      // The square of the rook
        pos.get_occupied_squares(),     // All pieces on the board act as blockers
        true                            // is_rook = true
    );
    */

    bitboard_t friendly_pieces = pos.get_pieces_by_color(color);

    bitboard_t valid_landing_squares = rook_attack_squares & ~friendly_pieces;
    
    bitboard_t temp_valid_landings = valid_landing_squares;
    while (temp_valid_landings) {
        square_e to_sq = static_cast<square_e>(pop_lsb(temp_valid_landings));
        int piece_on_to_sq_val = pos.get_piece_on_square(to_sq);

        if (piece_on_to_sq_val == EMPTY_MAILBOX_VAL) {
            move_list.push_back(Move::make_normal(from_sq, to_sq, P_ROOK));
        } else {
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(piece_on_to_sq_val);
            move_list.push_back(Move::make_capture(from_sq, to_sq, P_ROOK, captured_piece));
        }
    }
}

void MoveGenerator::add_queen_moves(const Position& pos, int color, square_e from_sq, std::vector<Move>& move_list) {
    
    
    
    bitboard_t queen_attack_squares = get_rook_slider_attacks(from_sq, pos.get_occupied_squares()) | get_bishop_slider_attacks(from_sq, pos.get_occupied_squares());
    // --- RAY CASTING ---
    /*
    bitboard_t queen_attack_squares =
        hyperion::core::generate_attacks_slow_internal(
            static_cast<int>(from_sq),
            pos.get_occupied_squares(),
            true  // Rook-like moves
        ) |
        hyperion::core::generate_attacks_slow_internal(
            static_cast<int>(from_sq),
            pos.get_occupied_squares(),
            false // Bishop-like moves
        );
    */

    bitboard_t friendly_pieces = pos.get_pieces_by_color(color);

    bitboard_t valid_landing_squares = queen_attack_squares & ~friendly_pieces;

    bitboard_t temp_valid_landings = valid_landing_squares;
    while (temp_valid_landings) {
        square_e to_sq = static_cast<square_e>(pop_lsb(temp_valid_landings));
        int piece_on_to_sq_val = pos.get_piece_on_square(to_sq);

        if (piece_on_to_sq_val == EMPTY_MAILBOX_VAL) {
            move_list.push_back(Move::make_normal(from_sq, to_sq, P_QUEEN));
        } else {
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(piece_on_to_sq_val);
            move_list.push_back(Move::make_capture(from_sq, to_sq, P_QUEEN, captured_piece));
        }
    }
}

void MoveGenerator::add_king_moves(const Position& pos, int color, square_e from_sq, std::vector<Move>& move_list) {
    bitboard_t king_attack_squares = hyperion::core::king_attacks[static_cast<int>(from_sq)];
    bitboard_t friendly_pieces = pos.get_pieces_by_color(color);

    bitboard_t valid_landing_squares = king_attack_squares & ~friendly_pieces;

    bitboard_t temp_valid_landings = valid_landing_squares;
    while (temp_valid_landings) {
        square_e to_sq = static_cast<square_e>(pop_lsb(temp_valid_landings));
        int piece_on_to_sq_val = pos.get_piece_on_square(to_sq);

        if (piece_on_to_sq_val == EMPTY_MAILBOX_VAL) {
            move_list.push_back(Move::make_normal(from_sq, to_sq, P_KING));
        } else {
            piece_type_e captured_piece = pos.get_piece_type_from_mailbox_val(piece_on_to_sq_val);
            move_list.push_back(Move::make_capture(from_sq, to_sq, P_KING, captured_piece));
        }
    }
}

void MoveGenerator::add_castling_moves(const Position& pos, int color, std::vector<Move>& move_list) {
    int current_castling_rights = pos.castling_rights; //
    int opponent_color = (color == WHITE) ? BLACK : WHITE;

    if (color == WHITE) {
        // Kingside Castle (O-O)
        if (current_castling_rights & WK_CASTLE_FLAG) {
            if (!get_bit(pos.occupied_bb, F1) && !get_bit(pos.occupied_bb, G1)) {
                if (!pos.is_square_attacked(static_cast<square_e>(E1), opponent_color) &&
                    !pos.is_square_attacked(static_cast<square_e>(F1), opponent_color) &&
                    !pos.is_square_attacked(static_cast<square_e>(G1), opponent_color)) {
                    move_list.push_back(Move(static_cast<square_e>(E1), static_cast<square_e>(G1), P_KING, P_NONE, CASTLING_KINGSIDE));
                }
            }
        }
        // Queenside Castle (O-O-O)
        if (current_castling_rights & WQ_CASTLE_FLAG) {
            if (!get_bit(pos.occupied_bb, B1) && !get_bit(pos.occupied_bb, C1) && !get_bit(pos.occupied_bb, D1)) {
                if (!pos.is_square_attacked(static_cast<square_e>(E1), opponent_color) &&
                    !pos.is_square_attacked(static_cast<square_e>(D1), opponent_color) &&
                    !pos.is_square_attacked(static_cast<square_e>(C1), opponent_color)) {
                    move_list.push_back(Move(static_cast<square_e>(E1), static_cast<square_e>(C1), P_KING, P_NONE, CASTLING_QUEENSIDE));
                }
            }
        }
    } else {
        // Kingside Castle (O-O)
        if (current_castling_rights & BK_CASTLE_FLAG) {
            if (!get_bit(pos.occupied_bb, F8) && !get_bit(pos.occupied_bb, G8)) {
                if (!pos.is_square_attacked(static_cast<square_e>(E8), opponent_color) &&
                    !pos.is_square_attacked(static_cast<square_e>(F8), opponent_color) &&
                    !pos.is_square_attacked(static_cast<square_e>(G8), opponent_color)) {
                    move_list.push_back(Move(static_cast<square_e>(E8),static_cast<square_e>(G8), P_KING, P_NONE, CASTLING_KINGSIDE));
                }
            }
        }
        // Queenside Castle (O-O-O)
        if (current_castling_rights & BQ_CASTLE_FLAG) {
            if (!get_bit(pos.occupied_bb, B8) && !get_bit(pos.occupied_bb, C8) && !get_bit(pos.occupied_bb, D8)) {
                if (!pos.is_square_attacked(static_cast<square_e>(E8), opponent_color) &&
                    !pos.is_square_attacked(static_cast<square_e>(D8), opponent_color) &&
                    !pos.is_square_attacked(static_cast<square_e>(C8), opponent_color)) {
                     move_list.push_back(Move(static_cast<square_e>(E8),static_cast<square_e>(C8), P_KING, P_NONE, CASTLING_QUEENSIDE));
                }
            }
        }
    }
}


void MoveGenerator::add_pawn_promotion_moves(square_e from_sq, square_e to_sq,
                                           piece_type_e captured_piece, bool is_capture,
                                           std::vector<Move>& move_list) {
    // Pawns promote to Queen, Rook, Bishop, or Knight
    move_list.push_back(Move::make_promotion(from_sq, to_sq, P_PAWN, P_QUEEN, is_capture, captured_piece)); //
    move_list.push_back(Move::make_promotion(from_sq, to_sq, P_PAWN, P_ROOK, is_capture, captured_piece));
    move_list.push_back(Move::make_promotion(from_sq, to_sq, P_PAWN, P_BISHOP, is_capture, captured_piece));
    move_list.push_back(Move::make_promotion(from_sq, to_sq, P_PAWN, P_KNIGHT, is_capture, captured_piece));
}

} // namespace core
} // namespace hyperion