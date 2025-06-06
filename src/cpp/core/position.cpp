// hyperion/src/cpp/core/position.cpp
#include "position.hpp"  
#include "constants.hpp" 
#include "bitboard.hpp"  
#include "zobrist.hpp"  
#include "move.hpp"     
#include <sstream>      
#include <algorithm>    
#include <cctype>        

/* piece_info_from_fen_char */
// Converts a FEN character (e.g., 'P', 'n', 'k') to a pair of piece type and color.
// Returns P_NONE and -1 if the character is invalid.

std::pair<hyperion::core::piece_type_e, int> piece_info_from_fen_char(char c) { 
    using namespace hyperion::core;
    switch (c) {
        case 'P': return {P_PAWN, WHITE};   //before looking at these switch cases look at the switch cases below
        case 'N': return {P_KNIGHT, WHITE}; //it basically divides it into BLACK or WHITE --> and then divides it by piece type
        case 'B': return {P_BISHOP, WHITE}; //we are then setting it to the pieces used in constants.hpp with the color info
        case 'R': return {P_ROOK, WHITE};   
        case 'Q': return {P_QUEEN, WHITE};
        case 'K': return {P_KING, WHITE};
        case 'p': return {P_PAWN, BLACK};
        case 'n': return {P_KNIGHT, BLACK};
        case 'b': return {P_BISHOP, BLACK};
        case 'r': return {P_ROOK, BLACK};
        case 'q': return {P_QUEEN, BLACK};
        case 'k': return {P_KING, BLACK};
        default:  return {P_NONE, -1}; // Invalid (there would be like no way this ever get set but its good to have a default in switch cases)
    }                                  // If it were set then that would probaly indicate we copy pasted fen wrong
}

/* fen_char_from_piece_info */
// Converts a piece type and color to its corresponding FEN character.
// Returns EMPTY_SQUARE_CHAR for P_NONE or '?' for invalid combinations.

char fen_char_from_piece_info(hyperion::core::piece_type_e pt, int color) {
    using namespace hyperion::core;
    if (pt == P_NONE) return EMPTY_SQUARE_CHAR; // Should not happen if piece exists
    if (color == WHITE) {// divides it by white first
        switch (pt) {
            case P_PAWN: return W_PAWN; case P_KNIGHT: return W_KNIGHT; case P_BISHOP: return W_BISHOP;
            case P_ROOK: return W_ROOK; case P_QUEEN: return W_QUEEN;   case P_KING: return W_KING;
            default: return '?'; //questionmark for default, as said in above comment on line 41-42 there is no way this should ever be set
        }
    } else { // BLACK
        switch (pt) {
            case P_PAWN: return B_PAWN; case P_KNIGHT: return B_KNIGHT; case P_BISHOP: return B_BISHOP;
            case P_ROOK: return B_ROOK; case P_QUEEN: return B_QUEEN;   case P_KING: return B_KING;
            default: return '?';// same here
        }
    }
}


namespace hyperion {
namespace core {

//--
/* Position::make_mailbox_entry */
//--
// Creates an integer value for the mailbox array representation from a piece type and color.
// Encodes piece type and color into a single integer for storage in board_mailbox.
// White pieces P_PAWN=0 to P_KING=5, Black pieces P_PAWN=6 to P_KING=11.
// Returns EMPTY_MAILBOX_VAL if the piece type is P_NONE.

int Position::make_mailbox_entry(piece_type_e type, int color) const {
    if (type == P_NONE) return EMPTY_MAILBOX_VAL;
    // Example: White pieces P_PAWN=0 to P_KING=5, Black pieces P_PAWN=6 to P_KING=11
    return static_cast<int>(type) + (color == BLACK ? NUM_PIECE_TYPES : 0);
}
//--
/* Position::get_piece_type_from_mailbox_val */
//--
// Extracts the piece_type_e from an integer value stored in the mailbox array.
// Decodes the piece type from the mailbox entry.
// Returns P_NONE if the mailbox value is invalid or represents an empty square.

piece_type_e Position::get_piece_type_from_mailbox_val(int mb_val) const {
    if (mb_val == EMPTY_MAILBOX_VAL || mb_val < 0 || mb_val >= (NUM_PIECE_TYPES * 2)) return P_NONE;
    return static_cast<piece_type_e>(mb_val % NUM_PIECE_TYPES);
}
//--
/* Position::get_color_from_mailbox_val */
//-
// Extracts the color (WHITE or BLACK) from an integer value stored in the mailbox array.
// Decodes the color from the mailbox entry.
// Returns -1 if the mailbox value is invalid or represents an empty square.

int Position::get_color_from_mailbox_val(int mb_val) const {
    if (mb_val == EMPTY_MAILBOX_VAL || mb_val < 0 || mb_val >= (NUM_PIECE_TYPES * 2)) return -1; // Invalid
    return (mb_val >= NUM_PIECE_TYPES) ? BLACK : WHITE;
}
//--
/* Position::Position */
//--
// Default constructor for the Position class.
// Initializes the board to the standard chess starting position.
// Calls set_from_fen with the starting FEN string.
// IMPORTANT: Zobrist::initialize_keys() must be called globally before creating a Position object.

Position::Position() {
    //below is the starting position fen
    set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

//--
/* Position::clear_board_state */
//--
// Resets the entire board state to an empty configuration.
// Clears all piece bitboards, color bitboards, and the occupied bitboard.
// Resets the board_mailbox to empty values.
// Resets game state variables: side_to_move to WHITE, castling_rights to 0, en_passant_square to NO_SQ, halfmove_clock to 0, fullmove_number to 1.
// Clears the current_hash and the history_stack.

void Position::clear_board_state() {
    for (int pt = 0; pt < NUM_PIECE_TYPES; ++pt) {
        piece_bbs[pt][WHITE] = EMPTY_BB;
        piece_bbs[pt][BLACK] = EMPTY_BB;
    }
    color_bbs[WHITE] = EMPTY_BB;
    color_bbs[BLACK] = EMPTY_BB;
    occupied_bb = EMPTY_BB;

    std::fill(board_mailbox.begin(), board_mailbox.end(), EMPTY_MAILBOX_VAL);

    side_to_move = WHITE;
    castling_rights = 0;
    en_passant_square = square_e::NO_SQ;
    halfmove_clock = 0;
    fullmove_number = 1;
    current_hash = 0ULL;
    history_stack.clear();
     history_stack.reserve(256);
}
//--
/* Position::update_derived_bitboards_and_mailbox */
//--
// Updates the derived board representations (color_bbs, occupied_bb, board_mailbox) based on the current state of piece_bbs.
// Iterates through all piece types and colors in piece_bbs.
// Populates color_bbs[WHITE] and color_bbs[BLACK].
// Populates the board_mailbox with entries derived from piece_bbs.
// Sets the occupied_bb as the union of color_bbs[WHITE] and color_bbs[BLACK].

/* information about update_derived_bitboards_and_mailbox() */
void Position::update_derived_bitboards_and_mailbox() {
    color_bbs[WHITE] = EMPTY_BB;
    color_bbs[BLACK] = EMPTY_BB;
    std::fill(board_mailbox.begin(), board_mailbox.end(), EMPTY_MAILBOX_VAL);

    for (int p_type_idx = 0; p_type_idx < NUM_PIECE_TYPES; ++p_type_idx) {
        piece_type_e p_type = static_cast<piece_type_e>(p_type_idx);

        bitboard_t white_pieces_of_type = piece_bbs[p_type_idx][WHITE];
        color_bbs[WHITE] |= white_pieces_of_type;
        bitboard_t temp_bb_w = white_pieces_of_type;
        while (temp_bb_w) {
            int sq_idx = pop_lsb(temp_bb_w);
            board_mailbox[sq_idx] = make_mailbox_entry(p_type, WHITE);
        }

        bitboard_t black_pieces_of_type = piece_bbs[p_type_idx][BLACK];
        color_bbs[BLACK] |= black_pieces_of_type;
        bitboard_t temp_bb_b = black_pieces_of_type;
        while (temp_bb_b) {
            int sq_idx = pop_lsb(temp_bb_b);
            board_mailbox[sq_idx] = make_mailbox_entry(p_type, BLACK);
        }
    }
    occupied_bb = color_bbs[WHITE] | color_bbs[BLACK];
}
//--
/* Position::compute_initial_hash */
//--
// Computes the Zobrist hash for the current board position from scratch.
// Initializes current_hash to 0.
// XORs keys for all pieces on the board based on their type, color, and square.
// XORs the key for the side to move if it's black's turn.
// XORs the key for the current castling rights state.
// XORs the key for the en passant file if an en passant square is set.

void Position::compute_initial_hash() {
    current_hash = 0ULL;

    // 1. Pieces on board
    for (int p_type_idx = 0; p_type_idx < NUM_PIECE_TYPES; ++p_type_idx) {
        bitboard_t bb_w = piece_bbs[p_type_idx][WHITE];
        while (bb_w) {
            int sq_idx = pop_lsb(bb_w);
            current_hash ^= Zobrist::piece_square_keys[p_type_idx][WHITE][sq_idx];
        }
        bitboard_t bb_b = piece_bbs[p_type_idx][BLACK];
        while (bb_b) {
            int sq_idx = pop_lsb(bb_b);
            current_hash ^= Zobrist::piece_square_keys[p_type_idx][BLACK][sq_idx];
        }
    }

    // 2. Side to move
    if (side_to_move == BLACK) {
        current_hash ^= Zobrist::black_to_move_key;
    }

    // 3. Castling rights (using 16 keys, indexed by the bitmask state)
    current_hash ^= Zobrist::castling_keys[castling_rights];


    // 4. En passant square
    if (en_passant_square != square_e::NO_SQ) {
        int ep_file = static_cast<int>(en_passant_square) % 8; // File index 0-7
        current_hash ^= Zobrist::en_passant_file_keys[ep_file];
    }
}
//--
/* Position::set_from_fen */
//--
// Sets the board position from a FEN string.
// Calls clear_board_state() first to reset the position.
// Parses the FEN string for: piece placement, side to move, castling availability, en passant target square, halfmove clock, and fullmove number.
// Updates piece_bbs based on the FEN piece placement.
// Sets side_to_move, castling_rights, en_passant_square, halfmove_clock, and fullmove_number.
// Calls update_derived_bitboards_and_mailbox() and compute_initial_hash() after parsing. 

void Position::set_from_fen(const std::string& fen_string) {
    clear_board_state();
    std::istringstream fen_stream(fen_string);
    std::string part;

    // 1. Piece placement
    fen_stream >> part;
    int rank = 7; //fen ranks are 8..1
    int file = 0; //fen files are a..h
    for (char c : part) {
        if (c == '/') {
            rank--;
            file = 0;
        } else if (std::isdigit(c)) {
            file += (c - '0');
        } else {
            auto p_info = piece_info_from_fen_char(c);
            if (p_info.first != P_NONE) {
                set_bit(piece_bbs[static_cast<int>(p_info.first)][p_info.second], rank * 8 + file);
            }
            file++;
        }
    }

    // 2. Side to move
    fen_stream >> part;
    side_to_move = (part == "w") ? WHITE : BLACK;

    // 3. Castling availability
    fen_stream >> part;
    this->castling_rights = 0; // Start with no rights
    if (part != "-") {
        for (char c : part) {
            if (c == 'K') this->castling_rights |= WK_CASTLE_FLAG;
            else if (c == 'Q') this->castling_rights |= WQ_CASTLE_FLAG;
            else if (c == 'k') this->castling_rights |= BK_CASTLE_FLAG;
            else if (c == 'q') this->castling_rights |= BQ_CASTLE_FLAG;
        }
    }

    // 4. En passant target square
    fen_stream >> part;
    if (part != "-") {
        int ep_file = part[0] - 'a';
        int ep_rank = part[1] - '1';
        this->en_passant_square = static_cast<square_e>(ep_rank * 8 + ep_file);
    } else {
        this->en_passant_square = square_e::NO_SQ;
    }

    // 5. Halfmove clock
    if (!(fen_stream >> this->halfmove_clock)) {
        this->halfmove_clock = 0; // Default iff missing
    }

    // 6. Fullmove number
    if (!(fen_stream >> this->fullmove_number)) {
        this->fullmove_number = 1; // Default if missing
    }

    update_derived_bitboards_and_mailbox();
    compute_initial_hash(); // Calculate hash from scratch for the new position
}
//--
/* Position::to_fen */
//--
// Converts the current board position to a FEN string.
// Iterates through the board_mailbox (from rank 8 down to 1, file 'a' to 'h') to construct the piece placement part of the FEN.
// Appends side to move ('w' or 'b').
// Appends castling availability (e.g., "KQkq", "-", "K").
// Appends the en passant target square in algebraic notation (e.g., "e3") or "-" if none.
// Appends the halfmove clock and fullmove number.
// Returns the generated FEN string.

std::string Position::to_fen() const {
    std::ostringstream fen;
    for (int r = 7; r >= 0; --r) {
        int empty_count = 0;
        for (int f = 0; f < 8; ++f) {
            square_e sq = static_cast<square_e>(r * 8 + f);
            int mb_val = board_mailbox[static_cast<int>(sq)];
            if (mb_val == EMPTY_MAILBOX_VAL) {
                empty_count++;
            } else {
                if (empty_count > 0) {
                    fen << empty_count;
                    empty_count = 0;
                }
                fen << fen_char_from_piece_info(get_piece_type_from_mailbox_val(mb_val), get_color_from_mailbox_val(mb_val));
            }
        }
        if (empty_count > 0) fen << empty_count;
        if (r > 0) fen << '/';
    }

    fen << " " << (side_to_move == WHITE ? "w" : "b");

    fen << " ";
    std::string castle_str;
    if (castling_rights & WK_CASTLE_FLAG) castle_str += 'K';
    if (castling_rights & WQ_CASTLE_FLAG) castle_str += 'Q';
    if (castling_rights & BK_CASTLE_FLAG) castle_str += 'k';
    if (castling_rights & BQ_CASTLE_FLAG) castle_str += 'q';
    fen << (castle_str.empty() ? "-" : castle_str);

    fen << " ";
    if (en_passant_square == square_e::NO_SQ) {
        fen << "-";
    } else {
        fen << square_to_algebraic(en_passant_square);
    }

    fen << " " << halfmove_clock;
    fen << " " << fullmove_number;
    return fen.str();
}


// --- Accessors ---

/* Position::get_side_to_move */
// Accessor method to get the current side to move.
// Returns an integer representing the side: WHITE or BLACK.
//int Position::get_side_to_move() const { return side_to_move; } <-- in header file

/* Position::get_pieces */
// Accessor method to get the bitboard for a specific piece type and color.
// Takes a piece_type_e (e.g., P_PAWN) and an integer p_color (WHITE or BLACK).
// Returns the corresponding bitboard from the piece_bbs array.
bitboard_t Position::get_pieces(piece_type_e p_type, int p_color) const {
    return piece_bbs[static_cast<int>(p_type)][p_color];
}

/* Position::get_pieces_by_type */
// Accessor method to get the bitboard for all pieces of a specific type, regardless of color.
// Takes a piece_type_e (e.g., P_ROOK).
// Returns the union of the bitboards for that piece type for both WHITE and BLACK.
bitboard_t Position::get_pieces_by_type(piece_type_e p_type) const {
    return piece_bbs[static_cast<int>(p_type)][WHITE] | piece_bbs[static_cast<int>(p_type)][BLACK];
}

/* Position::get_pieces_by_color */
// Accessor method to get the bitboard for all pieces of a specific color.
// Takes an integer p_color (WHITE or BLACK).
// Returns the corresponding bitboard from the color_bbs array.
bitboard_t Position::get_pieces_by_color(int p_color) const { return color_bbs[p_color]; }

/* Position::get_occupied_squares */
// Accessor method to get the bitboard for all occupied squares on the board.
// Returns the occupied_bb, which is the union of all white and black pieces.
bitboard_t Position::get_occupied_squares() const { return occupied_bb; }

/* Position::get_king_square */
// Accessor method to find the square of the king for a given color.
// Takes an integer king_color (WHITE or BLACK).
// Retrieves the king's bitboard for that color and returns the square_e of its least significant bit (LSB).
// Returns square_e::NO_SQ if the king is not found (which should not happen in a valid game state).
square_e Position::get_king_square(int king_color) const {
    bitboard_t king_bb = piece_bbs[static_cast<int>(P_KING)][king_color];
    if (king_bb == 0) return square_e::NO_SQ; // Should not happen in valid game
    return static_cast<square_e>(get_lsb_index(king_bb));
}

/* Position::get_piece_on_square */
// Accessor method to get the piece (type and color encoded) on a specific square using the mailbox representation.
// Takes a square_e sq.
// Returns the integer value from board_mailbox at the given square index.
// Returns EMPTY_MAILBOX_VAL if the square is out of bounds or empty.

int Position::get_piece_on_square(square_e sq) const {
    if (sq == square_e::NO_SQ || static_cast<int>(sq) >= NUM_SQUARES) return EMPTY_MAILBOX_VAL;
    return board_mailbox[static_cast<int>(sq)];
}

// --- Move Execution ---
//--
/* Position::make_move */
//--
// Applies a given move to the current board position.
// Updates all relevant board state: piece bitboards, mailbox, Zobrist hash, side to move, castling rights, en passant square, halfmove clock, and fullmove number.
// Saves the previous state information (castling rights, EP square, halfmove clock, hash, captured piece type) onto the history_stack for unmake_move.
// Handles normal moves, captures (including en passant), promotions, and castling.
// Updates Zobrist hash incrementally based on changes.
// Updates derived bitboards (color_bbs, occupied_bb) at the end.

void Position::make_move(const Move& m) {
    // 1. Save current state for unmake_move
    StateInfo prev_state;
    prev_state.castling_rights = this->castling_rights;
    prev_state.en_passant_square = this->en_passant_square;
    prev_state.halfmove_clock = this->halfmove_clock;
    prev_state.hash = this->current_hash;
    prev_state.captured_piece_type = m.piece_captured;
    history_stack.push_back(prev_state);

    int mover_color = this->side_to_move;
    int opponent_color = (mover_color == WHITE) ? BLACK : WHITE;

    square_e from_sq = m.from_sq;
    square_e to_sq = m.to_sq;
    piece_type_e moved_piece_type = m.piece_moved;
    int moved_piece_idx = static_cast<int>(moved_piece_type);
    int from_sq_idx = static_cast<int>(from_sq);
    int to_sq_idx = static_cast<int>(to_sq);

    // --- Update Zobrist hash (start) ---
    current_hash ^= Zobrist::castling_keys[this->castling_rights];
    if (this->en_passant_square != square_e::NO_SQ) {
        current_hash ^= Zobrist::en_passant_file_keys[static_cast<int>(this->en_passant_square) % 8];
    }

    // --- Update board (Incremental Updates) ---

    // A. Remove the piece from the starting square (from_sq)
    clear_bit(piece_bbs[moved_piece_idx][mover_color], from_sq);
    clear_bit(color_bbs[mover_color], from_sq);
    clear_bit(occupied_bb, from_sq); // from_sq is now empty
    current_hash ^= Zobrist::piece_square_keys[moved_piece_idx][mover_color][from_sq_idx];
    board_mailbox[from_sq_idx] = EMPTY_MAILBOX_VAL;

    // B. Handle capture
    if (m.is_capture()) {
        this->halfmove_clock = 0;
        piece_type_e captured_type = m.piece_captured;
        int captured_type_idx = static_cast<int>(captured_type);

        square_e actual_capture_sq = to_sq; // For normal captures
        int actual_capture_sq_idx = to_sq_idx;

        if (m.is_en_passant()) {
            actual_capture_sq = (mover_color == WHITE)
                               ? static_cast<square_e>(to_sq_idx - 8)
                               : static_cast<square_e>(to_sq_idx + 8);
            actual_capture_sq_idx = static_cast<int>(actual_capture_sq);
            // Note: The moving pawn lands on to_sq, which was empty.
            // The captured pawn is on actual_capture_sq.
        }
        // Remove captured piece from its square
        clear_bit(piece_bbs[captured_type_idx][opponent_color], actual_capture_sq);
        clear_bit(color_bbs[opponent_color], actual_capture_sq);
        clear_bit(occupied_bb, actual_capture_sq); // This square becomes empty
        current_hash ^= Zobrist::piece_square_keys[captured_type_idx][opponent_color][actual_capture_sq_idx];
        board_mailbox[actual_capture_sq_idx] = EMPTY_MAILBOX_VAL; // Only if actual_capture_sq is different from to_sq (EP)
                                                               // If normal capture, to_sq mailbox will be overwritten by landing piece.
                                                               // For EP, mailbox on actual_capture_sq must be cleared.
    } else if (moved_piece_type == P_PAWN) {
        this->halfmove_clock = 0;
    } else {
        this->halfmove_clock++;
    }

    // C. Place the piece on the destination square (to_sq)
    piece_type_e piece_to_place = moved_piece_type;
    if (m.is_promotion()) {
        piece_to_place = m.get_promotion_piece();
        this->halfmove_clock = 0;
    }
    int piece_to_place_idx = static_cast<int>(piece_to_place);

    set_bit(piece_bbs[piece_to_place_idx][mover_color], to_sq);
    set_bit(color_bbs[mover_color], to_sq);
    set_bit(occupied_bb, to_sq); // to_sq is now occupied by the moved/promoted piece
    current_hash ^= Zobrist::piece_square_keys[piece_to_place_idx][mover_color][to_sq_idx];
    board_mailbox[to_sq_idx] = make_mailbox_entry(piece_to_place, mover_color);

    // D. Handle castling (moving the rook)
    if (m.is_castling()) {
        this->halfmove_clock = 0;
        square_e rook_from_sq, rook_to_sq;
        int rook_from_sq_idx, rook_to_sq_idx;
        int rook_type_idx = static_cast<int>(P_ROOK);

        if (m.is_kingside_castle()) {
            rook_from_sq = (mover_color == WHITE) ? static_cast<square_e>(H1) : static_cast<square_e>(H8); // Use defined constants if available
            rook_to_sq = (mover_color == WHITE) ? static_cast<square_e>(F1) : static_cast<square_e>(F8);
        } else { // Queenside
            rook_from_sq = (mover_color == WHITE) ? static_cast<square_e>(A1) : static_cast<square_e>(A8);
            rook_to_sq = (mover_color == WHITE) ? static_cast<square_e>(D1): static_cast<square_e>(D8);
        }
        rook_from_sq_idx = static_cast<int>(rook_from_sq);
        rook_to_sq_idx = static_cast<int>(rook_to_sq);

        // Move the rook
        clear_bit(piece_bbs[rook_type_idx][mover_color], rook_from_sq);
        clear_bit(color_bbs[mover_color], rook_from_sq); // Rook leaves its color bb
        clear_bit(occupied_bb, rook_from_sq);           // Rook leaves occupied
        current_hash ^= Zobrist::piece_square_keys[rook_type_idx][mover_color][rook_from_sq_idx];
        board_mailbox[rook_from_sq_idx] = EMPTY_MAILBOX_VAL;

        set_bit(piece_bbs[rook_type_idx][mover_color], rook_to_sq);
        set_bit(color_bbs[mover_color], rook_to_sq);    // Rook joins its color bb
        set_bit(occupied_bb, rook_to_sq);               // Rook joins occupied
        current_hash ^= Zobrist::piece_square_keys[rook_type_idx][mover_color][rook_to_sq_idx];
        board_mailbox[rook_to_sq_idx] = make_mailbox_entry(P_ROOK, mover_color);
    }

    // --- Update game state ---

    // E. Update castling rights
    const int const_H1 = static_cast<int>(square_e::SQ_H1);
    const int const_A1 = static_cast<int>(square_e::SQ_A1);
    //const int const_E1 = static_cast<int>(square_e::SQ_E1);
    const int const_H8 = static_cast<int>(square_e::SQ_H8);
    const int const_A8 = static_cast<int>(square_e::SQ_A8);
    //const int const_E8 = static_cast<int>(square_e::SQ_E8);


    if (moved_piece_type == P_KING) {
        if (mover_color == WHITE) {
            this->castling_rights &= ~(WK_CASTLE_FLAG | WQ_CASTLE_FLAG);
        } else {
            this->castling_rights &= ~(BK_CASTLE_FLAG | BQ_CASTLE_FLAG);
        }
    } else if (moved_piece_type == P_ROOK) {
        if (from_sq_idx == const_H1) this->castling_rights &= ~WK_CASTLE_FLAG;
        else if (from_sq_idx == const_A1) this->castling_rights &= ~WQ_CASTLE_FLAG;
        else if (from_sq_idx == const_H8) this->castling_rights &= ~BK_CASTLE_FLAG;
        else if (from_sq_idx == const_A8) this->castling_rights &= ~BQ_CASTLE_FLAG;
    }
    // If a rook is captured on its starting square
    if (m.is_capture() && m.piece_captured == P_ROOK) {
        if (to_sq_idx == const_H1) this->castling_rights &= ~WK_CASTLE_FLAG; // Opponent's rook captured on H1
        else if (to_sq_idx == const_A1) this->castling_rights &= ~WQ_CASTLE_FLAG;
        else if (to_sq_idx == const_H8) this->castling_rights &= ~BK_CASTLE_FLAG;
        else if (to_sq_idx == const_A8) this->castling_rights &= ~BQ_CASTLE_FLAG;
    }
    current_hash ^= Zobrist::castling_keys[this->castling_rights];


    // F. Set new en passant square
    if (m.is_double_pawn_push()) {
        this->en_passant_square = (mover_color == WHITE)
                                 ? static_cast<square_e>(from_sq_idx + 8)
                                 : static_cast<square_e>(from_sq_idx - 8);
        current_hash ^= Zobrist::en_passant_file_keys[static_cast<int>(this->en_passant_square) % 8];
    } else {
        this->en_passant_square = square_e::NO_SQ;
    }

    // G. Update fullmove number
    if (mover_color == BLACK) {
        this->fullmove_number++;
    }

    // H. Switch side to move
    this->side_to_move = opponent_color;
    current_hash ^= Zobrist::black_to_move_key;
}

//--
/* Position::unmake_move */
//--
// Reverts the last move made on the board, restoring the previous position state.
// Uses the information stored in the history_stack (StateInfo) to undo the changes made by make_move.
// This function uses the provided 'Move' object and undos the move that was just done
// and pops a 'statinfo' object from the history_stack, which contains the variables
// (like castling rights, EP square, halfmove clock, caputred piece, and the zobrist hash)
// All board representations (piece_bbs, color_bbs, occupied_bb, board_mailbox
// are restored to there previous state

void Position::unmake_move(const Move& m) {
    if (history_stack.empty()) {
        return;
    }
    StateInfo prev_state = history_stack.back();
    history_stack.pop_back();

    // Restore game state variables that are fully reset
    int original_side_that_made_move = (this->side_to_move == WHITE) ? BLACK : WHITE; // Side that made 'm'
    this->side_to_move = original_side_that_made_move;

    if (original_side_that_made_move == BLACK) {
        this->fullmove_number--;
    }

    this->castling_rights = prev_state.castling_rights;
    this->en_passant_square = prev_state.en_passant_square;
    this->halfmove_clock = prev_state.halfmove_clock;
    // Zobrist hash is restored at the end

    // --- Undo piece movements and captures (Incremental Updates) ---
    int mover_color = original_side_that_made_move; // Color of the piece that was moved by 'm'
    int opponent_color = (mover_color == WHITE) ? BLACK : WHITE;

    square_e from_sq = m.from_sq;
    square_e to_sq = m.to_sq;
    int from_sq_idx = static_cast<int>(from_sq);
    int to_sq_idx = static_cast<int>(to_sq);

    piece_type_e original_moved_piece = m.piece_moved;
    int original_moved_piece_idx = static_cast<int>(original_moved_piece);

    piece_type_e piece_that_landed = m.is_promotion() ? m.get_promotion_piece() : original_moved_piece;
    int piece_that_landed_idx = static_cast<int>(piece_that_landed);

    // A. Remove the piece that landed on to_sq
    clear_bit(piece_bbs[piece_that_landed_idx][mover_color], to_sq);
    clear_bit(color_bbs[mover_color], to_sq);
    clear_bit(occupied_bb, to_sq); // to_sq is now empty (before restoring potential capture)
    board_mailbox[to_sq_idx] = EMPTY_MAILBOX_VAL; // Will be overwritten if capture restored

    // B. Place the original piece back on from_sq
    set_bit(piece_bbs[original_moved_piece_idx][mover_color], from_sq);
    set_bit(color_bbs[mover_color], from_sq);
    set_bit(occupied_bb, from_sq); // from_sq is now occupied
    board_mailbox[from_sq_idx] = make_mailbox_entry(original_moved_piece, mover_color);

    // C. Restore captured piece, if any
    piece_type_e captured_piece_type = prev_state.captured_piece_type;
    if (captured_piece_type != P_NONE) {
        int captured_piece_idx = static_cast<int>(captured_piece_type);
        square_e actual_capture_sq = to_sq; // Default for normal captures
        int actual_capture_sq_idx = to_sq_idx;

        if (m.is_en_passant()) {
            actual_capture_sq = (mover_color == WHITE)
                               ? static_cast<square_e>(to_sq_idx - 8)
                               : static_cast<square_e>(to_sq_idx + 8);
            actual_capture_sq_idx = static_cast<int>(actual_capture_sq);
            // For EP, the `to_sq` was empty before the move, so `occupied_bb` at `to_sq` remains clear.
            // The captured pawn is restored on `actual_capture_sq`.
        }
        // Place the captured piece back
        set_bit(piece_bbs[captured_piece_idx][opponent_color], actual_capture_sq);
        set_bit(color_bbs[opponent_color], actual_capture_sq);
        set_bit(occupied_bb, actual_capture_sq); // actual_capture_sq is now occupied
        board_mailbox[actual_capture_sq_idx] = make_mailbox_entry(captured_piece_type, opponent_color);
    }

    // D. Undo castling (move the rook back)
    if (m.is_castling()) {
        square_e rook_landed_sq, rook_original_sq;
        int rook_landed_sq_idx, rook_original_sq_idx;
        int rook_type_idx = static_cast<int>(P_ROOK);
        
        // Use defined constants if available
        const int const_F1 = static_cast<int>(square_e::SQ_F1);
        const int const_H1 = static_cast<int>(square_e::SQ_H1);
        const int const_D1 = static_cast<int>(square_e::SQ_D1);
        const int const_A1 = static_cast<int>(square_e::SQ_A1);
        const int const_F8 = static_cast<int>(square_e::SQ_F8);
        const int const_H8 = static_cast<int>(square_e::SQ_H8);
        const int const_D8 = static_cast<int>(square_e::SQ_D8);
        const int const_A8 = static_cast<int>(square_e::SQ_A8);


        if (m.is_kingside_castle()) {
            rook_landed_sq = (mover_color == WHITE) ? static_cast<square_e>(const_F1) : static_cast<square_e>(const_F8);
            rook_original_sq = (mover_color == WHITE) ? static_cast<square_e>(const_H1) : static_cast<square_e>(const_H8);
        } else { // Queenside
            rook_landed_sq = (mover_color == WHITE) ? static_cast<square_e>(const_D1) : static_cast<square_e>(const_D8);
            rook_original_sq = (mover_color == WHITE) ? static_cast<square_e>(const_A1) : static_cast<square_e>(const_A8);
        }
        rook_landed_sq_idx = static_cast<int>(rook_landed_sq);
        rook_original_sq_idx = static_cast<int>(rook_original_sq);

        // Remove rook from its landing square
        clear_bit(piece_bbs[rook_type_idx][mover_color], rook_landed_sq);
        clear_bit(color_bbs[mover_color], rook_landed_sq);
        clear_bit(occupied_bb, rook_landed_sq);
        board_mailbox[rook_landed_sq_idx] = EMPTY_MAILBOX_VAL;

        // Place rook back on its original square
        set_bit(piece_bbs[rook_type_idx][mover_color], rook_original_sq);
        set_bit(color_bbs[mover_color], rook_original_sq);
        set_bit(occupied_bb, rook_original_sq);
        board_mailbox[rook_original_sq_idx] = make_mailbox_entry(P_ROOK, mover_color);
    }

    // F. Restore Zobrist hash
    this->current_hash = prev_state.hash;
}

//--
/* Position::is_square_attacked */
//--
// Checks if a given square is attacked by any piece of the specified attacker's color.
// Requires generation of attack bitboards for all piece types of the attacker_color that could attack sq_to_check.
//
bool Position::is_square_attacked(square_e sq_to_check, int by_attacker_color) const {
    if (sq_to_check == square_e::NO_SQ) { // this would be an invalid input square
        return false;
    }

    int sq_idx = static_cast<int>(sq_to_check);

    // 1. Check attacks by Pawns of 'by_attacker_color'
    // To find if 'sq_to_check' is attacked by a pawn of 'by_attacker_color',
    // we look at the squares from which such a pawn would attack 'sq_to_check'.
    // These squares are equivalent to the squares a pawn of the *opposite* color
    // on 'sq_to_check' would attack.
    int defender_color = (by_attacker_color == WHITE) ? BLACK : WHITE;
    // pawn_attacks[defender_color][sq_idx] gives squares a 'defender_color' pawn on 'sq_idx' attacks.
    // If any of these squares are occupied by a 'by_attacker_color' pawn, then 'sq_to_check' is attacked.
    if ((hyperion::core::pawn_attacks[defender_color][sq_idx] & get_pieces(P_PAWN, by_attacker_color)) != 0) {
        return true;
    }

    // 2. Check attacks by Knights of 'by_attacker_color'
    // knight_attacks[sq_idx] gives all squares a knight on 'sq_idx' attacks.
    // If any of these squares are occupied by a 'by_attacker_color' knight, then 'sq_idx' is attacked.
    if ((hyperion::core::knight_attacks[sq_idx] & get_pieces(P_KNIGHT, by_attacker_color)) != 0) {
        return true;
    }

    // 3. Check attacks by the King of 'by_attacker_color'
    // king_attacks[sq_idx] gives all squares a king on 'sq_idx' attacks.
    if ((hyperion::core::king_attacks[sq_idx] & get_pieces(P_KING, by_attacker_color)) != 0) {
        return true;
    }

    // For slider pieces (rooks, bishops, queens), we use the magic bitboard functions.
    // this->occupied_bb should be up-to-date, representing all pieces on the board.

    // 4. Check attacks by Rooks (and rook-like Queen moves) of 'by_attacker_color'
    // get_rook_slider_attacks(sq_to_check, this->occupied_bb) returns a bitboard of squares
    // that a rook on 'sq_to_check'' would attack, considering current board occupancy.
    // If any of these squares are occupied by 'by_attacker_color's rooks or queen,
    // then 'sq_to_check' is attacked by that rook/queen.
    bitboard_t rook_attack_potential = hyperion::core::get_rook_slider_attacks(sq_to_check, this->occupied_bb);
    if ((rook_attack_potential & (get_pieces(P_ROOK, by_attacker_color) | get_pieces(P_QUEEN, by_attacker_color))) != 0) {
        return true;
    }

    // 5. Check attacks by Bishops (and bishop-like Queen moves) of 'by_attacker_color'
    // Similar logic to rooks.
    bitboard_t bishop_attack_potential = hyperion::core::get_bishop_slider_attacks(sq_to_check, this->occupied_bb);
    if ((bishop_attack_potential & (get_pieces(P_BISHOP, by_attacker_color) | get_pieces(P_QUEEN, by_attacker_color))) != 0) {
        return true;
    }

    // If no piece of 'by_attacker_color' attacks 'sq_to_check'
    return false;
}

/* RAY CASTING IS SQUARE ATTACK*/
// TEMP RAY CASTING USE BECAUSE I CANT FOR THE LIFE OF ME FIGURE OUT MAGIC NUMBERS BITS SHIFTS MASK BS
/*
bool Position::is_square_attacked(square_e sq_to_check, int by_attacker_color) const {
    if (sq_to_check == square_e::NO_SQ) {
        return false;
    }
    int sq_idx = static_cast<int>(sq_to_check);

    // 1. Check attacks by Pawns (no change needed here)
    int defender_color = (by_attacker_color == WHITE) ? BLACK : WHITE;
    if ((hyperion::core::pawn_attacks[defender_color][sq_idx] & get_pieces(P_PAWN, by_attacker_color)) != 0) {
        return true;
    }

    // 2. Check attacks by Knights (no change needed here)
    if ((hyperion::core::knight_attacks[sq_idx] & get_pieces(P_KNIGHT, by_attacker_color)) != 0) {
        return true;
    }

    // 3. Check attacks by the King (no change needed here)
    if ((hyperion::core::king_attacks[sq_idx] & get_pieces(P_KING, by_attacker_color)) != 0) {
        return true;
    }

    // 4. Check attacks by Rooks (and rook-like Queen moves) of 'by_attacker_color'
    // OLD LINE:
    // bitboard_t rook_attack_potential = hyperion::core::get_rook_slider_attacks(sq_to_check, this->occupied_bb);
    // NEW LINE:
    bitboard_t rook_attack_potential = hyperion::core::generate_attacks_slow_internal(
        static_cast<int>(sq_to_check),  // The square to check if attacked
        this->occupied_bb,              // All pieces on the board act as blockers
        true                            // is_rook = true
    );
    if ((rook_attack_potential & (get_pieces(P_ROOK, by_attacker_color) | get_pieces(P_QUEEN, by_attacker_color))) != 0) {
        return true;
    }

    // 5. Check attacks by Bishops (and bishop-like Queen moves) of 'by_attacker_color'
    // OLD LINE:
    // bitboard_t bishop_attack_potential = hyperion::core::get_bishop_slider_attacks(sq_to_check, this->occupied_bb);
    // NEW LINE:
    bitboard_t bishop_attack_potential = hyperion::core::generate_attacks_slow_internal(
        static_cast<int>(sq_to_check),  // The square to check if attacked
        this->occupied_bb,              // All pieces on the board act as blockers
        false                           // is_rook = false (so it's bishop)
    );
    if ((bishop_attack_potential & (get_pieces(P_BISHOP, by_attacker_color) | get_pieces(P_QUEEN, by_attacker_color))) != 0) {
        return true;
    }

    return false;
}
*/

//--
/* Position::is_in_check */
//--
// Checks if the king of the current side_to_move is in check.
// Calls is_king_in_check with the current side_to_move.
bool Position::is_in_check() const {
    return is_king_in_check(this->side_to_move);
}
//--
/* Position::is_king_in_check */
//--
// Checks if the king of the specified color is in check.
// Takes an integer king_color_to_check (WHITE or BLACK).
// Finds the king's square for that color using get_king_square.
// If the king is found, calls is_square_attacked to see if the opponent is attacking the king's square.
// Returns true if the king is not found (should be an error state) or if the king's square is attacked.

bool Position::is_king_in_check(int king_color_to_check) const {
    square_e k_sq = get_king_square(king_color_to_check);
    if (k_sq == square_e::NO_SQ) {
        return true;
    }
    return is_square_attacked(k_sq, (king_color_to_check == WHITE) ? BLACK : WHITE);
}


} // namespace core
} // namespace hyperion