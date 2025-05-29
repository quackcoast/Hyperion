// hyperion/src/cpp/core/position.cpp
#include "position.hpp"
#include "constants.hpp"
#include "bitboard.hpp"
#include "zobrist.hpp"    
#include "move.hpp"       
#include <sstream>        
#include <algorithm>      
#include <cctype>         

// This is a very big file so it can be difficult to work through.
// These are the main parts of the files, with any header includes shown whenever used by it
//
// 1) There are 2 helper functions that get info from the fen characters (mainly color and piece info)
//
// 2) There are 3 member functions specifically for the mailbox representation, this is essentially keeping track of the current board
//    with an overall board showing all the current pieces (with colors). Useful for debugging, but during an actually UCI game we should probably
//    turn it off for maximum porformance (its probably minimal but still).
//
// 3) Position constructor just sets the position to a normal start position, nothing funny going on here.
//
// 4) The rest I will have comments before each of the member function to save space up here, and so you dont have to scroll much !
//
//

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
    return '?';
}


namespace hyperion {
namespace core {

const int EMPTY_MAILBOX_VAL = -1; // this is equivent to static_cast<int>(P_NONE)

int Position::make_mailbox_entry(piece_type_e type, int color) const {
    if (type == P_NONE) return EMPTY_MAILBOX_VAL;
    // Example: White pieces P_PAWN=0 to P_KING=5, Black pieces P_PAWN=6 to P_KING=11
    return static_cast<int>(type) + (color == BLACK ? NUM_PIECE_TYPES : 0);
}
piece_type_e Position::get_piece_type_from_mailbox_val(int mb_val) const {
    if (mb_val == EMPTY_MAILBOX_VAL || mb_val < 0 || mb_val >= (NUM_PIECE_TYPES * 2)) return P_NONE;
    return static_cast<piece_type_e>(mb_val % NUM_PIECE_TYPES);
}
int Position::get_color_from_mailbox_val(int mb_val) const {
    if (mb_val == EMPTY_MAILBOX_VAL || mb_val < 0 || mb_val >= (NUM_PIECE_TYPES * 2)) return -1; // Invalid
    return (mb_val >= NUM_PIECE_TYPES) ? BLACK : WHITE;
}

Position::Position() {
    // IMPORTANT: Zobrist::initialize_keys() must have been called once globally
    // this should be done in main when the program starts, but for any debugging or
    // before any Position object is created or set_from_fen is called use member function.
    set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); //this is the start position of a normal chess game
}

/* information on clear_board_state() */
// sets all bitboards to empty, clears the history, and resets castling_rights to 0, en_passant_square = ::NO_SQ, halfmove clock to 0, fullmove_number to 1

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
}

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
int Position::get_side_to_move() const { return side_to_move; }
bitboard_t Position::get_pieces(piece_type_e p_type, int p_color) const {
    return piece_bbs[static_cast<int>(p_type)][p_color];
}
bitboard_t Position::get_pieces_by_type(piece_type_e p_type) const {
    return piece_bbs[static_cast<int>(p_type)][WHITE] | piece_bbs[static_cast<int>(p_type)][BLACK];
}
bitboard_t Position::get_pieces_by_color(int p_color) const { return color_bbs[p_color]; }
bitboard_t Position::get_occupied_squares() const { return occupied_bb; }

square_e Position::get_king_square(int king_color) const {
    bitboard_t king_bb = piece_bbs[static_cast<int>(P_KING)][king_color];
    if (king_bb == 0) return square_e::NO_SQ; // Should not happen in valid game
    return static_cast<square_e>(get_lsb_index(king_bb));
}

int Position::get_piece_on_square(square_e sq) const {
    if (sq == square_e::NO_SQ || static_cast<int>(sq) >= NUM_SQUARES) return EMPTY_MAILBOX_VAL;
    return board_mailbox[static_cast<int>(sq)];
}

// --- Move Execution ---
void Position::make_move(const Move& m) {
    //makemove stuff
}

void Position::unmake_move(const Move& m) {
    //unmakemove stuff
}


bool Position::is_square_attacked(square_e sq_to_check, int by_attacker_color) const {
 //need attack boards
}


bool Position::is_in_check() const {
    return is_king_in_check(this->side_to_move);
}

bool Position::is_king_in_check(int king_color_to_check) const {
    square_e k_sq = get_king_square(king_color_to_check);
    if (k_sq == square_e::NO_SQ) {
        return true;
    }
    return is_square_attacked(k_sq, (king_color_to_check == WHITE) ? BLACK : WHITE);
}


} // namespace core
} // namespace hyperion