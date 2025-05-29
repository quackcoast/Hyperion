// hyperion/src/cpp/core/bitboard.cpp
#include "bitboard.hpp"
#include "constants.hpp"
#include "position.hpp"
                           

// namespace is used for organization
namespace hyperion { 
namespace core {

// Sources
// https://en.wikipedia.org/wiki/Intrinsic_function <- look for more info on general Intrinsic functions
// https://www.chessprogramming.org/BitScan <- bitscan info

/*
Compilers for C and C++, of Microsoft, Intel, and the GNU Compiler Collection (GCC) implement intrinsics
that map directly to the x86 single instruction. Intrinsics allow mapping to standard assembly instructions
that are not normally accessible through C/C++, e.g., bit scan.

Dev Note: Im going to try to use them for there performace gain, but if things get too
messy or weird. I will just remove them and then update functions. For now I will elif because I dont
know if we use clang or GCC (ive used GCC more in the past). In the end, higher performace = higher elo
*/

#if defined(_MSC_VER)
    #include <intrin.h> // For _BitScanForward64, __popcnt64
#elif defined(__GNUC__) || defined(__clang__)
    // GCC and Clang have built-in functions (no specific include needed for __builtin_popcountll, __builtin_ctzll)
#else
    // Fallback implementations if no intrinsics, we should TRY use them though for performance gain
#endif

// --- Core Bitboard Operations ---

// Using bitboard_t as defined in bitboard.hpp
void set_bit(bitboard_t& bb, int square_index) {
    if(square_index >= 0 && square_index < hyperion::core::NUM_SQUARES) { //bounds check
        bb |= (1ULL << square_index); // unsigned long long
    }
}

void clear_bit(bitboard_t& bb, int square_index) {
    if(square_index >= 0 && square_index < hyperion::core::NUM_SQUARES) { //bounds check
        bb &= ~(1ULL << square_index); // unsigned long long
    }
}

bool get_bit(bitboard_t bb, int square_index) {
    if (square_index < 0 || square_index >= hyperion::core::NUM_SQUARES) { //bounds check
        return false;
    }
    return (bb & (1ULL << square_index)) != 0;
}

// --- Utility Functions ---

// Function name should match the declaration in bitboard.hpp (print_bitboard)
void print_bitboard(bitboard_t bb) {
    std::cout << " Bitboard 0x" << std::hex << bb << std::dec << std::endl;
    for (int rank_idx = 7; rank_idx >= 0; --rank_idx) { //ranks 8 down to 1
        std::cout << (rank_idx + 1) << "  ";
        for (int file_idx = 0; file_idx < 8; ++file_idx){ // Files A to H
            // This is where the square index calculation using the global
            // constants in constant.hpp are helpful.
            // How to get the index num? general formal is right here:
            //     int square_index = rank_idx * 8 + file_idx;
            // Then go through and get_bit(bb, square_index) through all 64 indexs
            int current_square_idx = rank_idx * 8 + file_idx;
            if (get_bit(bb,current_square_idx)) {
                std::cout << "1 ";
            }
            else {
                std::cout << hyperion::core::EMPTY_SQUARE_CHAR << " "; //
            }
        } 
        std::cout << std::endl;
    } 
    std::cout << "   a b c d e f g h" << std::endl; 
    std::cout << std::endl;

    // example empty bitboard:
    //
    // 8  . . . . . . . .
    // 7  . . . . . . . .
    // 6  . . . . . . . .
    // 5  . . . . . . . .
    // 4  . . . . . . . .
    // 3  . . . . . . . .
    // 2  . . . . . . . .
    // 1  . . . . . . . .
    //    a b c d e f g h
}

int count_set_bits(bitboard_t bb) {
    // this is where possible optimization is potentially used
    // first try _MSC_VER, second try GCC or CLANG, third fallback is
    // Brain Kernighans Algorithm:
    //     https://www.techiedelight.com/brian-kernighans-algorithm-count-set-bits-integer/
    //
    //     for more info on that algorithm read above^
#if defined(_MSC_VER)
    return static_cast<int>(__popcnt64(bb));
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(bb);
#else
    // Brian Kernighan's Algorithm (fallback/ failsafe)
    int count = 0;
    while (bb > 0) {
        bb &= (bb - 1);
        count++;
    }
    return count;
#endif
}

int get_lsb_index(bitboard_t bb) {
    if (bb == 0){
        return static_cast<int>(hyperion::core::square_e::NO_SQ);
    }
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanForward64(&index, bb);
    return static_cast<int>(index);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(bb); // Count Trailing Zeros
#else
    // Fallback/ failsafe:
    int count = 0;
    // This loop assumes bb != 0 (guaranteed by the check above)
    while (!((bb >> count) & 1ULL)) {
        count++;
    }
    return count;
#endif
}

int pop_lsb(bitboard_t& bb) {
    int lsb_idx = get_lsb_index(bb); 
    if (lsb_idx != static_cast<int>(hyperion::core::square_e::NO_SQ)) {
        // this should be faster then calling the clear_bit()
        // LSB: bb &= (bb -1); <-- should be faster if LSB was
        // found via intrinsics
        // If we decide not to use intrinsics, this can all be chaned to just call
        // clear_bit
        bb &= (bb - 1); // Efficiently clear LSB
    }
    return lsb_idx;
}

std::string square_to_algebraic(int square_index) {
    if (square_index < 0 || square_index >= hyperion::core::NUM_SQUARES) {
        return "??"; //bounds check, this would be a completly invalid square
    }
    char file_char = 'a' + (square_index % 8); 
    char rank_char = '1' + (square_index / 8); 
    std::string s = "";
    s += file_char;
    s += rank_char;
    return s;
}

// --- attacks ---

// Define the actual storage for the extern declared arrays
std::array<std::array<bitboard_t, NUM_SQUARES>, 2> pawn_attacks;
std::array<bitboard_t, NUM_SQUARES> knight_attacks;
std::array<bitboard_t, NUM_SQUARES> king_attacks;

// Helper to safely add an attack if the target square is on the board
void add_attack(bitboard_t& attacks_bb, int current_file, int current_rank, int df, int dr) {
    int to_file = current_file + df;
    int to_rank = current_rank + dr;
    if (to_file >= 0 && to_file < 8 && to_rank >= 0 && to_rank < 8) {
        set_bit(attacks_bb, static_cast<square_e>(to_rank * 8 + to_file));
    }
}

void initialize_attack_tables() {
    for (int sq_idx = 0; sq_idx < NUM_SQUARES; ++sq_idx) {
        square_e sq = static_cast<square_e>(sq_idx);
        int rank = sq_idx / 8;
        int file = sq_idx % 8;

        // Pawn attacks
        pawn_attacks[WHITE][sq_idx] = EMPTY_BB;
        if (rank < 7) { // White pawns can't attack from 8th rank
            if (file > 0) set_bit(pawn_attacks[WHITE][sq_idx], static_cast<square_e>((rank + 1) * 8 + (file - 1))); // Capture left
            if (file < 7) set_bit(pawn_attacks[WHITE][sq_idx], static_cast<square_e>((rank + 1) * 8 + (file + 1))); // Capture right
        }
        pawn_attacks[BLACK][sq_idx] = EMPTY_BB;
        if (rank > 0) { // Black pawns can't attack from 1st rank
            if (file > 0) set_bit(pawn_attacks[BLACK][sq_idx], static_cast<square_e>((rank - 1) * 8 + (file - 1))); // Capture left
            if (file < 7) set_bit(pawn_attacks[BLACK][sq_idx], static_cast<square_e>((rank - 1) * 8 + (file + 1))); // Capture right
        }

        // Knight attacks
        knight_attacks[sq_idx] = EMPTY_BB;
        const int knight_moves[8][2] = {
            {1, 2}, {1, -2}, {-1, 2}, {-1, -2},
            {2, 1}, {2, -1}, {-2, 1}, {-2, -1}
        };
        for (const auto& move : knight_moves) {
            add_attack(knight_attacks[sq_idx], file, rank, move[0], move[1]);
        }

        // King attacks
        king_attacks[sq_idx] = EMPTY_BB;
        const int king_moves[8][2] = {
            {0, 1}, {0, -1}, {1, 0}, {-1, 0},
            {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
        };
        for (const auto& move : king_moves) {
            add_attack(king_attacks[sq_idx], file, rank, move[0], move[1]);
        }
    }
    // Slider attack tables (e.g., magic bitboards) would be initialized here too.
}
// --- Basic Slider Attack Generation (Ray Casting - Slower, for illustration) ---
// For high performance, you'd use Magic Bitboards.

// Directions: N, S, E, W, NE, NW, SE, SW
const int RAYS_DELTAS[8][2] = {
    {0, 1}, {0, -1}, {1, 0}, {-1, 0}, // Rook-like (Orthogonal)
    {1, 1}, {-1, 1}, {1, -1}, {-1, -1} // Bishop-like (Diagonal)
};

bitboard_t get_rook_attacks(square_e sq, bitboard_t occupied) {
    bitboard_t attacks = EMPTY_BB;
    int r = static_cast<int>(sq) / 8;
    int f = static_cast<int>(sq) % 8;

    for (int i = 0; i < 4; ++i) { // First 4 are orthogonal directions
        for (int d = 1; d < 8; ++d) {
            int next_f = f + RAYS_DELTAS[i][0] * d;
            int next_r = r + RAYS_DELTAS[i][1] * d;

            if (next_f >= 0 && next_f < 8 && next_r >= 0 && next_r < 8) {
                square_e target_sq = static_cast<square_e>(next_r * 8 + next_f);
                set_bit(attacks, target_sq);
                if (get_bit(occupied, target_sq)) { // Stop if piece encountered
                    break;
                }
            } else { // Off board
                break;
            }
        }
    }
    return attacks;
}

bitboard_t get_bishop_attacks(square_e sq, bitboard_t occupied) {
    bitboard_t attacks = EMPTY_BB;
    int r = static_cast<int>(sq) / 8;
    int f = static_cast<int>(sq) % 8;

    for (int i = 4; i < 8; ++i) { // Last 4 are diagonal directions
        for (int d = 1; d < 8; ++d) {
            int next_f = f + RAYS_DELTAS[i][0] * d;
            int next_r = r + RAYS_DELTAS[i][1] * d;

            if (next_f >= 0 && next_f < 8 && next_r >= 0 && next_r < 8) {
                square_e target_sq = static_cast<square_e>(next_r * 8 + next_f);
                set_bit(attacks, target_sq);
                if (get_bit(occupied, target_sq)) { // Stop if piece encountered
                    break;
                }
            } else { // Off board
                break;
            }
        }
    }
    return attacks;
}
} // namespace core   
} // namespace hyperion 