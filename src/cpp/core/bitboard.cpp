// hyperion/src/cpp/core/bitboard.cpp
#include "bitboard.hpp"
#include "constants.hpp"
                           


namespace hyperion { // Ensure this is lowercase
namespace core {   // Ensure this is lowercase

// https://en.wikipedia.org/wiki/Intrinsic_function <- look for more info on general Intrinsic functions

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

} // namespace core   
} // namespace hyperion 