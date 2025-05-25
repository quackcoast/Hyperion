// hyperion/src/cpp/core/bitboard.hpp
#ifndef HYPERION_CORE_BITBOARD_HPP
#define HYPERION_CORE_BITBOARD_HPP

#include "constants.hpp" 

#include <cstdint>  // For uint64_t
#include <string>   // For std::string
#include <iostream> // For std::cout

namespace hyperion {
namespace core {   

// Defining a type for the bitboards, using snake_case
using bitboard_t = uint64_t;

enum class square_e {
    SQ_A1, SQ_B1, SQ_C1, SQ_D1, SQ_E1, SQ_F1, SQ_G1, SQ_H1,
    SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2,
    SQ_A3, SQ_B3, SQ_C3, SQ_D3, SQ_E3, SQ_F3, SQ_G3, SQ_H3,
    SQ_A4, SQ_B4, SQ_C4, SQ_D4, SQ_E4, SQ_F4, SQ_G4, SQ_H4,
    SQ_A5, SQ_B5, SQ_C5, SQ_D5, SQ_E5, SQ_F5, SQ_G5, SQ_H5,
    SQ_A6, SQ_B6, SQ_C6, SQ_D6, SQ_E6, SQ_F6, SQ_G6, SQ_H6,
    SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7,
    SQ_A8, SQ_B8, SQ_C8, SQ_D8, SQ_E8, SQ_F8, SQ_G8, SQ_H8,
    NO_SQ = 64 // Matches NUM_SQUARES from constants.hpp
               
};

// --- Constants ---
const bitboard_t EMPTY_BB = 0ULL;    
const bitboard_t UNIVERSAL_BB = ~0ULL;

// --- Core Bitboard Operations ---

void set_bit(bitboard_t& bb, int square_index);
inline void set_bit(bitboard_t& bb, square_e sq) {
    if (sq != square_e::NO_SQ) set_bit(bb, static_cast<int>(sq));
}

void clear_bit(bitboard_t& bb, int square_index);
inline void clear_bit(bitboard_t& bb, square_e sq) {
    if (sq != square_e::NO_SQ) clear_bit(bb, static_cast<int>(sq));
}

bool get_bit(bitboard_t bb, int square_index);
inline bool get_bit(bitboard_t bb, square_e sq) {
    return (sq != square_e::NO_SQ) ? get_bit(bb, static_cast<int>(sq)) : false;
}

// --- Utility Functions ---

void print_bitboard(bitboard_t bb);

int count_set_bits(bitboard_t bb);

// Returns static_cast<int>(square_e::NO_SQ) if bitboard is empty
int get_lsb_index(bitboard_t bb);

// Returns static_cast<int>(square_e::NO_SQ) if bitboard was empty
int pop_lsb(bitboard_t& bb);

std::string square_to_algebraic(int square_index);
inline std::string square_to_algebraic(square_e sq) { 
    return (sq != square_e::NO_SQ) ? square_to_algebraic(static_cast<int>(sq)) : "NO_SQ";
}

inline bitboard_t square_to_bitboard(square_e sq) {
    if (sq == square_e::NO_SQ) return EMPTY_BB;
    return 1ULL << static_cast<int>(sq);
}
inline bitboard_t square_to_bitboard(int square_index) {
    if (square_index < 0 || square_index >= hyperion::core::NUM_SQUARES) return EMPTY_BB;
    return 1ULL << square_index;
}

} // namespace core 
} // namespace hyperion 

#endif // HYPERION_CORE_BITBOARD_HPP