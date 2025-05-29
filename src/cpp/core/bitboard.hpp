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


// --- Precomputed Attack Tables ---
// Initialized once in a Zobrist::initialize()
// Pawn attacks: [color][from_square]
extern std::array<std::array<bitboard_t, NUM_SQUARES>, 2> pawn_attacks;

// Knight attacks: [from_square]
extern std::array<bitboard_t, NUM_SQUARES> knight_attacks;

// King attacks: [from_square]
extern std::array<bitboard_t, NUM_SQUARES> king_attacks;

// Functions to generate slider attacks
// TODO MAGIC BITBOARDS
// Queen attacks can combine rook and bishop attacks

// Call this once at program startup
//void initialize_attack_tables();

} // namespace core 
} // namespace hyperion 

#endif // HYPERION_CORE_BITBOARD_HPP