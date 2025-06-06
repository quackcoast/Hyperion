#ifndef HYPERION_CORE_ZOBRIST_HPP
#define HYPERION_CORE_ZOBRIST_HPP

#include "constants.hpp"
#include <cstdint>
#include <array>

namespace hyperion {
namespace core {

using zobrist_key_t = uint64_t;

class Zobrist {
public:
    // Keys for each piece type, on each color, on each square
    static std::array<std::array<std::array<zobrist_key_t, NUM_SQUARES>, 2>, NUM_PIECE_TYPES> piece_square_keys;
    // Key for side to move (e.g., one key if it's black's turn)
    static zobrist_key_t black_to_move_key;
    // Keys for each castling right (e.g., WK_CASTLE_FLAG maps to a key)
    static std::array<zobrist_key_t, 16> castling_keys; // Index by castling_rights bitmask
    // Keys for en passant file (one for each file 'a' through 'h')
    static std::array<zobrist_key_t, 8> en_passant_file_keys;

    // Initializes all the random Zobrist keys
    static void initialize_keys();

private:
    // Private constructor to prevent instantiation, or make all members static
    Zobrist() = delete;
};

} // namespace core
} // namespace hyperion

#endif // HYPERION_CORE_ZOBRIST_HPP