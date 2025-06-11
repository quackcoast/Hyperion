#include "zobrist.hpp"
#include <random>

namespace hyperion {
namespace core {

// Define static members
std::array<std::array<std::array<zobrist_key_t, NUM_SQUARES>, 2>, NUM_PIECE_TYPES> Zobrist::piece_square_keys;
zobrist_key_t Zobrist::black_to_move_key;
std::array<zobrist_key_t, 16> Zobrist::castling_keys;
std::array<zobrist_key_t, 8> Zobrist::en_passant_file_keys;

void Zobrist::initialize_keys() {
    std::mt19937_64 rng(0x2C0DE2DE);
    std::uniform_int_distribution<zobrist_key_t> dist(0, UINT64_MAX);

    for (int piece = 0; piece < NUM_PIECE_TYPES; ++piece) {
        for (int color = 0; color < 2; ++color) {
            for (int sq = 0; sq < NUM_SQUARES; ++sq) {
                piece_square_keys[piece][color][sq] = dist(rng);
            }
        }
    }

    black_to_move_key = dist(rng);

    for (int i = 0; i < 16; ++i) { // For all combinations of castling rights
        Zobrist::castling_keys[i] = dist(rng);
    }
    for (int file = 0; file < 8; ++file) {
        Zobrist::en_passant_file_keys[file] = dist(rng);
    }
}

} // namespace core
} // namespace hyperion