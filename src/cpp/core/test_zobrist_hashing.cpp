// hyperion/src/cpp/core/test_zobrist_hash.cpp
#include "zobrist.hpp"
#include "constants.hpp"
#include <iostream>
#include <vector>
#include <set>
#include <cassert>
#include <numeric>

// Helper function to make tests more readable
inline void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "Assertion failed: " << message << std::endl;
        assert(condition);
    }
}

/*
---
* Zobrist Hashing tests can be compiled with the following command:

    *g++ -std=c++17 -Wall -Wextra -Icore -o test_zobrist.exe src/cpp/core/zobrist.cpp src/cpp/core/test_zobrist_hashing.cpp*

* Run it:
    *./test_zobrist.exe*
---
*/


void test_initialization() {
    std::cout << "Running test_initialization..." << std::endl;

    // 1. Check for non-zero keys (it's statistically very unlikely any key is 0 (like super duper unlikely), but good check nevertheless)
    check(hyperion::core::Zobrist::black_to_move_key != 0, "black_to_move_key should be non-zero.");

    std::set<hyperion::core::zobrist_key_t> all_keys;
    all_keys.insert(hyperion::core::Zobrist::black_to_move_key);

    for (int piece = 0; piece < hyperion::core::NUM_PIECE_TYPES; ++piece) {
        for (int color = 0; color < 2; ++color) {
            for (int sq = 0; sq < hyperion::core::NUM_SQUARES; ++sq) {
                hyperion::core::zobrist_key_t key = hyperion::core::Zobrist::piece_square_keys[piece][color][sq];
                check(key != 0, "Piece-square key should be non-zero.");
                all_keys.insert(key);
            }
        }
    }

    for (int i = 0; i < 16; ++i) {
        hyperion::core::zobrist_key_t key = hyperion::core::Zobrist::castling_keys[i];
        // Note: castling_keys[0] (no castling rights) is also a random number.
        // If it were 0, it would mean "no rights" doesn't contribute to hash, which is also a valid design.
        // But current init gives it a random value.
        check(key != 0, "Castling key should be non-zero.");
        all_keys.insert(key);
    }

    for (int i = 0; i < 8; ++i) {
        hyperion::core::zobrist_key_t key = hyperion::core::Zobrist::en_passant_file_keys[i];
        check(key != 0, "En passant file key should be non-zero.");
        all_keys.insert(key);
    }

    // 2. Check for uniqueness
    // Total keys = 1 (black_to_move) + (6*2*64) (piece_square) + 16 (castling) + 8 (en_passant_file)
    //              = 1 + 768 + 16 + 8 = 793
    size_t expected_unique_keys = 1 + (hyperion::core::NUM_PIECE_TYPES * 2 * hyperion::core::NUM_SQUARES) + 16 + 8;
    check(all_keys.size() == expected_unique_keys, "All Zobrist keys should be unique.");

    // 3. Check reproducibility (if initialized again with same seed, keys should be identical)
    // This is implicitly tested if tests pass consistently.
    // We can explicitly store a few keys, re-initialize, and check they are the same.
    hyperion::core::zobrist_key_t sample_key1 = hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_PAWN][hyperion::core::WHITE][hyperion::core::E2];
    hyperion::core::zobrist_key_t sample_key2 = hyperion::core::Zobrist::black_to_move_key;

    hyperion::core::Zobrist::initialize_keys(); // Reinitialize

    check(hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_PAWN][hyperion::core::WHITE][hyperion::core::E2] == sample_key1, "Key for White Pawn on E2 mismatch after re-initialization.");
    check(hyperion::core::Zobrist::black_to_move_key == sample_key2, "black_to_move_key mismatch after re-initialization.");


    std::cout << "test_initialization passed." << std::endl;
}

void test_individual_components() {
    std::cout << "Running test_individual_components..." << std::endl;
    hyperion::core::zobrist_key_t hash = 0;
    hyperion::core::zobrist_key_t initial_hash = hash;

    // 1. Piece operations
    hyperion::core::zobrist_key_t pawn_e2_key = hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_PAWN][hyperion::core::WHITE][hyperion::core::E2];
    
    // Add pawn to E2
    hash ^= pawn_e2_key;
    check(hash == pawn_e2_key, "Hash incorrect after adding pawn to E2.");

    // Remove pawn from E2
    hash ^= pawn_e2_key;
    check(hash == initial_hash, "Hash incorrect after removing pawn from E2 (should be initial).");

    // Move pawn from E2 to E4
    hyperion::core::zobrist_key_t pawn_e4_key = hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_PAWN][hyperion::core::WHITE][hyperion::core::E4];
    hash ^= pawn_e2_key; // Add at E2
    hash ^= pawn_e2_key; // Remove from E2
    hash ^= pawn_e4_key; // Add at E4
    check(hash == pawn_e4_key, "Hash incorrect after moving pawn E2-E4 (from empty board).");
    hash = initial_hash; // Reset for next test

    // 2. Side to move
    check(hash == initial_hash, "Hash should be initial before side_to_move test.");
    // Initially White to move (no XOR with black_to_move_key). Switch to Black.
    hash ^= hyperion::core::Zobrist::black_to_move_key;
    check(hash == hyperion::core::Zobrist::black_to_move_key, "Hash incorrect after switching to Black to move.");
    // Switch back to White
    hash ^= hyperion::core::Zobrist::black_to_move_key;
    check(hash == initial_hash, "Hash incorrect after switching back to White to move.");

    // 3. Castling rights
    int castling_rights = hyperion::core::WK_CASTLE_FLAG | hyperion::core::WQ_CASTLE_FLAG | hyperion::core::BK_CASTLE_FLAG | hyperion::core::BQ_CASTLE_FLAG; // All rights
    hyperion::core::zobrist_key_t all_castling_key = hyperion::core::Zobrist::castling_keys[castling_rights];
    
    hash ^= all_castling_key; // Add all castling rights
    check(hash == all_castling_key, "Hash incorrect after setting all castling rights.");
    
    // White loses kingside castling
    int new_castling_rights = hyperion::core::WQ_CASTLE_FLAG | hyperion::core::BK_CASTLE_FLAG | hyperion::core::BQ_CASTLE_FLAG;
    hyperion::core::zobrist_key_t new_castling_key = hyperion::core::Zobrist::castling_keys[new_castling_rights];

    hash ^= all_castling_key;   // XOR out old rights
    hash ^= new_castling_key;   // XOR in new rights
    check(hash == new_castling_key, "Hash incorrect after White loses kingside castling.");
    
    hash = initial_hash; // Reset

    // 4. En passant
    // Assume EP on e3 (file E, index 4)
    int ep_file_idx = hyperion::core::E2 % 8; // File E is 4
    hyperion::core::zobrist_key_t ep_e_file_key = hyperion::core::Zobrist::en_passant_file_keys[ep_file_idx];

    hash ^= ep_e_file_key; // Set EP on E-file
    check(hash == ep_e_file_key, "Hash incorrect after setting EP on E-file.");

    hash ^= ep_e_file_key; // Clear EP on E-file
    check(hash == initial_hash, "Hash incorrect after clearing EP on E-file.");

    std::cout << "test_individual_components passed." << std::endl;
}

void test_make_unmake_scenario() {
    std::cout << "Running test_make_unmake_scenario..." << std::endl;

    hyperion::core::zobrist_key_t current_hash = 0; // Start with an "empty" hash for this scenario

    // Initial state for this scenario: White Pawn on E2, Black King on E8, White to move, all castling rights, no EP
    // 1. Setup initial position components
    current_hash ^= hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_PAWN][hyperion::core::WHITE][hyperion::core::E2];
    current_hash ^= hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_KING][hyperion::core::BLACK][hyperion::core::E8];
    
    int castling_state = hyperion::core::WK_CASTLE_FLAG | hyperion::core::WQ_CASTLE_FLAG | hyperion::core::BK_CASTLE_FLAG | hyperion::core::BQ_CASTLE_FLAG;
    current_hash ^= hyperion::core::Zobrist::castling_keys[castling_state];
    // Side is White to move, so no XOR with black_to_move_key yet.
    // No EP square, so no XOR with en_passant_file_keys.

    hyperion::core::zobrist_key_t hash_before_move = current_hash;

    // 2. Simulate a move: White pawn e2-e4
    // This move involves:
    // - Piece P_PAWN WHITE moves from E2 to E4
    // - Side to move becomes BLACK
    // - En passant square becomes E3 (file E, index 4)
    // - Castling rights remain unchanged for this specific move.

    hyperion::core::zobrist_key_t move_delta_key = 0;
    // Piece movement
    move_delta_key ^= hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_PAWN][hyperion::core::WHITE][hyperion::core::E2]; // remove from e2
    move_delta_key ^= hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_PAWN][hyperion::core::WHITE][hyperion::core::E4]; // add to e4
    // Side to move
    move_delta_key ^= hyperion::core::Zobrist::black_to_move_key; // toggle to black
    // En passant (e4 is square 28, file is 28 % 8 = 4, which is 'e')
    int ep_file_idx = hyperion::core::E4 % 8;
    move_delta_key ^= hyperion::core::Zobrist::en_passant_file_keys[ep_file_idx]; // add EP on file 'e'

    current_hash ^= move_delta_key;
    hyperion::core::zobrist_key_t hash_after_move = current_hash;
    check(hash_after_move != hash_before_move, "Hash should change after a move.");

    // 3. Simulate unmaking the move: White pawn e4-e2
    // This involves reversing the changes:
    // - Piece P_PAWN WHITE moves from E4 to E2
    // - Side to move becomes WHITE
    // - En passant square E3 is removed

    // The delta key for unmaking should be identical to move_delta_key due to XOR properties
    hyperion::core::zobrist_key_t unmake_delta_key = 0;
    // Piece movement
    unmake_delta_key ^= hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_PAWN][hyperion::core::WHITE][hyperion::core::E4]; // remove from e4
    unmake_delta_key ^= hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_PAWN][hyperion::core::WHITE][hyperion::core::E2]; // add to e2
    // Side to move
    unmake_delta_key ^= hyperion::core::Zobrist::black_to_move_key; // toggle to white
    // En passant
    unmake_delta_key ^= hyperion::core::Zobrist::en_passant_file_keys[ep_file_idx]; // remove EP on file 'e'

    check(move_delta_key == unmake_delta_key, "Move delta key and Unmake delta key must be identical.");

    current_hash ^= unmake_delta_key; // Apply unmake operations
    
    check(current_hash == hash_before_move, "Hash after unmaking move should be identical to hash before move.");

    // Scenario 2: King move affecting castling rights
    // State: White King on E1, White to move, WK_CASTLE_FLAG is set. Hash includes this.
    current_hash = 0; // Reset hash for new scenario
    current_hash ^= hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_KING][hyperion::core::WHITE][hyperion::core::E1];
    int initial_castling = hyperion::core::WK_CASTLE_FLAG | hyperion::core::WQ_CASTLE_FLAG; // Say white has both
    current_hash ^= hyperion::core::Zobrist::castling_keys[initial_castling];
    // White to move, no black_to_move_key XORed. No EP.

    hash_before_move = current_hash;

    // Move: King E1 to F1 (loses both WK and WQ castling rights if E1 is king's initial square)
    move_delta_key = 0;
    move_delta_key ^= hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_KING][hyperion::core::WHITE][hyperion::core::E1];
    move_delta_key ^= hyperion::core::Zobrist::piece_square_keys[hyperion::core::P_KING][hyperion::core::WHITE][hyperion::core::F1];
    move_delta_key ^= hyperion::core::Zobrist::black_to_move_key; // Switch to black

    int new_castling_after_king_move = 0; // Lost all rights
    move_delta_key ^= hyperion::core::Zobrist::castling_keys[initial_castling];      // XOR out old
    move_delta_key ^= hyperion::core::Zobrist::castling_keys[new_castling_after_king_move]; // XOR in new (no rights)

    current_hash ^= move_delta_key;
    hash_after_move = current_hash;

    // Unmake Move: King F1 to E1
    // For unmake, the delta is the same.
    current_hash ^= move_delta_key; // XORing with the same delta effectively reverses it.

    check(current_hash == hash_before_move, "Hash mismatch after king move and unmove (castling test).");


    std::cout << "test_make_unmake_scenario passed." << std::endl;
}


int main() {
    hyperion::core::Zobrist::initialize_keys(); // <-- this will initialize the zobrist hash BEFORE the tests starts -super duper important !!

    test_initialization();
    test_individual_components();
    test_make_unmake_scenario();

    std::cout << "\nAll Zobrist tests passed successfully!" << std::endl;

    return 0;
}
