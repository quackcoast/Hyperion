// hyperion/src/cpp/core/test_bitboard.cpp
#include "bitboard.hpp"  
#include "constants.hpp" 
#include <iostream>      

/*
---
* BitBoard tests can be compiled with the following command:

    *g++ -std=c++17 -Wall -Wextra -Icore -o test_bitboard.exe src/cpp/core/test_bitboard.cpp src/cpp/core/bitboard.cpp*

* Run it:
    *./test_bitboard.exe*
---
*/

int main() { // testing bitboard_t
    std::cout << "---===--- BitBoard test ---===---" << std::endl;
    hyperion::core::bitboard_t my_board = hyperion::core::EMPTY_BB;

    std::cout << "Initial empty board:" << std::endl;
    hyperion::core::print_bitboard(my_board);

    std::cout << "Setting A1 (using hyperion::core::A1 from constants.hpp) and H8:" << std::endl;
    hyperion::core::set_bit(my_board, hyperion::core::A1);
    hyperion::core::set_bit(my_board, hyperion::core::H8);

    std::cout << "Setting D4 (using hyperion::core::square_e::SQ_D4 enum from bitboard.hpp):" << std::endl;
    hyperion::core::set_bit(my_board, hyperion::core::square_e::SQ_D4);
    hyperion::core::print_bitboard(my_board);

    std::cout << "Is D4 set? " << (hyperion::core::get_bit(my_board, hyperion::core::D4) ? "Yes" : "No") << std::endl; // Uses int constant D4
    std::cout << "Is E5 set? " << (hyperion::core::get_bit(my_board, hyperion::core::square_e::SQ_E5) ? "Yes" : "No") << std::endl; // <--- Use SQ_E5

    std::cout << "Number of set bits: " << hyperion::core::count_set_bits(my_board) << std::endl;

    std::cout << "Clearing D4 (using hyperion::core::D4):" << std::endl;
    hyperion::core::clear_bit(my_board, hyperion::core::D4);
    hyperion::core::print_bitboard(my_board);
    std::cout << "Number of set bits after clearing D4: " << hyperion::core::count_set_bits(my_board) << std::endl;

    std::cout << "\nTesting LSB and pop_lsb:" << std::endl;
    hyperion::core::bitboard_t test_board = hyperion::core::EMPTY_BB; 
    hyperion::core::set_bit(test_board, hyperion::core::C2);
    hyperion::core::set_bit(test_board, hyperion::core::F5);
    hyperion::core::set_bit(test_board, hyperion::core::A8);

    std::cout << "Test board for pop_lsb:" << std::endl;
    hyperion::core::print_bitboard(test_board);

    while (test_board != hyperion::core::EMPTY_BB) {
        int lsb_sq_idx = hyperion::core::pop_lsb(test_board);
        if (lsb_sq_idx == static_cast<int>(hyperion::core::square_e::NO_SQ)) {
             std::cout << "Board was already empty or pop_lsb returned no_sq unexpectedly." << std::endl;
             break;
        }
        std::cout << "Popped LSB: " << lsb_sq_idx
                  << " (" << hyperion::core::square_to_algebraic(lsb_sq_idx) << ")" << std::endl;
        hyperion::core::print_bitboard(test_board);
    }
    if (test_board == hyperion::core::EMPTY_BB) {
        std::cout << "Board is now empty." << std::endl;
    }

    std::cout << "\nTesting pop_lsb on an empty board:" << std::endl;
    hyperion::core::bitboard_t empty_test_board = hyperion::core::EMPTY_BB;
    hyperion::core::print_bitboard(empty_test_board);
    int result_on_empty = hyperion::core::pop_lsb(empty_test_board);
    if (result_on_empty == static_cast<int>(hyperion::core::square_e::NO_SQ)) {
        std::cout << "pop_lsb on empty board correctly returned no_sq (" << result_on_empty << ")." << std::endl;
    } else {
        std::cout << "Error: pop_lsb on empty board returned " << result_on_empty << " instead of no_sq." << std::endl;
    }
    hyperion::core::print_bitboard(empty_test_board);


    std::cout << "\nUsing square_to_bitboard with integer constants:" << std::endl;
    hyperion::core::bitboard_t e4_bb = hyperion::core::square_to_bitboard(hyperion::core::E4);
    hyperion::core::print_bitboard(e4_bb);

    std::cout << "Using square_to_bitboard with enum:" << std::endl;
    hyperion::core::bitboard_t g7_bb = hyperion::core::square_to_bitboard(hyperion::core::square_e::SQ_G7);
    hyperion::core::print_bitboard(g7_bb);

    std::cout << "\nPiece constant example (from constants.hpp):" << std::endl;
    std::cout << "White Pawn char: '" << hyperion::core::W_PAWN << "'" << std::endl;
 
    std::cout << "Empty Square char: '" << hyperion::core::EMPTY_SQUARE_CHAR << "'" << std::endl;
    std::cout << "NUM_SQUARES: " << hyperion::core::NUM_SQUARES << std::endl;
    std::cout << "---===--- ---===--- ---===---" << std::endl;
    return 0;
}