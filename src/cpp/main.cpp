#include "core/zobrist.hpp" 
#include "core/bitboard.hpp" 
#include <iostream>          

int main() {

    std::cout << "Initializing Zobrist keys..." << std::endl;
    hyperion::core::Zobrist::initialize_keys();
    std::cout << "Initializing attack tables..." << std::endl;
    hyperion::core::initialize_attack_tables();
    std::cout << "Initialization complete." << std::endl;

    std::cout << "Hyperion Chess Engine" << std::endl;

    std::cout << "Hyperion Engine finished." << std::endl;

    return 0;
}

// look into these: gprof, Valgrind/Callgrind, Visual Studio Profiler, XCode Instruments
// They will show where the bottle necks are in the code. very important to look at AFTER the code works and compiles