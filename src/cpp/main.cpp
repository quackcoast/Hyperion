#include "core/zobrist.hpp"
#include "core/bitboard.hpp"
#include "core/position.hpp"
#include "core/movegen.hpp"
#include "search/search.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <sstream>

// helper function to convert our Move object to a UCI-compliant string
std::string move_to_uci_string(const hyperion::core::Move& move) {
    using namespace hyperion::core;

    std::string uci_move = square_to_algebraic(static_cast<int>(move.from_sq)) +
                           square_to_algebraic(static_cast<int>(move.to_sq));

    if (move.is_promotion()) {
        switch (move.get_promotion_piece()) {
            case P_QUEEN:  uci_move += 'q'; break;
            case P_ROOK:   uci_move += 'r'; break;
            case P_BISHOP: uci_move += 'b'; break;
            case P_KNIGHT: uci_move += 'n'; break;
            default: break;
        }
    }
    return uci_move;
}

// Main UCI loop
void uci_loop() {
    using namespace hyperion;

    core::Position pos; 
    engine::Search search_handler;

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "uci") {
            std::cout << "id name Hyperion 0.1.0-beta" << std::endl;
            std::cout << "id author Tom and LJ" << std::endl;
            std::cout << "uciok" << std::endl;
        } 
        else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        } 
        else if (token == "position") {
            std::string fen_string;
            iss >> token;

            if (token == "startpos") {
                fen_string = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
                pos.set_from_fen(fen_string);
                iss >> token; 
            } 
            else if (token == "fen") {
                fen_string = "";
                while (iss >> token && token != "moves") {
                    fen_string += token + " ";
                }
                pos.set_from_fen(fen_string);
            }
            
            if (token == "moves") {
                core::MoveGenerator move_gen;
                std::vector<core::Move> legal_moves;
                
                while (iss >> token) {
                    legal_moves.clear();
                    move_gen.generate_legal_moves(pos, legal_moves);
                    bool move_found = false;
                    for (const auto& legal_move : legal_moves) {
                        if (move_to_uci_string(legal_move) == token) {
                            pos.make_move(legal_move);
                            move_found = true;
                            break;
                        }
                    }
                    if (!move_found) {
                        std::cerr << "info string Error: GUI sent illegal move " << token << " for FEN " << pos.to_fen() << std::endl;
                    }
                }
            }
        } 
        else if (token == "go") {
            // --- MODIFIED SECTION FOR TIME MANAGEMENT ---

            // Default values for UCI time control parameters
            long wtime = -1, btime = -1, movetime = -1;
            
            // Parse the 'go' command line for time controls
            std::string go_token;
            while (iss >> go_token) {
                if (go_token == "wtime") iss >> wtime;
                else if (go_token == "btime") iss >> btime;
                else if (go_token == "movetime") iss >> movetime;
                // Note: winc, binc, movestogo could also be parsed here for more advanced time management
            }

            int time_to_allocate_ms = 10000; // Default time (10s) if no time controls are sent

            if (movetime != -1) {
                // Use the exact time specified by 'movetime'
                time_to_allocate_ms = movetime;
            } else if (wtime != -1 && btime != -1) {
                // Determine time left for the current player
                long time_left_ms = (pos.get_side_to_move() == core::WHITE) ? wtime : btime;
                
                // Allocate 1/50th of the remaining time for the current move
                time_to_allocate_ms = time_left_ms / 50;
                
                // Safety check: if time is very low, don't use more than half of it
                if (time_to_allocate_ms >= time_left_ms) {
                    time_to_allocate_ms = time_left_ms / 2;
                }
            }

            std::cout << "info string search started with a time limit of " << time_to_allocate_ms << "ms" << std::endl;
            core::Move best_move = search_handler.find_best_move(pos, time_to_allocate_ms);
            std::cout << "bestmove " << move_to_uci_string(best_move) << std::endl;
        } 
        else if (token == "quit") {
            break;
        }
        
        std::cout << std::flush;
    }
}

int main() {
    hyperion::core::Zobrist::initialize_keys();
    hyperion::core::initialize_attack_tables();
    
    uci_loop();

    return 0;
}