#include "eval.hpp"
#include "../core/movegen.hpp"
#include "search.hpp"
#include <vector>
#include <random>

namespace hyperion {
namespace engine {

// NOTE: THIS WILL BE COMPLETLY REWORKED ONCE WE HAVE A WORKING NN TO EVALUATE THE POS

//--
/* random_playout */
//--
// Simulates a complete game from a given position by making random moves for both sides
// This function is the core of the "simulation" phase in Monte Carlo Tree Searc, the point of this whole thing
// The simulation ends when a terminal state (checkmate, stalemate, or 50-move rule draw) is reached
    //  position: The board state from which the random playout will begin. It is passed by value to avoid modifying the original
    //  gen: A reference to a Mersenne Twister random number generator for selecting moves
    // The result of the game from the perspective of the starting player: 1.0 for a win, -1.0 for a loss, and 0.0 for a draw
double random_playout(core::Position position, std::mt19937& gen) {
    core::MoveGenerator move_gen;
    std::vector<core::Move> move_list;
    // Store the side to move at the beginning of the playout to correctly evaluate the final score
    int initial_player = position.get_side_to_move();

    // The main game loop for the random simulation
    while (true) {
        move_list.clear();
        // Generate all legal moves for the current player
        move_gen.generate_legal_moves(position, move_list);

        // --- Check for Game Over conditions ---
        if (move_list.empty()) {
            // No legal moves available
            if (position.is_in_check()) {
                // If the current player is in check and has no moves, it's checkmate
                // A loss for the current player (-1.0), a win for the other (+1.0)
                return (position.get_side_to_move() == initial_player) ? -1.0 : 1.0;
            } else {
                // If the current player is not in check and has no moves, it's a stalemate
                return 0.0;
            }
        }
        
        // --- Check for draw by the 50-move rule ---
        // The game is a draw if 50 full moves (100 half-moves) occur without a capture or pawn move
        if (position.halfmove_clock >= 100) {
            return 0.0;
        }

        // --- Pick and play a random move ---
        // Create a uniform distribution to select a random move index
        std::uniform_int_distribution<> distrib(0, move_list.size() - 1);
        // Select a random move from the list of legal moves
        const core::Move& random_move = move_list[distrib(gen)];
        // Apply the chosen move to the board to advance the position
        position.make_move(random_move);
    }
    // This line is un reachable as the loop only terminates via a return, but it prevents compiler warnings
    return 0.0;
}

} // namespace engine
} // namespace hyperion