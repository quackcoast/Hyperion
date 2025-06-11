#include "search.hpp"
#include "eval.hpp"
#include "../core/movegen.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <iostream>

namespace hyperion {
namespace engine {

// UCT exploration constant. Higher values favor exploring lessvisited nodes
// constexpr double UCT_C = 1.414; // sqrt(2)
// constexpr double UCT_C = 3.14; // pi
// you kinda just need to try around these values and see what works
// to be honest ive changed this number so much and it is SUPPOSED to do something, but doesnt really do much :/
// possible bug maybe?
constexpr double UCT_C = .6; // number I pulled out of my ass

//--
/* Search::Search */
//--
// Constructs a Search object, initializing the random number generator
// The random generator is used for the simulation (playout) phase of MCTS
Search::Search() : random_generator(std::random_device{}()) {
}

//--
/* Search::find_best_move */
//--
// The main entry point for the Monte Carlo Tree Search (MCTS)
// It iteratively builds a game tree for a specified duration, then selects the best move
    //  root_pos: The starting position of the search
    //  time_limit_ms: The maximum time in milliseconds to run the search
    // The best core::Move found for the root_pos
core::Move Search::find_best_move(core::Position& root_pos, int time_limit_ms) {
    // --- Setup ---
    // Initialize the search tree with a root node
    root_node = std::make_unique<Node>();
    
    // Clear the transposition table from any previous search
    tt.clear();
    // Store the root node in the transposition table
    tt.store(root_pos.current_hash, root_node.get());

    auto start_time = std::chrono::steady_clock::now();
    int iterations = 0;

    // --- Main MCTS Loop ---
    // The loop continues until the time limit is exceeded
    while (true) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed >= time_limit_ms) {
            break;
        }

        // Create a copy of the position to modify during this iteration's traversal
        core::Position search_pos = root_pos; 
        
        // MCTS consists of four main phases per iteration:
        // 1. Selection: Traverse the tree to find a promising leaf node
        Node* node = select(root_node.get(), search_pos);
        // 2. Expansion: Add a new child to the selected node
        node = expand(node, search_pos);
        // 3. Simulation: Run a random playout from the new node
        double result = simulate(search_pos);
        // 4. Backpropagation: Update node statistics back up the tree
        backpropagate(node, result);
        
        iterations++;
    }
    
    // Output search statistics
    std::cout << "info depth " << iterations << " nodes " << tt.size() << std::endl;

    // After the search, determine the best move from the root
    return get_best_move_from_root();
}

//--
/* Search::select */
//--
// Performs the selection phase of MCTS
// It traverses the tree from a given node, choosing the child with the highest UCT score at each step,
// until it reaches a leaf node (a node that is not fully expanded or is terminal)
    //  node: The starting node for the selection process (usually the root)
    //  pos: The board position, which is updated as the selection traverses the tree
    // A pointer to the selected leaf Node
Node* Search::select(Node* node, core::Position& pos) {
    core::MoveGenerator move_gen;
    std::vector<core::Move> legal_moves;

    while (true) {
        move_gen.generate_legal_moves(pos, legal_moves);

        // If the node is terminal (no legal moves) or not yet fully expanded,
        // we have found our leaf node and stop the selection phase
        if (legal_moves.empty() || !node->is_fully_expanded(legal_moves.size())) {
            return node;
        }

        // --- Find the best child using UCT ---
        Node* best_child = nullptr;
        double max_score = -std::numeric_limits<double>::infinity();

        // Iterate through all children to find the one with the highest UCT score
        for (const auto& child : node->children) {
            double score = uct_score(child.get(), node->visits);
            if (score > max_score) {
                max_score = score;
                best_child = child.get();
            }
        }
        
        // This case should not be reached if the node is fully expanded
        if (!best_child) { 
            return node;
        }

        // Descend the tree by making the move of the best child
        pos.make_move(best_child->move);
        node = best_child;

        // Clear the move list for the next iteration
        legal_moves.clear();
    }
}

//--
/* Search::expand */
//--
// Performs the expansion phase of MCTS
// If the given node is not terminal, it adds one new child node to it,
// corresponding to the next unexplored move from this position
    //  node: The leaf node to expand
    //  pos: The board position corresponding to the leaf node
    // A pointer to the newly created child node. If the node is terminal, returns the original node
Node* Search::expand(Node* node, core::Position& pos) {
    core::MoveGenerator move_gen;
    std::vector<core::Move> legal_moves;
    move_gen.generate_legal_moves(pos, legal_moves);

    // If the node is terminal (a checkmate or stalemate), we can't expand it further
    if (legal_moves.empty()) {
        return node;
    }

    // Expand by picking the next unexplored move
    const core::Move& move_to_expand = legal_moves[node->children.size()];
    
    // Apply the move to the board position
    pos.make_move(move_to_expand);

    // Create a new child node representing the new position
    node->children.push_back(std::make_unique<Node>(node, move_to_expand));
    Node* new_child = node->children.back().get();

    // Store the new node in the transposition table for future lookups
    tt.store(pos.current_hash, new_child);
    
    // Return the newly created node for the simulation phase
    return new_child;
}

//--
/* Search::simulate */
//--
// Performs the simulation (playout) phase of MCTS
// From the given position, it plays a random game to its conclusion
    //  pos: The starting position for the simulation
    // The result of the game from the perspective of the current player (+1 for win, -1 loss, 0 draw)
double Search::simulate(core::Position& pos) {
    // Delegate the simulation to a random playout function
    return random_playout(pos, this->random_generator);
}

//--
/* Search::backpropagate */
//--
// Performs the backpropagation phase of MCTS
// It updates the visit counts and outcome statistics of all nodes from the simulation's start node up to the root
    //  node The node from which the simulation was run
    //  result The result of the simulation
void Search::backpropagate(Node* node, double result) {
    // The simulation result is from the perspective of the player who just moved to 'node'
    // We traverse up the tree to the root
    while (node != nullptr) {
        // Increment the visit count for the current node
        node->visits++;
        // The result must be inverted for the parent, as it's from the opponent's perspective
        result = -result; 
        // Update the node's value with the result
        node->value += result;
        // Move up to the parent node
        node = node->parent;
    }
}

//--
/* Search::uct_score */
//--
// Calculates the UCT (Upper Confidence Bound for Trees) score for a given node
// This score balances exploitation (choosing known good moves) and exploration (trying new moves)
    //  node: The child node for which to calculate the score
    //  parent_visits: The number of times the parent of 'node' has been visited
    // The calculated UCT score as a double
double Search::uct_score(const Node* node, int parent_visits) const {
    // If a node has not been visited, prioritize it by giving it an infinite score
    if (node->visits == 0) {
        return std::numeric_limits<double>::infinity();
    }
    // Exploitation term: the average value of the node from the parent's perspective
    double q_value = node->value / node->visits;
    // Exploration term: encourages visiting less-explored nodes
    double u_value = UCT_C * std::sqrt(std::log(parent_visits) / node->visits);
    
    // The final score is the sum of the exploitation and exploration terms
    // The node's value is already stored from the parent's perspective, so no negation is needed here
    return q_value + u_value;
}

//--
/* Search::get_best_move_from_root */
//--
// Determines the best move from the root node after the MCTS search is complete
// The most robust move is the one that was explored the most times
    // The core::Move corresponding to the most visited child of the root node
core::Move Search::get_best_move_from_root() {
    int max_visits = -1;
    core::Move best_move; // Default-constructs a "null" move

    // A sanity check to ensure the root node exists
    if (!root_node) {
        return best_move;
    }

    // The best move is the one corresponding to the most visited child
    for (const auto& child : root_node->children) {
        if (child->visits > max_visits) {
            max_visits = child->visits;
            best_move = child->move;
        }
    }
    return best_move;
}

} // namespace engine
} // namespace hyperion