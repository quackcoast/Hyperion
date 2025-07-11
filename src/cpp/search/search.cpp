#include "search.hpp"
#include "eval.hpp"
#include "../core/movegen.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <iostream>

namespace hyperion { 
namespace engine {

// Ive included both the tracitional MCTS and a hand crafted evalutation MCTS. THe handcrafted evaluation is a side project, but it is effective at seeing
// what a traditional nn-mcts would get. I did include the handcrafted evaluation below. You much just comment / uncomment everything shown below

// UCT exploration constant. Higher values favor exploring lessvisited nodes
constexpr double UCT_C = 1.414; // sqrt(2)
// constexpr double UCT_C = 3.14; // pi
// you kinda just need to try around these values and see what works
// to be honest ive changed this number so much and it is SUPPOSED to do something, but doesnt really do much :/
// possible bug maybe?
// constexpr double UCT_C = .6; // number I pulled out of my ass


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
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= time_limit_ms) {
            break;
        }

        // Create a copy of the position to modify during this iteration's traversal
        core::Position search_pos = root_pos; 
        
        // MCTS consists of four main phases per iteration:
        // 1. Selection: Traverse the tree to find a promising leaf node
        Node* node = select(root_node.get(), search_pos);
        // 2. Expansion: Add a new child to the selected node
        node = expand(node, search_pos);
        // 2. Expansion
        /*
        if (!is_terminal(search_pos)) { // Only expand if not a terminal node
            node = expand(node, search_pos);
        }
        */
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
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
/*
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
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= time_limit_ms) {
            break;
        }

        // Create a copy of the position to modify during this iteration's traversal
        core::Position search_pos = root_pos; 
        
        // MCTS consists of four main phases per iteration:
        // 1. Selection: Traverse the tree to find a promising leaf node
        Node* node = select(root_node.get(), search_pos);
        // 2. Expansion: Add a new child to the selected node
        node = expand(node, search_pos);
        // 2. Expansion
        if (!is_terminal(search_pos)) { // Only expand if not a terminal node
            node = expand(node, search_pos);
        }
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
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABOVE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================

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
    // ======================================================================================
    // ======================================================================================
    // ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
    // ======================================================================================
    // ======================================================================================
}/*
     while (true) {
        // If the node is not fully expanded, we must expand it first. Selection ends.
        if (!node->is_fully_expanded()) {
            return node;
        }

        // If it's a leaf node (no children after being fully expanded), it's terminal.
        if (node->children.empty()) {
            return node;
        }

        // --- Find the best child using UCT ---
        Node* best_child = nullptr;
        double max_score = -std::numeric_limits<double>::infinity();

        for (const auto& child : node->children) {
            double score = uct_score(child.get(), node->visits);
            if (score > max_score) {
                max_score = score;
                best_child = child.get();
            }
        }
        
        // This case should not be reached if node has children.
        if (!best_child) { 
            return node;
        }

        // Descend to the best child
        pos.make_move(best_child->move);
        node = best_child;
    }
}
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABOVE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================

// ======================================================================================
// ======================================================================================
// ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================

*/
//--
// Search::expand
//--
// Performs the expansion phase of MCTS
// If the given node is not terminal, it adds one new child node to it,
// corresponding to the next unexplored move from this position
    //  node: The leaf node to expand
    //  pos: The board position corresponding to the leaf node
    // A pointer to the newly created child node. If the node is terminal, returns the original node
/*
Node* Search::expand(Node* node, core::Position& pos) {
    // Generate moves for this node IF they haven't been generated yet.
    if (!node->moves_generated) {
        core::MoveGenerator move_gen;
        // The expensive legal move generation is now called only ONCE per node.
        move_gen.generate_legal_moves(pos, node->untried_moves);
        node->moves_generated = true;
        // Optional: Shuffle untried_moves to add randomness
        // std::shuffle(node->untried_moves.begin(), node->untried_moves.end(), random_generator);
    }
    
    // If there are still untried moves, expand one.
    if (!node->untried_moves.empty()) {
        core::Move move_to_expand = node->untried_moves.back();
        node->untried_moves.pop_back();

        pos.make_move(move_to_expand);

        // Create the new child node
        node->children.push_back(std::make_unique<Node>(node, move_to_expand));
        Node* new_child = node->children.back().get();

        // Check the transposition table. If we've seen this position before,
        // we should ideally use the existing node. This is a more advanced optimization.
        // For now, we'll just store it.
        tt.store(pos.current_hash, new_child);
        
        return new_child;
    }

    // This can happen if the node is terminal (no legal moves).
    return node;
}
*/
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABOVE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
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
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
//--
/* Search::simulate */
//--
// Performs the simulation (playout) phase of MCTS
// From the given position, it plays a random game to its conclusion
    //  pos: The starting position for the simulation
    // The result of the game from the perspective of the current player (+1 for win, -1 loss, 0 draw)
/*
double Search::simulate(core::Position& pos) {
    core::Position sim_pos = pos;
    return limited_depth_playout(sim_pos, this->random_generator);
}
*/
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABOVE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================

double Search::simulate(core::Position& pos) {
    // Delegate the simulation to a random playout function
    return random_playout(pos, this->random_generator);
}

// ======================================================================================
// ======================================================================================
// ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
//--
/* Search::backpropagate */
//--
// Performs the backpropagation phase of MCTS
// It updates the visit counts and outcome statistics of all nodes from the simulation's start node up to the root
//  result The result of the simulation
/*
void Search::backpropagate(Node* node, double result) {
    while (node != nullptr) {
        // Atomically increment the visit count (this is fine for integers)
        node->visits.fetch_add(1, std::memory_order_relaxed);

        // Use a compare-and-swap (CAS) loop for atomic floating-point addition
        double current_value = node->value.load(std::memory_order_relaxed);
        double new_value;
        do {
            new_value = current_value + result;
        } while (!node->value.compare_exchange_weak(current_value, new_value, 
                                                     std::memory_order_release, 
                                                     std::memory_order_relaxed));
        // `current_value` is automatically updated by compare_exchange_weak on failure,
        // so the loop retries with the latest value

        // Invert the result for the parent node
        result = -result;
        
        // Move up to the parent
        node = node->parent;
    }
}
    */
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABVOE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
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
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
//--
/* Search::uct_score */
//--
/*
double Search::uct_score(const Node* node, int parent_visits) const {
    if (node->visits == 0) {
        return std::numeric_limits<double>::infinity();
    }
    // Exploitation term: Average value of the node from its own perspective
    // The parent wants to choose the move that is best for it, which means
    // choosing the child node with the lowest value (since the child's value
    // represents the opponent's advantage)
    double exploitation_term = -node->value / node->visits;
    
    // Exploration term
    double exploration_term = UCT_C * std::sqrt(std::log(static_cast<double>(parent_visits)) / node->visits);
    
    return exploitation_term + exploration_term;
}
*/
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABOVE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================

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
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
/*
// herlper function to check for terminal nodes in the search
bool Search::is_terminal(core::Position& pos) {
    core::MoveGenerator move_gen;
    std::vector<core::Move> legal_moves;
    move_gen.generate_legal_moves(pos, legal_moves);
    return legal_moves.empty() || pos.halfmove_clock >= 100;
}
*/
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABOVE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
} // namespace engine
} // namespace hyperion