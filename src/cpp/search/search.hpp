#ifndef HYPERION_ENGINE_SEARCH_HPP
#define HYPERION_ENGINE_SEARCH_HPP

#include "../core/position.hpp"
#include "../core/move.hpp"
#include "tt.hpp"

#include <vector>
#include <memory>
#include <random>
#include <atomic>

namespace hyperion {
namespace engine {

// ======================================================================================
// ======================================================================================
// ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
// New node stucture
/*
struct Node {
    Node* parent = nullptr;
    core::Move move; // The move that led to this node
    std::vector<std::unique_ptr<Node>> children;
    std::atomic<int> visits = 0;
    std::atomic<double> value = 0.0;
    std::vector<core::Move> untried_moves;
    bool moves_generated = false; // Flag to check if weve generated moves for this node

    Node() = default;
    Node(Node* p, core::Move m) : parent(p), move(m) {}

    bool is_fully_expanded() const {
        return moves_generated && untried_moves.empty();
    }
};
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABOVE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================

*/
//--
/* struct Node */
//--

struct Node {
    Node* parent = nullptr;
    std::vector<std::unique_ptr<Node>> children;
    core::Move move;
    int visits = 0;
    double value = 0.0;
    Node() = default;
    Node(Node* p, core::Move m) : parent(p), move(m) {}
    bool is_fully_expanded(size_t num_legal_moves) const {
        return children.size() >= num_legal_moves;
    }
};
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
/*
class Search {
public:
    Search();
    core::Move find_best_move(core::Position& root_pos, int time_limit_ms);

private:
    std::unique_ptr<Node> root_node;
    TranspositionTable tt;
    std::mt19937 random_generator;

    Node* select(Node* node, core::Position& pos);
    Node* expand(Node* node, core::Position& pos);
    double simulate(core::Position& pos);
    void backpropagate(Node* node, double result);
    double uct_score(const Node* node, int parent_visits) const;
    core::Move get_best_move_from_root();
    bool is_terminal(core::Position& pos);
};
*/
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABOVE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
class Search {
public:
    Search();

    // The main function to find the best move
    core::Move find_best_move(core::Position& root_pos, int time_limit_ms);
    

private:
    std::unique_ptr<Node> root_node;
    TranspositionTable tt;
    std::mt19937 random_generator;

    // The four core MCTS steps
    Node* select(Node* node, core::Position& pos);
    Node* expand(Node* node, core::Position& pos);
    double simulate(core::Position& pos);
    void backpropagate(Node* node, double result);

    // Helper to calculate the UCT score for a node
    double uct_score(const Node* node, int parent_visits) const;

    // Helper to pick the final move after the search is complete
    core::Move get_best_move_from_root();
};

} // namespace engine
} // namespace hyperion

#endif // HYPERION_ENGINE_SEARCH_HPP
