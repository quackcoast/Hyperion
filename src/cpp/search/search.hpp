#ifndef HYPERION_ENGINE_SEARCH_HPP
#define HYPERION_ENGINE_SEARCH_HPP

#include "../core/position.hpp"
#include "../core/move.hpp"
#include "tt.hpp"

#include <vector>
#include <memory>
#include <random>

namespace hyperion {
namespace engine {

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
