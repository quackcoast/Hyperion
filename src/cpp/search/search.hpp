#ifndef HYPERION_ENGINE_SEARCH_HPP
#define HYPERION_ENGINE_SEARCH_HPP

#include "../core/position.hpp"
#include "../core/move.hpp"
#include "../nn_inference/nn_inference.hpp"
#include "tt.hpp"

#include <vector>
#include <memory>
#include <random>
#include <atomic>

namespace hyperion {
    namespace engine {

        class NeuralNetwork;

        //--
        /* struct Node */
        //--

        struct Node {
            Node* parent = nullptr;
            std::vector<std::unique_ptr<Node>> children;
            core::Move move;

            int visits = 0;
            double value = 0.0;
            double prior_probability = 0.0;

            Node() = default;
            Node(Node* p, core::Move m) : parent(p), move(m) {};
        };

        class Search {
        public:
            // The compiler now knows NeuralNetwork is a class
            Search(NeuralNetwork* network);

            // The main function to find the best move
            core::Move find_best_move(core::Position& root_pos, int time_limit_ms);

        private:
            NeuralNetwork* nn; // No longer triggers syntax errors
            std::unique_ptr<Node> root_node;
            TranspositionTable tt;
            std::mt19937 random_generator;

            // The four core MCTS steps
            Node* select(Node* node, core::Position& pos);
            double expand(Node* node, core::Position& pos);
            void backpropagate(Node* node, double result);

            // Helper to calculate the UCT score for a node
            double uct_score(const Node* node, int parent_visits) const;

            // Helper to pick the final move after the search is complete
            core::Move get_best_move_from_root();
        };

    } // namespace engine
} // namespace hyperion

#endif // HYPERION_ENGINE_SEARCH_HPP