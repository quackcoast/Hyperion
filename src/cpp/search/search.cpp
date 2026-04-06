#include "search.hpp"
#include "eval.hpp"
#include "move_encoder.hpp"
#include "../core/movegen.hpp"
#include "../nn_inference/nn_inference.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <iostream>

namespace hyperion {
    namespace engine {

        constexpr double UCT_C = 1.414;

        Search::Search(NeuralNetwork* network) : nn(network), random_generator(std::random_device{}()) {
        }

        core::Move Search::find_best_move(core::Position& root_pos, int time_limit_ms) {
            root_node = std::make_unique<Node>();

            tt.clear();
            tt.store(root_pos.current_hash, root_node.get());

            auto start_time = std::chrono::steady_clock::now();
            int iterations = 0;

            while (true) {
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count() >= time_limit_ms) {
                    break;
                }

                core::Position search_pos = root_pos;

                // 1. selection
                Node* leaf = select(root_node.get(), search_pos);

                // 2 & 3. expansion and evaluation
                double value = 0.0;
                if (leaf->children.empty()) {
                    value = expand(leaf, search_pos);
                }

                // 4. backpropagation
                backpropagate(leaf, value);

                iterations++;
            }

            auto end_time = std::chrono::steady_clock::now();
            long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            long long nps = (elapsed_ms > 0) ? (iterations * 1000LL / elapsed_ms) : 0;
            
            // this calculates a rough centipawn score from the root node's Q value for Cutechess (-1.0 to 1.0 mapped to cp)
            double root_q = (root_node && root_node->visits > 0) ? (root_node->value / root_node->visits) : 0.0;
            int cp_score = static_cast<int>(root_q * 1000.0);

            std::cout << "info depth " << iterations 
                      << " score cp " << cp_score 
                      << " nodes " << iterations 
                      << " nps " << nps 
                      << " time " << elapsed_ms 
                      << std::endl;
                      
            return get_best_move_from_root();
        }

        Node* Search::select(Node* node, core::Position& pos) {
            while (true) {
                // if a node has no children, it's either terminal or needs to be expanded
                if (node->children.empty()) {
                    return node;
                }

                // find the best child using PUCT
                Node* best_child = nullptr;
                double max_score = -std::numeric_limits<double>::infinity();

                for (const auto& child : node->children) {
                    double score = uct_score(child.get(), node->visits);
                    
                    // add a debug check for NaN
                    if (std::isnan(score)) {
                        std::cerr << "CRASH LOG: uct_score returned NaN! Neural network likely output NaN values." << std::endl;
                        exit(1);
                    }

                    if (score > max_score) {
                        max_score = score;
                        best_child = child.get();
                    }
                }

                if (best_child == nullptr) {
                    std::cerr << "CRASH LOG: best_child is nullptr! All child scores evaluated to NaN or -infinity." << std::endl;
                    exit(1);
                }

                // descend the tree
                pos.make_move(best_child->move);
                node = best_child;
            }
        }

        double Search::expand(Node* node, core::Position& pos) {
            core::MoveGenerator move_gen;
            std::vector<core::Move> legal_moves;
            move_gen.generate_legal_moves(pos, legal_moves);

            if (legal_moves.empty()) {
                return pos.is_in_check() ? -1.0 : 0.0;
            }
            
            InferenceResult nn_result = nn->infer(pos);

            
            const std::vector<float>& raw_policy = nn_result.policy;
            double nn_value = nn_result.value;
           

            std::vector<double> legal_probabilities;
            double sum_probabilities = 0.0;

            for (const auto& move : legal_moves) {
                try {
                    int policy_idx = get_policy_index(move, pos);

                    if (policy_idx < 0 || policy_idx >= raw_policy.size()) {
                        throw std::out_of_range("policy_idx out of bounds! Index: " + std::to_string(policy_idx) + ", size: " + std::to_string(raw_policy.size()));
                    }

                    double prob = raw_policy[policy_idx];
                    legal_probabilities.push_back(prob);
                    sum_probabilities += prob;
                } catch (const std::exception& e) {
                    std::cerr << "CRASH LOG: Failed processing move " << move_to_uci_string(move) << "\nReason: " << e.what() << std::endl;
                    exit(1);
                }
            }

            if (sum_probabilities <= 1e-8) {
                for (size_t i = 0; i < legal_moves.size(); ++i) {
                    legal_probabilities[i] = 1.0 / legal_moves.size();
                }
            }
            else {
                for (size_t i = 0; i < legal_moves.size(); ++i) {
                    legal_probabilities[i] /= sum_probabilities;
                }
            }

            for (size_t i = 0; i < legal_moves.size(); ++i) {
                auto new_child = std::make_unique<Node>(node, legal_moves[i]);
                new_child->prior_probability = legal_probabilities[i];
                node->children.push_back(std::move(new_child));
            }

            return nn_value;
        }

        void Search::backpropagate(Node* node, double result) {
            while (node != nullptr) {
                node->visits++;
                result = -result;
                node->value += result;
                node = node->parent;
            }
        }

        double Search::uct_score(const Node* node, int parent_visits) const {
            double q_value = (node->visits == 0) ? 0.0 : (node->value / node->visits);
            double u_value = UCT_C * node->prior_probability * (std::sqrt(parent_visits) / (1.0 + node->visits));

            return q_value + u_value;
        }

        core::Move Search::get_best_move_from_root() {
            int max_visits = -1;
            core::Move best_move;

            if (!root_node) {
                return best_move;
            }

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