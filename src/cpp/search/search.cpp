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

            // Initial root expansion
            expand_with_dummy_priors(root_node.get(), root_pos);
            if (!root_node->is_terminal) {
                std::vector<core::Position> init_pos = {root_pos};
                auto init_results = nn->infer_batch(init_pos);
                if (!init_results.empty() && !root_node->children.empty()) {
                    update_priors_from_nn(root_node.get(), init_results[0].policy, root_pos);
                    root_node->value = init_results[0].value;
                    root_node->visits = 1;
                }
            }

            while (true) {
                auto now = std::chrono::steady_clock::now();
                long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                if (elapsed_ms >= time_limit_ms) break;

                // Smart early stop logic
                if (elapsed_ms > time_limit_ms * 0.25 && root_node->children.size() >= 2) {
                    int max_v = -1, second_max_v = -1;
                    for (const auto& child : root_node->children) {
                        if (child->visits > max_v) {
                            second_max_v = max_v;
                            max_v = child->visits;
                        } else if (child->visits > second_max_v) {
                            second_max_v = child->visits;
                        }
                    }
                    if (max_v > 300 && second_max_v > 0 && max_v > second_max_v * 5) {
                        break; 
                    }
                }

                std::vector<Node*> all_selected_leaves;
                std::vector<Node*> batch_leaves;
                std::vector<core::Position> batch_positions;

                for (int b = 0; b < BATCH_SIZE; ++b) {
                    core::Position search_pos = root_pos;
                    Node* leaf = select(root_node.get(), search_pos);

                    if (leaf->is_terminal) {
                        backpropagate(leaf, leaf->terminal_value);
                        continue;
                    }

                    if (leaf->is_evaluating) {
                        all_selected_leaves.push_back(leaf);
                        apply_virtual_loss(leaf);
                        continue;
                    }

                    if (leaf->children.empty()) {
                        expand_with_dummy_priors(leaf, search_pos);
                        
                        if (leaf->is_terminal) {
                            backpropagate(leaf, leaf->terminal_value);
                            continue;
                        }

                        leaf->is_evaluating = true;
                        batch_leaves.push_back(leaf);
                        batch_positions.push_back(search_pos);
                        
                        all_selected_leaves.push_back(leaf);
                        apply_virtual_loss(leaf);
                    }
                }

                if (!batch_leaves.empty()) {
                    auto nn_results = nn->infer_batch(batch_positions);
                    
                    std::map<Node*, double> evaluated_values;
                    for (size_t i = 0; i < batch_leaves.size(); ++i) {
                        Node* leaf = batch_leaves[i];
                        leaf->is_evaluating = false;
                        update_priors_from_nn(leaf, nn_results[i].policy, batch_positions[i]);
                        evaluated_values[leaf] = nn_results[i].value;
                    }

                    for (Node* leaf : all_selected_leaves) {
                        double real_value = evaluated_values[leaf];
                        revert_virtual_loss_and_backpropagate(leaf, real_value);
                    }
                    
                    iterations += all_selected_leaves.size();
                } else {
                    // Handled terminal/purely virtual nodes
                    iterations += BATCH_SIZE;
                }
            }

            auto end_time = std::chrono::steady_clock::now();
            long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            long long nps = (elapsed_ms > 0) ? (iterations * 1000LL / elapsed_ms) : 0;
            
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
                if (node->is_terminal) return node;
                if (node->is_evaluating) return node;
                if (node->children.empty()) return node;

                Node* best_child = nullptr;
                double max_score = -std::numeric_limits<double>::infinity();
                
                int total_parent_visits = node->visits + node->virtual_visits;

                for (const auto& child : node->children) {
                    double score = uct_score(child.get(), total_parent_visits);
                    if (score > max_score) {
                        max_score = score;
                        best_child = child.get();
                    }
                }

                if (best_child == nullptr) {
                    best_child = node->children[0].get();
                }

                pos.make_move(best_child->move);
                node = best_child;
            }
        }

        // Batching helpers
        void Search::expand_with_dummy_priors(Node* node, core::Position& pos) {
            core::MoveGenerator move_gen;
            std::vector<core::Move> legal_moves;
            move_gen.generate_legal_moves(pos, legal_moves);

            if (legal_moves.empty()) {
                node->is_terminal = true;
                node->terminal_value = pos.is_in_check() ? -1.0 : 0.0;
                return;
            }

            double uniform_prob = 1.0 / legal_moves.size();
            for (const auto& move : legal_moves) {
                auto new_child = std::make_unique<Node>(node, move);
                new_child->prior_probability = uniform_prob;
                node->children.push_back(std::move(new_child));
            }
        }

        void Search::update_priors_from_nn(Node* node, const std::vector<float>& policy, core::Position& pos) {
            if (node->is_terminal) return;

            double sum_probabilities = 0.0;
            for (auto& child : node->children) {
                try {
                    int policy_idx = get_policy_index(child->move, pos);
                    if (policy_idx >= 0 && policy_idx < policy.size()) {
                        child->prior_probability = policy[policy_idx];
                        sum_probabilities += child->prior_probability;
                    }
                } catch(...) {}
            }

            if (sum_probabilities <= 1e-8) {
                double uniform = 1.0 / node->children.size();
                for (auto& child : node->children) child->prior_probability = uniform;
            } else {
                for (auto& child : node->children) child->prior_probability /= sum_probabilities;
            }
        }

        void Search::apply_virtual_loss(Node* node) {
            double v_result = 1.0; 
            while (node != nullptr) {
                node->virtual_visits++;
                v_result = -v_result;
                node->virtual_loss_value += v_result;
                node = node->parent;
            }
        }

        void Search::revert_virtual_loss_and_backpropagate(Node* node, double result) {
            Node* temp = node;
            double v_result = 1.0;
            while (temp != nullptr) {
                temp->virtual_visits--;
                v_result = -v_result;
                temp->virtual_loss_value -= v_result;
                temp = temp->parent;
            }

            backpropagate(node, result);
        }

        // Old expand, kept for backwards compatibility if needed elsewhere
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
                } catch (...) { exit(1); }
            }

            if (sum_probabilities <= 1e-8) {
                for (size_t i = 0; i < legal_moves.size(); ++i) {
                    legal_probabilities[i] = 1.0 / legal_moves.size();
                }
            } else {
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
            int total_visits = node->visits + node->virtual_visits;
            double q_value = (total_visits == 0) ? 0.0 : ((node->value + node->virtual_loss_value) / total_visits);
            double u_value = UCT_C * node->prior_probability * (std::sqrt(parent_visits) / (1.0 + total_visits));

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