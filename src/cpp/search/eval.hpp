#ifndef HYPERION_ENGINE_EVAL_HPP
#define HYPERION_ENGINE_EVAL_HPP
#include <random>

#include "../core/position.hpp"

namespace hyperion {
namespace engine {

double random_playout(core::Position position, std::mt19937& gen);

// ======================================================================================
// ======================================================================================
// ====================UNCOMENT BELOW FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
/*
// satic evaluation function
double static_evaluate(const core::Position& pos);

// Simulates a limited number of pseudo-legal random moves, then evaluates.
double limited_depth_playout(core::Position position, std::mt19937& gen);
*/
// ======================================================================================
// ======================================================================================
// ====================UNCOMENT ABOVE FOR MCTS WITH STATIC EVALUATION====================
// ======================================================================================
// ======================================================================================
} // namespace engine
} // namespace hyperion

#endif // HYPERION_ENGINE_EVAL_HPP