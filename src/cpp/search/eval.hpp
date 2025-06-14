#ifndef HYPERION_ENGINE_EVAL_HPP
#define HYPERION_ENGINE_EVAL_HPP
#include <random>

#include "../core/position.hpp"

namespace hyperion {
namespace engine {

// A simple static evaluation function
double static_evaluate(const core::Position& pos);

// Simulates a limited number of pseudo-legal random moves, then evaluates.
double limited_depth_playout(core::Position position, std::mt19937& gen);

} // namespace engine
} // namespace hyperion

#endif // HYPERION_ENGINE_EVAL_HPP