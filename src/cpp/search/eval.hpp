#ifndef HYPERION_ENGINE_EVAL_HPP
#define HYPERION_ENGINE_EVAL_HPP
#include <random>

#include "../core/position.hpp"

namespace hyperion {
namespace engine {

double random_playout(core::Position position, std::mt19937& gen);

} // namespace engine
} // namespace hyperion

#endif // HYPERION_ENGINE_EVAL_HPP