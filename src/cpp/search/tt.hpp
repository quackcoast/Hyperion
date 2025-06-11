#ifndef HYPERION_ENGINE_TT_HPP
#define HYPERION_ENGINE_TT_HPP

#include <cstdint>
#include <unordered_map>

namespace hyperion {
namespace engine {

struct Node;

class TranspositionTable {
public:

    // Finds a node by its Zobrist hash. Returns nullptr if not found
    Node* find(uint64_t hash);

    // Stores a pointer to a node with its hash as the key
    void store(uint64_t hash, Node* node);

    // Clears the table
    void clear();

    // get current size for debugging
    int size();

private:

    std::unordered_map<uint64_t, Node*> table;
};

} // namespace engine
} // namespace hyperion

#endif // HYPERION_ENGINE_TT_HPP