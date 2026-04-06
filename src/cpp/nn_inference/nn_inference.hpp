#ifndef NN_INFERENCE_HPP
#define NN_INFERENCE_HPP

#include <string>
#include <vector>
#include <memory>
#include <torch/script.h>
#include <torch/torch.h>

namespace hyperion {
    namespace core {
        class Position;
    }
}

namespace hyperion {
namespace engine {

struct InferenceResult {
    float value;
    std::vector<float> policy;
};

class NeuralNetwork {
public:
    explicit NeuralNetwork(const std::string& model_path);
    ~NeuralNetwork();

    InferenceResult infer(const hyperion::core::Position& pos);
    std::vector<InferenceResult> infer_batch(const std::vector<hyperion::core::Position>& pos_list);

private:
    torch::Tensor position_to_tensor(const hyperion::core::Position& pos);
    torch::Tensor positions_to_tensor(const std::vector<hyperion::core::Position>& pos_list);
    std::unique_ptr<torch::jit::Module> module_;
    torch::Device device_;
};

} // namespace engine
} // namespace hyperion

#endif // NN_INFERENCE_HPP