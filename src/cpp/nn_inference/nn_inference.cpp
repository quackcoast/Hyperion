#include "nn_inference.hpp"
#include "core/position.hpp"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <fstream> // For std::ifstream

namespace hyperion {
namespace engine {

NeuralNetwork::NeuralNetwork(const std::string& model_path) :
    device_(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU)
{
    if (torch::cuda::is_available()) {
        std::cout << "INFO: CUDA is available! Using GPU for inference." << std::endl;
    }
    else {
        std::cout << "INFO: CUDA not available. Using CPU for inference." << std::endl;
    }

    std::ifstream f(model_path.c_str(), std::ios::binary);
    if (!f.good()) {
        std::cerr << "FATAL ERROR: Cannot open model file at " << model_path << "\n";
        exit(1);
    }

    try {
        module_ = std::make_unique<torch::jit::Module>(torch::jit::load(f));
        module_->to(device_);
        module_->eval();
        std::cout << "Successfully loaded NN model from: " << model_path << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "FATAL ERROR: Failed to load the model from " << model_path << "\n";
        std::cerr << e.what() << std::endl;
        exit(1);
    }
}

NeuralNetwork::~NeuralNetwork() = default;

// InferenceResult NeuralNetwork::infer(const hyperion::core::Position& pos) {
//     torch::NoGradGuard no_grad;
//     torch::Tensor input_tensor = position_to_tensor(pos);
//     input_tensor = input_tensor.to(device_);
//     std::vector<torch::jit::IValue> inputs;
//     inputs.push_back(input_tensor);
//     auto output_tuple = module_->forward(inputs).toTuple();
//     at::Tensor policy_tensor = output_tuple->elements()[0].toTensor();
//     at::Tensor value_tensor = output_tuple->elements()[1].toTensor();
//     InferenceResult result;
//     result.value = value_tensor.item<float>();
//     result.policy.assign(policy_tensor.data_ptr<float>(), policy_tensor.data_ptr<float>() + policy_tensor.numel());
//     return result;
// }

InferenceResult NeuralNetwork::infer(const hyperion::core::Position& pos) {
    try {
        torch::NoGradGuard no_grad;
        torch::Tensor input_tensor = position_to_tensor(pos);
        input_tensor = input_tensor.to(device_);
        
        // Run the model (using PyTorch C++)
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        torch::jit::IValue output_val = module_->forward(inputs);
        
        if (!output_val.isTuple()) {
            std::cerr << "CRASH LOG: Model output is NOT a tuple! It is of type: " << output_val.tagKind() << std::endl;
            exit(1);
        }

        auto output_tuple = output_val.toTuple();

        if (output_tuple->elements().size() < 2) {
            std::cerr << "CRASH LOG: Model output tuple has fewer than 2 elements! Size: " << output_tuple->elements().size() << std::endl;
            exit(1);
        }

        at::Tensor policy_tensor_gpu = output_tuple->elements()[0].toTensor();
        at::Tensor value_tensor_gpu = output_tuple->elements()[1].toTensor();

        at::Tensor policy_tensor_cpu = policy_tensor_gpu.to(torch::kCPU);
        at::Tensor value_tensor_cpu = value_tensor_gpu.to(torch::kCPU);

        InferenceResult result;

        result.value = value_tensor_cpu.item<float>();

        result.policy.assign(policy_tensor_cpu.data_ptr<float>(), policy_tensor_cpu.data_ptr<float>() + policy_tensor_cpu.numel());

        return result;
    } catch (const c10::Error& e) {
        std::cerr << "CRASH LOG: c10::Error during inference! " << e.what() << std::endl;
        exit(1);
    } catch (const std::exception& e) {
        std::cerr << "CRASH LOG: std::exception during inference! " << e.what() << std::endl;
        exit(1);
    } catch (...) {
        std::cerr << "CRASH LOG: Unknown exception during inference!" << std::endl;
        exit(1);
    }
}


torch::Tensor NeuralNetwork::position_to_tensor(const hyperion::core::Position& pos) {
    const int num_planes = 20;
    const int plane_size = 64;

    torch::Tensor tensor = torch::zeros({1, num_planes, 8, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* features = tensor.data_ptr<float>();

    auto populate_plane = [&](int plane_idx, hyperion::core::bitboard_t bb) {
        while (bb) {
            int sq = hyperion::core::pop_lsb(bb);
            features[plane_idx * plane_size + sq] = 1.0f;
        }
    };

    auto fill_plane = [&](int plane_idx, float value) {
        std::fill_n(&features[plane_idx * plane_size], plane_size, value);
    };

    using namespace hyperion::core;
    populate_plane(0, pos.get_pieces(P_PAWN, WHITE));
    populate_plane(1, pos.get_pieces(P_KNIGHT, WHITE));
    populate_plane(2, pos.get_pieces(P_BISHOP, WHITE));
    populate_plane(3, pos.get_pieces(P_ROOK, WHITE));
    populate_plane(4, pos.get_pieces(P_QUEEN, WHITE));
    populate_plane(5, pos.get_pieces(P_KING, WHITE));
    populate_plane(6, pos.get_pieces(P_PAWN, BLACK));
    populate_plane(7, pos.get_pieces(P_KNIGHT, BLACK));
    populate_plane(8, pos.get_pieces(P_BISHOP, BLACK));
    populate_plane(9, pos.get_pieces(P_ROOK, BLACK));
    populate_plane(10, pos.get_pieces(P_QUEEN, BLACK));
    populate_plane(11, pos.get_pieces(P_KING, BLACK));

    if (pos.get_side_to_move() == WHITE) {
        fill_plane(12, 1.0f);
    }
    if (pos.castling_rights & WK_CASTLE_FLAG) fill_plane(13, 1.0f);
    if (pos.castling_rights & WQ_CASTLE_FLAG) fill_plane(14, 1.0f);
    if (pos.castling_rights & BK_CASTLE_FLAG) fill_plane(15, 1.0f);
    if (pos.castling_rights & BQ_CASTLE_FLAG) fill_plane(16, 1.0f);

    square_e ep_sq = pos.en_passant_square;
    if (ep_sq != square_e::NO_SQ) {
        features[17 * plane_size + static_cast<int>(ep_sq)] = 1.0f;
    }

    float fifty_move_val = std::min(1.0f, static_cast<float>(pos.halfmove_clock) / 100.0f);
    fill_plane(18, fifty_move_val);
    float fullmove_val = std::min(1.0f, static_cast<float>(pos.fullmove_number) / 200.0f);
    fill_plane(19, fullmove_val);

    return tensor;
}

} // namespace engine
} // namespace hyperion