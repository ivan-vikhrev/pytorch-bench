#include "logger.hpp"
#include "statistics.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using HighresClock = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

static inline uint64_t sec_to_ns(uint32_t duration) {
    return duration * 1000000000LL;
}

static inline double ns_to_ms(std::chrono::nanoseconds duration) {
    return static_cast<double>(duration.count()) * 0.000001;
}

std::string format_double(const double number) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << number;
    return ss.str();
};

template <class T>
std::vector<T> read_data(const std::string &file_path, const std::vector<int> &shape) {
    auto tensor_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    auto input_size = tensor_size * sizeof(T);
    std::vector<T> tensor_data(tensor_size);
    char *data = reinterpret_cast<char *>(tensor_data.data());
    for (int b = 0; b < shape[0]; ++b) {
        std::ifstream binary_file(file_path, std::ios_base::binary | std::ios_base::ate);
        if (!binary_file) {
            throw std::runtime_error("Can't open " + file_path);
        }

        auto file_size = static_cast<std::size_t>(binary_file.tellg());
        if (file_size != input_size) {
            throw std::invalid_argument("File " + file_path + " contains " + std::to_string(file_size) +
                                        " bytes but the mdoel expects " + std::to_string(input_size));
        }

        binary_file.seekg(0, std::ios_base::beg);
        if (!binary_file.good()) {
            throw std::runtime_error("Can't read " + file_path);
        }

        binary_file.read(&data[b * input_size], input_size);
    }

    return tensor_data;
}

int main(int argc, const char *argv[]) {
    try {
        logger::info << "PyTorch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "."
                     << TORCH_VERSION_PATCH << logger::endl;
        logger::info << "Loading model " << argv[1] << logger::endl;
        torch::jit::script::Module module;
        module = torch::jit::load(argv[1]);
        module.eval();

        logger::info << "Set no grad guard" << logger::endl;
        torch::NoGradGuard no_grad;

        logger::info << "Prepare input tensors" << logger::endl;
        std::vector<std::vector<torch::jit::IValue>> inputs;
        std::vector<std::vector<int64_t>> data_buffer;
        {
            auto input_ids = read_data<int64_t>("/home/ivikhrev/dev/data/bert_base_ner/input_ids_1_128.bin", {1, 128});
            auto attention_mask =
                read_data<int64_t>("/home/ivikhrev/dev/data/bert_base_ner/attention_mask_1_128.bin", {1, 128});
            auto token_type_ids =
                read_data<int64_t>("/home/ivikhrev/dev/data/bert_base_ner/token_type_ids_1_128.bin", {1, 128});

            // auto input_ids_t = torch::from_blob(input_ids.data(),
            //                                     {1, 128},
            //                                     torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
            auto input_ids_t = torch::from_file("/home/ivikhrev/dev/data/bert_base_ner/input_ids_1_128.bin",
                                                false,
                                                128,
                                                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
            auto attention_mask_t = torch::from_blob(attention_mask.data(),
                                                     {1, 128},
                                                     torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
            auto token_type_ids_t = torch::from_blob(token_type_ids.data(),
                                                     {1, 128},
                                                     torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
            data_buffer.push_back(std::move(attention_mask));
            data_buffer.push_back(std::move(token_type_ids));
            inputs.push_back({input_ids_t.view({1,128}), attention_mask_t, token_type_ids_t});
        }
        std::vector<double> latencies;
        // set time limit
        uint32_t time_limit_sec = 5;
        uint64_t time_limit_ns = sec_to_ns(time_limit_sec);

        logger::info << "Measuring model performance" << logger::endl;

        auto infer_start_time = HighresClock::now();
        module.forward(inputs[0]);
        auto first_inference_time = ns_to_ms(HighresClock::now() - infer_start_time);
        logger::info << "Warming up inference took " << format_double(first_inference_time) << " ms" << logger::endl;

        int64_t iteration = 0;
        auto start_time = HighresClock::now();
        auto uptime = std::chrono::duration_cast<ns>(HighresClock::now() - start_time).count();
        while (static_cast<uint64_t>(uptime) < time_limit_ns) {
            infer_start_time = HighresClock::now();
            module.forward(inputs[iteration % inputs.size()]);
            latencies.push_back(ns_to_ms(HighresClock::now() - infer_start_time));
            uptime = std::chrono::duration_cast<ns>(HighresClock::now() - start_time).count();
            ++iteration;
        }
        auto output = module.forward(inputs[0]).toTuple()->elements()[0].toTensor();
        logger::info << output << logger::endl;
        Metrics metrics(latencies, 1);

        // Performance metrics report
        logger::info << "Count: " << iteration << " iterations" << logger::endl;
        logger::info << "Duration: " << format_double(uptime * 0.000001) << " ms" << logger::endl;
        logger::info << "Latency:" << logger::endl;
        logger::info << "\tMedian   " << format_double(metrics.latency.median) << " ms" << logger::endl;
        logger::info << "\tAverage: " << format_double(metrics.latency.avg) << " ms" << logger::endl;
        logger::info << "\tMin:     " << format_double(metrics.latency.min) << " ms" << logger::endl;
        logger::info << "\tMax:     " << format_double(metrics.latency.max) << " ms" << logger::endl;
        logger::info << "Throughput: " << format_double(metrics.fps) << " FPS" << logger::endl;

    } catch (const std::exception &ex) {
        logger::err << ex.what() << logger::endl;
        return -1;
    }
    return 0;
}