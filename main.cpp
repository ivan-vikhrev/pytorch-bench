#include "args_handler.hpp"
#include "logger.hpp"
#include "statistics.hpp"

#include <gflags/gflags.h>
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

namespace {
constexpr char help_msg[] = "show the help message and exit";
DEFINE_bool(h, false, help_msg);

constexpr char model_msg[] = "path to an .onnx file with a trained model";
DEFINE_string(m, "", model_msg);

constexpr char input_msg[] =
    "path to an input to process. The input must be an image and/or binaries, a folder of images and/or binaries.\n"
    "                                                     Ex, \"input1:file1 input2:file2 input3:file3\" or just path "
    "to the file or folder if model has one input";
DEFINE_string(i, "", input_msg);

constexpr char shape_msg[] = "shape for network input.\n"
                             "                                                     Ex., "
                             "\"input1[1,128],input2[1,128],input3[1,128]\" or just \"[1,3,224,224]\"";
DEFINE_string(shape, "", shape_msg);

constexpr char data_type_msg[] = "model input data type (options: f32, int32, int64).";
DEFINE_string(dtype, "", data_type_msg);

constexpr char time_msg[] = "time limit for inference in seconds";
DEFINE_uint32(t, 0, time_msg);

void parse(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (FLAGS_h || 1 == argc) {
        std::cout << "onnxruntime_benchmark"
                  << "\nOptions:"
                  << "\n\t[-h]                                         " << help_msg
                  << "\n\t[-help]                                      print help on all arguments"
                  << "\n\t -m <MODEL FILE>                             " << model_msg
                  << "\n\t[-i <INPUT>]                                 " << input_msg
                  << "\n\t[-shape <[N,C,H,W]>]                         " << shape_msg
                  << "\n\t[-dtype <dtype>]                             " << data_type_msg
                  << "\n\t[-t <NUMBER>]                                " << time_msg;
    }
    if (FLAGS_m.empty()) {
        throw std::invalid_argument{"-m <MODEL FILE> can't be empty"};
    }
}

inline uint64_t sec_to_ns(uint32_t duration) {
    return duration * 1000000000LL;
}

inline double ns_to_ms(std::chrono::nanoseconds duration) {
    return static_cast<double>(duration.count()) * 0.000001;
}

std::string format_double(const double number) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << number;
    return ss.str();
}

std::vector<std::vector<torch::jit::IValue>> get_input_tensors(
    const std::vector<std::pair<std::string, std::vector<std::string>>> &input_files,
    const std::map<std::string, std::vector<int>> &shapes,
    const std::map<std::string, torch::Dtype> &dtypes) {
    int inputs_num = input_files[0].second.size();
    std::vector<std::vector<torch::jit::IValue>> inputs(inputs_num);
    for (int i = 0; i < inputs_num; ++i) {
        logger::info << "Input config " << i << logger::endl;
        for (const auto &[input_name, files_list] : input_files) {
            logger::info << "\t" << input_name << logger::endl;
            logger::info << "\t\t" << files_list[i] << logger::endl;
            auto size =
                std::accumulate(shapes.at(input_name).begin(), shapes.at(input_name).end(), 1, std::multiplies<int>());
            auto tensor = torch::from_file(files_list[i],
                                           false,
                                           size,
                                           torch::TensorOptions().dtype(dtypes.at(input_name)).device(torch::kCPU));
            inputs[i].push_back(tensor.view({1, 128}));
        }
    }
    return inputs;
}
} // namespace

int main(int argc, char *argv[]) {
    try {
        logger::info << "Parsing input arguments" << logger::endl;
        parse(argc, argv);
        std::map<std::string, std::vector<int>> shapes;
        for (const auto &[input_name, shape] : args::parse_parameter_string(FLAGS_shape)) {
            shapes.emplace(input_name, args::string_to_vec<int>(shape, ','));
        }

        std::map<std::string, torch::Dtype> dtypes;
        for (const auto &[input_name, type] : args::parse_parameter_string(FLAGS_dtype)) {
            if (type == "fp32") {
                dtypes.emplace(input_name, torch::kFloat32);
            }
            else if (type == "int32") {
                dtypes.emplace(input_name, torch::kInt32);
            }
            else if (type == "int64") {
                dtypes.emplace(input_name, torch::kInt64);
            }
        }

        logger::info << "Reading input files" << logger::endl;
        auto input_files = args::parse_input_files_arguments(gflags::GetArgvs());

        logger::info << "PyTorch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << "."
                     << TORCH_VERSION_PATCH << logger::endl;
        logger::info << "Loading model " << FLAGS_m << logger::endl;

        torch::jit::script::Module module;
        module = torch::jit::load(FLAGS_m);
        module.eval();

        logger::info << "Disabling the gradient computation" << logger::endl;
        torch::NoGradGuard no_grad;

        logger::info << "Prepare input tensors" << logger::endl;
        auto inputs = get_input_tensors(input_files, shapes, dtypes);

        // set time limit
        uint32_t time_limit_sec = FLAGS_t;
        if (time_limit_sec == 0) {
            time_limit_sec = 60;
        }
        uint64_t time_limit_ns = sec_to_ns(time_limit_sec);

        logger::info << "Measuring model performance" << logger::endl;

        auto infer_start_time = HighresClock::now();
        module.forward(inputs[0]);
        auto first_inference_time = ns_to_ms(HighresClock::now() - infer_start_time);
        logger::info << "Warming up inference took " << format_double(first_inference_time) << " ms" << logger::endl;

        std::vector<double> latencies;
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

        // check outputs
        // auto output = module.forward(inputs[0]).toTuple()->elements()[0].toTensor();
        // logger::info << output << logger::endl;
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