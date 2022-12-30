// Copyright 2020-2022 Junekey Jeon
//
// The contents of this file may be used under the terms of
// the Apache License v2.0 with LLVM Exceptions.
//
//    (See accompanying file LICENSE-Apache or copy at
//     https://llvm.org/foundation/relicensing/LICENSE.txt)
//
// Alternatively, the contents of this file may be used under the terms of
// the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE-Boost or copy at
//     https://www.boost.org/LICENSE_1_0.txt)
//
// Unless required by applicable law or agreed to in writing, this software
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.

#include "benchmark.h"
#include "random_float.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>
#include <utility>
#include <unordered_map>
#include <vector>

template <class Float>
class benchmark_holder {
public:
    static benchmark_holder& get_instance() {
        static benchmark_holder<Float> inst;
        return inst;
    }

    // Generate random samples
    void prepare_samples(std::size_t number_of_samples) {
        samples_.resize(number_of_samples);
        for (auto& sample : samples_) {
            sample = uniformly_randomly_generate_finite_float<Float>(rg_);
        }
    }

    // { "name" : [measured_time_for_each_precision] }
    using output_type = std::unordered_map<std::string, std::vector<double>>;
    void run(double duration_per_each_precision_in_sec, std::string_view float_name,
             int max_digits, output_type& out) {
        assert(max_digits >= 1);
        auto buffer = std::make_unique<char[]>(10000);
        auto dur = std::chrono::duration<double>{duration_per_each_precision_in_sec};

        for (int digits = 1; digits <= max_digits; ++digits) {
            std::cout << "Benchmark for digits = " << digits << " with uniformly random "
                      << float_name << "'s...\n";

            for (auto const& name_func_pair : name_func_pairs_) {
                auto& measured_times = out[name_func_pair.first];
                if (measured_times.empty()) {
                    measured_times.resize(max_digits);
                }

                std::size_t iterations = 0;
                std::size_t sample_idx = 0;
                auto from = std::chrono::steady_clock::now();
                auto to = from + dur;
                auto now = std::chrono::steady_clock::now();

                while (now <= to) {
                    name_func_pair.second(samples_[sample_idx], digits - 1, buffer.get());

                    if (++sample_idx == samples_.size()) {
                        sample_idx = 0;
                    }
                    ++iterations;
                    now = std::chrono::steady_clock::now();
                }

                measured_times[digits - 1] =
                    double(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(now - from).count()) /
                    iterations;
            }
        }
    }

    output_type run(double duration_per_each_precision_in_sec, std::string_view float_name,
                    int max_precision) {
        output_type out;
        run(duration_per_each_precision_in_sec, float_name, max_precision, out);
        return out;
    }

    void register_function(std::string_view name, void (*func)(Float, int, char*)) {
        name_func_pairs_.emplace(name, func);
    }

private:
    benchmark_holder() : rg_(generate_correctly_seeded_mt19937_64()) {}

    std::vector<Float> samples_;
    std::mt19937_64 rg_;
    std::unordered_map<std::string, void (*)(Float, int, char*)> name_func_pairs_;
};

register_function_for_to_chars_fixed_precision_benchmark::
    register_function_for_to_chars_fixed_precision_benchmark(std::string_view name,
                                                             void (*func_float)(float, int, char*)) {
    benchmark_holder<float>::get_instance().register_function(name, func_float);
};

register_function_for_to_chars_fixed_precision_benchmark::
    register_function_for_to_chars_fixed_precision_benchmark(std::string_view name,
                                                             void (*func_double)(double, int, char*)) {
    benchmark_holder<double>::get_instance().register_function(name, func_double);
};

register_function_for_to_chars_fixed_precision_benchmark::
    register_function_for_to_chars_fixed_precision_benchmark(std::string_view name,
                                                             void (*func_float)(float, int, char*),
                                                             void (*func_double)(double, int, char*)) {
    benchmark_holder<float>::get_instance().register_function(name, func_float);
    benchmark_holder<double>::get_instance().register_function(name, func_double);
};


#define RUN_MATLAB
#ifdef RUN_MATLAB
    #include <cstdlib>

void run_matlab() {
    std::system("matlab -nosplash -r \"cd('matlab');"
                "plot_fixed_precision_benchmark(\'../results/"
                "to_chars_fixed_precision_benchmark_binary64.csv\');\"");
}
#endif

template <class Float>
static void benchmark_test(std::string_view float_name, std::size_t number_of_samples,
                           double duration_per_each_precision_in_sec, int max_digits) {
    auto& inst = benchmark_holder<Float>::get_instance();
    std::cout << "Generating random samples...\n";
    inst.prepare_samples(number_of_samples);
    auto out = inst.run(duration_per_each_precision_in_sec, float_name, max_digits);

    std::cout << "Benchmarking done.\n"
              << "Now writing to files...\n";

    // Write benchmark results
    auto filename = std::string("results/to_chars_fixed_precision_benchmark_");
    filename += float_name;
    filename += ".csv";
    std::ofstream out_file{filename};
    out_file << "number_of_samples," << number_of_samples << std::endl;
    ;
    out_file << "name,digits,time\n";

    for (auto const& name_result_pair : out) {
        for (int digits = 1; digits <= max_digits; ++digits) {
            out_file << "\"" << name_result_pair.first << "\"," << digits<< ","
                     << name_result_pair.second[digits - 1] << "\n";
        }
    }
}

int main() {
    constexpr std::size_t number_of_benchmark_samples_double = 1000000;
    constexpr double duration_per_each_precision_in_sec_double = 0.2;
    constexpr int max_digits_double = 770; // max_nonzero_decimal_digits = 767


    std::cout << "[Running fixed-precision formatting benchmark for binary64...]\n";
    benchmark_test<double>("binary64", number_of_benchmark_samples_double,
                           duration_per_each_precision_in_sec_double, max_digits_double);
    std::cout << "Done.\n\n\n";


#ifdef RUN_MATLAB
    run_matlab();
#endif
}