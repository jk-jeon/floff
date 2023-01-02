// Copyright 2023 Paul Dreik (edited by Junekey Jeon)
//
// The contents of this file may be used under the terms of
// the Apache License v2.0 with LLVM Exceptions.
//
//    (See accompanying file LICENSE-Apache or copy at
//     https://urldefense.com/v3/__https://llvm.org/foundation/relicensing/LICENSE.txt__;!!Mih3wA!GIM8rmjUd_OXeHEHfmQYSyzKgavvIRu0Y6gtrVXQKj5kXET8-qFFDUSb5kVw5HJC7lswQMRxktBow6WqbBU$
//     )
//
// Alternatively, the contents of this file may be used under the terms of
// the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE-Boost or copy at
//     https://urldefense.com/v3/__https://www.boost.org/LICENSE_1_0.txt__;!!Mih3wA!GIM8rmjUd_OXeHEHfmQYSyzKgavvIRu0Y6gtrVXQKj5kXET8-qFFDUSb5kVw5HJC7lswQMRxktBoYTKDsMI$
//     )
//
// Unless required by applicable law or agreed to in writing, this software
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.

#include <cstdint>
#include <cstdio>
#include <future>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>

#include "floff/floff.h"

namespace {
    template <std::size_t N>
    std::string_view with_floff(float x, int precision, char (&buffer)[N]) {
        auto* end = jkj::floff::floff<jkj::floff::main_cache_full, jkj::floff::extended_cache_long>(
            x, precision, buffer);
        *end = '\0';
        return {buffer, std::size_t(end - buffer)};
    }

    template <std::size_t N>
    std::string_view with_cstdio(float x, int precision, char (&buffer)[N]) {
        auto const ret = std::snprintf(buffer, sizeof(buffer), "%.*e", precision, x);
        if (ret > 0 && static_cast<std::size_t>(ret) < sizeof(buffer)) {
            return {buffer, static_cast<size_t>(ret)};
        }
        throw std::runtime_error("insufficient buffer length");
    }
}

int main(int /*argc*/, char* /*argv*/[]) {

    int precision = 10;
    std::mutex mtx;

    // Find the optimal number of threads
    auto const hw_concurrency = std::max(1u, std::thread::hardware_concurrency());

    // Divide the range of inputs
    std::vector<std::uint64_t> ranges(hw_concurrency + 1);
    ranges[0] = 0;

    auto q = (std::uint64_t(1) << 32) / hw_concurrency;
    auto r = (std::uint64_t(1) << 32) % hw_concurrency;

    for (std::size_t i = 1; i <= hw_concurrency; ++i) {
        ranges[i] = std::uint64_t(q * i);
    }
    for (std::size_t i = 0; i < r; ++i) {
        ranges[i + 1] += i + 1;
    }
    assert(ranges[hw_concurrency] == (std::uint64_t(1) << 32));

    // Spawn threads
    using ieee754_bits = jkj::floff::float_bits<float>;
    struct failure_case_t {
        ieee754_bits input;
        std::string reference_result;
        std::string floff_result;
    };
    std::vector<std::future<std::vector<failure_case_t>>> tasks(hw_concurrency);
    for (std::size_t i = 0; i < hw_concurrency; ++i) {
        tasks[i] =
            std::async([&mtx, precision, thread_id = i, from = ranges[i], to = ranges[i + 1]]() {
                std::vector<failure_case_t> failure_cases;
                char buffer1[1024];
                char buffer2[1024];

                for (std::uint64_t u = from; u < to; ++u) {
                    // Exclude non-finite inputs.
                    ieee754_bits x{std::uint32_t(u)};
                    if (x.is_finite()) {
                        auto const reference_result = with_cstdio(x.to_float(), precision, buffer1);
                        auto const floff_result = with_floff(x.to_float(), precision, buffer2);

                        if (reference_result != floff_result) {
                            failure_cases.push_back(
                                {x, std::string(reference_result), std::string(floff_result)});
                        }
                    }

                    if ((u & 0xffffff) == 0xffffff) {
                        std::lock_guard lg{mtx};
                        std::cerr << "Thread " << thread_id << ": " << (u - from + 1) << " / "
                                  << (to - from) << " done.\n";
                    }
                }

                if (((to - 1) & 0xffffff) != 0xffffff) {
                    std::lock_guard lg{mtx};
                    std::cerr << "Thread " << thread_id << ": " << (to - from) << " / "
                              << (to - from) << " done.\n";
                }

                return failure_cases;
            });
    }

    // Get merged list of failure cases.
    std::vector<failure_case_t> failure_cases;
    for (auto& task : tasks) {
        auto failure_cases_subset = task.get();
        failure_cases.insert(failure_cases.end(), failure_cases_subset.begin(),
                             failure_cases_subset.end());
    }

    if (failure_cases.empty()) {
        std::cerr << "No error case was found.\n";
    }
    else {
        for (auto const& failure_case : failure_cases) {
            std::cerr << "Results differ. x=" << failure_case.input.to_float()
                      << " interpreted as an integer:" << failure_case.input.u
                      << " floff=" << failure_case.floff_result
                      << " reference=" << failure_case.reference_result << '\n';
        }

        std::cerr << "Done.\n\n\n";
        return -1;
    }

    std::cerr << "Done.\n\n\n";
}