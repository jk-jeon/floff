// Copyright 2022 Junekey Jeon
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
#include "floff/floff.h"

namespace {
    using jkj::floff::main_cache_full;
    using jkj::floff::main_cache_compressed;
    using jkj::floff::extended_cache_long;
    using jkj::floff::extended_cache_super_compact;

    void floff_full_long(double x, int precision, char* buffer) noexcept {
        *(jkj::floff::floff<main_cache_full, extended_cache_long>(x, precision, buffer)) = '\0';
    }
    void floff_compressed_super_compact(double x, int precision, char* buffer) noexcept {
        *(jkj::floff::floff<main_cache_compressed, extended_cache_super_compact>(x, precision, buffer)) = '\0';
    }

    auto dummy1 = []() -> register_function_for_to_chars_fixed_precision_benchmark {
        return {"floff_full_long", floff_full_long};
    }();
    auto dummy2 = []() -> register_function_for_to_chars_fixed_precision_benchmark {
        return {"floff_compressed_super_compact", floff_compressed_super_compact};
    }();
}