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
#define FMT_HEADER_ONLY 1
#include "fmt/compile.h"

namespace {
    void double_to_chars(double x, int precision, char* buffer) {
        fmt::format_to(buffer, FMT_COMPILE("{:.{}e}"), x, precision);
    }

    auto dummy = []() -> register_function_for_to_chars_fixed_precision_benchmark {
        return {"fmt", double_to_chars};
    }();
}