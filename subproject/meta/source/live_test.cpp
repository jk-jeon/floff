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

#include "floff/floff.h"

#include <iostream>
#include <iomanip>
#include <string>

template <class Float>
static void live_test() {
    std::string buffer;

    using jkj::floff::main_cache_full;
    using jkj::floff::main_cache_compressed;
    using jkj::floff::extended_cache_long;
    using jkj::floff::extended_cache_super_compact;

    while (true) {
        Float x;
        std::string x_str;
        while (true) {
            std::getline(std::cin, x_str);
            try {
                if constexpr (std::is_same_v<Float, float>) {
                    x = std::stof(x_str);
                }
                else {
                    x = std::stod(x_str);
                }
            }
            catch (...) {
                std::cout << "Not a valid input; input again.\n";
                continue;
            }
            break;
        }

        auto xx = jkj::floff::float_bits<Float>{x};
        std::cout << "              sign: " << (xx.is_negative() ? "-" : "+") << std::endl;
        std::cout << "     exponent bits: "
                  << "0x" << std::hex << std::setfill('0') << xx.extract_exponent_bits() << std::dec
                  << " (value: " << xx.binary_exponent() << ")\n";
        std::cout << "  significand bits: "
                  << "0x" << std::hex << std::setfill('0');
        if constexpr (std::is_same_v<Float, float>) {
            std::cout << std::setw(8);
        }
        else {
            std::cout << std::setw(16);
        }
        std::cout << xx.extract_significand_bits() << " (value: 0x" << xx.binary_significand()
                  << ")\n"
                  << std::dec;

        std::size_t number_of_digits = 0;
        std::cout << "Input number of digits to print:\n";
        std::cin >> number_of_digits;

        std::cout << "std::cout output: " << std::scientific
                  << std::setprecision(number_of_digits - 1) << x << "\n\n";

        if (number_of_digits + 7 > buffer.size()) {
            buffer.resize(number_of_digits + 7);
        }
        buffer.resize(jkj::floff::floff<main_cache_full, extended_cache_super_compact>(x, number_of_digits - 1,
                                                                         buffer.data()) -
                      buffer.data() + 1);
        buffer.back() = '\0';
        std::cout << "    floff output: " << buffer << "\n";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

int main() {
    std::cout << "[Start live test]\n";
    live_test<double>();
}
