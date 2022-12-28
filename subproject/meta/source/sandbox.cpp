#include "floff/floff.h"

#include <iostream>
#include <iomanip>

#define NNN 2

int main() {
    std::cout << jkj::floff::detail::compute_power(10, 0) << "\n"
              << jkj::floff::detail::compute_power(10, 1) << "\n"
              << jkj::floff::detail::compute_power(10, 2) << "\n"
              << jkj::floff::detail::compute_power(10, 3) << "\n"
              << jkj::floff::detail::compute_power(10, 4) << "\n"
              << jkj::floff::detail::compute_power(10, 5) << "\n"
              << jkj::floff::detail::compute_power(10, 6) << "\n";

#if NNN == 0
    auto const x = 1.61231e-276;
    std::cout << std::endl << std::scientific << std::setprecision(699) << x << "\n\n";

    char buffer[1024];
    *(jkj::floff::floff<jkj::floff::extended_cache_long>(x, buffer, 700)) = '\0';
    std::cout << buffer << "\n\n";

#elif NNN == 1
    std::uint64_t left_max = 0, right_min = 0 - std::uint64_t(1);
    constexpr std::uint32_t L = 16;
    for (std::uint32_t n = 1; n <= 9999'9999; ++n) {
        auto left = std::uint64_t(n) << 32;
        if (left % 100'0000 != 0) {
            left /= 100'0000;
            ++left;
        }
        else {
            left /= 100'0000;
        }
        --left;
        left <<= L;
        if (left % n != 0) {
            left /= n;
            ++left;
        }
        else {
            left /= n;
        }

        if (left > left_max) {
            left_max = left;
        }

        auto right = std::uint64_t(n + 1) << 32;
        if (right % 100'0000 != 0) {
            right /= 100'0000;
            ++right;
        }
        else {
            right /= 100'0000;
        }
        --right;
        right <<= L;
        if (right % n == 0) {
            right /= n;
            --right;
        }
        else {
            right /= n;
        }
        if (right < right_min) {
            right_min = right;
        }
    }

    std::cout << left_max << ", " << right_min << std::endl;
#endif
}
