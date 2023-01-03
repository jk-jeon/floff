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

#include "best_rational_approx.h"
#include "rational_continued_fractions.h"
#include "big_uint.h"
#include "floff/floff.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>

struct pair_hash {
    std::size_t operator()(std::pair<int, int> const& p) const {
        return std::hash<std::uint64_t>{}((std::uint64_t(p.first) << 32) | std::uint64_t(p.second));
    }
};

template <class CacheUnitType>
struct multiplier_info {
    using cache_unit_type = CacheUnitType;

    // The smallest e which might need this multiplier.
    int first_exponent;

    // The difference between the position of the first bit needed by the first exponent and the
    // position of the first bit actually stored in the table.
    int min_shift;

    struct bit_range {
        int first;
        int last;
        std::size_t cache_block_count;
    };

    bit_range range_union;
    std::vector<bit_range> individual_ranges;
    std::vector<int> base_exponents;

    int cache_bits;
    jkj::big_uint multiplier;
};

template <class CacheUnitType>
struct extended_cache_result {
    std::size_t max_cache_blocks = 0;
    std::size_t cache_bits_unit = 0;
    int segment_length = 0;
    int collapse_factor = 0;
    int e_min = 0;
    int e_max = 0;
    int k_min = 0;
    std::vector<multiplier_info<CacheUnitType>> mul_info;
};

#include <iostream>

// collapse_factor == 0 means constant cache block count.
template <class Float, class FloatTraits = jkj::floff::default_float_traits<Float>>
auto generate_extended_cache(int segment_length, int collapse_factor, unsigned int k_min_shift)
    -> extended_cache_result<typename FloatTraits::carrier_uint> {
    assert(segment_length > 0);
    assert(collapse_factor >= 0);

    using float_format = typename FloatTraits::format;
    using rational = jkj::unsigned_rational<jkj::big_uint>;

    int constexpr kappa = std::is_same_v<Float, float> ? 1 : 2;
    constexpr std::size_t cache_bits_unit = FloatTraits::carrier_bits;
    static_assert(jkj::big_uint::element_number_of_bits == cache_bits_unit);

    auto constexpr e_min = float_format::min_exponent - float_format::significand_bits;
    auto constexpr e_max = float_format::max_exponent - float_format::significand_bits;

    auto const k_min = kappa - jkj::floff::detail::log::floor_log10_pow2(e_max) + segment_length -
                       int(k_min_shift);
    // When k = k_min + s * segment_length > segment_length - e_min, the segment we get is always
    // zero, so the largest s is floor((-e_min - k_min + segment_length) / segment_length).
    auto const number_of_multipliers = (-e_min - k_min) / segment_length + 2;

    auto const n_max =
        jkj::big_uint::power_of_2(std::size_t(float_format::significand_bits + 2)) - 1;
    auto segment_divisor = jkj::big_uint::pow(10, std::size_t(segment_length));

    auto compute_pow2_pow5 = [pow2_pow5_cache =
                                  std::unordered_map<std::pair<int, int>, rational, pair_hash>{}](
                                 int exp2, int exp5) mutable -> rational const& {
        auto itr = pow2_pow5_cache.find({exp2, exp5});
        if (itr == pow2_pow5_cache.end()) {
            jkj::big_uint num = 1, den = 1;
            if (exp2 < 0) {
                den *= jkj::big_uint::power_of_2(std::size_t(-exp2));
            }
            else {
                num *= jkj::big_uint::power_of_2(std::size_t(exp2));
            }
            if (exp5 < 0) {
                den *= jkj::big_uint::pow(5, std::size_t(-exp5));
            }
            else {
                num *= jkj::big_uint::pow(5, std::size_t(exp5));
            }
            itr = pow2_pow5_cache
                      .emplace(std::piecewise_construct, std::forward_as_tuple(exp2, exp5),
                               std::forward_as_tuple(std::move(num), std::move(den)))
                      .first;
        }
        return itr->second;
    };
    auto floor = [](rational const& r) -> jkj::big_uint { return r.numerator / r.denominator; };
    auto ceil = [](rational const& r) -> jkj::big_uint {
        auto num = r.numerator;
        auto q = num.long_division(r.denominator);
        if (num.is_zero()) {
            return q;
        }
        else {
            return ++q;
        }
    };

    using cache_unit_type = typename FloatTraits::carrier_uint;
    extended_cache_result<cache_unit_type> result;

    result.max_cache_blocks = 1;
    result.cache_bits_unit = cache_bits_unit;
    result.segment_length = segment_length;
    result.collapse_factor = collapse_factor;
    result.e_min = e_min;
    result.e_max = e_max;
    result.k_min = k_min;

    auto& mul_info = result.mul_info;
    mul_info.resize(number_of_multipliers,
                    {e_max + 1,
                     0,
                     {std::numeric_limits<int>::max(), std::numeric_limits<int>::min()},
                     {},
                     {},
                     0,
                     {}});

    std::size_t max_cache_bits = 0;
    for (int e = e_min; e <= e_max; ++e) {
        std::cout << "Inspecting necessary bits for e = " << e << "...\n";

        // We assume we already know the digits we obtain by multiplying
        // 10^(kappa - floor(e*log10(2))), so the first k is the smallest k that is
        // strictly larger than kappa - floor(e*log10(2)).
        auto const k_min_local =
            kappa - jkj::floff::detail::log::floor_log10_pow2(e) + segment_length;
        auto multiplier_index = (k_min_local - k_min) / segment_length;
        auto k = k_min + multiplier_index * segment_length;

        while (k < std::max(0, -e + 1) + segment_length) {
            auto const& two_x = compute_pow2_pow5(e + k, k);

            std::size_t cache_bits = cache_bits_unit;
            std::size_t binary_search_length = cache_bits_unit;
            // m = ceil(2^Q * x / D) = ceil(2^(Q + e - 1 + k - eta) * 5^(k - eta)).
            jkj::big_uint m;
            int const first_bit_index = e + k - segment_length;
            int last_bit_index;
            auto bigger_than_two_xi = [&](rational const& r) {
                return 2 * m * segment_divisor * r.denominator <
                       jkj::big_uint::power_of_2(cache_bits) * r.numerator;
            };

            bool got_upper_bound = false;
            while (true) {
                last_bit_index = int(cache_bits) + e - 1 + k - segment_length;
                m = ceil(compute_pow2_pow5(last_bit_index, k - segment_length));

                bool enough_precision;
                if (two_x.denominator == 1) {
                    enough_precision = bigger_than_two_xi(two_x + rational{1, n_max});
                }
                else if (two_x.denominator <= n_max) {
                    // Compute the greatest v <= n_max such that vp == -1 (mod q).
                    // To obtain such v, we first find the smallest positive v0 such that
                    // v0 * p == -1 (mod q). Then v = v0 + floor((n_max - v0)/q) * q.
                    auto v = jkj::find_best_rational_approx<
                                 jkj::rational_continued_fractions<jkj::big_uint>>(
                                 two_x, two_x.denominator - 1)
                                 .above.denominator;
                    v += ((n_max - v) / two_x.denominator) * two_x.denominator;

                    // Compare xi against target + 1/vq = (vp + 1)/vq.
                    // Note that vp + 1 is guaranteed to be a multiple of q.
                    auto numerator = (v * two_x.numerator + 1).long_division(two_x.denominator);
                    enough_precision = bigger_than_two_xi({std::move(numerator), std::move(v)});
                }
                else {
                    enough_precision = bigger_than_two_xi(
                        jkj::find_best_rational_approx<
                            jkj::rational_continued_fractions<jkj::big_uint>>(two_x, n_max)
                            .above);
                }

                if (enough_precision) {
                    got_upper_bound = true;
                    binary_search_length >>= 1;
                    if (binary_search_length == 0) {
                        break;
                    }

                    cache_bits -= binary_search_length;
                }
                else {
                    if (got_upper_bound) {
                        binary_search_length >>= 1;
                        if (binary_search_length == 0) {
                            ++cache_bits;
                            break;
                        }
                    }
                    cache_bits += binary_search_length;
                }
            }
            mul_info[multiplier_index].range_union.first =
                std::min(mul_info[multiplier_index].range_union.first, first_bit_index);
            mul_info[multiplier_index].range_union.last =
                std::max(mul_info[multiplier_index].range_union.last, last_bit_index);
            if (mul_info[multiplier_index].individual_ranges.empty()) {
                mul_info[multiplier_index].first_exponent = e;
            }
            mul_info[multiplier_index].individual_ranges.push_back(
                {first_bit_index, last_bit_index});

            max_cache_bits = std::max(max_cache_bits, cache_bits);
            ++multiplier_index;
            k += segment_length;
        }
    }
    result.max_cache_blocks = (max_cache_bits + cache_bits_unit - 1) / cache_bits_unit;

    std::size_t number_of_e_base_k_pairs = 0;
    // For average computation.
    // In practice, we might get rid of some cache blocks
    // (as we can prove those are all-zero), so
    // the real average is a bit smaller than the computed average.
    std::size_t accumulated_number_of_cache_blocks = 0;
    std::size_t number_of_e_k_pairs = 0;

    std::cout << "\nGenerating multipliers...\n\n";
    for (int multiplier_index = 0; multiplier_index < number_of_multipliers; ++multiplier_index) {
        auto& r = mul_info[multiplier_index];
        int const k = k_min + multiplier_index * segment_length;

        // Skip multipliers that do not have actual exponents attached.
        if (r.individual_ranges.empty()) {
            r.cache_bits = 0;
            continue;
        }

        if (collapse_factor != 0) {
            for (int e_base = r.first_exponent;
                 e_base < r.first_exponent + int(r.individual_ranges.size());) {
                auto max_bits = r.individual_ranges[e_base - r.first_exponent].last -
                                r.individual_ranges[e_base - r.first_exponent].first + 1;
                r.base_exponents.push_back(e_base);

                int e = e_base + 1;
                for (; ((e - e_min) % collapse_factor) != 0 &&
                       e < r.first_exponent + int(r.individual_ranges.size());
                     ++e) {
                    max_bits =
                        std::max(max_bits, r.individual_ranges[e - r.first_exponent].last -
                                               r.individual_ranges[e - r.first_exponent].first + 1);
                }

                auto const cache_block_count = (max_bits + cache_bits_unit - 1) / cache_bits_unit;
                accumulated_number_of_cache_blocks += cache_block_count * (e - e_base);
                number_of_e_k_pairs += (e - e_base);
                ++number_of_e_base_k_pairs;
                auto const new_e_base = e;
                for (e = e_base; e < new_e_base; ++e) {
                    r.individual_ranges[e - r.first_exponent].cache_block_count = cache_block_count;
                }
                e_base = new_e_base;
            }
        }
        else {
            for (int e = r.first_exponent; e < r.first_exponent + r.individual_ranges.size(); ++e) {
                r.individual_ranges[e - r.first_exponent].cache_block_count =
                    result.max_cache_blocks;
            }

            number_of_e_k_pairs += r.individual_ranges.size();
            accumulated_number_of_cache_blocks +=
                r.individual_ranges.size() * result.max_cache_blocks;
        }

        // Generate multiplier.
        r.multiplier = floor(compute_pow2_pow5(r.range_union.last, k - segment_length));
        r.multiplier.long_division(
            jkj::big_uint::power_of_2(std::size_t(r.range_union.last - r.range_union.first + 1)));
        r.cache_bits = int(log2p1(r.multiplier));
        assert(r.cache_bits > 0);

        r.min_shift = (r.range_union.last - r.range_union.first + 1) - r.cache_bits;

        for (std::size_t i = 0; i < r.individual_ranges.size(); ++i) {
            // Check if the naive computation of ceiling can succeed for every exponent.
            auto const last_bit_index = r.individual_ranges[i].first +
                                        r.individual_ranges[i].cache_block_count * cache_bits_unit -
                                        1;
            auto excessive_bits_to_right = last_bit_index > r.range_union.last
                                               ? std::uint32_t(last_bit_index - r.range_union.last)
                                               : std::uint32_t(0);
            auto const cache_bits = r.individual_ranges[i].cache_block_count * cache_bits_unit -
                                    excessive_bits_to_right;

            auto const number_of_trailing_zero_blocks = excessive_bits_to_right / cache_bits_unit;
            excessive_bits_to_right %= cache_bits_unit;

            // Should we round-up the loaded cache?
            if (k < segment_length ||
                (r.first_exponent + i) + k + cache_bits < segment_length + 1) {
                // This is the last bits of the cache that will be loaded.
                auto relevant_block = floor(compute_pow2_pow5(last_bit_index, k - segment_length));
                relevant_block /=
                    jkj::big_uint::power_of_2(number_of_trailing_zero_blocks * cache_bits_unit);
                relevant_block.long_division(jkj::big_uint::power_of_2(cache_bits_unit));

                relevant_block += jkj::big_uint::power_of_2(excessive_bits_to_right);

                if (relevant_block == jkj::big_uint::power_of_2(cache_bits_unit)) {
                    std::cout << "[Failure] Ceiling can possibly fail!\n";
                    return {};
                }
            }
        }
    }


    std::cout << "   Total bytes for powers of 5: "
              << (std::accumulate(mul_info.cbegin(), mul_info.cend(), 0,
                                  [](int v, multiplier_info<cache_unit_type> const& r) {
                                      return v + r.cache_bits;
                                  }) +
                  7) /
                     8
              << "\n";
    std::cout << "    Number of (e_base,k) pairs: " << number_of_e_base_k_pairs << "\n";
    std::cout << "         Number of powers of 5: " << number_of_multipliers << "\n";
    std::cout << "Maximum number of cache blocks: " << result.max_cache_blocks << " ("
              << max_cache_bits << " bits)\n";
    std::cout << "Average number of cache blocks: "
              << double(accumulated_number_of_cache_blocks) / number_of_e_k_pairs << "\n";
    std::cout << "         Number of (e,k) pairs: " << number_of_e_k_pairs << "\n";

    return result;
}


std::size_t print_uint_typename(std::ostream& out, std::size_t max_value) {
    if (max_value < (1ull << 8)) {
        out << "std::uint8_t";
        return 1;
    }
    else if (max_value < (1ull << 16)) {
        out << "std::uint16_t";
        return 2;
    }
    else if (max_value < (1ull << 32)) {
        out << "std::uint32_t";
        return 4;
    }
    else {
        out << "std::uint64_t";
        return 8;
    }
}

#include <iomanip>

template <class CacheUnitType>
bool print_cache(std::ostream& out, extended_cache_result<CacheUnitType> const& cache) {
    if (cache.max_cache_blocks >= 16) {
        std::cout << "[Failure] Too many cache blocks needed!\n";
        return false;
    }

    // Print out header.
    out << "static constexpr std::size_t max_cache_blocks = " << cache.max_cache_blocks << ";\n";
    out << "static constexpr std::size_t cache_bits_unit = " << cache.cache_bits_unit << ";\n";
    out << "static constexpr int segment_length = " << cache.segment_length << ";\n";
    if (cache.collapse_factor == 0) {
        out << "static constexpr bool constant_block_count = true;\n";
    }
    else {
        out << "static constexpr bool constant_block_count = false;\n";
        out << "static constexpr int collapse_factor = " << cache.collapse_factor << ";\n";
    }
    out << "static constexpr int e_min = " << cache.e_min << ";\n";
    out << "static constexpr int k_min = " << cache.k_min << ";\n";
    int const cache_bit_index_offset_base =
        cache.mul_info[0].first_exponent + cache.mul_info[0].min_shift;
    out << "static constexpr int cache_bit_index_offset_base = " << cache_bit_index_offset_base
        << ";\n";
    int cache_block_count_offset_base = 0;
    if (cache.collapse_factor != 0) {
        cache_block_count_offset_base =
            (cache.mul_info[0].first_exponent - cache.e_min) / cache.collapse_factor;
        out << "static constexpr int cache_block_count_offset_base = "
            << cache_block_count_offset_base << ";\n\n";
    }

    std::size_t total_data_size_in_bytes = 0;

    // Print out the powers of 5 table.
    using cache_unit_type = CacheUnitType;
    std::vector<int> cache_bits_prefix_sums{0};
    std::vector<cache_unit_type> cache_blocks;
    cache_unit_type current_block = 0;
    std::size_t number_of_written_bits_in_current_block = 0;
    assert(jkj::big_uint::element_number_of_bits == cache.cache_bits_unit);

    for (auto const& r : cache.mul_info) {
        if (r.cache_bits != 0) {
            // Align the MSB of the multiplier to the MSB of the first element in the multiplier,
            // to simplify the procedure.
            auto const shift_amount =
                (r.multiplier.size() * cache.cache_bits_unit) - int(log2p1(r.multiplier));
            auto const aligned_multiplier = r.multiplier * jkj::big_uint::power_of_2(shift_amount);
            assert(!aligned_multiplier.is_zero());

            for (std::size_t idx = aligned_multiplier.size() - 1; idx > 0; --idx) {
                cache_blocks.push_back(current_block | (aligned_multiplier[idx] >>
                                                        number_of_written_bits_in_current_block));
                if (number_of_written_bits_in_current_block == 0) {
                    current_block = 0;
                }
                else {
                    current_block =
                        (aligned_multiplier[idx]
                         << (cache.cache_bits_unit - number_of_written_bits_in_current_block));
                }
            }
            auto const number_of_remaining_bits = cache.cache_bits_unit - shift_amount;

            if (number_of_remaining_bits + number_of_written_bits_in_current_block >=
                cache.cache_bits_unit) {
                cache_blocks.push_back(current_block | (aligned_multiplier[0] >>
                                                        number_of_written_bits_in_current_block));

                current_block =
                    (aligned_multiplier[0]
                     << (cache.cache_bits_unit - number_of_written_bits_in_current_block));

                number_of_written_bits_in_current_block = number_of_remaining_bits +
                                                          number_of_written_bits_in_current_block -
                                                          cache.cache_bits_unit;
            }
            else {
                current_block |= (aligned_multiplier[0] >> number_of_written_bits_in_current_block);
                number_of_written_bits_in_current_block += number_of_remaining_bits;
            }
        }        

        cache_bits_prefix_sums.push_back(cache_bits_prefix_sums.back() + int(r.cache_bits));
    }
    if (number_of_written_bits_in_current_block != 0) {
        cache_blocks.push_back(current_block);
    }

    // Judge if an additional zero block is needed.
    if (cache.mul_info.size() > 0) {
        auto const& last_mul_info = cache.mul_info.back();
        assert(cache_bits_prefix_sums.size() > 1);

        // For each exponent in associated to the last multiplier,
        for (int exp_index = 0; exp_index < last_mul_info.individual_ranges.size(); ++exp_index) {
            // Compute the largest cache block index that can arise.
            // Follow the cache load procedure of the actual algorithm.
            auto cache_block_count = last_mul_info.individual_ranges[exp_index].cache_block_count;

            std::uint32_t number_of_leading_zero_blocks;
            std::uint32_t first_cache_block_index;
            std::uint32_t bit_offset;

            // The request window starting/ending positions.
            auto start_bit_index = cache_bits_prefix_sums[cache_bits_prefix_sums.size() - 2] +
                                   exp_index - last_mul_info.min_shift;
            auto end_bit_index = start_bit_index + cache_block_count * int(cache.cache_bits_unit);

            // The source window starting/ending positions.
            auto const src_start_bit_index =
                cache_bits_prefix_sums[cache_bits_prefix_sums.size() - 2];
            auto const src_end_bit_index =
                cache_bits_prefix_sums[cache_bits_prefix_sums.size() - 1];

            // If the request window goes further than the left boundary of the source window,
            if (start_bit_index < src_start_bit_index) {
                number_of_leading_zero_blocks =
                    std::uint32_t(src_start_bit_index - start_bit_index) /
                    std::uint32_t(cache.cache_bits_unit);

                start_bit_index += number_of_leading_zero_blocks * int(cache.cache_bits_unit);

                auto const src_start_block_index =
                    int(std::uint32_t(src_start_bit_index) / std::uint32_t(cache.cache_bits_unit));
                auto const src_start_block_bit_index =
                    src_start_block_index * int(cache.cache_bits_unit);

                first_cache_block_index = src_start_block_index;

                if (start_bit_index < src_start_block_bit_index) {
                    auto shift_amount = src_start_block_bit_index - start_bit_index;
                    ++number_of_leading_zero_blocks;
                    bit_offset = std::uint32_t(int(cache.cache_bits_unit) - shift_amount);
                }
                else {
                    bit_offset = std::uint32_t(start_bit_index - src_start_block_bit_index);
                }
            }
            else {
                number_of_leading_zero_blocks = 0;
                first_cache_block_index =
                    std::uint32_t(start_bit_index) / std::uint32_t(cache.cache_bits_unit);
                bit_offset = std::uint32_t(start_bit_index) % std::uint32_t(cache.cache_bits_unit);
            }

            // If the request window goes further than the right boundary of the source window,
            if (end_bit_index > src_end_bit_index) {
                auto const number_of_trailing_zero_blocks =
                    std::uint32_t(end_bit_index - src_end_bit_index) /
                    std::uint32_t(cache.cache_bits_unit);
                cache_block_count -= number_of_trailing_zero_blocks;
            }

            auto const number_of_blocks_to_load = cache_block_count - number_of_leading_zero_blocks;
            auto const last_block_index =
                bit_offset == 0 ? first_cache_block_index + number_of_blocks_to_load - 1
                                : first_cache_block_index + number_of_blocks_to_load;

            if (last_block_index >= cache_blocks.size()) {
                assert(last_block_index == cache_blocks.size());
                cache_blocks.push_back(0);
                break;
            }
        }
    }


    out << "static constexpr ";
    print_uint_typename(out, std::numeric_limits<cache_unit_type>::max());
    out << " cache[] = {" << std::hex;
    for (std::size_t i = 0; i < cache_blocks.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << "\n\t0x" << std::setfill('0');
        if constexpr (std::is_same_v<cache_unit_type, std::uint32_t>) {
            out << std::setw(8);
        }
        else {
            out << std::setw(16);
        }
        out << cache_blocks[i];
    }
    out << std::dec << "\n};\n\n";

    total_data_size_in_bytes += cache_blocks.size() * sizeof(cache_unit_type);

    // Compute the cache block count table.
    std::vector<int> cache_block_counts_prefix_sums{0};
    std::vector<std::uint32_t> cache_block_counts;
    if (cache.collapse_factor != 0) {
        for (auto const& r : cache.mul_info) {
            for (auto const& e_base : r.base_exponents) {
                auto const count = r.individual_ranges[e_base - r.first_exponent].cache_block_count;
                cache_block_counts.push_back(count);
            }
            cache_block_counts_prefix_sums.push_back(cache_block_counts_prefix_sums.back() +
                                                     r.base_exponents.size());
        }
    }

    // Print out the multiplier index info table.
    out << "struct multiplier_index_info {\n\t";

    std::size_t info_alignment_size = 0;

    info_alignment_size =
        std::max(info_alignment_size,
                 print_uint_typename(out, cache_bits_prefix_sums[cache.mul_info.size() - 1]));
    out << " first_cache_bit_index;\n\t";

    info_alignment_size = print_uint_typename(
        out, cache_bits_prefix_sums[cache.mul_info.size() - 1] -
                 cache.mul_info.back().first_exponent + cache.mul_info[0].first_exponent);
    out << " cache_bit_index_offset;\n";

    if (cache.collapse_factor != 0) {
        out << "\t";
        info_alignment_size = std::max(
            info_alignment_size,
            print_uint_typename(out, cache_block_counts_prefix_sums[cache.mul_info.size() - 1] -
                                         cache.mul_info.back().first_exponent +
                                         cache.mul_info[0].first_exponent));
        out << " cache_block_count_index_offset;\n};\n\n";
    }
    else {
        out << "};\n\n";
    }

    // This computation is of course wrong in general, but it is correct for all specific cases we
    // are considering.
    std::size_t info_struct_size = info_alignment_size * (cache.collapse_factor != 0 ? 3 : 2);

    out << "static constexpr multiplier_index_info multiplier_index_info_table[] = {\n\t";
    for (std::size_t multiplier_index = 0; multiplier_index < cache.mul_info.size();
         ++multiplier_index) {
        auto const first_cache_bit_index = cache_bits_prefix_sums[multiplier_index];

        auto const cache_bit_offset = cache_bits_prefix_sums[multiplier_index] -
                                      cache.mul_info[multiplier_index].first_exponent -
                                      cache.mul_info[multiplier_index].min_shift +
                                      cache_bit_index_offset_base;
        assert(cache_bit_offset >= 0);

        if (cache.collapse_factor == 0) {
            out << "{" << first_cache_bit_index << ", " << cache_bit_offset << "},\n\t";
        }
        else {
            auto const cache_block_count_offset =
                cache_block_counts_prefix_sums[multiplier_index] -
                (cache.mul_info[multiplier_index].first_exponent - cache.e_min) /
                    cache.collapse_factor +
                cache_block_count_offset_base;

            out << "{" << first_cache_bit_index << ", " << cache_bit_offset << ", "
                << cache_block_count_offset << "},\n\t";
        }
    }
    if (cache.collapse_factor == 0) {
        out << "{" << cache_bits_prefix_sums.back() << ", 0}\n};";
    }
    else {
        out << "{" << cache_bits_prefix_sums.back() << ", 0, 0}\n};\n\n";
    }
    total_data_size_in_bytes += info_struct_size * (cache.mul_info.size() + 1);

    // Print out the cache block count table.
    if (cache.collapse_factor != 0) {
        out << "static constexpr std::uint8_t cache_block_counts[] = {" << std::hex;
        if (cache.max_cache_blocks < 3) {
            for (std::size_t i = 0; i < cache_block_counts.size(); i += 8) {
                if (i != 0) {
                    out << ",";
                }
                out << "\n\t0x" << std::setfill('0');

                std::uint8_t value = cache_block_counts[i] - 1;
                if (i + 1 < cache_block_counts.size()) {
                    value |= ((cache_block_counts[i + 1] - 1) << 1);
                }
                if (i + 2 < cache_block_counts.size()) {
                    value |= ((cache_block_counts[i + 2] - 1) << 2);
                }
                if (i + 3 < cache_block_counts.size()) {
                    value |= ((cache_block_counts[i + 3] - 1) << 3);
                }
                if (i + 4 < cache_block_counts.size()) {
                    value |= ((cache_block_counts[i + 4] - 1) << 4);
                }
                if (i + 5 < cache_block_counts.size()) {
                    value |= ((cache_block_counts[i + 5] - 1) << 5);
                }
                if (i + 6 < cache_block_counts.size()) {
                    value |= ((cache_block_counts[i + 6] - 1) << 6);
                }
                if (i + 7 < cache_block_counts.size()) {
                    value |= ((cache_block_counts[i + 7] - 1) << 7);
                }
                out << std::setw(2) << std::uint32_t(value);

                ++total_data_size_in_bytes;
            }
        }
        else if (cache.max_cache_blocks < 4) {
            for (std::size_t i = 0; i < cache_block_counts.size(); i += 4) {
                if (i != 0) {
                    out << ",";
                }
                out << "\n\t0x" << std::setfill('0');

                std::uint8_t value = cache_block_counts[i];
                if (i + 1 < cache_block_counts.size()) {
                    value |= (cache_block_counts[i + 1] << 2);
                }
                if (i + 2 < cache_block_counts.size()) {
                    value |= (cache_block_counts[i + 2] << 4);
                }
                if (i + 3 < cache_block_counts.size()) {
                    value |= (cache_block_counts[i + 3] << 6);
                }
                out << std::setw(2) << std::uint32_t(value);

                ++total_data_size_in_bytes;
            }
        }
        else {
            assert(cache.max_cache_blocks < 16);
            for (std::size_t i = 0; i < cache_block_counts.size(); i += 2) {
                if (i != 0) {
                    out << ",";
                }
                out << "\n\t0x" << std::setfill('0');

                std::uint8_t value = cache_block_counts[i];
                if (i + 1 < cache_block_counts.size()) {
                    value |= (cache_block_counts[i + 1] << 4);
                }
                out << std::setw(2) << std::uint32_t(value);

                ++total_data_size_in_bytes;
            }
        }

        out << std::dec << "\n};";
    }

    std::cout << "Total static data size: " << total_data_size_in_bytes << " bytes.\n";
    return true;
}

#include <fstream>

void generate_extended_cache_and_write_to_file(char const* filename, int segment_length,
                                               int collapse_factor, unsigned int k_min_shift) {
    std::ofstream out{filename};
    auto cache = generate_extended_cache<double>(segment_length, collapse_factor, k_min_shift);
    std::cout << "\n";
    print_cache(out, cache);
}

int main() {
    constexpr bool generate_long = true;
    constexpr bool generate_compact = true;
    constexpr bool generate_super_compact = true;

    if constexpr (generate_long) {
        std::cout << "[Generating long extended cache...]\n";
        generate_extended_cache_and_write_to_file( //
            "results/binary64_generated_extended_cache_long.txt", 22, 0, 3);
        std::cout << "Done.\n\n\n";
    }

    if constexpr (generate_compact) {
        std::cout << "[Generating compact extended cache...]\n";
        generate_extended_cache_and_write_to_file( //
            "results/binary64_generated_extended_cache_compact.txt", 80, 64, 0);
        std::cout << "Done.\n\n\n";
    }

    if constexpr (generate_super_compact) {
        std::cout << "[Generating super compact extended cache...]\n";
        generate_extended_cache_and_write_to_file( //
            "results/binary64_generated_extended_cache_super_compact.txt", 252, 128, 27);
        std::cout << "Done.\n\n\n";
    }
}
