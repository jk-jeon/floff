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

// Some parts are copied from Dragonbox project.


#ifndef JKJ_HEADER_FLOFF
#define JKJ_HEADER_FLOFF

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

// Suppress additional buffer overrun check.
// I have no idea why MSVC thinks some functions here are vulnerable to the buffer overrun
// attacks. No, they aren't.
#if defined(__GNUC__) || defined(__clang__)
    #define JKJ_SAFEBUFFERS
    #define JKJ_FORCEINLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define JKJ_SAFEBUFFERS __declspec(safebuffers)
    #define JKJ_FORCEINLINE __forceinline
#else
    #define JKJ_SAFEBUFFERS
    #define JKJ_FORCEINLINE inline
#endif

#if defined(__has_builtin)
    #define JKJ_HAS_BUILTIN(x) __has_builtin(x)
#else
    #define JKJ_HAS_BUILTIN(x) false
#endif

#if __has_cpp_attribute(assume)
    #define JKJ_UNRECHABLE [[assume(false)]]
#elif defined(__GNUC__) && JKJ_HAS_BUILTIN(__builtin_unreachable)
    #define JKJ_UNRECHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
    #define JKJ_UNRECHABLE __assume(false)
#else
    #define JKJ_UNRECHABLE
#endif

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

namespace jkj::floff {
    namespace detail {
        template <class T>
        constexpr std::size_t physical_bits =
            sizeof(T) * std::numeric_limits<unsigned char>::digits;

        template <class T>
        constexpr std::size_t value_bits =
            std::numeric_limits<std::enable_if_t<std::is_unsigned_v<T>, T>>::digits;
    }

    // These classes expose encoding specs of IEEE-754-like floating-point formats.
    // Currently available formats are IEEE754-binary32 & IEEE754-binary64.

    struct ieee754_binary32 {
        static constexpr int significand_bits = 23;
        static constexpr int exponent_bits = 8;
        static constexpr int min_exponent = -126;
        static constexpr int max_exponent = 127;
        static constexpr int exponent_bias = -127;
        static constexpr int decimal_digits = 9;
    };
    struct ieee754_binary64 {
        static constexpr int significand_bits = 52;
        static constexpr int exponent_bits = 11;
        static constexpr int min_exponent = -1022;
        static constexpr int max_exponent = 1023;
        static constexpr int exponent_bias = -1023;
        static constexpr int decimal_digits = 17;
    };

    // A floating-point traits class defines ways to interpret a bit pattern of given size as an
    // encoding of floating-point number. This is a default implementation of such a traits class,
    // supporting ways to interpret 32-bits into a binary32-encoded floating-point number and to
    // interpret 64-bits into a binary64-encoded floating-point number. Users might specialize this
    // class to change the default behavior for certain types.
    template <class T>
    struct default_float_traits {
        // I don't know if there is a truly reliable way of detecting
        // IEEE-754 binary32/binary64 formats; I just did my best here.
        static_assert(std::numeric_limits<T>::is_iec559 && std::numeric_limits<T>::radix == 2 &&
                          (detail::physical_bits<T> == 32 || detail::physical_bits<T> == 64),
                      "default_ieee754_traits only works for 32-bits or 64-bits types "
                      "supporting binary32 or binary64 formats!");

        // The type that is being viewed.
        using type = T;

        // Refers to the format specification class.
        using format =
            std::conditional_t<detail::physical_bits<T> == 32, ieee754_binary32, ieee754_binary64>;

        // Defines an unsigned integer type that is large enough to carry a variable of type T.
        // Most of the operations will be done on this integer type.
        using carrier_uint =
            std::conditional_t<detail::physical_bits<T> == 32, std::uint32_t, std::uint64_t>;
        static_assert(sizeof(carrier_uint) == sizeof(T));

        // Number of bits in the above unsigned integer type.
        static constexpr int carrier_bits = int(detail::physical_bits<carrier_uint>);

        // Convert from carrier_uint into the original type.
        // Depending on the floating-point encoding format, this operation might not be possible for
        // some specific bit patterns. However, the contract is that u always denotes a
        // valid bit pattern, so this function must be assumed to be noexcept.
        static T carrier_to_float(carrier_uint u) noexcept {
            T x;
            std::memcpy(&x, &u, sizeof(carrier_uint));
            return x;
        }

        // Same as above.
        static carrier_uint float_to_carrier(T x) noexcept {
            carrier_uint u;
            std::memcpy(&u, &x, sizeof(carrier_uint));
            return u;
        }

        // Extract exponent bits from a bit pattern.
        // The result must be aligned to the LSB so that there is no additional zero paddings
        // on the right. This function does not do bias adjustment.
        static constexpr unsigned int extract_exponent_bits(carrier_uint u) noexcept {
            constexpr int significand_bits = format::significand_bits;
            constexpr int exponent_bits = format::exponent_bits;
            static_assert(detail::value_bits<unsigned int> > exponent_bits);
            constexpr auto exponent_bits_mask =
                (unsigned int)(((unsigned int)(1) << exponent_bits) - 1);
            return (unsigned int)(u >> significand_bits) & exponent_bits_mask;
        }

        // Extract significand bits from a bit pattern.
        // The result must be aligned to the LSB so that there is no additional zero paddings
        // on the right. The result does not contain the implicit bit.
        static constexpr carrier_uint extract_significand_bits(carrier_uint u) noexcept {
            constexpr auto mask = carrier_uint((carrier_uint(1) << format::significand_bits) - 1);
            return carrier_uint(u & mask);
        }

        // Remove the exponent bits and extract significand bits together with the sign bit.
        static constexpr carrier_uint remove_exponent_bits(carrier_uint u,
                                                           unsigned int exponent_bits) noexcept {
            return u ^ (carrier_uint(exponent_bits) << format::significand_bits);
        }

        // Shift the obtained signed significand bits to the left by 1 to remove the sign bit.
        static constexpr carrier_uint remove_sign_bit_and_shift(carrier_uint u) noexcept {
            return carrier_uint(carrier_uint(u) << 1);
        }

        // The actual value of exponent is obtained by adding this value to the extracted exponent
        // bits.
        static constexpr int exponent_bias =
            1 - (1 << (carrier_bits - format::significand_bits - 2));

        // Obtain the actual value of the binary exponent from the extracted exponent bits.
        static constexpr int binary_exponent(unsigned int exponent_bits) noexcept {
            if (exponent_bits == 0) {
                return format::min_exponent;
            }
            else {
                return int(exponent_bits) + format::exponent_bias;
            }
        }

        // Obtain the actual value of the binary exponent from the extracted significand bits and
        // exponent bits.
        static constexpr carrier_uint binary_significand(carrier_uint significand_bits,
                                                         unsigned int exponent_bits) noexcept {
            if (exponent_bits == 0) {
                return significand_bits;
            }
            else {
                return significand_bits | (carrier_uint(1) << format::significand_bits);
            }
        }


        /* Various boolean observer functions */

        static constexpr bool is_nonzero(carrier_uint u) noexcept { return (u << 1) != 0; }
        static constexpr bool is_positive(carrier_uint u) noexcept {
            constexpr auto sign_bit = carrier_uint(1)
                                      << (format::significand_bits + format::exponent_bits);
            return u < sign_bit;
        }
        static constexpr bool is_negative(carrier_uint u) noexcept { return !is_positive(u); }
        static constexpr bool is_finite(unsigned int exponent_bits) noexcept {
            constexpr unsigned int exponent_bits_all_set = (1u << format::exponent_bits) - 1;
            return exponent_bits != exponent_bits_all_set;
        }
        static constexpr bool has_all_zero_significand_bits(carrier_uint u) noexcept {
            return (u << 1) == 0;
        }
        static constexpr bool has_even_significand_bits(carrier_uint u) noexcept {
            return u % 2 == 0;
        }
    };

    // Convenient wrappers for floating-point traits classes.
    // In order to reduce the argument passing overhead, these classes should be as simple as
    // possible (e.g., no inheritance, no private non-static data member, etc.; this is an
    // unfortunate fact about common ABI convention).

    template <class T, class Traits = default_float_traits<T>>
    struct float_bits;

    template <class T, class Traits = default_float_traits<T>>
    struct signed_significand_bits;

    template <class T, class Traits>
    struct float_bits {
        using type = T;
        using traits_type = Traits;
        using carrier_uint = typename traits_type::carrier_uint;

        carrier_uint u;

        float_bits() = default;
        constexpr explicit float_bits(carrier_uint bit_pattern) noexcept : u{bit_pattern} {}
        constexpr explicit float_bits(T float_value) noexcept
            : u{traits_type::float_to_carrier(float_value)} {}

        constexpr T to_float() const noexcept { return traits_type::carrier_to_float(u); }

        // Extract exponent bits from a bit pattern.
        // The result must be aligned to the LSB so that there is no additional zero paddings
        // on the right. This function does not do bias adjustment.
        constexpr unsigned int extract_exponent_bits() const noexcept {
            return traits_type::extract_exponent_bits(u);
        }

        // Extract significand bits from a bit pattern.
        // The result must be aligned to the LSB so that there is no additional zero paddings
        // on the right. The result does not contain the implicit bit.
        constexpr carrier_uint extract_significand_bits() const noexcept {
            return traits_type::extract_significand_bits(u);
        }

        // Remove the exponent bits and extract significand bits together with the sign bit.
        constexpr auto remove_exponent_bits(unsigned int exponent_bits) const noexcept {
            return signed_significand_bits<type, traits_type>(
                traits_type::remove_exponent_bits(u, exponent_bits));
        }

        // Obtain the actual value of the binary exponent from the extracted exponent bits.
        static constexpr int binary_exponent(unsigned int exponent_bits) noexcept {
            return traits_type::binary_exponent(exponent_bits);
        }
        constexpr int binary_exponent() const noexcept {
            return binary_exponent(extract_exponent_bits());
        }

        // Obtain the actual value of the binary exponent from the extracted significand bits and
        // exponent bits.
        static constexpr carrier_uint binary_significand(carrier_uint significand_bits,
                                                         unsigned int exponent_bits) noexcept {
            return traits_type::binary_significand(significand_bits, exponent_bits);
        }
        constexpr carrier_uint binary_significand() const noexcept {
            return binary_significand(extract_significand_bits(), extract_exponent_bits());
        }

        constexpr bool is_nonzero() const noexcept { return traits_type::is_nonzero(u); }
        constexpr bool is_positive() const noexcept { return traits_type::is_positive(u); }
        constexpr bool is_negative() const noexcept { return traits_type::is_negative(u); }
        constexpr bool is_finite(unsigned int exponent_bits) const noexcept {
            return traits_type::is_finite(exponent_bits);
        }
        constexpr bool is_finite() const noexcept {
            return traits_type::is_finite(extract_exponent_bits());
        }
        constexpr bool has_even_significand_bits() const noexcept {
            return traits_type::has_even_significand_bits(u);
        }
    };

    template <class T, class Traits>
    struct signed_significand_bits {
        using type = T;
        using traits_type = Traits;
        using carrier_uint = typename traits_type::carrier_uint;

        carrier_uint u;

        signed_significand_bits() = default;
        constexpr explicit signed_significand_bits(carrier_uint bit_pattern) noexcept
            : u{bit_pattern} {}

        // Shift the obtained signed significand bits to the left by 1 to remove the sign bit.
        constexpr carrier_uint remove_sign_bit_and_shift() const noexcept {
            return traits_type::remove_sign_bit_and_shift(u);
        }

        constexpr bool is_positive() const noexcept { return traits_type::is_positive(u); }
        constexpr bool is_negative() const noexcept { return traits_type::is_negative(u); }
        constexpr bool has_all_zero_significand_bits() const noexcept {
            return traits_type::has_all_zero_significand_bits(u);
        }
        constexpr bool has_even_significand_bits() const noexcept {
            return traits_type::has_even_significand_bits(u);
        }
    };

    namespace detail {
        ////////////////////////////////////////////////////////////////////////////////////////
        // Bit operation intrinsics.
        ////////////////////////////////////////////////////////////////////////////////////////

        namespace bits {
            // Most compilers should be able to optimize this into the ROR instruction.
            inline std::uint32_t rotr(std::uint32_t n, std::uint32_t r) noexcept {
                r &= 31;
                return (n >> r) | (n << (32 - r));
            }
            inline std::uint64_t rotr(std::uint64_t n, std::uint32_t r) noexcept {
                r &= 63;
                return (n >> r) | (n << (64 - r));
            }

            // Count leading zero bits.
            // Undefined behavior for x == 0.
            inline int countl_zero(std::uint64_t x) noexcept {
#if JKJ_HAS_BUILTIN(__builtin_clzll)
                return __builtin_clzll(x);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
                unsigned long index;
                _BitScanReverse64(&index, x);
                return 63 - int(index);
#else
                // We use the 4-bit de Brujin sequence 0x0f65.
                // The corresponding sequence is:
                // 0, 1, 3, 7, 15, 14, 13, 11, 6, 12, 9, 2, 5, 10, 4, 8
                constexpr std::uint32_t de_brujin = 0x0f650000;
                // 16-bit de Brujin packed in 4-bits:
                constexpr std::uint64_t lookup = 0xcba79361d842e5f0;

                int count;
                std::uint32_t x32;
                if ((x >> 32) == 0) {
                    count = 32;
                    x32 = std::uint32_t(x);
                }
                else {
                    count = 0;
                    x32 = std::uint32_t(x >> 32);
                }
                std::uint32_t x16;
                if ((x32 >> 16) == 0) {
                    count += 16;
                    x16 = std::uint16_t(x32);
                }
                else {
                    x16 = (x32 >> 16);
                }

                // Set one bit above the leading 1 and clear all other bits.
                x16 |= (x16 >> 1);
                x16 |= (x16 >> 2);
                x16 |= (x16 >> 4);
                x16 |= (x16 >> 8);
                ++x16;

                return count + int((lookup >> (((x16 * de_brujin) >> 28) << 2)) & 0xf);
#endif
            }

            // Count trailing zero bits.
            // Undefined behavior for x == 0.
            inline int countr_zero(std::uint64_t x) noexcept {
#if JKJ_HAS_BUILTIN(__builtin_ctzll)
                return __builtin_ctzll(x);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
                unsigned long index;
                _BitScanForward64(&index, x);
                return int(index);
#else
                // We use the 4-bit de Brujin sequence 0x0f65.
                // The corresponding sequence is:
                // 0, 1, 3, 7, 15, 14, 13, 11, 6, 12, 9, 2, 5, 10, 4, 8
                constexpr std::uint32_t de_brujin = 0x0f650000;
                // 16-bit de Brujin packed in 4-bits:
                constexpr std::uint64_t lookup = 0x45697daf38ce2b10;

                // In (-x & x), only the least significant set bit is 1.
                x &= (0 - x);
                int count = 0;
                std::uint32_t x32 = std::uint32_t(x);
                if (x32 == 0) {
                    count = 32;
                    x32 = std::uint32_t(x >> 32);
                }
                std::uint32_t x16 = std::uint16_t(x32);
                if (x16 == 0) {
                    count += 16;
                    x16 = x32 >> 16;
                }
                return count + int((lookup >> (((x16 * de_brujin) >> 28) << 2)) & 0xf);
#endif
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////
        // Utilities for wide unsigned integer arithmetic.
        ////////////////////////////////////////////////////////////////////////////////////////

        namespace wuint {
            // Compilers might support built-in 128-bit integer types. However, it seems that
            // emulating them with a pair of 64-bit integers actually produces a better code,
            // so we avoid using those built-ins. That said, they are still useful for
            // implementing 64-bit x 64-bit -> 128-bit multiplication.

            // clang-format off
    #if defined(__SIZEOF_INT128__)
            // To silence "error: ISO C++ does not support '__int128' for 'type name'
            // [-Wpedantic]"
        #if defined(__GNUC__)
            __extension__
        #endif
            using builtin_uint128_t = unsigned __int128;
    #endif
            // clang-format on

            struct uint128 {
                uint128() = default;

                std::uint64_t high_;
                std::uint64_t low_;

                constexpr uint128(std::uint64_t high, std::uint64_t low) noexcept
                    : high_{high}, low_{low} {}

                constexpr std::uint64_t high() const noexcept { return high_; }
                constexpr std::uint64_t low() const noexcept { return low_; }

                uint128& operator+=(std::uint64_t n) & noexcept {
#if JKJ_HAS_BUILTIN(__builtin_addcll)
                    unsigned long long carry;
                    low_ = __builtin_addcll(low_, n, 0, &carry);
                    high_ = __builtin_addcll(high_, 0, carry, &carry);
#elif JKJ_HAS_BUILTIN(__builtin_ia32_addcarryx_u64)
                    unsigned long long result;
                    auto carry = __builtin_ia32_addcarryx_u64(0, low_, n, &result);
                    low_ = result;
                    __builtin_ia32_addcarryx_u64(carry, high_, 0, &result);
                    high_ = result;
#elif defined(_MSC_VER) && defined(_M_X64)
                    auto carry = _addcarry_u64(0, low_, n, &low_);
                    _addcarry_u64(carry, high_, 0, &high_);
#else
                    auto sum = low_ + n;
                    high_ += (sum < low_ ? 1 : 0);
                    low_ = sum;
#endif
                    return *this;
                }
            };

            static inline std::uint64_t umul64(std::uint32_t x, std::uint32_t y) noexcept {
#if defined(_MSC_VER) && defined(_M_IX86)
                return __emulu(x, y);
#else
                return x * std::uint64_t(y);
#endif
            }

            // Get 128-bit result of multiplication of two 64-bit unsigned integers.
            JKJ_SAFEBUFFERS inline uint128 umul128(std::uint64_t x, std::uint64_t y) noexcept {
#if defined(__SIZEOF_INT128__)
                auto result = builtin_uint128_t(x) * builtin_uint128_t(y);
                return {std::uint64_t(result >> 64), std::uint64_t(result)};
#elif defined(_MSC_VER) && defined(_M_X64)
                uint128 result;
                result.low_ = _umul128(x, y, &result.high_);
                return result;
#else
                auto a = std::uint32_t(x >> 32);
                auto b = std::uint32_t(x);
                auto c = std::uint32_t(y >> 32);
                auto d = std::uint32_t(y);

                auto ac = umul64(a, c);
                auto bc = umul64(b, c);
                auto ad = umul64(a, d);
                auto bd = umul64(b, d);

                auto intermediate = (bd >> 32) + std::uint32_t(ad) + std::uint32_t(bc);

                return {ac + (intermediate >> 32) + (ad >> 32) + (bc >> 32),
                        (intermediate << 32) + std::uint32_t(bd)};
#endif
            }

            JKJ_SAFEBUFFERS inline std::uint64_t umul128_upper64(std::uint64_t x,
                                                                 std::uint64_t y) noexcept {
#if defined(__SIZEOF_INT128__)
                auto result = builtin_uint128_t(x) * builtin_uint128_t(y);
                return std::uint64_t(result >> 64);
#elif defined(_MSC_VER) && defined(_M_X64)
                return __umulh(x, y);
#else
                auto a = std::uint32_t(x >> 32);
                auto b = std::uint32_t(x);
                auto c = std::uint32_t(y >> 32);
                auto d = std::uint32_t(y);

                auto ac = umul64(a, c);
                auto bc = umul64(b, c);
                auto ad = umul64(a, d);
                auto bd = umul64(b, d);

                auto intermediate = (bd >> 32) + std::uint32_t(ad) + std::uint32_t(bc);

                return ac + (intermediate >> 32) + (ad >> 32) + (bc >> 32);
#endif
            }

            // Get upper 128-bits of multiplication of a 64-bit unsigned integer and a 128-bit
            // unsigned integer.
            JKJ_SAFEBUFFERS inline uint128 umul192_upper128(std::uint64_t x, uint128 y) noexcept {
                auto r = umul128(x, y.high());
                r += umul128_upper64(x, y.low());
                return r;
            }

            // Get upper 64-bits of multiplication of a 32-bit unsigned integer and a 64-bit
            // unsigned integer.
            inline std::uint64_t umul96_upper64(std::uint32_t x, std::uint64_t y) noexcept {
#if defined(__SIZEOF_INT128__) || (defined(_MSC_VER) && defined(_M_X64))
                return umul128_upper64(std::uint64_t(x) << 32, y);
#else
                auto yh = std::uint32_t(y >> 32);
                auto yl = std::uint32_t(y);

                auto xyh = umul64(x, yh);
                auto xyl = umul64(x, yl);

                return xyh + (xyl >> 32);
#endif
            }

            // Get lower 128-bits of multiplication of a 64-bit unsigned integer and a 128-bit
            // unsigned integer.
            JKJ_SAFEBUFFERS inline uint128 umul192_lower128(std::uint64_t x, uint128 y) noexcept {
                auto high = x * y.high();
                auto high_low = umul128(x, y.low());
                return {high + high_low.high(), high_low.low()};
            }

            // Get lower 64-bits of multiplication of a 32-bit unsigned integer and a 64-bit
            // unsigned integer.
            inline std::uint64_t umul96_lower64(std::uint32_t x, std::uint64_t y) noexcept {
                return x * y;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////
        // Some simple utilities for constexpr computation.
        ////////////////////////////////////////////////////////////////////////////////////////

        template <class Int>
        constexpr Int compute_power(Int a, unsigned int exp) noexcept {
            Int res = 1;
            while (exp > 0) {
                if (exp % 2 != 0) {
                    res *= a;
                }
                a *= a;
                exp >>= 1;
            }
            return res;
        }

        template <unsigned int exp>
        struct power_of_10_impl {
            static_assert(exp <= 19);
            using type = std::conditional_t<exp <= 9, std::uint32_t, std::uint64_t>;

            static constexpr type value = compute_power(type(10), exp);
        };

        template <unsigned int exp>
        inline constexpr auto power_of_10 = power_of_10_impl<exp>::value;

        ////////////////////////////////////////////////////////////////////////////////////////
        // Utilities for fast/constexpr log computation.
        ////////////////////////////////////////////////////////////////////////////////////////

        namespace log {
            static_assert((-1 >> 1) == -1, "right-shift for signed integers must be arithmetic");

            // Compute floor(e * c - s).
            enum class multiply : std::uint32_t {};
            enum class subtract : std::uint32_t {};
            enum class shift : std::size_t {};
            enum class min_exponent : std::int32_t {};
            enum class max_exponent : std::int32_t {};

            template <multiply m, subtract f, shift k, min_exponent e_min, max_exponent e_max>
            constexpr int compute(int e) noexcept {
                assert(std::int32_t(e_min) <= e && e <= std::int32_t(e_max));
                return int((std::int32_t(e) * std::int32_t(m) - std::int32_t(f)) >> std::size_t(k));
            }

            // For constexpr computation.
            // Returns -1 when n = 0.
            template <class UInt>
            constexpr int floor_log2(UInt n) noexcept {
                int count = -1;
                while (n != 0) {
                    ++count;
                    n >>= 1;
                }
                return count;
            }

            static constexpr int floor_log10_pow2_min_exponent = -2620;
            static constexpr int floor_log10_pow2_max_exponent = 2620;
            constexpr int floor_log10_pow2(int e) noexcept {
                using namespace log;
                return compute<multiply(315653), subtract(0), shift(20),
                               min_exponent(floor_log10_pow2_min_exponent),
                               max_exponent(floor_log10_pow2_max_exponent)>(e);
            }

            static constexpr int floor_log2_pow10_min_exponent = -1233;
            static constexpr int floor_log2_pow10_max_exponent = 1233;
            constexpr int floor_log2_pow10(int e) noexcept {
                using namespace log;
                return compute<multiply(1741647), subtract(0), shift(19),
                               min_exponent(floor_log2_pow10_min_exponent),
                               max_exponent(floor_log2_pow10_max_exponent)>(e);
            }

            static constexpr int floor_log10_pow2_minus_log10_4_over_3_min_exponent = -2985;
            static constexpr int floor_log10_pow2_minus_log10_4_over_3_max_exponent = 2936;
            constexpr int floor_log10_pow2_minus_log10_4_over_3(int e) noexcept {
                using namespace log;
                return compute<multiply(631305), subtract(261663), shift(21),
                               min_exponent(floor_log10_pow2_minus_log10_4_over_3_min_exponent),
                               max_exponent(floor_log10_pow2_minus_log10_4_over_3_max_exponent)>(e);
            }

            static constexpr int floor_log5_pow2_min_exponent = -1831;
            static constexpr int floor_log5_pow2_max_exponent = 1831;
            constexpr int floor_log5_pow2(int e) noexcept {
                using namespace log;
                return compute<multiply(225799), subtract(0), shift(19),
                               min_exponent(floor_log5_pow2_min_exponent),
                               max_exponent(floor_log5_pow2_max_exponent)>(e);
            }

            static constexpr int floor_log5_pow2_minus_log5_3_min_exponent = -3543;
            static constexpr int floor_log5_pow2_minus_log5_3_max_exponent = 2427;
            constexpr int floor_log5_pow2_minus_log5_3(int e) noexcept {
                using namespace log;
                return compute<multiply(451597), subtract(715764), shift(20),
                               min_exponent(floor_log5_pow2_minus_log5_3_min_exponent),
                               max_exponent(floor_log5_pow2_minus_log5_3_max_exponent)>(e);
            }
        }

        template <std::size_t max_blocks>
        struct fixed_point_calculator {
            static_assert(1 < max_blocks);

            // Multiply multiplier to the fractional blocks and take the resulting integer part.
            // The fractional blocks are updated.
            template <class MultiplierType>
            JKJ_FORCEINLINE static MultiplierType generate(MultiplierType multiplier,
                                                           std::uint64_t* blocks_ptr,
                                                           std::size_t number_of_blocks) noexcept {
                assert(0 < number_of_blocks && number_of_blocks <= max_blocks);

                if constexpr (max_blocks == 3) {
                    wuint::uint128 mul_result;
                    std::uint64_t carry = 0;

                    switch (number_of_blocks) {
                    case 3:
                        mul_result = wuint::umul128(blocks_ptr[2], multiplier);
                        blocks_ptr[2] = mul_result.low();
                        carry = mul_result.high();
                        [[fallthrough]];

                    case 2:
                        mul_result = wuint::umul128(blocks_ptr[1], multiplier);
                        mul_result += carry;
                        blocks_ptr[1] = mul_result.low();
                        carry = mul_result.high();
                        [[fallthrough]];

                    case 1:
                        mul_result = wuint::umul128(blocks_ptr[0], multiplier);
                        mul_result += carry;
                        blocks_ptr[0] = mul_result.low();
                        return mul_result.high();

                    default:
                        JKJ_UNRECHABLE;
                    }
                }
                else {
                    auto mul_result = wuint::umul128(blocks_ptr[number_of_blocks - 1], multiplier);
                    blocks_ptr[number_of_blocks - 1] = mul_result.low();
                    auto carry = mul_result.high();
                    for (std::size_t i = 1; i < number_of_blocks; ++i) {
                        mul_result =
                            wuint::umul128(blocks_ptr[number_of_blocks - i - 1], multiplier);
                        mul_result += carry;
                        blocks_ptr[number_of_blocks - i - 1] = mul_result.low();
                        carry = mul_result.high();
                    }

                    return MultiplierType(carry);
                }
            }

            // Multiply multiplier to the fractional blocks and discard the resulting integer part.
            // The fractional blocks are updated.
            template <class MultiplierType>
            JKJ_FORCEINLINE static void discard_upper(MultiplierType multiplier,
                                                      std::uint64_t* blocks_ptr,
                                                      std::size_t number_of_blocks) noexcept {
                assert(0 < number_of_blocks && number_of_blocks <= max_blocks);

                blocks_ptr[0] *= multiplier;
                if (number_of_blocks > 1) {
                    if constexpr (max_blocks == 3) {
                        wuint::uint128 mul_result;
                        std::uint64_t carry = 0;

                        if (number_of_blocks > 2) {
                            mul_result = wuint::umul128(multiplier, blocks_ptr[2]);
                            blocks_ptr[2] = mul_result.low();
                            carry = mul_result.high();
                        }

                        mul_result = wuint::umul128(multiplier, blocks_ptr[1]);
                        mul_result += carry;
                        blocks_ptr[1] = mul_result.low();
                        blocks_ptr[0] += mul_result.high();
                    }
                    else {
                        auto mul_result =
                            wuint::umul128(multiplier, blocks_ptr[number_of_blocks - 1]);
                        blocks_ptr[number_of_blocks - 1] = mul_result.low();
                        auto carry = mul_result.high();

                        for (std::uint8_t i = 2; i < number_of_blocks; ++i) {
                            mul_result =
                                wuint::umul128(multiplier, blocks_ptr[number_of_blocks - i]);
                            mul_result += carry;
                            blocks_ptr[number_of_blocks - i] = mul_result.low();
                            carry = mul_result.high();
                        }
                        blocks_ptr[0] += carry;
                    }
                }
            }

            // Multiply multiplier to the fractional blocks and take the resulting integer part.
            // Don't care about what happens to the fractional blocks.
            template <class MultiplierType>
            JKJ_FORCEINLINE static MultiplierType
            generate_and_discard_lower(MultiplierType multiplier, std::uint64_t* blocks_ptr,
                                       std::size_t number_of_blocks) noexcept {
                assert(0 < number_of_blocks && number_of_blocks <= max_blocks);

                if constexpr (max_blocks == 3) {
                    wuint::uint128 mul_result;
                    std::uint64_t carry = 0;

                    switch (number_of_blocks) {
                    case 3:
                        mul_result = wuint::umul128(blocks_ptr[2], multiplier);
                        carry = mul_result.high();
                        [[fallthrough]];

                    case 2:
                        mul_result = wuint::umul128(blocks_ptr[1], multiplier);
                        mul_result += carry;
                        carry = mul_result.high();
                        [[fallthrough]];

                    case 1:
                        mul_result = wuint::umul128(blocks_ptr[0], multiplier);
                        mul_result += carry;
                        return mul_result.high();

                    default:
                        JKJ_UNRECHABLE;
                    }
                }
                else {
                    auto mul_result = wuint::umul128(blocks_ptr[number_of_blocks - 1], multiplier);
                    auto carry = mul_result.high();
                    for (std::size_t i = 1; i < number_of_blocks; ++i) {
                        mul_result =
                            wuint::umul128(blocks_ptr[number_of_blocks - i - 1], multiplier);
                        mul_result += carry;
                        carry = mul_result.high();
                    }

                    return MultiplierType(carry);
                }
            }
        };

        template <class UInt, unsigned int count>
        struct fractional_part_rounding_thresholds_holder {
            UInt values[count];
            constexpr UInt operator[](unsigned int n) const noexcept { return values[n]; }
        };

        template <class UInt, unsigned int count>
        static constexpr fractional_part_rounding_thresholds_holder<UInt, count>
        generate_fractional_part_rounding_thresholds_holder() noexcept {
            constexpr std::size_t bit_width = sizeof(UInt) * 8;
            constexpr UInt msb = UInt(UInt(1) << (bit_width - 1));
            fractional_part_rounding_thresholds_holder<UInt, count> ret_value{};
            UInt divisor = 5;
            for (unsigned int i = 0; i < count; ++i) {
                ret_value.values[i] = (msb | (msb / divisor)) + 1;
                divisor *= 10;
            }
            return ret_value;
        }

        struct additional_static_data_holder {
            static constexpr char radix_100_table[] = {
                '0', '0', '0', '1', '0', '2', '0', '3', '0', '4', //
                '0', '5', '0', '6', '0', '7', '0', '8', '0', '9', //
                '1', '0', '1', '1', '1', '2', '1', '3', '1', '4', //
                '1', '5', '1', '6', '1', '7', '1', '8', '1', '9', //
                '2', '0', '2', '1', '2', '2', '2', '3', '2', '4', //
                '2', '5', '2', '6', '2', '7', '2', '8', '2', '9', //
                '3', '0', '3', '1', '3', '2', '3', '3', '3', '4', //
                '3', '5', '3', '6', '3', '7', '3', '8', '3', '9', //
                '4', '0', '4', '1', '4', '2', '4', '3', '4', '4', //
                '4', '5', '4', '6', '4', '7', '4', '8', '4', '9', //
                '5', '0', '5', '1', '5', '2', '5', '3', '5', '4', //
                '5', '5', '5', '6', '5', '7', '5', '8', '5', '9', //
                '6', '0', '6', '1', '6', '2', '6', '3', '6', '4', //
                '6', '5', '6', '6', '6', '7', '6', '8', '6', '9', //
                '7', '0', '7', '1', '7', '2', '7', '3', '7', '4', //
                '7', '5', '7', '6', '7', '7', '7', '8', '7', '9', //
                '8', '0', '8', '1', '8', '2', '8', '3', '8', '4', //
                '8', '5', '8', '6', '8', '7', '8', '8', '8', '9', //
                '9', '0', '9', '1', '9', '2', '9', '3', '9', '4', //
                '9', '5', '9', '6', '9', '7', '9', '8', '9', '9'  //
            };

            static constexpr auto fractional_part_rounding_thresholds32 =
                generate_fractional_part_rounding_thresholds_holder<std::uint32_t, 8>();
            static constexpr auto fractional_part_rounding_thresholds64 =
                generate_fractional_part_rounding_thresholds_holder<std::uint64_t, 18>();
        };

        struct compute_mul_result {
            std::uint64_t result;
            bool is_integer;
        };

        // Load the necessary bits into blocks_ptr and then return the number of cache blocks
        // loaded. The most significant block is loaded into blocks_ptr[0].
        template <class ExtendedCache, bool zero_out,
                  class CacheBlockType = std::decay_t<decltype(ExtendedCache::cache[0])>>
        JKJ_FORCEINLINE std::uint8_t load_extended_cache(CacheBlockType* blocks_ptr, int e, int k,
                                                         std::uint32_t multiplier_index) noexcept {
            if constexpr (zero_out) {
                std::memset(blocks_ptr, 0,
                            sizeof(CacheBlockType) * ExtendedCache::max_cache_blocks);
            }

            auto const mul_info = ExtendedCache::multiplier_index_info_table[multiplier_index];

            std::uint8_t cache_block_count = [&] {
                if constexpr (ExtendedCache::constant_block_count) {
                    return std::uint8_t(ExtendedCache::max_cache_blocks);
                }
                else {
                    auto const cache_block_count_index =
                        mul_info.cache_block_count_index_offset +
                        std::uint32_t(e - ExtendedCache::e_min) / ExtendedCache::collapse_factor -
                        ExtendedCache::cache_block_count_offset_base;

                    if constexpr (ExtendedCache::max_cache_blocks < 3) {
                        // 1-bit packing.
                        return std::uint8_t(
                                   (ExtendedCache::cache_block_counts[cache_block_count_index /
                                                                      8] >>
                                    (cache_block_count_index % 8)) &
                                   0x1) +
                               1;
                    }
                    else if constexpr (ExtendedCache::max_cache_blocks < 4) {
                        // 2-bit packing.
                        return std::uint8_t(
                            (ExtendedCache::cache_block_counts[cache_block_count_index / 4] >>
                             (2 * (cache_block_count_index % 4))) &
                            0x3);
                    }
                    else {
                        // 4-bit packing.
                        return std::uint8_t(
                            (ExtendedCache::cache_block_counts[cache_block_count_index / 2] >>
                             (4 * (cache_block_count_index % 2))) &
                            0xf);
                    }
                }
            }();

            std::uint32_t number_of_leading_zero_blocks;
            std::uint32_t first_cache_block_index;
            std::uint32_t bit_offset;
            std::uint32_t excessive_bits_to_left;
            std::uint32_t excessive_bits_to_right;

            // The request window starting/ending positions.
            auto start_bit_index = int(mul_info.cache_bit_index_offset) + e -
                                   ExtendedCache::cache_bit_index_offset_base;
            auto end_bit_index =
                start_bit_index + cache_block_count * int(ExtendedCache::cache_bits_unit);

            // The source window starting/ending positions.
            auto const src_start_bit_index = int(mul_info.first_cache_bit_index);
            auto const src_end_bit_index =
                int(ExtendedCache::multiplier_index_info_table[multiplier_index + 1]
                        .first_cache_bit_index);

            // If the request window goes further than the left boundary of the source window,
            if (start_bit_index < src_start_bit_index) {
                number_of_leading_zero_blocks =
                    std::uint32_t(src_start_bit_index - start_bit_index) /
                    std::uint32_t(ExtendedCache::cache_bits_unit);
                excessive_bits_to_left = std::uint32_t(src_start_bit_index - start_bit_index) %
                                         std::uint32_t(ExtendedCache::cache_bits_unit);

                if constexpr (!zero_out) {
                    std::memset(blocks_ptr, 0,
                                number_of_leading_zero_blocks * sizeof(CacheBlockType));
                }
                start_bit_index +=
                    number_of_leading_zero_blocks * int(ExtendedCache::cache_bits_unit);

                auto const src_start_block_index =
                    int(std::uint32_t(src_start_bit_index) /
                        std::uint32_t(ExtendedCache::cache_bits_unit));
                auto const src_start_block_bit_index =
                    src_start_block_index * int(ExtendedCache::cache_bits_unit);

                first_cache_block_index = src_start_block_index;

                if (start_bit_index < src_start_block_bit_index) {
                    auto shift_amount = src_start_block_bit_index - start_bit_index;
                    assert(shift_amount >= 0 && shift_amount < int(ExtendedCache::cache_bits_unit));

                    blocks_ptr[number_of_leading_zero_blocks] =
                        ((ExtendedCache::cache[src_start_block_index] >> shift_amount) &
                         (CacheBlockType(CacheBlockType(0) - CacheBlockType(1)) >>
                          excessive_bits_to_left));

                    ++number_of_leading_zero_blocks;
                    bit_offset = std::uint32_t(int(ExtendedCache::cache_bits_unit) - shift_amount);
                    excessive_bits_to_left = 0;
                }
                else {
                    bit_offset = std::uint32_t(start_bit_index - src_start_block_bit_index);
                }
            }
            else {
                number_of_leading_zero_blocks = 0;
                first_cache_block_index =
                    std::uint32_t(start_bit_index) / std::uint32_t(ExtendedCache::cache_bits_unit);
                bit_offset =
                    std::uint32_t(start_bit_index) % std::uint32_t(ExtendedCache::cache_bits_unit);
                excessive_bits_to_left = 0;
            }

            // If the request window goes further than the right boundary of the source window,
            if (end_bit_index > src_end_bit_index) {
                auto const number_of_trailing_zero_blocks =
                    std::uint32_t(end_bit_index - src_end_bit_index) /
                    std::uint32_t(ExtendedCache::cache_bits_unit);
                excessive_bits_to_right = std::uint32_t(end_bit_index - src_end_bit_index) %
                                          std::uint32_t(ExtendedCache::cache_bits_unit);

                cache_block_count -= number_of_trailing_zero_blocks;
            }
            else {
                excessive_bits_to_right = 0;
            }

            // Load blocks.
            auto const number_of_blocks_to_load = cache_block_count - number_of_leading_zero_blocks;
            auto* const dst_ptr = blocks_ptr + number_of_leading_zero_blocks;
            if (bit_offset == 0) {
                if constexpr (ExtendedCache::max_cache_blocks == 3) {
                    switch (number_of_blocks_to_load) {
                    case 3:
                        std::memcpy(dst_ptr, ExtendedCache::cache + first_cache_block_index,
                                    3 * sizeof(CacheBlockType));
                        break;

                    case 2:
                        std::memcpy(dst_ptr, ExtendedCache::cache + first_cache_block_index,
                                    2 * sizeof(CacheBlockType));
                        break;

                    case 1:
                        std::memcpy(dst_ptr, ExtendedCache::cache + first_cache_block_index,
                                    1 * sizeof(CacheBlockType));
                        break;

                    case 0:
                        break;

                    default:
                        JKJ_UNRECHABLE;
                    }
                }
                else {
                    std::memcpy(dst_ptr, ExtendedCache::cache + first_cache_block_index,
                                number_of_blocks_to_load * sizeof(CacheBlockType));
                }
            }
            else {
                if constexpr (ExtendedCache::max_cache_blocks == 3) {
                    switch (number_of_blocks_to_load) {
                    case 3:
                        *(dst_ptr + 2) =
                            (ExtendedCache::cache[first_cache_block_index + 2] << bit_offset) |
                            (ExtendedCache::cache[first_cache_block_index + 3] >>
                             (ExtendedCache::cache_bits_unit - bit_offset));
                        [[fallthrough]];

                    case 2:
                        *(dst_ptr + 1) =
                            (ExtendedCache::cache[first_cache_block_index + 1] << bit_offset) |
                            (ExtendedCache::cache[first_cache_block_index + 2] >>
                             (ExtendedCache::cache_bits_unit - bit_offset));
                        [[fallthrough]];

                    case 1:
                        *dst_ptr = (ExtendedCache::cache[first_cache_block_index] << bit_offset) |
                                   (ExtendedCache::cache[first_cache_block_index + 1] >>
                                    (ExtendedCache::cache_bits_unit - bit_offset));

                    case 0:
                        break;

                    default:
                        JKJ_UNRECHABLE;
                    }
                }
                else {
                    for (std::uint8_t i = 0; i < number_of_blocks_to_load; ++i) {
                        *(dst_ptr + i) =
                            (ExtendedCache::cache[first_cache_block_index + i] << bit_offset) |
                            (ExtendedCache::cache[first_cache_block_index + i + 1] >>
                             (ExtendedCache::cache_bits_unit - bit_offset));
                    }
                }
            }
            
            // Remove possible flooding bits from adjacent entries.
            *dst_ptr &=
                (CacheBlockType(CacheBlockType(0) - CacheBlockType(1)) >> excessive_bits_to_left);

            blocks_ptr[cache_block_count - 1] &=
                (CacheBlockType(CacheBlockType(0) - CacheBlockType(1)) << excessive_bits_to_right);

            // To compute ceil(2^Q * x / D), we need to check if
            // 2^Q * x / D = 2^(Q + e + k - eta - 1) * 5^(k - eta) is an integer or not.
            if (k < ExtendedCache::segment_length ||
                e + k + cache_block_count * int(ExtendedCache::cache_bits_unit) -
                        excessive_bits_to_right <
                    ExtendedCache::segment_length + 1) {
                blocks_ptr[cache_block_count - 1] += (CacheBlockType(1) << excessive_bits_to_right);
                assert(blocks_ptr[cache_block_count - 1] != 0);
            }

            return cache_block_count;
        }

        template <bool constant_block_count, std::uint8_t max_cache_blocks>
        struct cache_block_count_t;

        template <std::uint8_t max_cache_blocks>
        struct cache_block_count_t<false, max_cache_blocks> {
            std::uint8_t value;
            operator std::uint8_t() const noexcept { return value; }
            cache_block_count_t& operator=(std::uint8_t new_value) noexcept {
                value = new_value;
                return *this;
            }
        };

        template <std::uint8_t max_cache_blocks>
        struct cache_block_count_t<true, max_cache_blocks> {
            static constexpr std::uint8_t value = max_cache_blocks;
            operator std::uint8_t() const noexcept { return value; }
            cache_block_count_t& operator=(std::uint8_t) noexcept {
                // Don't do anything.
                return *this;
            }
        };

        template <unsigned int n>
        inline constexpr auto uconst = std::integral_constant<unsigned int, n>{};

        template <unsigned int digits, bool dummy = (digits <= 9)>
        struct uint_with_known_number_of_digits;

        template <unsigned int digits_>
        struct uint_with_known_number_of_digits<digits_, true> {
            static constexpr auto digits = digits_;
            std::uint32_t value;
        };
        template <unsigned int digits_>
        struct uint_with_known_number_of_digits<digits_, false> {
            static constexpr auto digits = digits_;
            std::uint64_t value;
        };

        template <class HasFurtherDigits, class... Args>
        JKJ_FORCEINLINE bool check_rounding_condition_inside_subsegment(
            std::uint32_t current_digits, std::uint32_t fractional_part,
            int remaining_digits_in_the_current_subsegment, HasFurtherDigits has_further_digits,
            Args... args) noexcept {
            if (fractional_part >=
                additional_static_data_holder::fractional_part_rounding_thresholds32
                    [remaining_digits_in_the_current_subsegment - 1]) {
                return true;
            }

            if constexpr (std::is_same_v<decltype(has_further_digits), bool>) {
                return ((fractional_part >> 31) & ((current_digits & 1) | has_further_digits)) != 0;
            }
            else {
                return fractional_part >= 0x8000'0000 &&
                       ((current_digits & 1) != 0 || has_further_digits(args...));
            }
        }

        template <class HasFurtherDigits, class... Args>
        JKJ_FORCEINLINE bool
        check_rounding_condition_with_next_bit(std::uint32_t current_digits, bool next_bit,
                                               HasFurtherDigits has_further_digits,
                                               Args... args) noexcept {
            if (!next_bit) {
                return false;
            }

            if constexpr (std::is_same_v<decltype(has_further_digits), bool>) {
                return ((current_digits & 1) | has_further_digits) != 0;
            }
            else {
                return (current_digits & 1) != 0 || has_further_digits(args...);
            }
        }

        template <class UintWithKnownDigits, class HasFurtherDigits, class... Args>
        JKJ_FORCEINLINE bool check_rounding_condition_subsegment_boundary_with_next_subsegment(
            std::uint32_t current_digits, UintWithKnownDigits next_subsegment,
            HasFurtherDigits has_further_digits, Args... args) noexcept {
            if (next_subsegment.value > power_of_10<decltype(next_subsegment)::digits> / 2) {
                return true;
            }

            if constexpr (std::is_same_v<decltype(has_further_digits), bool>) {
                return next_subsegment.value ==
                           power_of_10<decltype(next_subsegment)::digits> / 2 &&
                       ((current_digits & 1) | has_further_digits) != 0;
            }
            else {
                return next_subsegment.value ==
                           power_of_10<decltype(next_subsegment)::digits> / 2 &&
                       ((current_digits & 1) != 0 || has_further_digits(args...));
            }
        }

        namespace has_further_digits_impl {
            template <int k_right_threshold, int additional_neg_exp_of_2>
            bool no_neg_k_can_be_integer(int k, int exp2_base) noexcept {
                return k < k_right_threshold || exp2_base + k < additional_neg_exp_of_2;
            }

            template <int k_left_threshold, int k_right_threshold, int additional_neg_exp_of_2,
                      int min_neg_exp_of_5, class SignificandType>
            bool only_one_neg_k_can_be_integer(int k, int exp2_base,
                                               SignificandType significand) noexcept {
                // Supposed to be k - additional_neg_exp_of_5_v < -min_neg_exp_of_5 || ...
                if (k < k_left_threshold || exp2_base + k < additional_neg_exp_of_2) {
                    return true;
                }
                // Supposed to be k - additional_neg_exp_of_5_v >= 0.
                if (k >= k_right_threshold) {
                    return false;
                }

                constexpr std::uint64_t mod_inv =
                    compute_power(0xcccccccccccccccd, (unsigned int)(min_neg_exp_of_5));
                constexpr std::uint64_t max_quot =
                    0xffffffffffffffff /
                    compute_power(std::uint64_t(5), (unsigned int)(min_neg_exp_of_5));

                return (significand * mod_inv) > max_quot;
            }

            template <int k_left_threshold, int k_middle_threshold, int k_right_threshold,
                      int additional_neg_exp_of_2, int min_neg_exp_of_5, int segment_length,
                      class SignificandType>
            bool only_two_neg_k_can_be_integer(int k, int exp2_base,
                                               SignificandType significand) noexcept {
                // Supposed to be k - additional_neg_exp_of_5_v < -min_neg_exp_of_5 - segment_length
                // || ...
                if (k < k_left_threshold || exp2_base + k < additional_neg_exp_of_2) {
                    return true;
                }
                // Supposed to be k - additional_neg_exp_of_5_v >= 0.
                if (k >= k_right_threshold) {
                    return false;
                }

                if (k >= k_middle_threshold) {
                    constexpr std::uint64_t mod_inv =
                        compute_power(0xcccccccccccccccd, (unsigned int)(min_neg_exp_of_5));
                    constexpr std::uint64_t max_quot =
                        0xffffffffffffffff /
                        compute_power(std::uint64_t(5), (unsigned int)(min_neg_exp_of_5));

                    return (significand * mod_inv) > max_quot;
                }
                else {
                    constexpr std::uint64_t mod_inv = compute_power(
                        0xcccccccccccccccd, (unsigned int)(min_neg_exp_of_5 + segment_length));
                    constexpr std::uint64_t max_quot =
                        0xffffffffffffffff /
                        compute_power(std::uint64_t(5),
                                      (unsigned int)(min_neg_exp_of_5 + segment_length));

                    return (significand * mod_inv) > max_quot;
                }
            }
        }

        inline void print_1_digit(std::uint32_t n, char* buffer) noexcept {
            if constexpr ('1' == '0' + 1 && '2' == '0' + 2 && '3' == '0' + 3 && '4' == '0' + 4 &&
                          '5' == '0' + 5 && '6' == '0' + 6 && '7' == '0' + 7 && '8' == '0' + 8 &&
                          '9' == '0' + 9) {
                if constexpr (('0' & 0xf) == 0) {
                    *buffer = char('0' | n);
                }
                else {
                    *buffer = char('0' + n);
                }
            }
            else {
                std::memcpy(buffer, additional_static_data_holder::radix_100_table + n * 2 + 1, 1);
            }
        }

        inline void print_2_digits(std::uint32_t n, char* buffer) noexcept {
            std::memcpy(buffer, additional_static_data_holder::radix_100_table + n * 2, 2);
        }

        inline void print_6_digits(std::uint32_t n, char* buffer) noexcept {
            // 429497 = ceil(2^32/10^4)
            auto prod = (n * std::uint64_t(429497)) + 1;
            print_2_digits(std::uint32_t(prod >> 32), buffer);
            for (int i = 0; i < 2; ++i) {
                prod = std::uint32_t(prod) * std::uint64_t(100);
                print_2_digits(std::uint32_t(prod >> 32), buffer + 2 + i * 2);
            }
        }

        inline void print_7_digits(std::uint32_t n, char* buffer) noexcept {
            // 17592187 = ceil(2^(32+12)/10^6)
            auto prod = ((n * std::uint64_t(17592187)) >> 12) + 1;
            print_1_digit(std::uint32_t(prod >> 32), buffer);
            for (int i = 0; i < 3; ++i) {
                prod = std::uint32_t(prod) * std::uint64_t(100);
                print_2_digits(std::uint32_t(prod >> 32), buffer + 1 + i * 2);
            }
        }

        inline void print_8_digits(std::uint32_t n, char* buffer) noexcept {
            // 140737489 = ceil(2^(32+15)/10^6)
            auto prod = ((n * std::uint64_t(140737489)) >> 15) + 1;
            print_2_digits(std::uint32_t(prod >> 32), buffer);
            for (int i = 0; i < 3; ++i) {
                prod = std::uint32_t(prod) * std::uint64_t(100);
                print_2_digits(std::uint32_t(prod >> 32), buffer + 2 + i * 2);
            }
        }

        inline void print_9_digits(std::uint32_t n, char* buffer) noexcept {
            // 1441151881 = ceil(2^(32+25)/10^8)
            auto prod = ((n * std::uint64_t(1441151881)) >> 25) + 1;
            print_1_digit(std::uint32_t(prod >> 32), buffer);
            for (int i = 0; i < 4; ++i) {
                prod = std::uint32_t(prod) * std::uint64_t(100);
                print_2_digits(std::uint32_t(prod >> 32), buffer + 1 + i * 2);
            }
        }

        template <class FloatFormat>
        struct main_cache_holder;

        template <>
        struct main_cache_holder<jkj::floff::ieee754_binary64> {
            using cache_entry_type = jkj::floff::detail::wuint::uint128;
            static constexpr int cache_bits = 128;
            static constexpr int min_k = -292;
            static constexpr int max_k = 326;
            static constexpr cache_entry_type cache[] = {
                {0xff77b1fcbebcdc4f, 0x25e8e89c13bb0f7b}, {0x9faacf3df73609b1, 0x77b191618c54e9ad},
                {0xc795830d75038c1d, 0xd59df5b9ef6a2418}, {0xf97ae3d0d2446f25, 0x4b0573286b44ad1e},
                {0x9becce62836ac577, 0x4ee367f9430aec33}, {0xc2e801fb244576d5, 0x229c41f793cda740},
                {0xf3a20279ed56d48a, 0x6b43527578c11110}, {0x9845418c345644d6, 0x830a13896b78aaaa},
                {0xbe5691ef416bd60c, 0x23cc986bc656d554}, {0xedec366b11c6cb8f, 0x2cbfbe86b7ec8aa9},
                {0x94b3a202eb1c3f39, 0x7bf7d71432f3d6aa}, {0xb9e08a83a5e34f07, 0xdaf5ccd93fb0cc54},
                {0xe858ad248f5c22c9, 0xd1b3400f8f9cff69}, {0x91376c36d99995be, 0x23100809b9c21fa2},
                {0xb58547448ffffb2d, 0xabd40a0c2832a78b}, {0xe2e69915b3fff9f9, 0x16c90c8f323f516d},
                {0x8dd01fad907ffc3b, 0xae3da7d97f6792e4}, {0xb1442798f49ffb4a, 0x99cd11cfdf41779d},
                {0xdd95317f31c7fa1d, 0x40405643d711d584}, {0x8a7d3eef7f1cfc52, 0x482835ea666b2573},
                {0xad1c8eab5ee43b66, 0xda3243650005eed0}, {0xd863b256369d4a40, 0x90bed43e40076a83},
                {0x873e4f75e2224e68, 0x5a7744a6e804a292}, {0xa90de3535aaae202, 0x711515d0a205cb37},
                {0xd3515c2831559a83, 0x0d5a5b44ca873e04}, {0x8412d9991ed58091, 0xe858790afe9486c3},
                {0xa5178fff668ae0b6, 0x626e974dbe39a873}, {0xce5d73ff402d98e3, 0xfb0a3d212dc81290},
                {0x80fa687f881c7f8e, 0x7ce66634bc9d0b9a}, {0xa139029f6a239f72, 0x1c1fffc1ebc44e81},
                {0xc987434744ac874e, 0xa327ffb266b56221}, {0xfbe9141915d7a922, 0x4bf1ff9f0062baa9},
                {0x9d71ac8fada6c9b5, 0x6f773fc3603db4aa}, {0xc4ce17b399107c22, 0xcb550fb4384d21d4},
                {0xf6019da07f549b2b, 0x7e2a53a146606a49}, {0x99c102844f94e0fb, 0x2eda7444cbfc426e},
                {0xc0314325637a1939, 0xfa911155fefb5309}, {0xf03d93eebc589f88, 0x793555ab7eba27cb},
                {0x96267c7535b763b5, 0x4bc1558b2f3458df}, {0xbbb01b9283253ca2, 0x9eb1aaedfb016f17},
                {0xea9c227723ee8bcb, 0x465e15a979c1cadd}, {0x92a1958a7675175f, 0x0bfacd89ec191eca},
                {0xb749faed14125d36, 0xcef980ec671f667c}, {0xe51c79a85916f484, 0x82b7e12780e7401b},
                {0x8f31cc0937ae58d2, 0xd1b2ecb8b0908811}, {0xb2fe3f0b8599ef07, 0x861fa7e6dcb4aa16},
                {0xdfbdcece67006ac9, 0x67a791e093e1d49b}, {0x8bd6a141006042bd, 0xe0c8bb2c5c6d24e1},
                {0xaecc49914078536d, 0x58fae9f773886e19}, {0xda7f5bf590966848, 0xaf39a475506a899f},
                {0x888f99797a5e012d, 0x6d8406c952429604}, {0xaab37fd7d8f58178, 0xc8e5087ba6d33b84},
                {0xd5605fcdcf32e1d6, 0xfb1e4a9a90880a65}, {0x855c3be0a17fcd26, 0x5cf2eea09a550680},
                {0xa6b34ad8c9dfc06f, 0xf42faa48c0ea481f}, {0xd0601d8efc57b08b, 0xf13b94daf124da27},
                {0x823c12795db6ce57, 0x76c53d08d6b70859}, {0xa2cb1717b52481ed, 0x54768c4b0c64ca6f},
                {0xcb7ddcdda26da268, 0xa9942f5dcf7dfd0a}, {0xfe5d54150b090b02, 0xd3f93b35435d7c4d},
                {0x9efa548d26e5a6e1, 0xc47bc5014a1a6db0}, {0xc6b8e9b0709f109a, 0x359ab6419ca1091c},
                {0xf867241c8cc6d4c0, 0xc30163d203c94b63}, {0x9b407691d7fc44f8, 0x79e0de63425dcf1e},
                {0xc21094364dfb5636, 0x985915fc12f542e5}, {0xf294b943e17a2bc4, 0x3e6f5b7b17b2939e},
                {0x979cf3ca6cec5b5a, 0xa705992ceecf9c43}, {0xbd8430bd08277231, 0x50c6ff782a838354},
                {0xece53cec4a314ebd, 0xa4f8bf5635246429}, {0x940f4613ae5ed136, 0x871b7795e136be9a},
                {0xb913179899f68584, 0x28e2557b59846e40}, {0xe757dd7ec07426e5, 0x331aeada2fe589d0},
                {0x9096ea6f3848984f, 0x3ff0d2c85def7622}, {0xb4bca50b065abe63, 0x0fed077a756b53aa},
                {0xe1ebce4dc7f16dfb, 0xd3e8495912c62895}, {0x8d3360f09cf6e4bd, 0x64712dd7abbbd95d},
                {0xb080392cc4349dec, 0xbd8d794d96aacfb4}, {0xdca04777f541c567, 0xecf0d7a0fc5583a1},
                {0x89e42caaf9491b60, 0xf41686c49db57245}, {0xac5d37d5b79b6239, 0x311c2875c522ced6},
                {0xd77485cb25823ac7, 0x7d633293366b828c}, {0x86a8d39ef77164bc, 0xae5dff9c02033198},
                {0xa8530886b54dbdeb, 0xd9f57f830283fdfd}, {0xd267caa862a12d66, 0xd072df63c324fd7c},
                {0x8380dea93da4bc60, 0x4247cb9e59f71e6e}, {0xa46116538d0deb78, 0x52d9be85f074e609},
                {0xcd795be870516656, 0x67902e276c921f8c}, {0x806bd9714632dff6, 0x00ba1cd8a3db53b7},
                {0xa086cfcd97bf97f3, 0x80e8a40eccd228a5}, {0xc8a883c0fdaf7df0, 0x6122cd128006b2ce},
                {0xfad2a4b13d1b5d6c, 0x796b805720085f82}, {0x9cc3a6eec6311a63, 0xcbe3303674053bb1},
                {0xc3f490aa77bd60fc, 0xbedbfc4411068a9d}, {0xf4f1b4d515acb93b, 0xee92fb5515482d45},
                {0x991711052d8bf3c5, 0x751bdd152d4d1c4b}, {0xbf5cd54678eef0b6, 0xd262d45a78a0635e},
                {0xef340a98172aace4, 0x86fb897116c87c35}, {0x9580869f0e7aac0e, 0xd45d35e6ae3d4da1},
                {0xbae0a846d2195712, 0x8974836059cca10a}, {0xe998d258869facd7, 0x2bd1a438703fc94c},
                {0x91ff83775423cc06, 0x7b6306a34627ddd0}, {0xb67f6455292cbf08, 0x1a3bc84c17b1d543},
                {0xe41f3d6a7377eeca, 0x20caba5f1d9e4a94}, {0x8e938662882af53e, 0x547eb47b7282ee9d},
                {0xb23867fb2a35b28d, 0xe99e619a4f23aa44}, {0xdec681f9f4c31f31, 0x6405fa00e2ec94d5},
                {0x8b3c113c38f9f37e, 0xde83bc408dd3dd05}, {0xae0b158b4738705e, 0x9624ab50b148d446},
                {0xd98ddaee19068c76, 0x3badd624dd9b0958}, {0x87f8a8d4cfa417c9, 0xe54ca5d70a80e5d7},
                {0xa9f6d30a038d1dbc, 0x5e9fcf4ccd211f4d}, {0xd47487cc8470652b, 0x7647c32000696720},
                {0x84c8d4dfd2c63f3b, 0x29ecd9f40041e074}, {0xa5fb0a17c777cf09, 0xf468107100525891},
                {0xcf79cc9db955c2cc, 0x7182148d4066eeb5}, {0x81ac1fe293d599bf, 0xc6f14cd848405531},
                {0xa21727db38cb002f, 0xb8ada00e5a506a7d}, {0xca9cf1d206fdc03b, 0xa6d90811f0e4851d},
                {0xfd442e4688bd304a, 0x908f4a166d1da664}, {0x9e4a9cec15763e2e, 0x9a598e4e043287ff},
                {0xc5dd44271ad3cdba, 0x40eff1e1853f29fe}, {0xf7549530e188c128, 0xd12bee59e68ef47d},
                {0x9a94dd3e8cf578b9, 0x82bb74f8301958cf}, {0xc13a148e3032d6e7, 0xe36a52363c1faf02},
                {0xf18899b1bc3f8ca1, 0xdc44e6c3cb279ac2}, {0x96f5600f15a7b7e5, 0x29ab103a5ef8c0ba},
                {0xbcb2b812db11a5de, 0x7415d448f6b6f0e8}, {0xebdf661791d60f56, 0x111b495b3464ad22},
                {0x936b9fcebb25c995, 0xcab10dd900beec35}, {0xb84687c269ef3bfb, 0x3d5d514f40eea743},
                {0xe65829b3046b0afa, 0x0cb4a5a3112a5113}, {0x8ff71a0fe2c2e6dc, 0x47f0e785eaba72ac},
                {0xb3f4e093db73a093, 0x59ed216765690f57}, {0xe0f218b8d25088b8, 0x306869c13ec3532d},
                {0x8c974f7383725573, 0x1e414218c73a13fc}, {0xafbd2350644eeacf, 0xe5d1929ef90898fb},
                {0xdbac6c247d62a583, 0xdf45f746b74abf3a}, {0x894bc396ce5da772, 0x6b8bba8c328eb784},
                {0xab9eb47c81f5114f, 0x066ea92f3f326565}, {0xd686619ba27255a2, 0xc80a537b0efefebe},
                {0x8613fd0145877585, 0xbd06742ce95f5f37}, {0xa798fc4196e952e7, 0x2c48113823b73705},
                {0xd17f3b51fca3a7a0, 0xf75a15862ca504c6}, {0x82ef85133de648c4, 0x9a984d73dbe722fc},
                {0xa3ab66580d5fdaf5, 0xc13e60d0d2e0ebbb}, {0xcc963fee10b7d1b3, 0x318df905079926a9},
                {0xffbbcfe994e5c61f, 0xfdf17746497f7053}, {0x9fd561f1fd0f9bd3, 0xfeb6ea8bedefa634},
                {0xc7caba6e7c5382c8, 0xfe64a52ee96b8fc1}, {0xf9bd690a1b68637b, 0x3dfdce7aa3c673b1},
                {0x9c1661a651213e2d, 0x06bea10ca65c084f}, {0xc31bfa0fe5698db8, 0x486e494fcff30a63},
                {0xf3e2f893dec3f126, 0x5a89dba3c3efccfb}, {0x986ddb5c6b3a76b7, 0xf89629465a75e01d},
                {0xbe89523386091465, 0xf6bbb397f1135824}, {0xee2ba6c0678b597f, 0x746aa07ded582e2d},
                {0x94db483840b717ef, 0xa8c2a44eb4571cdd}, {0xba121a4650e4ddeb, 0x92f34d62616ce414},
                {0xe896a0d7e51e1566, 0x77b020baf9c81d18}, {0x915e2486ef32cd60, 0x0ace1474dc1d122f},
                {0xb5b5ada8aaff80b8, 0x0d819992132456bb}, {0xe3231912d5bf60e6, 0x10e1fff697ed6c6a},
                {0x8df5efabc5979c8f, 0xca8d3ffa1ef463c2}, {0xb1736b96b6fd83b3, 0xbd308ff8a6b17cb3},
                {0xddd0467c64bce4a0, 0xac7cb3f6d05ddbdf}, {0x8aa22c0dbef60ee4, 0x6bcdf07a423aa96c},
                {0xad4ab7112eb3929d, 0x86c16c98d2c953c7}, {0xd89d64d57a607744, 0xe871c7bf077ba8b8},
                {0x87625f056c7c4a8b, 0x11471cd764ad4973}, {0xa93af6c6c79b5d2d, 0xd598e40d3dd89bd0},
                {0xd389b47879823479, 0x4aff1d108d4ec2c4}, {0x843610cb4bf160cb, 0xcedf722a585139bb},
                {0xa54394fe1eedb8fe, 0xc2974eb4ee658829}, {0xce947a3da6a9273e, 0x733d226229feea33},
                {0x811ccc668829b887, 0x0806357d5a3f5260}, {0xa163ff802a3426a8, 0xca07c2dcb0cf26f8},
                {0xc9bcff6034c13052, 0xfc89b393dd02f0b6}, {0xfc2c3f3841f17c67, 0xbbac2078d443ace3},
                {0x9d9ba7832936edc0, 0xd54b944b84aa4c0e}, {0xc5029163f384a931, 0x0a9e795e65d4df12},
                {0xf64335bcf065d37d, 0x4d4617b5ff4a16d6}, {0x99ea0196163fa42e, 0x504bced1bf8e4e46},
                {0xc06481fb9bcf8d39, 0xe45ec2862f71e1d7}, {0xf07da27a82c37088, 0x5d767327bb4e5a4d},
                {0x964e858c91ba2655, 0x3a6a07f8d510f870}, {0xbbe226efb628afea, 0x890489f70a55368c},
                {0xeadab0aba3b2dbe5, 0x2b45ac74ccea842f}, {0x92c8ae6b464fc96f, 0x3b0b8bc90012929e},
                {0xb77ada0617e3bbcb, 0x09ce6ebb40173745}, {0xe55990879ddcaabd, 0xcc420a6a101d0516},
                {0x8f57fa54c2a9eab6, 0x9fa946824a12232e}, {0xb32df8e9f3546564, 0x47939822dc96abfa},
                {0xdff9772470297ebd, 0x59787e2b93bc56f8}, {0x8bfbea76c619ef36, 0x57eb4edb3c55b65b},
                {0xaefae51477a06b03, 0xede622920b6b23f2}, {0xdab99e59958885c4, 0xe95fab368e45ecee},
                {0x88b402f7fd75539b, 0x11dbcb0218ebb415}, {0xaae103b5fcd2a881, 0xd652bdc29f26a11a},
                {0xd59944a37c0752a2, 0x4be76d3346f04960}, {0x857fcae62d8493a5, 0x6f70a4400c562ddc},
                {0xa6dfbd9fb8e5b88e, 0xcb4ccd500f6bb953}, {0xd097ad07a71f26b2, 0x7e2000a41346a7a8},
                {0x825ecc24c873782f, 0x8ed400668c0c28c9}, {0xa2f67f2dfa90563b, 0x728900802f0f32fb},
                {0xcbb41ef979346bca, 0x4f2b40a03ad2ffba}, {0xfea126b7d78186bc, 0xe2f610c84987bfa9},
                {0x9f24b832e6b0f436, 0x0dd9ca7d2df4d7ca}, {0xc6ede63fa05d3143, 0x91503d1c79720dbc},
                {0xf8a95fcf88747d94, 0x75a44c6397ce912b}, {0x9b69dbe1b548ce7c, 0xc986afbe3ee11abb},
                {0xc24452da229b021b, 0xfbe85badce996169}, {0xf2d56790ab41c2a2, 0xfae27299423fb9c4},
                {0x97c560ba6b0919a5, 0xdccd879fc967d41b}, {0xbdb6b8e905cb600f, 0x5400e987bbc1c921},
                {0xed246723473e3813, 0x290123e9aab23b69}, {0x9436c0760c86e30b, 0xf9a0b6720aaf6522},
                {0xb94470938fa89bce, 0xf808e40e8d5b3e6a}, {0xe7958cb87392c2c2, 0xb60b1d1230b20e05},
                {0x90bd77f3483bb9b9, 0xb1c6f22b5e6f48c3}, {0xb4ecd5f01a4aa828, 0x1e38aeb6360b1af4},
                {0xe2280b6c20dd5232, 0x25c6da63c38de1b1}, {0x8d590723948a535f, 0x579c487e5a38ad0f},
                {0xb0af48ec79ace837, 0x2d835a9df0c6d852}, {0xdcdb1b2798182244, 0xf8e431456cf88e66},
                {0x8a08f0f8bf0f156b, 0x1b8e9ecb641b5900}, {0xac8b2d36eed2dac5, 0xe272467e3d222f40},
                {0xd7adf884aa879177, 0x5b0ed81dcc6abb10}, {0x86ccbb52ea94baea, 0x98e947129fc2b4ea},
                {0xa87fea27a539e9a5, 0x3f2398d747b36225}, {0xd29fe4b18e88640e, 0x8eec7f0d19a03aae},
                {0x83a3eeeef9153e89, 0x1953cf68300424ad}, {0xa48ceaaab75a8e2b, 0x5fa8c3423c052dd8},
                {0xcdb02555653131b6, 0x3792f412cb06794e}, {0x808e17555f3ebf11, 0xe2bbd88bbee40bd1},
                {0xa0b19d2ab70e6ed6, 0x5b6aceaeae9d0ec5}, {0xc8de047564d20a8b, 0xf245825a5a445276},
                {0xfb158592be068d2e, 0xeed6e2f0f0d56713}, {0x9ced737bb6c4183d, 0x55464dd69685606c},
                {0xc428d05aa4751e4c, 0xaa97e14c3c26b887}, {0xf53304714d9265df, 0xd53dd99f4b3066a9},
                {0x993fe2c6d07b7fab, 0xe546a8038efe402a}, {0xbf8fdb78849a5f96, 0xde98520472bdd034},
                {0xef73d256a5c0f77c, 0x963e66858f6d4441}, {0x95a8637627989aad, 0xdde7001379a44aa9},
                {0xbb127c53b17ec159, 0x5560c018580d5d53}, {0xe9d71b689dde71af, 0xaab8f01e6e10b4a7},
                {0x9226712162ab070d, 0xcab3961304ca70e9}, {0xb6b00d69bb55c8d1, 0x3d607b97c5fd0d23},
                {0xe45c10c42a2b3b05, 0x8cb89a7db77c506b}, {0x8eb98a7a9a5b04e3, 0x77f3608e92adb243},
                {0xb267ed1940f1c61c, 0x55f038b237591ed4}, {0xdf01e85f912e37a3, 0x6b6c46dec52f6689},
                {0x8b61313bbabce2c6, 0x2323ac4b3b3da016}, {0xae397d8aa96c1b77, 0xabec975e0a0d081b},
                {0xd9c7dced53c72255, 0x96e7bd358c904a22}, {0x881cea14545c7575, 0x7e50d64177da2e55},
                {0xaa242499697392d2, 0xdde50bd1d5d0b9ea}, {0xd4ad2dbfc3d07787, 0x955e4ec64b44e865},
                {0x84ec3c97da624ab4, 0xbd5af13bef0b113f}, {0xa6274bbdd0fadd61, 0xecb1ad8aeacdd58f},
                {0xcfb11ead453994ba, 0x67de18eda5814af3}, {0x81ceb32c4b43fcf4, 0x80eacf948770ced8},
                {0xa2425ff75e14fc31, 0xa1258379a94d028e}, {0xcad2f7f5359a3b3e, 0x096ee45813a04331},
                {0xfd87b5f28300ca0d, 0x8bca9d6e188853fd}, {0x9e74d1b791e07e48, 0x775ea264cf55347e},
                {0xc612062576589dda, 0x95364afe032a819e}, {0xf79687aed3eec551, 0x3a83ddbd83f52205},
                {0x9abe14cd44753b52, 0xc4926a9672793543}, {0xc16d9a0095928a27, 0x75b7053c0f178294},
                {0xf1c90080baf72cb1, 0x5324c68b12dd6339}, {0x971da05074da7bee, 0xd3f6fc16ebca5e04},
                {0xbce5086492111aea, 0x88f4bb1ca6bcf585}, {0xec1e4a7db69561a5, 0x2b31e9e3d06c32e6},
                {0x9392ee8e921d5d07, 0x3aff322e62439fd0}, {0xb877aa3236a4b449, 0x09befeb9fad487c3},
                {0xe69594bec44de15b, 0x4c2ebe687989a9b4}, {0x901d7cf73ab0acd9, 0x0f9d37014bf60a11},
                {0xb424dc35095cd80f, 0x538484c19ef38c95}, {0xe12e13424bb40e13, 0x2865a5f206b06fba},
                {0x8cbccc096f5088cb, 0xf93f87b7442e45d4}, {0xafebff0bcb24aafe, 0xf78f69a51539d749},
                {0xdbe6fecebdedd5be, 0xb573440e5a884d1c}, {0x89705f4136b4a597, 0x31680a88f8953031},
                {0xabcc77118461cefc, 0xfdc20d2b36ba7c3e}, {0xd6bf94d5e57a42bc, 0x3d32907604691b4d},
                {0x8637bd05af6c69b5, 0xa63f9a49c2c1b110}, {0xa7c5ac471b478423, 0x0fcf80dc33721d54},
                {0xd1b71758e219652b, 0xd3c36113404ea4a9}, {0x83126e978d4fdf3b, 0x645a1cac083126ea},
                {0xa3d70a3d70a3d70a, 0x3d70a3d70a3d70a4}, {0xcccccccccccccccc, 0xcccccccccccccccd},
                {0x8000000000000000, 0x0000000000000000}, {0xa000000000000000, 0x0000000000000000},
                {0xc800000000000000, 0x0000000000000000}, {0xfa00000000000000, 0x0000000000000000},
                {0x9c40000000000000, 0x0000000000000000}, {0xc350000000000000, 0x0000000000000000},
                {0xf424000000000000, 0x0000000000000000}, {0x9896800000000000, 0x0000000000000000},
                {0xbebc200000000000, 0x0000000000000000}, {0xee6b280000000000, 0x0000000000000000},
                {0x9502f90000000000, 0x0000000000000000}, {0xba43b74000000000, 0x0000000000000000},
                {0xe8d4a51000000000, 0x0000000000000000}, {0x9184e72a00000000, 0x0000000000000000},
                {0xb5e620f480000000, 0x0000000000000000}, {0xe35fa931a0000000, 0x0000000000000000},
                {0x8e1bc9bf04000000, 0x0000000000000000}, {0xb1a2bc2ec5000000, 0x0000000000000000},
                {0xde0b6b3a76400000, 0x0000000000000000}, {0x8ac7230489e80000, 0x0000000000000000},
                {0xad78ebc5ac620000, 0x0000000000000000}, {0xd8d726b7177a8000, 0x0000000000000000},
                {0x878678326eac9000, 0x0000000000000000}, {0xa968163f0a57b400, 0x0000000000000000},
                {0xd3c21bcecceda100, 0x0000000000000000}, {0x84595161401484a0, 0x0000000000000000},
                {0xa56fa5b99019a5c8, 0x0000000000000000}, {0xcecb8f27f4200f3a, 0x0000000000000000},
                {0x813f3978f8940984, 0x4000000000000000}, {0xa18f07d736b90be5, 0x5000000000000000},
                {0xc9f2c9cd04674ede, 0xa400000000000000}, {0xfc6f7c4045812296, 0x4d00000000000000},
                {0x9dc5ada82b70b59d, 0xf020000000000000}, {0xc5371912364ce305, 0x6c28000000000000},
                {0xf684df56c3e01bc6, 0xc732000000000000}, {0x9a130b963a6c115c, 0x3c7f400000000000},
                {0xc097ce7bc90715b3, 0x4b9f100000000000}, {0xf0bdc21abb48db20, 0x1e86d40000000000},
                {0x96769950b50d88f4, 0x1314448000000000}, {0xbc143fa4e250eb31, 0x17d955a000000000},
                {0xeb194f8e1ae525fd, 0x5dcfab0800000000}, {0x92efd1b8d0cf37be, 0x5aa1cae500000000},
                {0xb7abc627050305ad, 0xf14a3d9e40000000}, {0xe596b7b0c643c719, 0x6d9ccd05d0000000},
                {0x8f7e32ce7bea5c6f, 0xe4820023a2000000}, {0xb35dbf821ae4f38b, 0xdda2802c8a800000},
                {0xe0352f62a19e306e, 0xd50b2037ad200000}, {0x8c213d9da502de45, 0x4526f422cc340000},
                {0xaf298d050e4395d6, 0x9670b12b7f410000}, {0xdaf3f04651d47b4c, 0x3c0cdd765f114000},
                {0x88d8762bf324cd0f, 0xa5880a69fb6ac800}, {0xab0e93b6efee0053, 0x8eea0d047a457a00},
                {0xd5d238a4abe98068, 0x72a4904598d6d880}, {0x85a36366eb71f041, 0x47a6da2b7f864750},
                {0xa70c3c40a64e6c51, 0x999090b65f67d924}, {0xd0cf4b50cfe20765, 0xfff4b4e3f741cf6d},
                {0x82818f1281ed449f, 0xbff8f10e7a8921a5}, {0xa321f2d7226895c7, 0xaff72d52192b6a0e},
                {0xcbea6f8ceb02bb39, 0x9bf4f8a69f764491}, {0xfee50b7025c36a08, 0x02f236d04753d5b5},
                {0x9f4f2726179a2245, 0x01d762422c946591}, {0xc722f0ef9d80aad6, 0x424d3ad2b7b97ef6},
                {0xf8ebad2b84e0d58b, 0xd2e0898765a7deb3}, {0x9b934c3b330c8577, 0x63cc55f49f88eb30},
                {0xc2781f49ffcfa6d5, 0x3cbf6b71c76b25fc}, {0xf316271c7fc3908a, 0x8bef464e3945ef7b},
                {0x97edd871cfda3a56, 0x97758bf0e3cbb5ad}, {0xbde94e8e43d0c8ec, 0x3d52eeed1cbea318},
                {0xed63a231d4c4fb27, 0x4ca7aaa863ee4bde}, {0x945e455f24fb1cf8, 0x8fe8caa93e74ef6b},
                {0xb975d6b6ee39e436, 0xb3e2fd538e122b45}, {0xe7d34c64a9c85d44, 0x60dbbca87196b617},
                {0x90e40fbeea1d3a4a, 0xbc8955e946fe31ce}, {0xb51d13aea4a488dd, 0x6babab6398bdbe42},
                {0xe264589a4dcdab14, 0xc696963c7eed2dd2}, {0x8d7eb76070a08aec, 0xfc1e1de5cf543ca3},
                {0xb0de65388cc8ada8, 0x3b25a55f43294bcc}, {0xdd15fe86affad912, 0x49ef0eb713f39ebf},
                {0x8a2dbf142dfcc7ab, 0x6e3569326c784338}, {0xacb92ed9397bf996, 0x49c2c37f07965405},
                {0xd7e77a8f87daf7fb, 0xdc33745ec97be907}, {0x86f0ac99b4e8dafd, 0x69a028bb3ded71a4},
                {0xa8acd7c0222311bc, 0xc40832ea0d68ce0d}, {0xd2d80db02aabd62b, 0xf50a3fa490c30191},
                {0x83c7088e1aab65db, 0x792667c6da79e0fb}, {0xa4b8cab1a1563f52, 0x577001b891185939},
                {0xcde6fd5e09abcf26, 0xed4c0226b55e6f87}, {0x80b05e5ac60b6178, 0x544f8158315b05b5},
                {0xa0dc75f1778e39d6, 0x696361ae3db1c722}, {0xc913936dd571c84c, 0x03bc3a19cd1e38ea},
                {0xfb5878494ace3a5f, 0x04ab48a04065c724}, {0x9d174b2dcec0e47b, 0x62eb0d64283f9c77},
                {0xc45d1df942711d9a, 0x3ba5d0bd324f8395}, {0xf5746577930d6500, 0xca8f44ec7ee3647a},
                {0x9968bf6abbe85f20, 0x7e998b13cf4e1ecc}, {0xbfc2ef456ae276e8, 0x9e3fedd8c321a67f},
                {0xefb3ab16c59b14a2, 0xc5cfe94ef3ea101f}, {0x95d04aee3b80ece5, 0xbba1f1d158724a13},
                {0xbb445da9ca61281f, 0x2a8a6e45ae8edc98}, {0xea1575143cf97226, 0xf52d09d71a3293be},
                {0x924d692ca61be758, 0x593c2626705f9c57}, {0xb6e0c377cfa2e12e, 0x6f8b2fb00c77836d},
                {0xe498f455c38b997a, 0x0b6dfb9c0f956448}, {0x8edf98b59a373fec, 0x4724bd4189bd5ead},
                {0xb2977ee300c50fe7, 0x58edec91ec2cb658}, {0xdf3d5e9bc0f653e1, 0x2f2967b66737e3ee},
                {0x8b865b215899f46c, 0xbd79e0d20082ee75}, {0xae67f1e9aec07187, 0xecd8590680a3aa12},
                {0xda01ee641a708de9, 0xe80e6f4820cc9496}, {0x884134fe908658b2, 0x3109058d147fdcde},
                {0xaa51823e34a7eede, 0xbd4b46f0599fd416}, {0xd4e5e2cdc1d1ea96, 0x6c9e18ac7007c91b},
                {0x850fadc09923329e, 0x03e2cf6bc604ddb1}, {0xa6539930bf6bff45, 0x84db8346b786151d},
                {0xcfe87f7cef46ff16, 0xe612641865679a64}, {0x81f14fae158c5f6e, 0x4fcb7e8f3f60c07f},
                {0xa26da3999aef7749, 0xe3be5e330f38f09e}, {0xcb090c8001ab551c, 0x5cadf5bfd3072cc6},
                {0xfdcb4fa002162a63, 0x73d9732fc7c8f7f7}, {0x9e9f11c4014dda7e, 0x2867e7fddcdd9afb},
                {0xc646d63501a1511d, 0xb281e1fd541501b9}, {0xf7d88bc24209a565, 0x1f225a7ca91a4227},
                {0x9ae757596946075f, 0x3375788de9b06959}, {0xc1a12d2fc3978937, 0x0052d6b1641c83af},
                {0xf209787bb47d6b84, 0xc0678c5dbd23a49b}, {0x9745eb4d50ce6332, 0xf840b7ba963646e1},
                {0xbd176620a501fbff, 0xb650e5a93bc3d899}, {0xec5d3fa8ce427aff, 0xa3e51f138ab4cebf},
                {0x93ba47c980e98cdf, 0xc66f336c36b10138}, {0xb8a8d9bbe123f017, 0xb80b0047445d4185},
                {0xe6d3102ad96cec1d, 0xa60dc059157491e6}, {0x9043ea1ac7e41392, 0x87c89837ad68db30},
                {0xb454e4a179dd1877, 0x29babe4598c311fc}, {0xe16a1dc9d8545e94, 0xf4296dd6fef3d67b},
                {0x8ce2529e2734bb1d, 0x1899e4a65f58660d}, {0xb01ae745b101e9e4, 0x5ec05dcff72e7f90},
                {0xdc21a1171d42645d, 0x76707543f4fa1f74}, {0x899504ae72497eba, 0x6a06494a791c53a9},
                {0xabfa45da0edbde69, 0x0487db9d17636893}, {0xd6f8d7509292d603, 0x45a9d2845d3c42b7},
                {0x865b86925b9bc5c2, 0x0b8a2392ba45a9b3}, {0xa7f26836f282b732, 0x8e6cac7768d7141f},
                {0xd1ef0244af2364ff, 0x3207d795430cd927}, {0x8335616aed761f1f, 0x7f44e6bd49e807b9},
                {0xa402b9c5a8d3a6e7, 0x5f16206c9c6209a7}, {0xcd036837130890a1, 0x36dba887c37a8c10},
                {0x802221226be55a64, 0xc2494954da2c978a}, {0xa02aa96b06deb0fd, 0xf2db9baa10b7bd6d},
                {0xc83553c5c8965d3d, 0x6f92829494e5acc8}, {0xfa42a8b73abbf48c, 0xcb772339ba1f17fa},
                {0x9c69a97284b578d7, 0xff2a760414536efc}, {0xc38413cf25e2d70d, 0xfef5138519684abb},
                {0xf46518c2ef5b8cd1, 0x7eb258665fc25d6a}, {0x98bf2f79d5993802, 0xef2f773ffbd97a62},
                {0xbeeefb584aff8603, 0xaafb550ffacfd8fb}, {0xeeaaba2e5dbf6784, 0x95ba2a53f983cf39},
                {0x952ab45cfa97a0b2, 0xdd945a747bf26184}, {0xba756174393d88df, 0x94f971119aeef9e5},
                {0xe912b9d1478ceb17, 0x7a37cd5601aab85e}, {0x91abb422ccb812ee, 0xac62e055c10ab33b},
                {0xb616a12b7fe617aa, 0x577b986b314d600a}, {0xe39c49765fdf9d94, 0xed5a7e85fda0b80c},
                {0x8e41ade9fbebc27d, 0x14588f13be847308}, {0xb1d219647ae6b31c, 0x596eb2d8ae258fc9},
                {0xde469fbd99a05fe3, 0x6fca5f8ed9aef3bc}, {0x8aec23d680043bee, 0x25de7bb9480d5855},
                {0xada72ccc20054ae9, 0xaf561aa79a10ae6b}, {0xd910f7ff28069da4, 0x1b2ba1518094da05},
                {0x87aa9aff79042286, 0x90fb44d2f05d0843}, {0xa99541bf57452b28, 0x353a1607ac744a54},
                {0xd3fa922f2d1675f2, 0x42889b8997915ce9}, {0x847c9b5d7c2e09b7, 0x69956135febada12},
                {0xa59bc234db398c25, 0x43fab9837e699096}, {0xcf02b2c21207ef2e, 0x94f967e45e03f4bc},
                {0x8161afb94b44f57d, 0x1d1be0eebac278f6}, {0xa1ba1ba79e1632dc, 0x6462d92a69731733},
                {0xca28a291859bbf93, 0x7d7b8f7503cfdcff}, {0xfcb2cb35e702af78, 0x5cda735244c3d43f},
                {0x9defbf01b061adab, 0x3a0888136afa64a8}, {0xc56baec21c7a1916, 0x088aaa1845b8fdd1},
                {0xf6c69a72a3989f5b, 0x8aad549e57273d46}, {0x9a3c2087a63f6399, 0x36ac54e2f678864c},
                {0xc0cb28a98fcf3c7f, 0x84576a1bb416a7de}, {0xf0fdf2d3f3c30b9f, 0x656d44a2a11c51d6},
                {0x969eb7c47859e743, 0x9f644ae5a4b1b326}, {0xbc4665b596706114, 0x873d5d9f0dde1fef},
                {0xeb57ff22fc0c7959, 0xa90cb506d155a7eb}, {0x9316ff75dd87cbd8, 0x09a7f12442d588f3},
                {0xb7dcbf5354e9bece, 0x0c11ed6d538aeb30}, {0xe5d3ef282a242e81, 0x8f1668c8a86da5fb},
                {0x8fa475791a569d10, 0xf96e017d694487bd}, {0xb38d92d760ec4455, 0x37c981dcc395a9ad},
                {0xe070f78d3927556a, 0x85bbe253f47b1418}, {0x8c469ab843b89562, 0x93956d7478ccec8f},
                {0xaf58416654a6babb, 0x387ac8d1970027b3}, {0xdb2e51bfe9d0696a, 0x06997b05fcc0319f},
                {0x88fcf317f22241e2, 0x441fece3bdf81f04}, {0xab3c2fddeeaad25a, 0xd527e81cad7626c4},
                {0xd60b3bd56a5586f1, 0x8a71e223d8d3b075}, {0x85c7056562757456, 0xf6872d5667844e4a},
                {0xa738c6bebb12d16c, 0xb428f8ac016561dc}, {0xd106f86e69d785c7, 0xe13336d701beba53},
                {0x82a45b450226b39c, 0xecc0024661173474}, {0xa34d721642b06084, 0x27f002d7f95d0191},
                {0xcc20ce9bd35c78a5, 0x31ec038df7b441f5}, {0xff290242c83396ce, 0x7e67047175a15272},
                {0x9f79a169bd203e41, 0x0f0062c6e984d387}, {0xc75809c42c684dd1, 0x52c07b78a3e60869},
                {0xf92e0c3537826145, 0xa7709a56ccdf8a83}, {0x9bbcc7a142b17ccb, 0x88a66076400bb692},
                {0xc2abf989935ddbfe, 0x6acff893d00ea436}, {0xf356f7ebf83552fe, 0x0583f6b8c4124d44},
                {0x98165af37b2153de, 0xc3727a337a8b704b}, {0xbe1bf1b059e9a8d6, 0x744f18c0592e4c5d},
                {0xeda2ee1c7064130c, 0x1162def06f79df74}, {0x9485d4d1c63e8be7, 0x8addcb5645ac2ba9},
                {0xb9a74a0637ce2ee1, 0x6d953e2bd7173693}, {0xe8111c87c5c1ba99, 0xc8fa8db6ccdd0438},
                {0x910ab1d4db9914a0, 0x1d9c9892400a22a3}, {0xb54d5e4a127f59c8, 0x2503beb6d00cab4c},
                {0xe2a0b5dc971f303a, 0x2e44ae64840fd61e}, {0x8da471a9de737e24, 0x5ceaecfed289e5d3},
                {0xb10d8e1456105dad, 0x7425a83e872c5f48}, {0xdd50f1996b947518, 0xd12f124e28f7771a},
                {0x8a5296ffe33cc92f, 0x82bd6b70d99aaa70}, {0xace73cbfdc0bfb7b, 0x636cc64d1001550c},
                {0xd8210befd30efa5a, 0x3c47f7e05401aa4f}, {0x8714a775e3e95c78, 0x65acfaec34810a72},
                {0xa8d9d1535ce3b396, 0x7f1839a741a14d0e}, {0xd31045a8341ca07c, 0x1ede48111209a051},
                {0x83ea2b892091e44d, 0x934aed0aab460433}, {0xa4e4b66b68b65d60, 0xf81da84d56178540},
                {0xce1de40642e3f4b9, 0x36251260ab9d668f}, {0x80d2ae83e9ce78f3, 0xc1d72b7c6b42601a},
                {0xa1075a24e4421730, 0xb24cf65b8612f820}, {0xc94930ae1d529cfc, 0xdee033f26797b628},
                {0xfb9b7cd9a4a7443c, 0x169840ef017da3b2}, {0x9d412e0806e88aa5, 0x8e1f289560ee864f},
                {0xc491798a08a2ad4e, 0xf1a6f2bab92a27e3}, {0xf5b5d7ec8acb58a2, 0xae10af696774b1dc},
                {0x9991a6f3d6bf1765, 0xacca6da1e0a8ef2a}, {0xbff610b0cc6edd3f, 0x17fd090a58d32af4},
                {0xeff394dcff8a948e, 0xddfc4b4cef07f5b1}, {0x95f83d0a1fb69cd9, 0x4abdaf101564f98f},
                {0xbb764c4ca7a4440f, 0x9d6d1ad41abe37f2}, {0xea53df5fd18d5513, 0x84c86189216dc5ee},
                {0x92746b9be2f8552c, 0x32fd3cf5b4e49bb5}, {0xb7118682dbb66a77, 0x3fbc8c33221dc2a2},
                {0xe4d5e82392a40515, 0x0fabaf3feaa5334b}, {0x8f05b1163ba6832d, 0x29cb4d87f2a7400f},
                {0xb2c71d5bca9023f8, 0x743e20e9ef511013}, {0xdf78e4b2bd342cf6, 0x914da9246b255417},
                {0x8bab8eefb6409c1a, 0x1ad089b6c2f7548f}, {0xae9672aba3d0c320, 0xa184ac2473b529b2},
                {0xda3c0f568cc4f3e8, 0xc9e5d72d90a2741f}, {0x8865899617fb1871, 0x7e2fa67c7a658893},
                {0xaa7eebfb9df9de8d, 0xddbb901b98feeab8}, {0xd51ea6fa85785631, 0x552a74227f3ea566},
                {0x8533285c936b35de, 0xd53a88958f872760}, {0xa67ff273b8460356, 0x8a892abaf368f138},
                {0xd01fef10a657842c, 0x2d2b7569b0432d86}, {0x8213f56a67f6b29b, 0x9c3b29620e29fc74},
                {0xa298f2c501f45f42, 0x8349f3ba91b47b90}, {0xcb3f2f7642717713, 0x241c70a936219a74},
                {0xfe0efb53d30dd4d7, 0xed238cd383aa0111}, {0x9ec95d1463e8a506, 0xf4363804324a40ab},
                {0xc67bb4597ce2ce48, 0xb143c6053edcd0d6}, {0xf81aa16fdc1b81da, 0xdd94b7868e94050b},
                {0x9b10a4e5e9913128, 0xca7cf2b4191c8327}, {0xc1d4ce1f63f57d72, 0xfd1c2f611f63a3f1},
                {0xf24a01a73cf2dccf, 0xbc633b39673c8ced}, {0x976e41088617ca01, 0xd5be0503e085d814},
                {0xbd49d14aa79dbc82, 0x4b2d8644d8a74e19}, {0xec9c459d51852ba2, 0xddf8e7d60ed1219f},
                {0x93e1ab8252f33b45, 0xcabb90e5c942b504}, {0xb8da1662e7b00a17, 0x3d6a751f3b936244},
                {0xe7109bfba19c0c9d, 0x0cc512670a783ad5}, {0x906a617d450187e2, 0x27fb2b80668b24c6},
                {0xb484f9dc9641e9da, 0xb1f9f660802dedf7}, {0xe1a63853bbd26451, 0x5e7873f8a0396974},
                {0x8d07e33455637eb2, 0xdb0b487b6423e1e9}, {0xb049dc016abc5e5f, 0x91ce1a9a3d2cda63},
                {0xdc5c5301c56b75f7, 0x7641a140cc7810fc}, {0x89b9b3e11b6329ba, 0xa9e904c87fcb0a9e},
                {0xac2820d9623bf429, 0x546345fa9fbdcd45}, {0xd732290fbacaf133, 0xa97c177947ad4096},
                {0x867f59a9d4bed6c0, 0x49ed8eabcccc485e}, {0xa81f301449ee8c70, 0x5c68f256bfff5a75},
                {0xd226fc195c6a2f8c, 0x73832eec6fff3112}, {0x83585d8fd9c25db7, 0xc831fd53c5ff7eac},
                {0xa42e74f3d032f525, 0xba3e7ca8b77f5e56}, {0xcd3a1230c43fb26f, 0x28ce1bd2e55f35ec},
                {0x80444b5e7aa7cf85, 0x7980d163cf5b81b4}, {0xa0555e361951c366, 0xd7e105bcc3326220},
                {0xc86ab5c39fa63440, 0x8dd9472bf3fefaa8}, {0xfa856334878fc150, 0xb14f98f6f0feb952},
                {0x9c935e00d4b9d8d2, 0x6ed1bf9a569f33d4}, {0xc3b8358109e84f07, 0x0a862f80ec4700c9},
                {0xf4a642e14c6262c8, 0xcd27bb612758c0fb}, {0x98e7e9cccfbd7dbd, 0x8038d51cb897789d},
                {0xbf21e44003acdd2c, 0xe0470a63e6bd56c4}, {0xeeea5d5004981478, 0x1858ccfce06cac75},
                {0x95527a5202df0ccb, 0x0f37801e0c43ebc9}, {0xbaa718e68396cffd, 0xd30560258f54e6bb},
                {0xe950df20247c83fd, 0x47c6b82ef32a206a}, {0x91d28b7416cdd27e, 0x4cdc331d57fa5442},
                {0xb6472e511c81471d, 0xe0133fe4adf8e953}, {0xe3d8f9e563a198e5, 0x58180fddd97723a7},
                {0x8e679c2f5e44ff8f, 0x570f09eaa7ea7649}, {0xb201833b35d63f73, 0x2cd2cc6551e513db},
                {0xde81e40a034bcf4f, 0xf8077f7ea65e58d2}, {0x8b112e86420f6191, 0xfb04afaf27faf783},
                {0xadd57a27d29339f6, 0x79c5db9af1f9b564}, {0xd94ad8b1c7380874, 0x18375281ae7822bd},
                {0x87cec76f1c830548, 0x8f2293910d0b15b6}, {0xa9c2794ae3a3c69a, 0xb2eb3875504ddb23},
                {0xd433179d9c8cb841, 0x5fa60692a46151ec}, {0x849feec281d7f328, 0xdbc7c41ba6bcd334},
                {0xa5c7ea73224deff3, 0x12b9b522906c0801}, {0xcf39e50feae16bef, 0xd768226b34870a01},
                {0x81842f29f2cce375, 0xe6a1158300d46641}, {0xa1e53af46f801c53, 0x60495ae3c1097fd1},
                {0xca5e89b18b602368, 0x385bb19cb14bdfc5}, {0xfcf62c1dee382c42, 0x46729e03dd9ed7b6},
                {0x9e19db92b4e31ba9, 0x6c07a2c26a8346d2}, {0xc5a05277621be293, 0xc7098b7305241886},
                {0xf70867153aa2db38, 0xb8cbee4fc66d1ea8}};
        };

        // Compressed cache for double
        struct compressed_cache_detail {
            static constexpr int compression_ratio = 27;
            static constexpr std::size_t compressed_table_size =
                (main_cache_holder<ieee754_binary64>::max_k -
                 main_cache_holder<ieee754_binary64>::min_k + compression_ratio) /
                compression_ratio;

            struct cache_holder_t {
                wuint::uint128 table[compressed_table_size];
            };
            static constexpr cache_holder_t cache = [] {
                cache_holder_t res{};
                for (std::size_t i = 0; i < compressed_table_size; ++i) {
                    res.table[i] =
                        main_cache_holder<ieee754_binary64>::cache[i * compression_ratio];
                }
                return res;
            }();

            struct pow5_holder_t {
                std::uint64_t table[compression_ratio];
            };
            static constexpr pow5_holder_t pow5 = [] {
                pow5_holder_t res{};
                std::uint64_t p = 1;
                for (std::size_t i = 0; i < compression_ratio; ++i) {
                    res.table[i] = p;
                    p *= 5;
                }
                return res;
            }();
        };
    }

    struct main_cache_full {
        template <class FloatFormat>
        static constexpr typename detail::main_cache_holder<FloatFormat>::cache_entry_type
        get_cache(int k) noexcept {
            assert(k >= detail::main_cache_holder<FloatFormat>::min_k &&
                   k <= detail::main_cache_holder<FloatFormat>::max_k);
            return detail::main_cache_holder<FloatFormat>::cache[std::size_t(
                k - detail::main_cache_holder<FloatFormat>::min_k)];
        }
    };

    struct main_cache_compressed {
        template <class FloatFormat>
        static constexpr typename detail::main_cache_holder<FloatFormat>::cache_entry_type
        get_cache(int k) noexcept {
            assert(k >= detail::main_cache_holder<FloatFormat>::min_k &&
                   k <= detail::main_cache_holder<FloatFormat>::max_k);

            if constexpr (std::is_same_v<FloatFormat, ieee754_binary64>) {
                // Compute the base index.
                auto const cache_index =
                    int(std::uint32_t(k - detail::main_cache_holder<FloatFormat>::min_k) /
                        detail::compressed_cache_detail::compression_ratio);
                auto const kb = cache_index * detail::compressed_cache_detail::compression_ratio +
                                detail::main_cache_holder<FloatFormat>::min_k;
                auto const offset = k - kb;

                // Get the base cache.
                auto const base_cache = detail::compressed_cache_detail::cache.table[cache_index];

                if (offset == 0) {
                    return base_cache;
                }
                else {
                    namespace log = detail::log;
                    namespace wuint = detail::wuint;

                    // Compute the required amount of bit-shift.
                    auto const alpha =
                        log::floor_log2_pow10(kb + offset) - log::floor_log2_pow10(kb) - offset;
                    assert(alpha > 0 && alpha < 64);

                    // Try to recover the real cache.
                    auto const pow5 = detail::compressed_cache_detail::pow5.table[offset];
                    auto recovered_cache = wuint::umul128(base_cache.high(), pow5);
                    auto const middle_low = wuint::umul128(base_cache.low(), pow5);

                    recovered_cache += middle_low.high();

                    auto const high_to_middle = recovered_cache.high() << (64 - alpha);
                    auto const middle_to_low = recovered_cache.low() << (64 - alpha);

                    recovered_cache =
                        wuint::uint128{(recovered_cache.low() >> alpha) | high_to_middle,
                                       ((middle_low.low() >> alpha) | middle_to_low)};

                    assert(recovered_cache.low() + 1 != 0);
                    recovered_cache = {recovered_cache.high(), recovered_cache.low() + 1};

                    return recovered_cache;
                }
            }
            else {
                // Just use the full cache for anything other than binary64
                return detail::main_cache_holder<FloatFormat>::cache[std::size_t(
                    k - detail::main_cache_holder<FloatFormat>::min_k)];
            }
        }
    };

    struct extended_cache_long {
        static constexpr std::size_t max_cache_blocks = 3;
        static constexpr std::size_t cache_bits_unit = 64;
        static constexpr int segment_length = 22;
        static constexpr bool constant_block_count = true;
        static constexpr int e_min = -1074;
        static constexpr int k_min = -271;
        static constexpr int cache_bit_index_offset_base = 974;
        static constexpr std::uint64_t cache[] = {
            0xcc5fc196fefd7d0c, 0x1e53ed49a96272c8, 0xb2e28cedd086d863, 0xb256369d4a4090be,
            0xd43e40076a82e056, 0x4966180f57bf27ab, 0x73482e74799ca38f, 0x350b22de909056fc,
            0x24f01ce8035c3a13, 0x56df68130e5614c6, 0xc0cfa8fbffb3c373, 0x49493583bd93ca52,
            0xe50f85e8af10f9bd, 0x6dec5eca4e77e794, 0x2a7085885afc5989, 0x27713fb587e5d5f5,
            0x3bbd3e40458cfad4, 0xa9c85a68d0a0f240, 0x35ecb8a3196ffb00, 0x5d0e6c51eda9db73,
            0xf1ae9bc8cb513f1c, 0xedfb556d0b3ce425, 0xde2347df72a867ed, 0xd7200175295c2a54,
            0xe47288046f4127ce, 0x9de50ff151a99f48, 0x2f93ca994bae1501, 0xcbacf92a3c5cb255,
            0x1b8c41fff3a71062, 0xa7a17e0791f14181, 0xda7f1272fad8ea8e, 0x243fcc407aa5c4d1,
            0x05881640c0db8b6c, 0xcf60100bcd51fee3, 0x41fc585cdb88fe1c, 0xf0bd574e5574ca2b,
            0xa1e3de8a47c813c5, 0xa404ec37045a05b8, 0xc249554416a0ab81, 0x7d346f75e7ef4f92,
            0xb4ae95375bff6a3e, 0xc42613286e18e342, 0x8c7c4bc7f39feb29, 0xf451130dbb6b8d67,
            0x4ed6ff12c528cb4e, 0xbc0398f27e2e7f02, 0xe43b4f2f30fc71df, 0xdb805653fb2e4549,
            0xcb75663b492c05d9, 0x279992430d135fdd, 0xb912ffbf694d473e, 0x7d0fb5cd0bc31c24,
            0x5a31b13a2305d635, 0x961a44e862d1253b, 0x942c7ff0054684d5, 0x1940f85b9619e4de,
            0xf56f6fd48770db54, 0x2048e3619d3ecbd1, 0x5ad267734b626ab2, 0xd497aae5da836756,
            0x257b47819a7268a2, 0xc0dee1778d0d99b7, 0x2760893b31197df2, 0x0f4b39615a1fef1c,
            0x3d28f4595f045f5e, 0x68198cc92d282646, 0xab840ed7f34aa207, 0x594af70a7c9a8466,
            0x0d3c447b36c4adaa, 0x73a9783d66bafd7c, 0xd57e97a24173229a, 0x3fedddcaf2d920ec,
            0x5ab0cf0f7fa05a5d, 0xfd431db310bef10c, 0x86906a275fe9d403, 0x35fe7c65c7e73794,
            0x70e557bb0b4edf3b, 0x0c354dcc39e877e3, 0xb2e1b87c2114a94c, 0x2d3b357c0692aa0a,
            0x078e2bad8d82c6bc, 0xd145ed52ec1f36b4, 0x311beb063cd5e1c9, 0x3749ae20c36e3685,
            0xe072f5b34ca93fd2, 0x235b11a17ec97da7, 0x06250127266ca206, 0x6680720c217f66c5,
            0xcf72ae1cbf4a6037, 0x48566c5cc9896704, 0x0f888f4e9059a4c1, 0x83bf4fa13d8895ec,
            0x852f3253b65a7f41, 0xe95fc7edbc424d2f, 0xcb6f4c2902395ee8, 0x19b77e8a5b4d988e,
            0xf34a7d1c3ee79f3c, 0x7ee67dfe30e8b317, 0x7df1d26a0827dbad, 0x1144e6575d72e369,
            0x6a1b48aca94c3fd3, 0x1c18223394282d5b, 0xa9d6c3b9491f2706, 0x0654a330021b79c4,
            0xa07d573c4af1c42f, 0x856fb5bb9d8f5fc7, 0x029239eeaf410c32, 0x3af35a890e51dba3,
            0xe10b9ccd95a5efea, 0x6b34767c12ddc8b0, 0x27408660be2df000, 0xcfb01909478fcb1c,
            0x64fef9a6c8de88bd, 0x53399a4e4227ca92, 0x6e9fc3e475398585, 0xa7ff464cb86b3a8b,
            0x1a1f274613e7a315, 0x7cb97c89a667bbb2, 0x85e23b46f90aa7cf, 0xdfd40cc7a241b280,
            0x227f15e9a4003db7, 0x573a0ceb3b9a35fe, 0x15e6aaa07898bcf1, 0x9825c14c84358b11,
            0x1f92b5ad7f29abca, 0xf485787a6520ec08, 0xd23699194119a5c3, 0x7387b71906614310,
            0xf6c848d98da56f90, 0xb1938b629f31890f, 0x07dabfd496c5a9c2, 0xdc890272a1074c24,
            0x9bc0bb46928b885e, 0xdeb2a85e587aeb58, 0xa90e81a302cee759, 0x400d36496828de45,
            0x9b14c3dc6342c577, 0x6df5a6e0a9f9632d, 0xccc1e5123df02597, 0x693db13b76ce9376,
            0x69e9a8a5a617565d, 0xdaf1afd498df0bdc, 0x21abb48db201e86d, 0x7fb942dc0970da82,
            0x00bc8db411d4f56d, 0x23b0de1593369d1b, 0x5fad34051767bdae, 0x347bf3f22ac4f809,
            0xc58edf98b59a373f, 0xec4724bd4189bd5e, 0xac6b5856629eedf5, 0x90abb3a1ebbdad97,
            0x45eb4d50ce6332f8, 0x40b7ba963646e043, 0xcbba5ba0cdaf5509, 0xffc3e5a10722b689,
            0x84c1dab405552d60, 0xdbd61fbe5b737542, 0x16f7ad8a8e05ba47, 0xa54e5f27a84b5da0,
            0xfcd9c0d8597087a7, 0xb78d63072b9ed4ca, 0xa0dfaba295941a9d, 0x0b03d63a2529dcea,
            0xf9de194ec523af42, 0xf067e145959167b6, 0x81f01d7b5721d9fd, 0x3448d7b1c390396c,
            0xe364b5d83b11154d, 0xf2607730e56a6b03, 0x038207de93a3f05c, 0x8b0cce521bd90a65,
            0x5296cd8e2d4d052b, 0xb162cb83424db7bb, 0x1acb4c8a011736f8, 0x6fc6c167a6a359d1,
            0x3c630164b93172d4, 0xbb22df25e590decb, 0xbbb6677dc03994e5, 0x014df95ff0ece4b7,
            0x2fde4568bb259c7d, 0xb0d32add9f0192ac, 0x64cf917fae4a4985, 0x70ea94e7e6f7019f,
            0x933cbdb13c062e37, 0x47bc3cf523a283f1, 0x10217cc4f3bd9f97, 0x6dcf0a47a23c3775,
            0xc07eef089dec2a40, 0xd56c8a963b01272f, 0x6b4be3385978b732, 0xbbaa3d4df50af0ac,
            0x62aa54e844fe7d4a, 0xcacccbbfde1ab76f, 0xdab42c1f3fca4472, 0xd4e20c2916a0eb76,
            0x4e00170c17c372a2, 0xbed384f0c7428321, 0xb5793de5a69eda73, 0xc401d3dba21a4c9f,
            0x8e86c05e1a63853b, 0xbd264515e7873f8a, 0x0396973fe09f1e7e, 0xe6786710a61a5152,
            0xd9902132f2e24ad6, 0xb17e17707c7e4452, 0x1ab18d106daffde8, 0xfd8f944f93bdec82,
            0x7b570dd17e07cb9f, 0xea591fb1f2db2afe, 0x278f761edeeea5d5, 0x0049814781858ccf,
            0xce06cac7426e6b70, 0xa63bb1951b672971, 0x3053c7c258aeb716, 0x66590d78a05d8282,
            0x2723fd36304c9dc5, 0x20cd299b8c40e3bd, 0x822f58eafad0994c, 0x84e7e08625c65000,
            0x9d2d30a9ccda4b25, 0xd3d90a2bf9ec583b, 0xdc7058848ce53c07, 0xbb3daf6a840f8643,
            0xc29c626efcc1af43, 0x49cf5f5127b3b5ad, 0xac314093db83ac44, 0xef2f8fb5cee0e1a2,
            0x7f9ba94bcf701709, 0x9d0243c163cdaf6d, 0x19fec8085b03bb5d, 0x86443e9d9fb3b0ec,
            0x76ecb60776a2e3bd, 0x7beec07060605a70, 0x8c4c58a31da02e8a, 0x85b2f70730aeb0f4,
            0xabb46f048cf0b22a, 0x7113a738f6fab1e4, 0xea27a055b82fb00e, 0xe59b0d47e4434641,
            0x2357c5dab8a49a58, 0x383c2f9f3ce238a0, 0xa1214e7123a90e99, 0xa3d216b25141e59a,
            0x3c07de164b0bb4f4, 0x81bb792f77d3f787, 0x7a4b198c6037a6c4, 0x29a821bfb144ce6c,
            0x8a61e074da415980, 0xac74be931bfb9cda, 0x155916a3016bf0da, 0xe2c347a9f3ed2b29,
            0x81ca36dbc5187d40, 0x62e3b48a9a9d3ec4, 0x422d497fb48c139c, 0x1e16aec906d67705,
            0xb0e0df9ed59268ab, 0xdbc1682b7d8c26e3, 0xf7dea7068bb3ca84, 0xc80a5af5e5764cb8,
            0x11d72e2e81654dde, 0x802c6f4727667ea8, 0x846d697e8f204fc3, 0x4686bb9eb24b1299,
            0x701833f5fa743c1f, 0x6e065e876e8649eb, 0x9638d6e8afb1b14e, 0x645631e21099e9b5,
            0x651b445d775ab0df, 0x5d9d842205f31974, 0x8eafbd51f9caa9f2, 0x628ec824cf30d215,
            0xd17a6a87687e9449, 0xdfee445f2ca513d3, 0x73a877686a9e006d, 0x29fc3d1470046708,
            0x5838d17199dc5b48, 0x721e0fd5b5449410, 0x960e81c832d435c1, 0x297b9da2405e0293,
            0xe1f35bfade12bc1d, 0xd015b4f5694dcead, 0x5ef3bf28ded7e8db, 0x1b375f2f54d4a2b1,
            0xbb6a21e606390529, 0xd9e3e9134da6c146, 0xf3d4236a7917e12c, 0xfe5b45cc2c1645c7,
            0xfae4459abfc08d22, 0x785629ccf73e3ad3, 0xe3737cb14f6f65e3, 0xd21a1489a90c9144,
            0x1a1e3319d4a38490, 0x73247bafc344b507, 0xa7ef8c1878053802, 0x5430a2e4c6a2ebe5,
            0x63d58b7200368e4d, 0x316017dd36646214, 0x672c5e884e85f5af, 0x73f2360290fbb262,
            0xceca95ebbd532720, 0x47993a1414efe278, 0x911623dba4b4e122, 0x71d83cf4e8da322e,
            0x5bccdbbfdb66cf2e, 0xf81887fb96ab91b4, 0xb1dd42e7736e1e3e, 0xd26979d473f3ca61,
            0x4e2d3e6794a0f529, 0x998de9b2dbf7654a, 0x3d0ff939ca26d64b, 0xecdc6e158acecfcd,
            0xe58976928af0ded6, 0x22d26cd93100ac14, 0xe6cc2ef95ef6dfe2, 0x216a24cd8d560525,
            0xe0a9c573f0f4fd05, 0x570dba55bd10be58, 0x055b13a71dac694c, 0x8579682baf0dfced,
            0x47dce842159e1c86, 0x72fb7156a2b1ce6d, 0x3a2efedc180103e3, 0x674acb05a0d6e351,
            0x3596e8a22a81b284, 0x8ecc4c4950cb8204, 0x7a1def9538f58c52, 0x6b7b7b3518deeafc,
            0x494b57cdf9903303, 0x52a986d963733488, 0x102c5a25c123a5bd, 0xca6e725c9b3f51a8,
            0x07d3f1bf70069eca, 0xa951148acb2efa5e, 0x47c44388998b914a, 0x3f9c83e746b8de39,
            0xcaeadaebdb698013, 0x1b02e11fb4e6dcaa, 0x2680b3f6f8bda5a4, 0xdf352de3aedc8dfd,
            0x630ea837fcb8385d, 0x7edd58d79c694158, 0x3fc5fd2b26fe54e8, 0x19a42d1ac8eadf5c,
            0x60bf3c991660f478, 0xf753c84ee01e98eb, 0xd0c6068832c3760f, 0x4eefa5976049a659,
            0xa102f0f1e43a1032, 0x43c71126ba90897a, 0x279ae100869594fb, 0x77de4c068f579b4b,
            0x2b2f0a6878a0d568, 0x7fdb1506d48a5818, 0x07a5410b686cecc6, 0x788235105c2f2974,
            0xd36aa9828b16c6c8, 0x230745dff250af51, 0x57db933c81e18cf0, 0x1636e615700efcab,
            0xf6e15568dd35f61a, 0x75e69b72ed8195ef, 0x0b1f9c27ed375a6b, 0x9e879260cfa804ef,
            0x261f8c458b92fa32, 0x3722228147032222, 0xf33e3bd5a08a9dde, 0x65dc8d8be09de195,
            0xb8b9f1b5c19fb284, 0x7c8fa1c02e9f0756, 0x0a065bc79283dad8, 0xf8ef9c8fafbd8af7,
            0x164911fafa645de0, 0xb2cc165d7ab3a327, 0xfeccbb72a578077e, 0x47a3d87dd24cee41,
            0xb98f69dc5c63ac88, 0xe7b1760566177a07, 0x266f5da3beae5570, 0x969f4627b63c01f4,
            0xb83a8e2af4f06148, 0x74235b38517382f9, 0x2d17c078a9a40d5b, 0x7f70adbcf8b3f22a,
            0xda1f2c9ca3011302, 0xb8e2dc6a371ad25a, 0x59af7f58d3e31c16, 0xfc4801b3283de5f7,
            0xd0c61b65eb2d93e0, 0x6bc4d9629adf3098, 0x26d20fdf6bab5ac7, 0x11cf4332de481638,
            0xb8f513d7289780d4, 0xbaf76387987e6ab9, 0x7b18f5430bf6f74c, 0x20843400a1d95cd7,
            0xaa26d11d6a10cc5a, 0x6899146442440955, 0xbb9bf72a5f214d32, 0x5e9ea055413ab751,
            0xcb1754690996a6a7, 0x08e741a38f3953b4, 0x43911052a7d9a7ec, 0x2ae6a7df9a0118c2,
            0x7a1756eb8ef7bfb4, 0x4bb1a4ddd13facb2, 0x951398938123e46e, 0xae357b83184a1621,
            0x8fab4ece3954b298, 0x72a88d7802ce64f0, 0xc511a7ba4702729c, 0xbd4b5ee7307bf70d,
            0x7b446437f0edb951, 0xf7b162a0a9d56cff, 0x32cd113af849e5bf, 0x1e9c99b2a88f8823,
            0xd917a9b0350f7834, 0x878cef6ad9bf40fb, 0x587614b72cd13ef2, 0x6dcf66541c634e44,
            0x7ac80ad28d730bef, 0x1e712bacc101dcba, 0xbd2ec0edbcf06a0f, 0x43a35e9d57fc82a0,
            0xb956fead7fb3bead, 0xa61225dbf0f0232b, 0xa89cd3f0aed7c235, 0x460fb00a84ba0000,
            0x0000000000000000};

        struct multiplier_index_info {
            std::uint16_t first_cache_bit_index;
            std::uint16_t cache_bit_index_offset;
        };

        static constexpr multiplier_index_info multiplier_index_info_table[] = {
            {0, 0},         {176, 249},     {427, 573},     {750, 969},     {1145, 1438},
            {1615, 1981},   {2155, 2594},   {2771, 3283},   {3459, 4044},   {4222, 4880},
            {5058, 5789},   {5961, 6765},   {6943, 7820},   {7959, 8909},   {9033, 10056},
            {9068, 10164},  {9154, 10323},  {9291, 10534},  {9480, 10796},  {9720, 11109},
            {10011, 11473}, {10353, 11888}, {10746, 12354}, {11190, 12871}, {11685, 13439},
            {12231, 14058}, {12828, 14728}, {13476, 15449}, {14175, 16221}, {14925, 16973},
            {15655, 17701}, {16361, 18408}, {17046, 19092}, {17708, 19756}, {18350, 20398},
            {18970, 21017}, {19567, 21614}, {20142, 22189}, {20695, 22739}, {21223, 23271},
            {21733, 23781}, {22221, 24269}, {22687, 24734}, {23130, 25178}, {23552, 25593},
            {23945, 25993}, {24323, 26370}, {24678, 26726}, {25012, 27060}, {25324, 27372},
            {25614, 27661}, {25881, 27929}, {26127, 28175}, {26351, 28398}, {26552, 28599},
            {26731, 28779}, {26889, 28937}, {27025, 29073}, {27139, 29187}, {27231, 29279},
            {27301, 29347}, {27347, 29393}, {27371, 29419}, {27375, 0}};
    };

    struct extended_cache_compact {
        static constexpr std::size_t max_cache_blocks = 6;
        static constexpr std::size_t cache_bits_unit = 64;
        static constexpr int segment_length = 80;
        static constexpr bool constant_block_count = false;
        static constexpr int collapse_factor = 64;
        static constexpr int e_min = -1074;
        static constexpr int k_min = -210;
        static constexpr int cache_bit_index_offset_base = 964;
        static constexpr int cache_block_count_offset_base = 27;

        static constexpr std::uint64_t cache[] = {
            0xc795830d75038c1d, 0xd59df5b9ef6a2417, 0xfeb13da03da3a72f, 0xa1be04416f774ca1,
            0x85f538987c4d8ab3, 0xde9c74098f2bafaa, 0xa14c221ad536f7af, 0x67d5fe0c0a0ff7f3,
            0x0de9d31958870e22, 0xc5ac162d4339a052, 0xd0d1d97e0d614173, 0x802b4105b336f99f,
            0xb2d984f4f566db5f, 0xb533e21c804345bd, 0x1f326438a9241b9f, 0x9c5ef58109bb5397,
            0xa8df5efabc5979c8, 0xfca8d3ffa1ef463c, 0x1d897747ed9d851e, 0x7e089e51c5849436,
            0x84991b0e100065e9, 0x5b739863e3229abf, 0x20ad3b847a221d5b, 0x54cfe86e61fe7942,
            0x52100905ac4685b8, 0xc9d071c69307a1b5, 0x443f79e206bc53fe, 0xad0d5e38f3213d42,
            0xc7e9ffc57eb87287, 0x899f2e376c6750de, 0x277c9dee7a4ad4b8, 0x1eef92c7ccd0b1ed,
            0xa88809578b3c883f, 0xacac074718d3a861, 0xc2e7a8075f6f48b7, 0xf7575b91c1228c74,
            0xa6156587fb4d33e7, 0x10f12886b5f4e7d3, 0xf1f8c78ab01e4c8e, 0x2ca5131d29e6d9bc,
            0x31e1fa72ff00a896, 0x0d72272b42d76b8d, 0x4ed9b2e8caa13ccd, 0xee30cdb21a96c453,
            0xcfb26c0c4ad47994, 0xa33653a0ffed981c, 0x1d6019d9050b2bd3, 0xf70f4cd944754727,
            0xcb2734119d3b7a9a, 0xa51823e34a7eedeb, 0xd4b46f0599fd4151, 0x0331d72aeaf71658,
            0x6ff327d536b74698, 0xfa475791a569d10f, 0x96e017d694487bcd, 0x671a0192ea53fcdf,
            0x4f028faa0591cee1, 0x10848f82d5dc343b, 0xf046f580a6dc57f2, 0x388f0f4ecd04a7c9,
            0x28069cf3cb733ef1, 0x8cece59cf233b0f6, 0x711334bcda45f1d3, 0x625ba277930142f2,
            0x9981c43f8a015832, 0x9b927206988a3207, 0x710462b1f77da9a0, 0xb36caa2112ab8b91,
            0x9b31ca21b110d5d1, 0x70c5cb8ef4ef21e0, 0xbfff827fd7fca291, 0x6083c84685eb1511,
            0xd86ad0063cca31d8, 0xbb805953a1297b88, 0xf63d7956e0d98578, 0x80976336e47a55a4,
            0xe26491b9c8dbbfc5, 0x405b99dac54b48d5, 0xf5033bae4a2c613a, 0xbce5f30e3c3ae7a5,
            0x9f0efb302818554f, 0x83412b3c79e33e50, 0x856a8777d6468b5d, 0x9bc1690affcda2f1,
            0xda1dfde453a3d469, 0x826c9a87361f35f8, 0x79c1e1fb5e140cbb, 0xd2e5d0703d1e4d7b,
            0x4e263e53ab13676f, 0xbae3ec77c07755a8, 0xc81109b422e30769, 0xfac9d2aca5844f05,
            0x87a7e9d9bb1d9c4f, 0x8c8f408fac935e4a, 0x635e65532989419f, 0x4efcc4d7ba319b9f,
            0xdf2809505710bd22, 0xb5e4edb89cda5b59, 0x12ad5b9910fe8ad6, 0xf8e45c246285362a,
            0x972bc5046accbe70, 0xb4de428c28ebec9d, 0x02e47593c24145ee, 0xaa90a540c28355e2,
            0x0367ac12d7fdaaeb, 0xe2187d19d06dde15, 0x649e162610192e14, 0xaf30e9d3c4b190b8,
            0xbd04d772b36d826a, 0x2d6275f5b7343bff, 0x7ca39b6c06ae7414, 0x227c995776fbefa9,
            0x95a4627d7d96416f, 0x1db95b9aed4e0161, 0x611d81ca92cc44fd, 0x8c42c7adfa974d94,
            0x729442cc23490c8f, 0x1ba514d154d1a3a5, 0x4f957b16e69a7395, 0xad7d0af4f070aa2c,
            0x7cbef9d2c3f5c30b, 0x66ec52ce16d31ef2, 0xe4dd4f9ca6b07c1f, 0xe3ffcc165d60011d,
            0x7955021b42f8a908, 0x7d3bbcd24c62b3b0, 0xe824b0a990000000, 0x0000000000000000};

        struct multiplier_index_info {
            std::uint16_t first_cache_bit_index;
            std::uint16_t cache_bit_index_offset;
            std::uint16_t cache_block_count_index_offset;
        };

        static constexpr multiplier_index_info multiplier_index_info_table[] = {
            {0, 0, 0},          {382, 648, 9},      {1028, 1560, 22},  {1939, 2736, 39},
            {3062, 4125, 60},   {3132, 4461, 70},   {3388, 4983, 84},  {3830, 5690, 98},
            {4457, 6494, 111},  {5181, 7217, 123},  {5824, 7860, 134}, {6387, 8425, 143},
            {6872, 8910, 151},  {7277, 9315, 158},  {7602, 9639, 164}, {7846, 9884, 168},
            {8011, 10049, 171}, {8096, 10133, 173}, {8100, 0, 0}};

        static constexpr std::uint8_t cache_block_counts[] = {
            0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
            0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x56, 0x34, 0x12, 0x66,
            0x66, 0x45, 0x23, 0x61, 0x66, 0x66, 0x66, 0x45, 0x23, 0x61, 0x66, 0x66, 0x66,
            0x56, 0x34, 0x12, 0x66, 0x66, 0x66, 0x56, 0x34, 0x12, 0x66, 0x66, 0x66, 0x45,
            0x23, 0x61, 0x66, 0x56, 0x34, 0x12, 0x66, 0x56, 0x34, 0x12, 0x66, 0x45, 0x23,
            0x61, 0x45, 0x23, 0x41, 0x23, 0x31, 0x12, 0x12, 0x01};
    };

    struct extended_cache_super_compact {
        static constexpr std::size_t max_cache_blocks = 15;
        static constexpr std::size_t cache_bits_unit = 64;
        static constexpr int segment_length = 252;
        static constexpr bool constant_block_count = false;
        static constexpr int collapse_factor = 128;
        static constexpr int e_min = -1074;
        static constexpr int k_min = -65;
        static constexpr int cache_bit_index_offset_base = 1054;
        static constexpr int cache_block_count_offset_base = 10;

        static constexpr std::uint64_t cache[] = {
            0xf712b443bbd52b7b, 0xa5e9ec7501d523e4, 0x6f99ee8b281c132a, 0x1c7262e905287f33,
            0xbf4f71a69f411989, 0xe95fb0bf35d5c518, 0x00d875ffe81c1457, 0x31f0fcb03c200323,
            0x6f64d6af592895a0, 0x45c073ee14c78fb0, 0x8744404cbdba226c, 0x8dbe2386885f0c74,
            0x279b6693e94ab813, 0x6df0a4a86ccbb52e, 0xa94baea98e947129, 0xfc2b4e9bb4cbe9a4,
            0x73bbc273e753c4ad, 0xc70c8ff8c19c1059, 0xb7da754b6db8b578, 0x5214cf7f2274988c,
            0x39b5c4db3b36b321, 0xda6f355441d9f234, 0x01ab018d850bd7e2, 0x36517c3f140b3bcf,
            0xd0e52375d8d125a7, 0xaf9709f49f3b8404, 0x022dd12dd219aa3f, 0x46e2ecebe43f459e,
            0xa428ebddeecd6636, 0x3a7d11bff7e2a722, 0xd35d40e9d3b97c7d, 0x60ef65c4478901f1,
            0x945301feb0da841a, 0x2028c054ab187f51, 0xbe94b1f686a8b684, 0x09c13fdc1c4868c9,
            0xf2325ac2bf88a4ce, 0x92980d8fa53b6888, 0x8f6e17c7572a3359, 0x2964c5bfdd7761f2,
            0xf60269fc4910b562, 0x3ca164c4a2183ab0, 0x13f4f9e5a06a95c9, 0xf75022e39380598a,
            0x0d3f3c870002ab76, 0x24a4beb4780b78ef, 0x17a59a8f5696d625, 0x0ad76de884cb489d,
            0x559d3d0681553d6a, 0x813dcf205788af76, 0xf42f9c3ad707bf72, 0x770d63ceb129026c,
            0xa604d413fc14c7c2, 0x3cfc19e01239c784, 0xec7ef19965cedd56, 0x7303dcb3b300b6fd,
            0x118059e1139c0f3c, 0x97097186308c91f7, 0x2ad91d77379dce42, 0xad396c61acbe15ec,
            0x728518461b5722b6, 0xb85c5bb1ed805ecd, 0x816abc04592a4974, 0x1866b17c7cfbd0d0,
            0x0000000000000000};

        struct multiplier_index_info {
            std::uint16_t first_cache_bit_index;
            std::uint16_t cache_bit_index_offset;
            std::uint16_t cache_block_count_index_offset;
        };

        static constexpr multiplier_index_info multiplier_index_info_table[] = {
            {0, 0, 0},        {860, 1698, 13},  {2506, 4181, 29}, {2941, 5069, 36},
            {3577, 5705, 41}, {3961, 6088, 44}, {4092, 0, 0}};

        static constexpr std::uint8_t cache_block_counts[] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xee,
                                                              0xee, 0xee, 0xee, 0xee, 0xac, 0x68,
                                                              0x24, 0x8a, 0x46, 0x62, 0x24, 0x13};
    };

    // precision means the number of decimal significand digits minus 1.
    // Assumes round-to-nearest, tie-to-even rounding.
    template <class MainCache = main_cache_full, class ExtendedCache>
    JKJ_SAFEBUFFERS char* floff(double const x, int const precision, char* buffer) noexcept {
        assert(precision >= 0);
        using namespace detail;

        std::uint64_t br = default_float_traits<double>::float_to_carrier(x);
        bool is_negative = ((br >> 63) != 0);
        br <<= 1;
        int e = int(br >> (ieee754_binary64::significand_bits + 1));
        auto significand = (br & ((std::uint64_t(1) << (ieee754_binary64::significand_bits + 1)) -
                                  1)); // shifted by 1-bit.

        // Infinities or NaN
        if (e == ((std::uint32_t(1) << ieee754_binary64::exponent_bits) - 1)) {
            if (significand == 0) {
                if (is_negative) {
                    *buffer = '-';
                    ++buffer;
                }
                std::memcpy(buffer, "Infinity", 8);
                return buffer + 8;
            }
            else {
                std::memcpy(buffer, "NaN", 3);
                return buffer + 3;
            }
        }
        else {
            if (is_negative) {
                *buffer = '-';
                ++buffer;
            }
            // Normal numbers.
            if (e != 0) {
                significand |=
                    (decltype(significand)(1) << (ieee754_binary64::significand_bits + 1));
                e += (ieee754_binary64::exponent_bias - ieee754_binary64::significand_bits);
            }
            // Subnormal numbers.
            else {
                // Zero
                if (significand == 0) {
                    if (precision == 0) {
                        std::memcpy(buffer, "0e+00", 5);
                        return buffer + 5;
                    }
                    else {
                        std::memcpy(buffer, "0.", 2);
                        std::memset(buffer + 2, '0', precision);
                        std::memcpy(buffer + 2 + precision, "e+00", 4);
                        return buffer + precision + 6;
                    }
                }
                // Nonzero
                e = ieee754_binary64::min_exponent - ieee754_binary64::significand_bits;
            }
        }

        constexpr int kappa = 2;
        int k = kappa - detail::log::floor_log10_pow2(e);
        std::uint32_t current_digits;
        char* const buffer_starting_pos = buffer;
        int decimal_exponent = -k;
        int remaining_digits = precision + 1;

        /////////////////////////////////////////////////////////////////////////////////////////////////
        /// Phase 1 - Print the first digit segment computed with the Dragonbox table.
        /////////////////////////////////////////////////////////////////////////////////////////////////

        {
            // Compute the first digit segment.
            auto const main_cache = MainCache::template get_cache<ieee754_binary64>(k);
            int const beta = e + log::floor_log2_pow10(k);

            // Integer check is okay for binary64.
            auto [first_segment, has_more_segments] = [&] {
                auto const r = wuint::umul192_upper128(significand << beta, main_cache);
                return compute_mul_result{r.high(), r.low() != 0};
            }();

            // The first segment can be up to 19 digits. It is in fact always of either 18 or 19
            // digits except when the input is a subnormal number. For subnormal numbers, the
            // smallest possible value of the first segment is 10^kappa, so it is of at least
            // kappa+1 digits.

            if (remaining_digits <= 2) {
                wuint::uint128 prod;
                std::uint64_t fractional_part64;
                std::uint64_t fractional_part_rounding_threshold64;
                std::uint32_t current_digits;

                // Convert to fixed-point form with 64/32-bit boundary for the fractional part.

                // 19 digits.
                if (first_segment >= 100'0000'0000'0000'0000ull) {
                    if (remaining_digits == 1) {
                        prod = wuint::umul128(first_segment, 1329227995784915873ull);
                        // ceil(2^63 + 2^64/10^18)
                        fractional_part_rounding_threshold64 = additional_static_data_holder::
                            fractional_part_rounding_thresholds64[17];
                    }
                    else {
                        prod = wuint::umul128(first_segment, 13292279957849158730ull);
                        // ceil(2^63 + 2^64/10^17)
                        fractional_part_rounding_threshold64 = additional_static_data_holder::
                            fractional_part_rounding_thresholds64[16];
                    }
                    fractional_part64 = (prod.low() >> 56) | (prod.high() << 8);
                    current_digits = std::uint32_t(prod.high() >> 56);
                    decimal_exponent += 18;
                }
                // 18 digits.
                else if (first_segment >= 10'0000'0000'0000'0000ull) {
                    if (remaining_digits == 1) {
                        prod = wuint::umul128(first_segment, 830767497365572421ull);
                        // ceil(2^63 + 2^64/10^17)
                        fractional_part_rounding_threshold64 = additional_static_data_holder::
                            fractional_part_rounding_thresholds64[16];
                    }
                    else {
                        prod = wuint::umul128(first_segment, 8307674973655724206ull);
                        // ceil(2^63 + 2^64/10^16)
                        fractional_part_rounding_threshold64 = additional_static_data_holder::
                            fractional_part_rounding_thresholds64[15];
                    }
                    fractional_part64 = (prod.low() >> 52) | (prod.high() << 12);
                    current_digits = std::uint32_t(prod.high() >> 52);
                    decimal_exponent += 17;
                }
                // This branch can be taken only for subnormal numbers.
                else {
                    // At least 10 digits.
                    if (first_segment >= 10'0000'0000) {
                        // 15 ~ 17 digits.
                        if (first_segment >= 100'0000'0000'0000ull) {
                            decimal_exponent += 6;
                        }
                        // 12 ~ 14 digits.
                        else if (first_segment >= 1000'0000'0000ull) {
                            first_segment *= 1000;
                            decimal_exponent += 3;
                        }
                        // 10 ~ 11 digits.
                        else {
                            first_segment *= 100'0000;
                        }

                        // 17 or 14 or 11 digits.
                        if (first_segment >= 1'0000'0000'0000'0000ull) {
                            decimal_exponent += 10;
                        }
                        // 16 or 13 or 10 digits.
                        else if (first_segment >= 1000'0000'0000'0000ull) {
                            first_segment *= 10;
                            decimal_exponent += 9;
                        }
                        // 15 or 12 digits.
                        else {
                            first_segment *= 100;
                            decimal_exponent += 8;
                        }

                        if (remaining_digits == 1) {
                            prod = wuint::umul128(first_segment, 32451855365842673ull);
                            // ceil(2^63 + 2^64/10^16)
                            fractional_part_rounding_threshold64 = additional_static_data_holder::
                                fractional_part_rounding_thresholds64[15];
                        }
                        else {
                            prod = wuint::umul128(first_segment, 324518553658426727ull);
                            // ceil(2^63 + 2^64/10^15)
                            fractional_part_rounding_threshold64 = additional_static_data_holder::
                                fractional_part_rounding_thresholds64[14];
                        }
                        fractional_part64 = (prod.low() >> 44) | (prod.high() << 20);
                        current_digits = std::uint32_t(prod.high() >> 44);
                    }
                    // At most 9 digits (and at least 3 digits).
                    else {
                        // The segment fits into 32-bits in this case.
                        auto segment32 = std::uint32_t(first_segment);

                        // 7 ~ 9 digits
                        if (segment32 >= 100'0000) {
                            decimal_exponent += 6;
                        }
                        // 4 ~ 6 digits
                        else if (segment32 >= 1000) {
                            segment32 *= 1000;
                            decimal_exponent += 3;
                        }
                        // 3 digits
                        else {
                            segment32 *= 100'0000;
                        }

                        // 9 or 6 or 3 digits
                        if (segment32 >= 1'0000'0000) {
                            decimal_exponent += 2;
                        }
                        // 8 or 5 digits
                        else if (segment32 >= 1000'0000) {
                            segment32 *= 10;
                            decimal_exponent += 1;
                        }
                        // 7 or 4 digits
                        else {
                            segment32 *= 100;
                        }

                        std::uint64_t prod;
                        if (remaining_digits == 1) {
                            prod = (segment32 * std::uint64_t(1441151882)) >> 25;
                            current_digits = std::uint32_t(prod >> 32);

                            if (check_rounding_condition_inside_subsegment(
                                    current_digits, std::uint32_t(prod), 8, has_more_segments)) {
                                if (++current_digits == 10) {
                                    *buffer = '1';
                                    ++buffer;
                                    ++decimal_exponent;
                                    goto print_exponent_and_return;
                                }
                            }
                            print_1_digit(current_digits, buffer);
                            ++buffer;
                        }
                        else {
                            prod = (segment32 * std::uint64_t(450359963)) >> 29;
                            current_digits = std::uint32_t(prod >> 32);

                            if (check_rounding_condition_inside_subsegment(
                                    current_digits, std::uint32_t(prod), 7, has_more_segments)) {
                                if (++current_digits == 100) {
                                    std::memcpy(buffer, "1.0", 3);
                                    buffer += 3;
                                    ++decimal_exponent;
                                    goto print_exponent_and_return;
                                }
                            }
                            buffer[0] =
                                additional_static_data_holder::radix_100_table[current_digits * 2];
                            buffer[1] = '.';
                            buffer[2] =
                                additional_static_data_holder::radix_100_table[current_digits * 2 +
                                                                               1];
                            buffer += 3;
                        }
                        goto print_exponent_and_return;
                    }
                }

                // Perform rounding, print the digit, and return.
                if (remaining_digits == 1) {
                    if (fractional_part64 >= fractional_part_rounding_threshold64 ||
                        ((fractional_part64 >> 63) & (has_more_segments | (current_digits & 1))) !=
                            0) {
                        if (++current_digits == 10) {
                            *buffer = '1';
                            ++buffer;
                            ++decimal_exponent;
                            goto print_exponent_and_return;
                        }
                    }
                    print_1_digit(current_digits, buffer);
                    ++buffer;
                }
                else {
                    if (fractional_part64 >= fractional_part_rounding_threshold64 ||
                        ((fractional_part64 >> 63) & (has_more_segments | (current_digits & 1))) !=
                            0) {
                        if (++current_digits == 100) {
                            std::memcpy(buffer, "1.0", 3);
                            buffer += 3;
                            ++decimal_exponent;
                            goto print_exponent_and_return;
                        }
                    }
                    buffer[0] = additional_static_data_holder::radix_100_table[current_digits * 2];
                    buffer[1] = '.';
                    buffer[2] =
                        additional_static_data_holder::radix_100_table[current_digits * 2 + 1];
                    buffer += 3;
                }
                goto print_exponent_and_return;
            } // remaining_digits <= 2

            // At this point, there are at least 3 digits to print.
            *buffer = '0'; // to simplify rounding.
            ++buffer;

            // We split the segment into three chunks, each consisting of 9 digits, 8 digits,
            // and 2 digits.

            // MSVC doesn't know how to do Grandlund-Montgomery for large 64-bit integers.
            // 7922816251426433760 = ceil(2^96/10^10) = floor(2^96*(10^9/(10^19 - 1)))
            auto const first_subsegment =
                std::uint32_t(wuint::umul128_upper64(first_segment, 7922816251426433760ull) >> 32);
            auto const second_third_subsegments =
                first_segment - first_subsegment * 100'0000'0000ull;
            assert(first_subsegment < 10'0000'0000);
            assert(second_third_subsegments < 100'0000'0000ull);

            int remaining_digits_in_the_current_subsegment;
            std::uint64_t prod; // holds intermediate values for digit generation.

            // Print the first subsegment.
            if (first_subsegment != 0) {
                // 9 digits (19 digits in total).
                if (first_subsegment >= 1'0000'0000) {
                    // 1441151882 = ceil(2^57 / 10^8) + 1
                    prod = first_subsegment * std::uint64_t(1441151882);
                    prod >>= 25;
                    remaining_digits_in_the_current_subsegment = 8;
                }
                // 7 or 8 digits (17 or 18 digits in total).
                else if (first_subsegment >= 100'0000) {
                    // 281474978 = ceil(2^48 / 10^6) + 1
                    prod = first_subsegment * std::uint64_t(281474978);
                    prod >>= 16;
                    remaining_digits_in_the_current_subsegment = 6;
                }
                // 5 or 6 digits (15 or 16 digits in total).
                else if (first_subsegment >= 1'0000) {
                    // 429497 = ceil(2^32 / 10^4)
                    prod = first_subsegment * std::uint64_t(429497);
                    remaining_digits_in_the_current_subsegment = 4;
                }
                // 3 or 4 digits (13 or 14 digits in total).
                else if (first_subsegment >= 100) {
                    // 42949673 = ceil(2^32 / 10^2)
                    prod = first_subsegment * std::uint64_t(42949673);
                    remaining_digits_in_the_current_subsegment = 2;
                }
                // 1 or 2 digits (11 or 12 digits in total).
                else {
                    prod = std::uint64_t(first_subsegment) << 32;
                    remaining_digits_in_the_current_subsegment = 0;
                }

                auto const initial_digits = std::uint32_t(prod >> 32);
                decimal_exponent += (11 - (initial_digits < 10 ? 1 : 0) +
                                     remaining_digits_in_the_current_subsegment);

                buffer -= (initial_digits < 10 ? 1 : 0);
                remaining_digits -= (2 - (initial_digits < 10 ? 1 : 0));
                print_2_digits(initial_digits, buffer);
                buffer += 2;

                if (remaining_digits > remaining_digits_in_the_current_subsegment) {
                    remaining_digits -= remaining_digits_in_the_current_subsegment;
                    for (; remaining_digits_in_the_current_subsegment > 0;
                         remaining_digits_in_the_current_subsegment -= 2) {
                        // Write next two digits.
                        prod = std::uint32_t(prod) * std::uint64_t(100);
                        print_2_digits(std::uint32_t(prod >> 32), buffer);
                        buffer += 2;
                    }
                }
                else {
                    for (int i = 0; i < (remaining_digits - 1) / 2; ++i) {
                        // Write next two digits.
                        prod = std::uint32_t(prod) * std::uint64_t(100);
                        print_2_digits(std::uint32_t(prod >> 32), buffer);
                        buffer += 2;
                    }

                    // Distinguish two cases of rounding.
                    if (remaining_digits_in_the_current_subsegment > remaining_digits) {
                        if ((remaining_digits & 1) != 0) {
                            prod = std::uint32_t(prod) * std::uint64_t(10);
                        }
                        else {
                            prod = std::uint32_t(prod) * std::uint64_t(100);
                        }
                        current_digits = std::uint32_t(prod >> 32);

                        if (check_rounding_condition_inside_subsegment(
                                current_digits, std::uint32_t(prod),
                                remaining_digits_in_the_current_subsegment - remaining_digits,
                                second_third_subsegments != 0 || has_more_segments)) {
                            goto round_up;
                        }
                        goto print_last_digits;
                    }
                    else {
                        prod = std::uint32_t(prod) * std::uint64_t(100);
                        current_digits = std::uint32_t(prod >> 32);

                        if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                current_digits,
                                uint_with_known_number_of_digits<10>{second_third_subsegments},
                                has_more_segments)) {
                            goto round_up_two_digits;
                        }
                        goto print_last_two_digits;
                    }
                }
            }

            // Print the second subsegment.
            // The second subsegment cannot be zero even for subnormal numbers.

            if (remaining_digits <= 2) {
                // In this case the first subsegment must be nonzero.

                if (remaining_digits == 1) {
                    auto const prod = wuint::umul128(second_third_subsegments, 18446744074ull);

                    current_digits = std::uint32_t(prod.high());
                    auto const fractional_part64 = prod.low() + 1;
                    // 18446744074 is even, so prod.low() cannot be equal to 2^64 - 1.
                    assert(fractional_part64 != 0);

                    if (fractional_part64 >= additional_static_data_holder::
                                                 fractional_part_rounding_thresholds64[8] ||
                        ((fractional_part64 >> 63) & (has_more_segments | (current_digits & 1))) !=
                            0) {
                        goto round_up_one_digit;
                    }
                    goto print_last_one_digit;
                } // remaining_digits == 1
                else {
                    auto const prod = wuint::umul128(second_third_subsegments, 184467440738ull);

                    current_digits = std::uint32_t(prod.high());
                    auto const fractional_part64 = prod.low() + 1;
                    // 184467440738 is even, so prod.low() cannot be equal to 2^64 - 1.
                    assert(fractional_part64 != 0);

                    if (fractional_part64 >= additional_static_data_holder::
                                                 fractional_part_rounding_thresholds64[7] ||
                        ((fractional_part64 >> 63) & (has_more_segments | (current_digits & 1))) !=
                            0) {
                        goto round_up_two_digits;
                    }
                    goto print_last_two_digits;
                }
            } // remaining_digits <= 2

            // Compilers are not aware of how to leverage the maximum value of
            // second_third_subsegments to find out a better magic number which allows us to
            // eliminate an additional shift.
            // 184467440737095517 = ceil(2^64/100) < floor(2^64*(10^8/(10^10 - 1))).
            auto const second_subsegment = std::uint32_t(
                wuint::umul128_upper64(second_third_subsegments, 184467440737095517ull));
            // Since the final result is of 2 digits, we can do the computation in 32-bits.
            auto const third_subsegment =
                std::uint32_t(second_third_subsegments) - second_subsegment * 100;
            assert(second_subsegment < 1'0000'0000);
            assert(third_subsegment < 100);
            {
                std::uint32_t initial_digits;
                if (first_subsegment != 0) {
                    prod = ((second_subsegment * std::uint64_t(281474977)) >> 16) + 1;
                    remaining_digits_in_the_current_subsegment = 6;

                    initial_digits = std::uint32_t(prod >> 32);
                    remaining_digits -= 2;
                }
                else {
                    // 7 or 8 digits (9 or 10 digits in total).
                    if (second_subsegment >= 100'0000) {
                        prod = (second_subsegment * std::uint64_t(281474978)) >> 16;
                        remaining_digits_in_the_current_subsegment = 6;
                    }
                    // 5 or 6 digits (7 or 8 digits in total).
                    else if (second_subsegment >= 1'0000) {
                        prod = second_subsegment * std::uint64_t(429497);
                        remaining_digits_in_the_current_subsegment = 4;
                    }
                    // 3 or 4 digits (5 or 6 digits in total).
                    else if (second_subsegment >= 100) {
                        prod = second_subsegment * std::uint64_t(42949673);
                        remaining_digits_in_the_current_subsegment = 2;
                    }
                    // 1 or 2 digits (3 or 4 digits in total).
                    else {
                        prod = std::uint64_t(second_subsegment) << 32;
                        remaining_digits_in_the_current_subsegment = 0;
                    }

                    initial_digits = std::uint32_t(prod >> 32);
                    decimal_exponent += (3 - (initial_digits < 10 ? 1 : 0) +
                                         remaining_digits_in_the_current_subsegment);

                    buffer -= (initial_digits < 10 ? 1 : 0);
                    remaining_digits -= (2 - (initial_digits < 10 ? 1 : 0));
                }

                print_2_digits(initial_digits, buffer);
                buffer += 2;

                if (remaining_digits > remaining_digits_in_the_current_subsegment) {
                    remaining_digits -= remaining_digits_in_the_current_subsegment;
                    for (; remaining_digits_in_the_current_subsegment > 0;
                         remaining_digits_in_the_current_subsegment -= 2) {
                        // Write next two digits.
                        prod = std::uint32_t(prod) * std::uint64_t(100);
                        print_2_digits(std::uint32_t(prod >> 32), buffer);
                        buffer += 2;
                    }
                }
                else {
                    for (int i = 0; i < (remaining_digits - 1) / 2; ++i) {
                        // Write next two digits.
                        prod = std::uint32_t(prod) * std::uint64_t(100);
                        print_2_digits(std::uint32_t(prod >> 32), buffer);
                        buffer += 2;
                    }

                    // Distinguish two cases of rounding.
                    if (remaining_digits_in_the_current_subsegment > remaining_digits) {
                        if ((remaining_digits & 1) != 0) {
                            prod = std::uint32_t(prod) * std::uint64_t(10);
                        }
                        else {
                            prod = std::uint32_t(prod) * std::uint64_t(100);
                        }
                        current_digits = std::uint32_t(prod >> 32);

                        if (check_rounding_condition_inside_subsegment(
                                current_digits, std::uint32_t(prod),
                                remaining_digits_in_the_current_subsegment - remaining_digits,
                                third_subsegment != 0 || has_more_segments)) {
                            goto round_up;
                        }
                        goto print_last_digits;
                    }
                    else {
                        prod = std::uint32_t(prod) * std::uint64_t(100);
                        current_digits = std::uint32_t(prod >> 32);

                        if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                current_digits,
                                uint_with_known_number_of_digits<2>{third_subsegment},
                                has_more_segments)) {
                            goto round_up_two_digits;
                        }
                        goto print_last_two_digits;
                    }
                }
            }

            // Print the third subsegment.
            {
                if (remaining_digits > 2) {
                    print_2_digits(third_subsegment, buffer);
                    buffer += 2;
                    remaining_digits -= 2;

                    // If there is no more segment, then fill remaining digits with 0's and return.
                    if (!has_more_segments) {
                        goto fill_remaining_digits_with_0s;
                    }
                }
                else if (remaining_digits == 1) {
                    prod = third_subsegment * std::uint64_t(429496730);
                    current_digits = std::uint32_t(prod >> 32);

                    if (check_rounding_condition_inside_subsegment(
                            current_digits, std::uint32_t(prod), 1, has_more_segments)) {
                        goto round_up_one_digit;
                    }
                    goto print_last_one_digit;
                }
                else {
                    // remaining_digits == 2.
                    // If there is no more segment, then print the current two digits and return.
                    if (!has_more_segments) {
                        print_2_digits(third_subsegment, buffer);
                        buffer += 2;
                        goto insert_decimal_dot;
                    }

                    // Otherwise, for performing the rounding, we have to wait until the next
                    // segment becomes available. This state can be detected afterwards by
                    // inspecting if remaining_digits == 0.
                    remaining_digits = 0;
                    current_digits = third_subsegment;
                }
            }
        }


        /////////////////////////////////////////////////////////////////////////////////////////////////
        /// Phase 2 - Print further digit segments computed with the extended cache table.
        /////////////////////////////////////////////////////////////////////////////////////////////////

        {
            auto multiplier_index =
                std::uint32_t(k + ExtendedCache::segment_length - ExtendedCache::k_min) /
                std::uint32_t(ExtendedCache::segment_length);
            int digits_in_the_second_segment;
            {
                auto const new_k =
                    ExtendedCache::k_min + int(multiplier_index) * ExtendedCache::segment_length;
                digits_in_the_second_segment = new_k - k;
                k = new_k;
            }
            auto const exp2_base = e + bits::countr_zero(significand);

            using cache_block_type = std::decay_t<decltype(ExtendedCache::cache[0])>;
            cache_block_type blocks[ExtendedCache::max_cache_blocks];
            cache_block_count_t<ExtendedCache::constant_block_count,
                                ExtendedCache::max_cache_blocks>
                cache_block_count;

            // Determine if 2^(e+k-e1) * 5^(k-k1) * n is not an integer, where e1, k1 are the first
            // and the second parameters, respectively.
            auto has_further_digits = [significand, exp2_base,
                                       &k](auto additional_neg_exp_of_2_c,
                                           auto additional_neg_exp_of_10_c) {
                constexpr auto additional_neg_exp_of_2_v =
                    int(decltype(additional_neg_exp_of_2_c)::value +
                        decltype(additional_neg_exp_of_10_c)::value);
                constexpr auto additional_neg_exp_of_5_v =
                    int(decltype(additional_neg_exp_of_10_c)::value);

                static_assert(additional_neg_exp_of_5_v < ExtendedCache::segment_length);


                constexpr auto min_neg_exp_of_5 =
                    (-ExtendedCache::k_min + additional_neg_exp_of_5_v) %
                    ExtendedCache::segment_length;

                // k >= k_right_threshold iff k - k1 >= 0.
                static_assert(additional_neg_exp_of_5_v + ExtendedCache::segment_length >=
                              1 + ExtendedCache::k_min);
                constexpr auto k_right_threshold =
                    ExtendedCache::k_min +
                    ((additional_neg_exp_of_5_v + ExtendedCache::segment_length - 1 -
                      ExtendedCache::k_min) /
                     ExtendedCache::segment_length) *
                        ExtendedCache::segment_length;

                // When the smallest absolute value of negative exponent for 5 is too big,
                // so whenever the exponent for 5 is negative, the result cannot be an
                // integer.
                if constexpr (min_neg_exp_of_5 > 23) {
                    return has_further_digits_impl::no_neg_k_can_be_integer<
                        k_right_threshold, additional_neg_exp_of_2_v>(k, exp2_base);
                }
                // When the smallest absolute value of negative exponent for 5 is big enough, so
                // the only negative exponent for 5 that allows the result to be an integer is the
                // smallest one.
                else if constexpr (min_neg_exp_of_5 + ExtendedCache::segment_length > 23) {
                    // k < k_left_threshold iff k - k1 < -min_neg_exp_of_5.
                    static_assert(additional_neg_exp_of_5_v + ExtendedCache::segment_length >=
                                  min_neg_exp_of_5 + 1 + ExtendedCache::k_min);
                    constexpr auto k_left_threshold =
                        ExtendedCache::k_min +
                        ((additional_neg_exp_of_5_v - min_neg_exp_of_5 +
                          ExtendedCache::segment_length - 1 - ExtendedCache::k_min) /
                         ExtendedCache::segment_length) *
                            ExtendedCache::segment_length;

                    return has_further_digits_impl::only_one_neg_k_can_be_integer<
                        k_left_threshold, k_right_threshold, additional_neg_exp_of_2_v,
                        min_neg_exp_of_5>(k, exp2_base, significand);
                }
                // When the smallest absolute value of negative exponent for 5 is big enough, so
                // the only negative exponents for 5 that allows the result to be an integer are the
                // smallest one and the next smallest one.
                else {
                    static_assert(min_neg_exp_of_5 + 2 * ExtendedCache::segment_length > 23);

                    constexpr auto k_left_threshold =
                        ExtendedCache::k_min +
                        ((additional_neg_exp_of_5_v - min_neg_exp_of_5 - 1 - ExtendedCache::k_min) /
                         ExtendedCache::segment_length) *
                            ExtendedCache::segment_length;
                    constexpr auto k_middle_threshold =
                        ExtendedCache::k_min +
                        ((additional_neg_exp_of_5_v - min_neg_exp_of_5 +
                          ExtendedCache::segment_length - 1 - ExtendedCache::k_min) /
                         ExtendedCache::segment_length) *
                            ExtendedCache::segment_length;

                    return has_further_digits_impl::only_two_neg_k_can_be_integer<
                        k_left_threshold, k_middle_threshold, k_right_threshold,
                        additional_neg_exp_of_2_v, min_neg_exp_of_5, ExtendedCache::segment_length>(
                        k, exp2_base, significand);
                }
            };

            // Deal with the second segment. The second segment is special because it can have
            // overlapping digits with the first segment. Note that we cannot just move the buffer
            // pointer backward and print the whole segment from there, because it may contain
            // leading zeros.
            {
                cache_block_count =
                    load_extended_cache<ExtendedCache, ExtendedCache::constant_block_count>(
                        blocks, e, k, multiplier_index);

                // Compute nm mod 2^Q.
                fixed_point_calculator<ExtendedCache::max_cache_blocks>::discard_upper(
                    significand, blocks, cache_block_count);

                if constexpr (ExtendedCache::segment_length == 22) {
                    // No rounding, continue.
                    if (remaining_digits > digits_in_the_second_segment) {
                        remaining_digits -= digits_in_the_second_segment;

                        if (digits_in_the_second_segment <= 2) {
                            assert(digits_in_the_second_segment != 0);

                            fixed_point_calculator<ExtendedCache::max_cache_blocks>::discard_upper(
                                power_of_10<19>, blocks, cache_block_count);

                            auto subsegment =
                                fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                    generate_and_discard_lower(power_of_10<3>, blocks,
                                                               cache_block_count);

                            if (digits_in_the_second_segment == 1) {
                                auto prod = subsegment * std::uint64_t(429496730);
                                prod = std::uint32_t(prod) * std::uint64_t(10);
                                print_1_digit(std::uint32_t(prod >> 32), buffer);
                                ++buffer;
                            }
                            else {
                                auto prod = subsegment * std::uint64_t(42949673);
                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                print_2_digits(std::uint32_t(prod >> 32), buffer);
                                buffer += 2;
                            }
                        } // digits_in_the_second_segment <= 2
                        else if (digits_in_the_second_segment <= 16) {
                            assert(22 - digits_in_the_second_segment <= 19);
                            fixed_point_calculator<ExtendedCache::max_cache_blocks>::discard_upper(
                                compute_power(std::uint64_t(10), 22 - digits_in_the_second_segment),
                                blocks, cache_block_count);

                            // When there are at most 9 digits, we can store them in 32-bits.
                            if (digits_in_the_second_segment <= 9) {
                                // The number of overlapping digits is in the range 13 ~ 19.
                                auto const subsegment =
                                    fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                        generate_and_discard_lower(power_of_10<9>, blocks,
                                                                   cache_block_count);

                                std::uint64_t prod;
                                if ((digits_in_the_second_segment & 1) != 0) {
                                    prod = ((subsegment * std::uint64_t(720575941)) >> 24) + 1;
                                    print_1_digit(std::uint32_t(prod >> 32), buffer);
                                    ++buffer;
                                }
                                else {
                                    prod = ((subsegment * std::uint64_t(450359963)) >> 20) + 1;
                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                }
                                for (; digits_in_the_second_segment > 2;
                                     digits_in_the_second_segment -= 2) {
                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                }
                            } // digits_in_the_second_segment <= 9
                            else {
                                // The number of digits in the segment is in the range 10 ~ 16.
                                auto const first_second_subsegments =
                                    fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                        generate_and_discard_lower(power_of_10<16>, blocks,
                                                                   cache_block_count);

                                // The first segment is of 8 digits, and the second segment is of
                                // 2 ~ 8 digits.
                                // ceil(2^(64+14)/10^8) = 3022314549036573
                                // = floor(2^(64+14)*(10^8/(10^16 - 1)))
                                auto const first_subsegment =
                                    std::uint32_t(wuint::umul128_upper64(first_second_subsegments,
                                                                         3022314549036573ull) >>
                                                  14);
                                auto const second_subsegment =
                                    std::uint32_t(first_second_subsegments) -
                                    1'0000'0000 * first_subsegment;

                                // Print the first subsegment.
                                print_8_digits(first_subsegment, buffer);
                                buffer += 8;

                                // Print the second subsegment.
                                // There are at least 2 digits in the second subsegment.
                                auto prod =
                                    ((second_subsegment * std::uint64_t(140737489)) >> 15) + 1;
                                print_2_digits(std::uint32_t(prod >> 32), buffer);
                                buffer += 2;
                                digits_in_the_second_segment -= 10;

                                for (; digits_in_the_second_segment > 1;
                                     digits_in_the_second_segment -= 2) {
                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                }
                                if (digits_in_the_second_segment != 0) {
                                    prod = std::uint32_t(prod) * std::uint64_t(10);
                                    print_1_digit(std::uint32_t(prod >> 32), buffer);
                                    ++buffer;
                                }
                            }
                        } // digits_in_the_second_segment <= 16
                        else {
                            // The number of digits in the segment is in the range 17 ~ 22.
                            auto const first_subsegment =
                                fixed_point_calculator<ExtendedCache::max_cache_blocks>::generate(
                                    power_of_10<6>, blocks, cache_block_count);

                            auto const second_third_subsegments =
                                fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                    generate_and_discard_lower(power_of_10<16>, blocks,
                                                               cache_block_count);

                            // ceil(2^(64+14)/10^8) = 3022314549036573
                            // = floor(2^(64+14)*(10^8/(10^16 - 1)))
                            auto const second_subsegment =
                                std::uint32_t(wuint::umul128_upper64(second_third_subsegments,
                                                                     3022314549036573ull) >>
                                              14);
                            auto const third_subsegment = std::uint32_t(second_third_subsegments) -
                                                          1'0000'0000 * second_subsegment;

                            // Print the first subsegment (1 ~ 6 digits).
                            std::uint64_t prod;
                            auto remaining_digits_in_the_current_subsegment =
                                digits_in_the_second_segment - 16;
                            switch (remaining_digits_in_the_current_subsegment) {
                            case 1:
                                prod = first_subsegment * std::uint64_t(429496730);
                                goto second_segment22_more_than_16_digits_first_subsegment_no_rounding_odd_remaining;

                            case 2:
                                prod = first_subsegment * std::uint64_t(42949673);
                                goto second_segment22_more_than_16_digits_first_subsegment_no_rounding_even_remaining;

                            case 3:
                                prod = first_subsegment * std::uint64_t(4294968);
                                goto second_segment22_more_than_16_digits_first_subsegment_no_rounding_odd_remaining;

                            case 4:
                                prod = first_subsegment * std::uint64_t(429497);
                                goto second_segment22_more_than_16_digits_first_subsegment_no_rounding_even_remaining;

                            case 5:
                                prod = ((first_subsegment * std::uint64_t(687195)) >> 4) + 1;
                                goto second_segment22_more_than_16_digits_first_subsegment_no_rounding_odd_remaining;

                            case 6:
                                prod = first_subsegment * std::uint64_t(429497);
                                print_2_digits(std::uint32_t(prod >> 32), buffer);
                                buffer += 2;
                                remaining_digits_in_the_current_subsegment = 4;
                                goto second_segment22_more_than_16_digits_first_subsegment_no_rounding_even_remaining;

                            default:
                                JKJ_UNRECHABLE;
                            }

                        second_segment22_more_than_16_digits_first_subsegment_no_rounding_odd_remaining
                            :
                            prod = std::uint32_t(prod) * std::uint64_t(10);
                            print_1_digit(std::uint32_t(prod >> 32), buffer);
                            ++buffer;

                        second_segment22_more_than_16_digits_first_subsegment_no_rounding_even_remaining
                            :
                            for (; remaining_digits_in_the_current_subsegment > 1;
                                 remaining_digits_in_the_current_subsegment -= 2) {
                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                print_2_digits(std::uint32_t(prod >> 32), buffer);
                                buffer += 2;
                            }

                            // Print the second and third subsegments (8 digits each).
                            print_8_digits(second_subsegment, buffer);
                            print_8_digits(third_subsegment, buffer + 8);
                            buffer += 16;
                        }
                    } // remaining_digits > digits_in_the_second_segment

                    // Perform rounding and return.
                    else {
                        if (digits_in_the_second_segment <= 2) {
                            fixed_point_calculator<ExtendedCache::max_cache_blocks>::discard_upper(
                                power_of_10<19>, blocks, cache_block_count);

                            // Get one more bit for potential rounding on the segment boundary.
                            auto subsegment =
                                fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                    generate_and_discard_lower(2000, blocks, cache_block_count);

                            bool segment_boundary_rounding_bit = ((subsegment & 1) != 0);
                            subsegment >>= 1;

                            if (digits_in_the_second_segment == 2) {
                                // Convert subsegment into fixed-point fractional form where the
                                // integer part is of one digit. The integer part is ignored.
                                // 42949673 = ceil(2^32/10^2)
                                auto prod = subsegment * std::uint64_t(42949673);

                                if (remaining_digits == 1) {
                                    prod = std::uint32_t(prod) * std::uint64_t(10);
                                    current_digits = std::uint32_t(prod >> 32);

                                    if (check_rounding_condition_inside_subsegment(
                                            current_digits, std::uint32_t(prod), 1,
                                            has_further_digits, uconst<1>, uconst<0>)) {
                                        goto round_up_one_digit;
                                    }
                                    goto print_last_one_digit;
                                }

                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                auto const next_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 0) {
                                    if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                            current_digits,
                                            uint_with_known_number_of_digits<2>{next_digits},
                                            has_further_digits, uconst<1>, uconst<0>)) {
                                        goto round_up_two_digits;
                                    }
                                    goto print_last_two_digits;
                                }
                                current_digits = next_digits;
                                assert(remaining_digits == 2);
                            }
                            else {
                                assert(digits_in_the_second_segment == 1);
                                // Convert subsegment into fixed-point fractional form where the
                                // integer part is of two digits. The integer part is ignored.
                                // 429496730 = ceil(2^32/10^1)
                                auto prod = subsegment * std::uint64_t(429496730);
                                prod = std::uint32_t(prod) * std::uint64_t(10);
                                auto const next_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 0) {
                                    if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                            current_digits,
                                            uint_with_known_number_of_digits<1>{next_digits},
                                            has_further_digits, uconst<1>, uconst<0>)) {
                                        goto round_up_two_digits;
                                    }
                                    goto print_last_two_digits;
                                }
                                current_digits = next_digits;
                                assert(remaining_digits == 1);
                            }

                            if (check_rounding_condition_with_next_bit(
                                    current_digits, segment_boundary_rounding_bit,
                                    has_further_digits, uconst<0>, uconst<0>)) {
                                goto round_up;
                            }
                            goto print_last_digits;
                        } // digits_in_the_second_segment <= 2

                        // When there are at most 9 digits in the segment.
                        if (digits_in_the_second_segment <= 9) {
                            // Throw away all overlapping digits.
                            assert(22 - digits_in_the_second_segment <= 19);
                            fixed_point_calculator<ExtendedCache::max_cache_blocks>::discard_upper(
                                compute_power(std::uint64_t(10), 22 - digits_in_the_second_segment),
                                blocks, cache_block_count);

                            // Get one more bit for potential rounding on the segment boundary.
                            auto segment = fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                generate_and_discard_lower(power_of_10<9> << 1, blocks,
                                                           cache_block_count);

                            std::uint64_t prod;
                            digits_in_the_second_segment -= remaining_digits;

                            if ((remaining_digits & 1) != 0) {
                                prod = ((segment * std::uint64_t(1441151881)) >> 26) + 1;
                                current_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 1) {
                                    goto second_segment22_at_most_9_digits_rounding;
                                }

                                print_1_digit(current_digits, buffer);
                                ++buffer;
                            }
                            else {
                                prod = ((segment * std::uint64_t(1801439851)) >> 23) + 1;
                                auto const next_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 0) {
                                    if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                            current_digits,
                                            uint_with_known_number_of_digits<2>{next_digits}, [&] {
                                                return std::uint32_t(prod) >=
                                                           (additional_static_data_holder::
                                                                fractional_part_rounding_thresholds32
                                                                    [digits_in_the_second_segment -
                                                                     1] &
                                                            0x7fffffff) ||
                                                       has_further_digits(uconst<1>, uconst<0>);
                                            })) {
                                        goto round_up_two_digits;
                                    }
                                    goto print_last_two_digits;
                                }
                                else if (remaining_digits == 2) {
                                    current_digits = next_digits;
                                    goto second_segment22_at_most_9_digits_rounding;
                                }

                                print_2_digits(next_digits, buffer);
                                buffer += 2;
                            }

                            assert(remaining_digits >= 3);
                            for (int i = 0; i < (remaining_digits - 3) / 2; ++i) {
                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                print_2_digits(std::uint32_t(prod >> 32), buffer);
                                buffer += 2;
                            }

                            if (digits_in_the_second_segment != 0) {
                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                current_digits = std::uint32_t(prod >> 32);
                                remaining_digits = 0;

                            second_segment22_at_most_9_digits_rounding:
                                if (check_rounding_condition_inside_subsegment(
                                        current_digits, std::uint32_t(prod),
                                        digits_in_the_second_segment, has_further_digits, uconst<1>,
                                        uconst<0>)) {
                                    goto round_up;
                                }
                                goto print_last_digits;
                            }
                            else {
                                prod = std::uint32_t(prod) * std::uint64_t(200);
                                current_digits = std::uint32_t(prod >> 32);
                                auto const segment_boundary_rounding_bit =
                                    (current_digits & 1) != 0;
                                current_digits >>= 1;

                                if (check_rounding_condition_with_next_bit(
                                        current_digits, segment_boundary_rounding_bit,
                                        has_further_digits, uconst<0>, uconst<0>)) {
                                    goto round_up_two_digits;
                                }
                                goto print_last_two_digits;
                            }
                        } // digits_in_the_second_segment <= 9

                        // first_second_subsegments is of 1 ~ 13 digits, and third_subsegment is
                        // of 9 digits.
                        // Get one more bit for potential rounding condition check.
                        auto first_second_subsegments =
                            fixed_point_calculator<ExtendedCache::max_cache_blocks>::generate(
                                power_of_10<13> << 1, blocks, cache_block_count);
                        bool first_bit_of_third_subsegment = ((first_second_subsegments & 1) != 0);
                        first_second_subsegments >>= 1;

                        // Compilers are not aware of how to leverage the maximum value of
                        // first_second_subsegments to find out a better magic number which
                        // allows us to eliminate an additional shift.
                        // 1844674407371 = ceil(2^64/10^7) = floor(2^64*(10^6/(10^13 - 1))).
                        auto const first_subsegment =
                            std::uint32_t(jkj::floff::detail::wuint::umul128_upper64(
                                first_second_subsegments, 1844674407371));
                        auto const second_subsegment =
                            std::uint32_t(first_second_subsegments) - 1000'0000 * first_subsegment;

                        int digits_in_the_second_subsegment;

                        // Print the first subsegment (0 ~ 6 digits) if exists.
                        if (digits_in_the_second_segment > 16) {
                            std::uint64_t prod;
                            int remaining_digits_in_the_current_subsegment =
                                digits_in_the_second_segment - 16;

                            // No rounding, continue.
                            if (remaining_digits > remaining_digits_in_the_current_subsegment) {
                                remaining_digits -= remaining_digits_in_the_current_subsegment;

                                // There is no overlap in the second subsegment.
                                digits_in_the_second_subsegment = 7;

                                // When there is no overlapping digit.
                                if (remaining_digits_in_the_current_subsegment == 6) {
                                    prod = (first_subsegment * std::uint64_t(429497)) + 1;
                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                    remaining_digits_in_the_current_subsegment -= 2;
                                }
                                // If there are overlapping digits, move all overlapping digits
                                // into the integer part.
                                else {
                                    prod = ((first_subsegment * std::uint64_t(687195)) >> 4) + 1;
                                    prod *= compute_power(
                                        std::uint64_t(10),
                                        5 - remaining_digits_in_the_current_subsegment);

                                    if ((remaining_digits_in_the_current_subsegment & 1) != 0) {
                                        prod = std::uint32_t(prod) * std::uint64_t(10);
                                        print_1_digit(std::uint32_t(prod >> 32), buffer);
                                        ++buffer;
                                    }
                                }

                                for (; remaining_digits_in_the_current_subsegment > 1;
                                     remaining_digits_in_the_current_subsegment -= 2) {
                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                }
                            }
                            // The first subsegment is the last subsegment to print.
                            else {
                                if ((remaining_digits & 1) != 0) {
                                    prod = ((first_subsegment * std::uint64_t(687195)) >> 4) + 1;

                                    // If there are overlapping digits, move all overlapping digits
                                    // into the integer part and then get the next digit.
                                    if (remaining_digits_in_the_current_subsegment < 6) {
                                        prod *= compute_power(
                                            std::uint64_t(10),
                                            5 - remaining_digits_in_the_current_subsegment);
                                        prod = std::uint32_t(prod) * std::uint64_t(10);
                                    }
                                    current_digits = std::uint32_t(prod >> 32);
                                    remaining_digits_in_the_current_subsegment -= remaining_digits;

                                    if (remaining_digits == 1) {
                                        goto second_segment22_more_than_9_digits_first_subsegment_rounding;
                                    }

                                    print_1_digit(current_digits, buffer);
                                    ++buffer;
                                }
                                else {
                                    // When there is no overlapping digit.
                                    if (remaining_digits_in_the_current_subsegment == 6) {
                                        if (remaining_digits == 0) {
                                            if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                                    current_digits,
                                                    uint_with_known_number_of_digits<6>{
                                                        first_subsegment},
                                                    has_further_digits, uconst<1>, uconst<16>)) {
                                                goto round_up_two_digits;
                                            }
                                            goto print_last_two_digits;
                                        }

                                        prod = (first_subsegment * std::uint64_t(429497)) + 1;
                                    }
                                    // Otherwise, convert the subsegment into a fixed-point
                                    // fraction form, move all overlapping digits into the
                                    // integer part, and then extract the next two digits.
                                    else {
                                        prod =
                                            ((first_subsegment * std::uint64_t(687195)) >> 4) + 1;
                                        prod *= compute_power(
                                            std::uint64_t(10),
                                            5 - remaining_digits_in_the_current_subsegment);

                                        if (remaining_digits == 0) {
                                            goto second_segment22_more_than_9_digits_first_subsegment_rounding_inside_subsegment;
                                        }

                                        prod = std::uint32_t(prod) * std::uint64_t(100);
                                    }
                                    current_digits = std::uint32_t(prod >> 32);
                                    remaining_digits_in_the_current_subsegment -= remaining_digits;

                                    if (remaining_digits == 2) {
                                        goto second_segment22_more_than_9_digits_first_subsegment_rounding;
                                    }

                                    print_2_digits(current_digits, buffer);
                                    buffer += 2;
                                }

                                assert(remaining_digits >= 3);
                                if (remaining_digits > 4) {
                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                }

                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                current_digits = std::uint32_t(prod >> 32);
                                remaining_digits = 0;

                            second_segment22_more_than_9_digits_first_subsegment_rounding:
                                if (remaining_digits_in_the_current_subsegment == 0) {
                                    if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                            current_digits,
                                            uint_with_known_number_of_digits<7>{second_subsegment},
                                            has_further_digits, uconst<1>, uconst<9>)) {
                                        goto round_up;
                                    }
                                }
                                else {
                                second_segment22_more_than_9_digits_first_subsegment_rounding_inside_subsegment
                                    :
                                    if (check_rounding_condition_inside_subsegment(
                                            current_digits, std::uint32_t(prod),
                                            remaining_digits_in_the_current_subsegment,
                                            has_further_digits, uconst<1>, uconst<16>)) {
                                        goto round_up;
                                    }
                                }
                                goto print_last_digits;
                            }
                        }
                        else {
                            digits_in_the_second_subsegment = digits_in_the_second_segment - 9;
                        }

                        // Print the second subsegment (1 ~ 7 digits).
                        {
                            // No rounding, continue.
                            if (remaining_digits > digits_in_the_second_subsegment) {
                                auto prod =
                                    ((second_subsegment * std::uint64_t(17592187)) >> 12) + 1;
                                remaining_digits -= digits_in_the_second_subsegment;

                                // When there is no overlapping digit.
                                if (digits_in_the_second_subsegment == 7) {
                                    print_1_digit(std::uint32_t(prod >> 32), buffer);
                                    ++buffer;
                                }
                                // If there are overlapping digits, move all overlapping digits
                                // into the integer part.
                                else {
                                    prod *= compute_power(std::uint64_t(10),
                                                          6 - digits_in_the_second_subsegment);

                                    if ((digits_in_the_second_subsegment & 1) != 0) {
                                        prod = std::uint32_t(prod) * std::uint64_t(10);
                                        print_1_digit(std::uint32_t(prod >> 32), buffer);
                                        ++buffer;
                                    }
                                }

                                for (; digits_in_the_second_subsegment > 1;
                                     digits_in_the_second_subsegment -= 2) {
                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                }
                            }
                            // The second subsegment is the last subsegment to print.
                            else {
                                std::uint64_t prod;

                                if ((remaining_digits & 1) != 0) {
                                    prod =
                                        ((second_subsegment * std::uint64_t(17592187)) >> 12) + 1;

                                    // If there are overlapping digits, move all overlapping digits
                                    // into the integer part and then get the next digit.
                                    if (digits_in_the_second_subsegment < 7) {
                                        prod *= compute_power(std::uint64_t(10),
                                                              6 - digits_in_the_second_subsegment);
                                        prod = std::uint32_t(prod) * std::uint64_t(10);
                                    }
                                    current_digits = std::uint32_t(prod >> 32);
                                    digits_in_the_second_subsegment -= remaining_digits;

                                    if (remaining_digits == 1) {
                                        goto second_segment22_more_than_9_digits_second_subsegment_rounding;
                                    }

                                    print_1_digit(current_digits, buffer);
                                    ++buffer;
                                }
                                else {
                                    // When there is no overlapping digit.
                                    if (digits_in_the_second_subsegment == 7) {
                                        if (remaining_digits == 0) {
                                            if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                                    current_digits,
                                                    uint_with_known_number_of_digits<7>{
                                                        second_subsegment},
                                                    has_further_digits, uconst<1>, uconst<9>)) {
                                                goto round_up_two_digits;
                                            }
                                            goto print_last_two_digits;
                                        }

                                        prod =
                                            ((second_subsegment * std::uint64_t(10995117)) >> 8) +
                                            1;
                                    }
                                    // Otherwise, convert the subsegment into a fixed-point
                                    // fraction form, move all overlapping digits into the
                                    // integer part, and then extract the next two digits.
                                    else {
                                        prod =
                                            ((second_subsegment * std::uint64_t(17592187)) >> 12) +
                                            1;
                                        prod *= compute_power(std::uint64_t(10),
                                                              6 - digits_in_the_second_subsegment);

                                        if (remaining_digits == 0) {
                                            goto second_segment22_more_than_9_digits_second_subsegment_rounding_inside_subsegment;
                                        }

                                        prod = std::uint32_t(prod) * std::uint64_t(100);
                                    }
                                    current_digits = std::uint32_t(prod >> 32);
                                    digits_in_the_second_subsegment -= remaining_digits;

                                    if (remaining_digits == 2) {
                                        goto second_segment22_more_than_9_digits_second_subsegment_rounding;
                                    }

                                    print_2_digits(current_digits, buffer);
                                    buffer += 2;
                                }

                                assert(remaining_digits >= 3);
                                if (remaining_digits > 4) {
                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                }

                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                current_digits = std::uint32_t(prod >> 32);
                                remaining_digits = 0;

                            second_segment22_more_than_9_digits_second_subsegment_rounding:
                                if (digits_in_the_second_subsegment == 0) {
                                    if (check_rounding_condition_with_next_bit(
                                            current_digits, first_bit_of_third_subsegment,
                                            has_further_digits, uconst<0>, uconst<9>)) {
                                        goto round_up;
                                    }
                                }
                                else {
                                second_segment22_more_than_9_digits_second_subsegment_rounding_inside_subsegment
                                    :
                                    if (check_rounding_condition_inside_subsegment(
                                            current_digits, std::uint32_t(prod),
                                            digits_in_the_second_subsegment, has_further_digits,
                                            uconst<1>, uconst<9>)) {
                                        goto round_up;
                                    }
                                }
                                goto print_last_digits;
                            }
                        }

                        // Print the third subsegment (9 digits).
                        {
                            // Get one more bit if we need to check rounding conditions on
                            // the segment boundary. We already have shifted by 1-bit in the
                            // computation of first & second subsegments, so here we don't
                            // shift the multiplier.
                            auto third_subsegment =
                                fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                    generate_and_discard_lower(power_of_10<9>, blocks,
                                                               cache_block_count);

                            bool segment_boundary_rounding_bit = ((third_subsegment & 1) != 0);
                            third_subsegment >>= 1;
                            third_subsegment += (first_bit_of_third_subsegment ? 5'0000'0000 : 0);

                            std::uint64_t prod;
                            if ((remaining_digits & 1) != 0) {
                                prod = ((third_subsegment * std::uint64_t(720575941)) >> 24) + 1;
                                current_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 1) {
                                    if (check_rounding_condition_inside_subsegment(
                                            current_digits, std::uint32_t(prod), 8,
                                            has_further_digits, uconst<1>, uconst<0>)) {
                                        goto round_up_one_digit;
                                    }
                                    goto print_last_one_digit;
                                }

                                print_1_digit(current_digits, buffer);
                                ++buffer;
                            }
                            else {
                                prod = ((third_subsegment * std::uint64_t(450359963)) >> 20) + 1;
                                current_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 2) {
                                    goto second_segment22_more_than_9_digits_third_subsegment_rounding;
                                }

                                print_2_digits(current_digits, buffer);
                                buffer += 2;
                            }

                            for (int i = 0; i < (remaining_digits - 3) / 2; ++i) {
                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                print_2_digits(std::uint32_t(prod >> 32), buffer);
                                buffer += 2;
                            }

                            prod = std::uint32_t(prod) * std::uint64_t(100);
                            current_digits = std::uint32_t(prod >> 32);

                            if (remaining_digits < 9) {
                            second_segment22_more_than_9_digits_third_subsegment_rounding:
                                if (check_rounding_condition_inside_subsegment(
                                        current_digits, std::uint32_t(prod), 9 - remaining_digits,
                                        has_further_digits, uconst<1>, uconst<0>)) {
                                    goto round_up_two_digits;
                                }
                            }
                            else {
                                if (check_rounding_condition_with_next_bit(
                                        current_digits, segment_boundary_rounding_bit,
                                        has_further_digits, uconst<0>, uconst<0>)) {
                                    goto round_up_two_digits;
                                }
                            }
                            goto print_last_two_digits;
                        }
                    }
                } // ExtendedCache::segment_length == 22

                else if constexpr (ExtendedCache::segment_length == 252) {
                    int overlapping_digits = 252 - digits_in_the_second_segment;
                    int remaining_subsegment_pairs = 14;

                    while (overlapping_digits >= 18) {
                        fixed_point_calculator<ExtendedCache::max_cache_blocks>::discard_upper(
                            power_of_10<18>, blocks, cache_block_count);
                        --remaining_subsegment_pairs;
                        overlapping_digits -= 18;
                    }

                    auto subsegment_pair =
                        fixed_point_calculator<ExtendedCache::max_cache_blocks>::generate(
                            power_of_10<18> << 1, blocks, cache_block_count);
                    auto subsegment_boundary_rounding_bit = (subsegment_pair & 1) != 0;
                    subsegment_pair >>= 1;

                    auto compute_has_further_digits = [&](auto additional_neg_exp_of_2,
                                                          auto additional_neg_exp_of_10) {
#define JKJ_FLOFF_252_HAS_FURTHER_DIGITS(n)                                                        \
case n:                                                                                            \
    return has_further_digits(additional_neg_exp_of_2,                                             \
                              uconst<decltype(additional_neg_exp_of_10)::value + (n - 1) * 18>);
                        switch (remaining_subsegment_pairs) {
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(1);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(2);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(3);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(4);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(5);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(6);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(7);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(8);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(9);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(10);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(11);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(12);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(13);
                            JKJ_FLOFF_252_HAS_FURTHER_DIGITS(14);

                        default:
                            JKJ_UNRECHABLE;
                        }
#undef JKJ_FLOFF_252_HAS_FURTHER_DIGITS
                    };

                    // Deal with the first subsegment pair.
                    {
                        // Divide it into two 9-digits subsegments.
                        auto const first_part = std::uint32_t(subsegment_pair / power_of_10<9>);
                        auto const second_part =
                            std::uint32_t(subsegment_pair) - power_of_10<9> * first_part;

                        auto print_subsegment = [&](auto subsegment, int digits_in_the_subsegment) {
                            remaining_digits -= digits_in_the_subsegment;

                            // Move all overlapping digits into the integer part.
                            auto prod = ((subsegment * std::uint64_t(720575941)) >> 24) + 1;
                            if (digits_in_the_subsegment < 9) {
                                prod *=
                                    compute_power(std::uint32_t(10), 8 - digits_in_the_subsegment);

                                if ((digits_in_the_subsegment & 1) != 0) {
                                    prod = std::uint32_t(prod) * std::uint64_t(10);
                                    print_1_digit(std::uint32_t(prod >> 32), buffer);
                                    ++buffer;
                                }
                            }
                            else {
                                print_1_digit(std::uint32_t(prod >> 32), buffer);
                                ++buffer;
                            }

                            for (; digits_in_the_subsegment > 1; digits_in_the_subsegment -= 2) {
                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                print_2_digits(std::uint32_t(prod >> 32), buffer);
                                buffer += 2;
                            }
                        };

                        // When the first part is not completely overlapping with the first segment.
                        int digits_in_the_second_part;
                        if (overlapping_digits < 9) {
                            int digits_in_the_first_part = 9 - overlapping_digits;

                            // No rounding, continue.
                            if (remaining_digits > digits_in_the_first_part) {
                                digits_in_the_second_part = 9;
                                print_subsegment(first_part, digits_in_the_first_part);
                            }
                            // Perform rounding and return.
                            else {
                                // When there is no overlapping digit.
                                std::uint64_t prod;
                                if (digits_in_the_first_part == 9) {
                                    if ((remaining_digits & 1) != 0) {
                                        prod = ((first_part * std::uint64_t(720575941)) >> 24) + 1;
                                    }
                                    else {
                                        if (remaining_digits == 0) {
                                            if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                                    current_digits,
                                                    uint_with_known_number_of_digits<9>{first_part},
                                                    compute_has_further_digits, uconst<1>,
                                                    uconst<9>)) {
                                                goto round_up_two_digits;
                                            }
                                            goto print_last_two_digits;
                                        }

                                        prod = ((first_part * std::uint64_t(450359963)) >> 20) + 1;
                                    }
                                }
                                else {
                                    prod = ((first_part * std::uint64_t(720575941)) >> 24) + 1;
                                    prod *= compute_power(std::uint32_t(10),
                                                          8 - digits_in_the_first_part);

                                    if ((remaining_digits & 1) != 0) {
                                        prod = std::uint32_t(prod) * std::uint64_t(10);
                                    }
                                    else {
                                        if (remaining_digits == 0) {
                                            goto second_segment252_first_subsegment_rounding_inside_subsegment;
                                        }

                                        prod = std::uint32_t(prod) * std::uint64_t(100);
                                    }
                                }
                                digits_in_the_first_part -= remaining_digits;
                                current_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits > 2) {
                                    if ((remaining_digits & 1) != 0) {
                                        print_1_digit(current_digits, buffer);
                                        ++buffer;
                                    }
                                    else {
                                        print_2_digits(current_digits, buffer);
                                        buffer += 2;
                                    }

                                    for (int i = 0; i < (remaining_digits - 3) / 2; ++i) {
                                        prod = std::uint32_t(prod) * std::uint64_t(100);
                                        print_2_digits(std::uint32_t(prod >> 32), buffer);
                                        buffer += 2;
                                    }

                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                    current_digits = std::uint32_t(prod >> 32);
                                    remaining_digits = 0;
                                }

                                if (digits_in_the_first_part != 0) {
                                second_segment252_first_subsegment_rounding_inside_subsegment:
                                    if (check_rounding_condition_inside_subsegment(
                                            current_digits, std::uint32_t(prod),
                                            digits_in_the_first_part, compute_has_further_digits,
                                            uconst<1>, uconst<9>)) {
                                        goto round_up;
                                    }
                                }
                                else {
                                    if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                            current_digits,
                                            uint_with_known_number_of_digits<9>{second_part},
                                            compute_has_further_digits, uconst<1>, uconst<0>)) {
                                        goto round_up;
                                    }
                                }
                                goto print_last_digits;
                            }
                        }
                        else {
                            digits_in_the_second_part = 18 - overlapping_digits;
                        }

                        // Print the second part.
                        // No rounding, continue.
                        if (remaining_digits > digits_in_the_second_part) {
                            print_subsegment(second_part, digits_in_the_second_part);
                        }
                        // Perform rounding and return.
                        else {
                            // When there is no overlapping digit.
                            std::uint64_t prod;
                            if (digits_in_the_second_part == 9) {
                                if ((remaining_digits & 1) != 0) {
                                    prod = ((second_part * std::uint64_t(720575941)) >> 24) + 1;
                                }
                                else {
                                    if (remaining_digits == 0) {
                                        if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                                current_digits,
                                                uint_with_known_number_of_digits<9>{second_part},
                                                compute_has_further_digits, uconst<1>, uconst<0>)) {
                                            goto round_up_two_digits;
                                        }
                                        goto print_last_two_digits;
                                    }

                                    prod = ((second_part * std::uint64_t(450359963)) >> 20) + 1;
                                }
                            }
                            else {
                                prod = ((second_part * std::uint64_t(720575941)) >> 24) + 1;
                                prod *=
                                    compute_power(std::uint32_t(10), 8 - digits_in_the_second_part);

                                if ((remaining_digits & 1) != 0) {
                                    prod = std::uint32_t(prod) * std::uint64_t(10);
                                }
                                else {
                                    if (remaining_digits == 0) {
                                        goto second_segment252_second_subsegment_rounding_inside_subsegment;
                                    }

                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                }
                            }
                            digits_in_the_second_part -= remaining_digits;
                            current_digits = std::uint32_t(prod >> 32);

                            if (remaining_digits > 2) {
                                if ((remaining_digits & 1) != 0) {
                                    print_1_digit(current_digits, buffer);
                                    ++buffer;
                                }
                                else {
                                    print_2_digits(current_digits, buffer);
                                    buffer += 2;
                                }

                                for (int i = 0; i < (remaining_digits - 3) / 2; ++i) {
                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                }

                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                current_digits = std::uint32_t(prod >> 32);
                                remaining_digits = 0;
                            }

                            if (digits_in_the_second_part != 0) {
                            second_segment252_second_subsegment_rounding_inside_subsegment:
                                if (check_rounding_condition_inside_subsegment(
                                        current_digits, std::uint32_t(prod),
                                        digits_in_the_second_part, compute_has_further_digits,
                                        uconst<1>, uconst<0>)) {
                                    goto round_up;
                                }
                            }
                            else {
                                if (check_rounding_condition_with_next_bit(
                                        current_digits, subsegment_boundary_rounding_bit,
                                        compute_has_further_digits, uconst<0>, uconst<0>)) {
                                    goto round_up;
                                }
                            }
                            goto print_last_digits;
                        }
                    }

                    // Remaining subsegment pairs do not have overlapping digits.
                    --remaining_subsegment_pairs;
                    for (; remaining_subsegment_pairs > 0; --remaining_subsegment_pairs) {
                        subsegment_pair =
                            fixed_point_calculator<ExtendedCache::max_cache_blocks>::generate(
                                power_of_10<18>, blocks, cache_block_count);

                        subsegment_pair += (subsegment_boundary_rounding_bit ? power_of_10<18> : 0);
                        subsegment_boundary_rounding_bit = (subsegment_pair & 1) != 0;
                        subsegment_pair >>= 1;

                        auto const first_part = std::uint32_t(subsegment_pair / power_of_10<9>);
                        auto const second_part =
                            std::uint32_t(subsegment_pair) - power_of_10<9> * first_part;

                        // The first part can be printed without rounding.
                        if (remaining_digits > 9) {
                            print_9_digits(first_part, buffer);

                            // The second part also can be printed without rounding.
                            if (remaining_digits > 18) {
                                print_9_digits(second_part, buffer + 9);
                            }
                            // Otherwise, perform rounding and return.
                            else {
                                buffer += 9;
                                remaining_digits -= 9;

                                std::uint64_t prod;
                                int remaining_digits_in_the_current_subsegment =
                                    9 - remaining_digits;
                                if ((remaining_digits & 1) != 0) {
                                    prod = ((second_part * std::uint64_t(720575941)) >> 24) + 1;
                                    current_digits = std::uint32_t(prod >> 32);

                                    if (remaining_digits == 1) {
                                        goto second_segment252_loop_second_subsegment_rounding;
                                    }

                                    print_1_digit(current_digits, buffer);
                                    ++buffer;
                                }
                                else {
                                    prod = ((second_part * std::uint64_t(450359963)) >> 20) + 1;
                                    current_digits = std::uint32_t(prod >> 32);

                                    if (remaining_digits == 2) {
                                        goto second_segment252_loop_second_subsegment_rounding;
                                    }

                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                }

                                for (int i = 0; i < (remaining_digits - 3) / 2; ++i) {
                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                }

                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                current_digits = std::uint32_t(prod >> 32);
                                remaining_digits = 0;

                                if (remaining_digits_in_the_current_subsegment != 0) {
                                second_segment252_loop_second_subsegment_rounding:
                                    if (check_rounding_condition_inside_subsegment(
                                            current_digits, std::uint32_t(prod),
                                            remaining_digits_in_the_current_subsegment,
                                            compute_has_further_digits, uconst<1>, uconst<0>)) {
                                        goto round_up;
                                    }
                                    goto print_last_digits;
                                }
                                else {
                                    if (check_rounding_condition_with_next_bit(
                                            current_digits, subsegment_boundary_rounding_bit,
                                            compute_has_further_digits, uconst<0>, uconst<0>)) {
                                        goto round_up_two_digits;
                                    }
                                    goto print_last_two_digits;
                                }
                            }
                        }
                        // Otherwise, perform rounding and return.
                        else {
                            std::uint64_t prod;
                            int remaining_digits_in_the_current_subsegment = 9 - remaining_digits;
                            if ((remaining_digits & 1) != 0) {
                                prod = ((first_part * std::uint64_t(720575941)) >> 24) + 1;
                                current_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 1) {
                                    goto second_segment252_loop_first_subsegment_rounding;
                                }

                                print_1_digit(current_digits, buffer);
                                ++buffer;
                            }
                            else {
                                prod = ((first_part * std::uint64_t(450359963)) >> 20) + 1;
                                current_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 2) {
                                    goto second_segment252_loop_first_subsegment_rounding;
                                }

                                print_2_digits(std::uint32_t(prod >> 32), buffer);
                                buffer += 2;
                            }

                            for (int i = 0; i < (remaining_digits - 3) / 2; ++i) {
                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                print_2_digits(std::uint32_t(prod >> 32), buffer);
                                buffer += 2;
                            }

                            prod = std::uint32_t(prod) * std::uint64_t(100);
                            current_digits = std::uint32_t(prod >> 32);
                            remaining_digits = 0;

                            if (remaining_digits_in_the_current_subsegment != 0) {
                            second_segment252_loop_first_subsegment_rounding:
                                if (check_rounding_condition_inside_subsegment(
                                        current_digits, std::uint32_t(prod),
                                        remaining_digits_in_the_current_subsegment,
                                        compute_has_further_digits, uconst<1>, uconst<9>)) {
                                    goto round_up;
                                }
                                goto print_last_digits;
                            }
                            else {
                                if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                        current_digits,
                                        uint_with_known_number_of_digits<9>{second_part},
                                        compute_has_further_digits, uconst<1>, uconst<9>)) {
                                    goto round_up_two_digits;
                                }
                                goto print_last_two_digits;
                            }
                        }

                        buffer += 18;
                        remaining_digits -= 18;
                    }
                } // ExtendedCache::segment_length == 252
            }

            // Print all remaining segments.
            while (has_further_digits(uconst<1>, uconst<0>)) {
                // Get new segment.
                ++multiplier_index;
                k += ExtendedCache::segment_length;

                cache_block_count =
                    load_extended_cache<ExtendedCache, ExtendedCache::constant_block_count>(
                        blocks, e, k, multiplier_index);

                // Compute nm mod 2^Q.
                fixed_point_calculator<ExtendedCache::max_cache_blocks>::discard_upper(
                    significand, blocks, cache_block_count);

                if constexpr (ExtendedCache::segment_length == 22) {
                    // When at least two subsegments left.
                    if (remaining_digits > 16) {
                        auto const first_second_subsegments =
                            fixed_point_calculator<ExtendedCache::max_cache_blocks>::generate(
                                power_of_10<16>, blocks, cache_block_count);

                        auto const first_subsegment =
                            std::uint32_t(jkj::floff::detail::wuint::umul128_upper64(
                                              first_second_subsegments, 3022314549036573ull) >>
                                          14);
                        auto const second_subsegment = std::uint32_t(first_second_subsegments) -
                                                       1'0000'0000 * first_subsegment;

                        print_8_digits(first_subsegment, buffer);
                        print_8_digits(second_subsegment, buffer + 8);

                        // When more segments left.
                        if (remaining_digits > 22) {
                            auto const third_subsegment =
                                fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                    generate_and_discard_lower(power_of_10<6>, blocks,
                                                               cache_block_count);

                            print_6_digits(third_subsegment, buffer + 16);
                            buffer += 22;
                            remaining_digits -= 22;
                        }
                        // When this is the last segment.
                        else {
                            buffer += 16;
                            remaining_digits -= 16;

                            auto third_subsegment =
                                fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                    generate_and_discard_lower(power_of_10<6> << 1, blocks,
                                                               cache_block_count);

                            bool segment_boundary_rounding_bit = ((third_subsegment & 1) != 0);
                            third_subsegment >>= 1;

                            std::uint64_t prod;
                            if ((remaining_digits & 1) != 0) {
                                prod = ((third_subsegment * std::uint64_t(687195)) >> 4) + 1;
                                current_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 1) {
                                    if (check_rounding_condition_inside_subsegment(
                                            current_digits, std::uint32_t(prod), 5,
                                            has_further_digits, uconst<1>, uconst<0>)) {
                                        goto round_up_one_digit;
                                    }
                                    goto print_last_one_digit;
                                }

                                print_1_digit(current_digits, buffer);
                                ++buffer;
                            }
                            else {
                                prod = (third_subsegment * std::uint64_t(429497)) + 1;
                                current_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 2) {
                                    goto segment_loop22_more_than_16_digits_rounding;
                                }

                                print_2_digits(current_digits, buffer);
                                buffer += 2;
                            }

                            if (remaining_digits > 4) {
                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                print_2_digits(std::uint32_t(prod >> 32), buffer);
                                buffer += 2;

                                if (remaining_digits == 6) {
                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                    current_digits = std::uint32_t(prod >> 32);

                                    if (check_rounding_condition_with_next_bit(
                                            current_digits, segment_boundary_rounding_bit,
                                            has_further_digits, uconst<0>, uconst<0>)) {
                                        goto round_up_two_digits;
                                    }
                                    goto print_last_two_digits;
                                }
                            }

                            prod = std::uint32_t(prod) * std::uint64_t(100);
                            current_digits = std::uint32_t(prod >> 32);

                        segment_loop22_more_than_16_digits_rounding:
                            if (check_rounding_condition_inside_subsegment(
                                    current_digits, std::uint32_t(prod), 6 - remaining_digits,
                                    has_further_digits, uconst<1>, uconst<0>)) {
                                goto round_up_two_digits;
                            }
                            goto print_last_two_digits;
                        }
                    }
                    // When two subsegments left.
                    else if (remaining_digits > 8) {
                        // Get one more bit for potential rounding conditions check.
                        auto first_second_subsegments =
                            fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                generate_and_discard_lower(power_of_10<16> << 1, blocks,
                                                           cache_block_count);

                        bool first_bit_of_third_subsegment = ((first_second_subsegments & 1) != 0);
                        first_second_subsegments >>= 1;

                        // 3022314549036573 = ceil(2^78/10^8) = floor(2^78*(10^8/(10^16 -
                        // 1))).
                        auto const first_subsegment =
                            std::uint32_t(jkj::floff::detail::wuint::umul128_upper64(
                                              first_second_subsegments, 3022314549036573ull) >>
                                          14);
                        auto const second_subsegment = std::uint32_t(first_second_subsegments) -
                                                       1'0000'0000 * first_subsegment;

                        print_8_digits(first_subsegment, buffer);
                        buffer += 8;
                        remaining_digits -= 8;

                        // Second subsegment (8 digits).
                        std::uint64_t prod;
                        if ((remaining_digits & 1) != 0) {
                            prod = ((second_subsegment * std::uint64_t(112589991)) >> 18) + 1;
                            current_digits = std::uint32_t(prod >> 32);

                            if (remaining_digits == 1) {
                                if (check_rounding_condition_inside_subsegment(
                                        current_digits, std::uint32_t(prod), 7, has_further_digits,
                                        uconst<1>, uconst<6>)) {
                                    goto round_up_one_digit;
                                }
                                goto print_last_one_digit;
                            }

                            print_1_digit(current_digits, buffer);
                            ++buffer;
                        }
                        else {
                            prod = ((second_subsegment * std::uint64_t(140737489)) >> 15) + 1;
                            current_digits = std::uint32_t(prod >> 32);

                            if (remaining_digits == 2) {
                                goto segment_loop22_more_than_8_digits_rounding;
                            }

                            print_2_digits(current_digits, buffer);
                            buffer += 2;
                        }

                        for (int i = 0; i < (remaining_digits - 3) / 2; ++i) {
                            prod = std::uint32_t(prod) * std::uint64_t(100);
                            print_2_digits(std::uint32_t(prod >> 32), buffer);
                            buffer += 2;
                        }

                        prod = std::uint32_t(prod) * std::uint64_t(100);
                        current_digits = std::uint32_t(prod >> 32);

                        if (remaining_digits < 8) {
                        segment_loop22_more_than_8_digits_rounding:
                            if (check_rounding_condition_inside_subsegment(
                                    current_digits, std::uint32_t(prod), 8 - remaining_digits,
                                    has_further_digits, uconst<1>, uconst<6>)) {
                                goto round_up_two_digits;
                            }
                        }
                        else {
                            if (check_rounding_condition_with_next_bit(
                                    current_digits, first_bit_of_third_subsegment,
                                    has_further_digits, uconst<0>, uconst<6>)) {
                                goto round_up_two_digits;
                            }
                        }
                        goto print_last_two_digits;
                    }
                    // remaining_digits is at most 8.
                    else {
                        // Get one more bit for potential rounding conditions check.
                        auto first_subsegment =
                            fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                generate_and_discard_lower(power_of_10<8> << 1, blocks,
                                                           cache_block_count);

                        bool first_bit_of_second_subsegment = ((first_subsegment & 1) != 0);
                        first_subsegment >>= 1;

                        std::uint64_t prod;
                        if ((remaining_digits & 1) != 0) {
                            prod = ((first_subsegment * std::uint64_t(112589991)) >> 18) + 1;
                            current_digits = std::uint32_t(prod >> 32);

                            if (remaining_digits == 1) {
                                if (check_rounding_condition_inside_subsegment(
                                        current_digits, std::uint32_t(prod), 7, has_further_digits,
                                        uconst<1>, uconst<14>)) {
                                    goto round_up_one_digit;
                                }
                                goto print_last_one_digit;
                            }

                            print_1_digit(current_digits, buffer);
                            ++buffer;
                        }
                        else {
                            prod = ((first_subsegment * std::uint64_t(140737489)) >> 15) + 1;
                            current_digits = std::uint32_t(prod >> 32);

                            if (remaining_digits == 2) {
                                goto segment_loop22_at_most_8_digits_rounding;
                            }

                            print_2_digits(current_digits, buffer);
                            buffer += 2;
                        }

                        for (int i = 0; i < (remaining_digits - 3) / 2; ++i) {
                            prod = std::uint32_t(prod) * std::uint64_t(100);
                            print_2_digits(std::uint32_t(prod >> 32), buffer);
                            buffer += 2;
                        }

                        prod = std::uint32_t(prod) * std::uint64_t(100);
                        current_digits = std::uint32_t(prod >> 32);

                        if (remaining_digits < 8) {
                        segment_loop22_at_most_8_digits_rounding:
                            if (check_rounding_condition_inside_subsegment(
                                    current_digits, std::uint32_t(prod), 8 - remaining_digits,
                                    has_further_digits, uconst<1>, uconst<14>)) {
                                goto round_up_two_digits;
                            }
                        }
                        else {
                            if (check_rounding_condition_with_next_bit(
                                    current_digits, first_bit_of_second_subsegment,
                                    has_further_digits, uconst<0>, uconst<14>)) {
                                goto round_up_two_digits;
                            }
                        }
                        goto print_last_two_digits;
                    }
                } // ExtendedCache::segment_length == 22
                else if (ExtendedCache::segment_length == 252) {
                    // Print as many 18-digits subsegment pairs as possible.
                    for (int remaining_subsegment_pairs = 14; remaining_subsegment_pairs > 0;
                         --remaining_subsegment_pairs) {
                        // No rounding, continue.
                        if (remaining_digits > 18) {
                            auto const subsegment_pair =
                                fixed_point_calculator<ExtendedCache::max_cache_blocks>::generate(
                                    power_of_10<18>, blocks, cache_block_count);
                            auto const first_part = std::uint32_t(subsegment_pair / power_of_10<9>);
                            auto const second_part =
                                std::uint32_t(subsegment_pair) - power_of_10<9> * first_part;

                            print_9_digits(first_part, buffer);
                            print_9_digits(second_part, buffer + 9);
                            buffer += 18;
                            remaining_digits -= 18;
                        }
                        // Final subsegment pair.
                        else {
                            auto compute_has_further_digits = [&](auto additional_neg_exp_of_2,
                                                                  auto additional_neg_exp_of_10) {
#define JKJ_FLOFF_252_HAS_FURTHER_DIGITS(n)                                                        \
case n:                                                                                            \
    return has_further_digits(additional_neg_exp_of_2,                                             \
                              uconst<decltype(additional_neg_exp_of_10)::value + (n - 1) * 18>);
                                switch (remaining_subsegment_pairs) {
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(1);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(2);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(3);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(4);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(5);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(6);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(7);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(8);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(9);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(10);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(11);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(12);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(13);
                                    JKJ_FLOFF_252_HAS_FURTHER_DIGITS(14);

                                default:
                                    JKJ_UNRECHABLE;
                                }
#undef JKJ_FLOFF_252_HAS_FURTHER_DIGITS
                            };

                            auto last_subsegment_pair =
                                fixed_point_calculator<ExtendedCache::max_cache_blocks>::
                                    generate_and_discard_lower(power_of_10<18> << 1, blocks,
                                                               cache_block_count);
                            bool const subsegment_boundary_rounding_bit =
                                ((last_subsegment_pair & 1) != 0);
                            last_subsegment_pair >>= 1;

                            auto const first_part =
                                std::uint32_t(last_subsegment_pair / power_of_10<9>);
                            auto const second_part =
                                std::uint32_t(last_subsegment_pair) - power_of_10<9> * first_part;

                            if (remaining_digits <= 9) {
                                std::uint64_t prod;

                                if ((remaining_digits & 1) != 0) {
                                    prod = ((first_part * std::uint64_t(1441151881)) >> 25) + 1;
                                    current_digits = std::uint32_t(prod >> 32);

                                    if (remaining_digits == 1) {
                                        if (check_rounding_condition_inside_subsegment(
                                                current_digits, std::uint32_t(prod), 8,
                                                compute_has_further_digits, uconst<1>, uconst<9>)) {
                                            goto round_up_one_digit;
                                        }
                                        goto print_last_one_digit;
                                    }

                                    print_1_digit(current_digits, buffer);
                                    ++buffer;
                                }
                                else {
                                    prod = ((first_part * std::uint64_t(450359963)) >> 20) + 1;
                                    current_digits = std::uint32_t(prod >> 32);

                                    if (remaining_digits == 2) {
                                        goto segment_loop252_final18_first_part_rounding;
                                    }

                                    print_2_digits(current_digits, buffer);
                                    buffer += 2;
                                }

                                for (int i = 0; i < (remaining_digits - 3) / 2; ++i) {
                                    prod = std::uint32_t(prod) * std::uint64_t(100);
                                    print_2_digits(std::uint32_t(prod >> 32), buffer);
                                    buffer += 2;
                                }

                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                current_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits < 9) {
                                segment_loop252_final18_first_part_rounding:
                                    if (check_rounding_condition_inside_subsegment(
                                            current_digits, std::uint32_t(prod),
                                            9 - remaining_digits, compute_has_further_digits,
                                            uconst<1>, uconst<9>)) {
                                        goto round_up_two_digits;
                                    }
                                }
                                else {
                                    if (check_rounding_condition_subsegment_boundary_with_next_subsegment(
                                            current_digits,
                                            uint_with_known_number_of_digits<9>{second_part},
                                            compute_has_further_digits, uconst<1>, uconst<0>)) {
                                        goto round_up_two_digits;
                                    }
                                }
                                goto print_last_two_digits;
                            } // remaining_digits <= 9

                            print_9_digits(first_part, buffer);
                            buffer += 9;
                            remaining_digits -= 9;

                            std::uint64_t prod;

                            if ((remaining_digits & 1) != 0) {
                                prod = ((second_part * std::uint64_t(1441151881)) >> 25) + 1;
                                current_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 1) {
                                    if (check_rounding_condition_inside_subsegment(
                                            current_digits, std::uint32_t(prod), 8,
                                            compute_has_further_digits, uconst<1>, uconst<0>)) {
                                        goto round_up_one_digit;
                                    }
                                    goto print_last_one_digit;
                                }

                                print_1_digit(current_digits, buffer);
                                ++buffer;
                            }
                            else {
                                prod = ((second_part * std::uint64_t(450359963)) >> 20) + 1;
                                current_digits = std::uint32_t(prod >> 32);

                                if (remaining_digits == 2) {
                                    goto segment_loop252_final18_second_part_rounding;
                                }

                                print_2_digits(current_digits, buffer);
                                buffer += 2;
                            }

                            for (int i = 0; i < (remaining_digits - 3) / 2; ++i) {
                                prod = std::uint32_t(prod) * std::uint64_t(100);
                                print_2_digits(std::uint32_t(prod >> 32), buffer);
                                buffer += 2;
                            }

                            prod = std::uint32_t(prod) * std::uint64_t(100);
                            current_digits = std::uint32_t(prod >> 32);

                            if (remaining_digits < 9) {
                            segment_loop252_final18_second_part_rounding:
                                if (check_rounding_condition_inside_subsegment(
                                        current_digits, std::uint32_t(prod), 9 - remaining_digits,
                                        compute_has_further_digits, uconst<1>, uconst<0>)) {
                                    goto round_up_two_digits;
                                }
                            }
                            else {
                                if (check_rounding_condition_with_next_bit(
                                        current_digits, subsegment_boundary_rounding_bit,
                                        compute_has_further_digits, uconst<0>, uconst<0>)) {
                                    goto round_up_two_digits;
                                }
                            }
                            goto print_last_two_digits;
                        }
                    }
                } // if (ExtendedCache::segment_length == 252)
            }
        }


        /////////////////////////////////////////////////////////////////////////////////////////////////
        /// Phase 3 - Fill remaining digits with 0's, insert decimal dot, print exponent, and
        /// return.
        /////////////////////////////////////////////////////////////////////////////////////////////////

    fill_remaining_digits_with_0s:
        std::memset(buffer, '0', remaining_digits);
        buffer += remaining_digits;

    insert_decimal_dot:
        buffer_starting_pos[0] = buffer_starting_pos[1];
        buffer_starting_pos[1] = '.';

    print_exponent_and_return:
        if (decimal_exponent >= 0) {
            std::memcpy(buffer, "e+", 2);
        }
        else {
            std::memcpy(buffer, "e-", 2);
            decimal_exponent = -decimal_exponent;
        }
        buffer += 2;
        if (decimal_exponent >= 100) {
            // d1 = decimal_exponent / 10; d2 = decimal_exponent % 10;
            // 6554 = ceil(2^16 / 10)
            auto prod = std::uint32_t(decimal_exponent) * std::uint32_t(6554);
            auto d1 = prod >> 16;
            prod = std::uint16_t(prod) * std::uint32_t(5); // * 10
            auto d2 = prod >> 15;                          // >> 16
            print_2_digits(d1, buffer);
            print_1_digit(d2, buffer + 2);
            buffer += 3;
        }
        else {
            print_2_digits(decimal_exponent, buffer);
            buffer += 2;
        }

        return buffer;

    round_up:
        if ((remaining_digits & 1) != 0) {
        round_up_one_digit:
            if (++current_digits == 10) {
                goto round_up_all_9s;
            }
            goto print_last_one_digit;
        }
        else {
        round_up_two_digits:
            if (++current_digits == 100) {
                goto round_up_all_9s;
            }
            goto print_last_two_digits;
        }

    print_last_digits:
        if ((remaining_digits & 1) != 0) {
        print_last_one_digit:
            print_1_digit(current_digits, buffer);
            ++buffer;
        }
        else {
        print_last_two_digits:
            print_2_digits(current_digits, buffer);
            buffer += 2;
        }
        goto insert_decimal_dot;

    round_up_all_9s:
        char* first_9_pos = buffer;
        buffer += (2 - (remaining_digits & 1));
        // Find all preceding 9's.
        while (true) {
            // '0' is written on buffer_starting_pos, so we have this:
            assert(first_9_pos != buffer_starting_pos);
            if (first_9_pos == buffer_starting_pos + 1) {
                break;
            }

            if (std::memcmp(first_9_pos - 2, "99", 2) != 0) {
                if (*(first_9_pos - 1) == '9') {
                    --first_9_pos;
                }

                if (first_9_pos == buffer_starting_pos + 1) {
                    break;
                }

                ++*(first_9_pos - 1);
                std::memset(first_9_pos, '0', buffer - first_9_pos);

                goto insert_decimal_dot;
            }
            first_9_pos -= 2;
        }

        // first_9_pos == buffer_starting_pos + 1 means every digit we wrote
        // so far are all 9's. In this case, we have to shift the whole thing by 1.
        ++decimal_exponent;
        std::memcpy(buffer_starting_pos, "1.", 2);
        std::memset(buffer_starting_pos + 2, '0', buffer - buffer_starting_pos - 2);

        goto print_exponent_and_return;
    }
}

#undef JKJ_UNRECHABLE
#undef JKJ_FORCEINLINE
#undef JKJ_SAFEBUFFERS
#undef JKJ_HAS_BUILTIN

#endif
