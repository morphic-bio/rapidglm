#pragma once
#include "inference.hpp"
#include "sparse.hpp"
#include <array>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif
#if defined(__aarch64__) && (defined(__GNUC__) || defined(__clang__))
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#endif
#endif
#include <boost/dynamic_bitset.hpp>
#include <boost/math/distributions/skew_normal.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <functional>
#include <map>
#include <memory>
#include <new>
#include <type_traits>
namespace rapidglm {
namespace detail {
// Philox4x32-10 for `lanes` consecutive counters (block, block + 1, ...) with
// the same draw, test and key. Word j of block l is written to out[j * lanes + l].
template <unsigned lanes>
inline void philox_blocks(std::uint64_t block, std::uint32_t draw, std::uint32_t test,
                          std::uint64_t seed, std::uint32_t *out) {
    std::uint32_t c[lanes][4];
    for (unsigned l = 0; l < lanes; ++l) {
        const std::uint64_t b = block + l;
        c[l][0] = static_cast<std::uint32_t>(b);
        c[l][1] = static_cast<std::uint32_t>(b >> 32);
        c[l][2] = draw;
        c[l][3] = test;
    }
    std::uint32_t k0 = static_cast<std::uint32_t>(seed),
                  k1 = static_cast<std::uint32_t>(seed >> 32);
    for (unsigned round = 0; round < 10; ++round) {
        for (unsigned l = 0; l < lanes; ++l) {
            const std::uint64_t a = 0xD2511F53ULL * c[l][0], b = 0xCD9E8D57ULL * c[l][2];
            const std::uint32_t n0 = static_cast<std::uint32_t>(b >> 32) ^ c[l][1] ^ k0,
                                n2 = static_cast<std::uint32_t>(a >> 32) ^ c[l][3] ^ k1;
            c[l][1] = static_cast<std::uint32_t>(b);
            c[l][3] = static_cast<std::uint32_t>(a);
            c[l][0] = n0;
            c[l][2] = n2;
        }
        k0 += 0x9E3779B9U;
        k1 += 0xBB67AE85U;
    }
    for (unsigned l = 0; l < lanes; ++l)
        for (unsigned j = 0; j < 4; ++j)
            out[j * lanes + l] = c[l][j];
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define RAPIDGLM_PHILOX_AVX2 1
// Low and high words of the 64-bit products of every 32-bit lane of c with m.
__attribute__((target("avx2"))) inline void philox_multiply_avx2(__m256i c, __m256i m, __m256i &low,
                                                                 __m256i &high) {
    const __m256i even = _mm256_mul_epu32(c, m),
                  odd = _mm256_mul_epu32(_mm256_srli_epi64(c, 32), m);
    low = _mm256_blend_epi32(even, _mm256_slli_epi64(odd, 32), 0xAA);
    high = _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0xAA);
}
// The same eight blocks with AVX2, one counter per 32-bit lane; the products
// are exact 32 x 32 -> 64-bit multiplications, so the words are identical.
__attribute__((target("avx2"))) inline void philox8_avx2(std::uint64_t block, std::uint32_t draw,
                                                         std::uint32_t test, std::uint64_t seed,
                                                         std::uint32_t *out) {
    __m256i c0, c1;
    if (static_cast<std::uint32_t>(block) <= UINT32_MAX - 7) {
        c0 =
            _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(static_cast<std::uint32_t>(block))),
                             _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        c1 = _mm256_set1_epi32(static_cast<int>(static_cast<std::uint32_t>(block >> 32)));
    } else {
        // The low word wraps within these eight counters.
        alignas(32) std::uint32_t lo[8], hi[8];
        for (unsigned l = 0; l < 8; ++l) {
            const std::uint64_t b = block + l;
            lo[l] = static_cast<std::uint32_t>(b);
            hi[l] = static_cast<std::uint32_t>(b >> 32);
        }
        c0 = _mm256_load_si256(reinterpret_cast<const __m256i *>(lo));
        c1 = _mm256_load_si256(reinterpret_cast<const __m256i *>(hi));
    }
    __m256i c2 = _mm256_set1_epi32(static_cast<int>(draw)),
            c3 = _mm256_set1_epi32(static_cast<int>(test));
    __m256i k0 = _mm256_set1_epi32(static_cast<int>(static_cast<std::uint32_t>(seed))),
            k1 = _mm256_set1_epi32(static_cast<int>(static_cast<std::uint32_t>(seed >> 32)));
    const __m256i m0 = _mm256_set1_epi64x(0xD2511F53LL), m1 = _mm256_set1_epi64x(0xCD9E8D57LL),
                  w0 = _mm256_set1_epi32(static_cast<int>(0x9E3779B9U)),
                  w1 = _mm256_set1_epi32(static_cast<int>(0xBB67AE85U));
    for (unsigned round = 0; round < 10; ++round) {
        __m256i alo, ahi, blo, bhi;
        philox_multiply_avx2(c0, m0, alo, ahi);
        philox_multiply_avx2(c2, m1, blo, bhi);
        const __m256i n0 = _mm256_xor_si256(_mm256_xor_si256(bhi, c1), k0),
                      n2 = _mm256_xor_si256(_mm256_xor_si256(ahi, c3), k1);
        c1 = blo;
        c3 = alo;
        c0 = n0;
        c2 = n2;
        k0 = _mm256_add_epi32(k0, w0);
        k1 = _mm256_add_epi32(k1, w1);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out), c0);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + 8), c1);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + 16), c2);
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(out + 24), c3);
}
inline bool philox_avx2_available() {
    static const bool available = __builtin_cpu_supports("avx2");
    return available;
}
#endif
// ARS (Salmon, Moraes, Dror & Shaw 2011): AES rounds applied to a 128-bit
// counter, with round keys from a Weyl sequence instead of the AES key
// schedule. With R rounds the counter is first combined with the key; each of
// R - 1 rounds then adds the Weyl constants to the two 64-bit halves of the
// key and applies a full AES round (SubBytes, ShiftRows, MixColumns, round
// key), and the last adds them again and applies the AES final round (no
// MixColumns). A 128-bit value is held as two 64-bit words, its bytes 0-7 in
// the low one. AES is exact integer arithmetic, so the three implementations
// below (x86 AES-NI, Arm AES instructions, table lookups) give the same bits.
inline constexpr unsigned ars_rounds = 10, ars_lanes = 8;
inline constexpr std::uint64_t ars_weyl_low = 0x9E3779B97F4A7C15ULL,
                               ars_weyl_high = 0xBB67AE8584CAA73BULL;
// The S-box and, for the rounds that mix columns, each byte's contribution to
// its output column by input row (row 0 in the low byte).
struct AesTables {
    std::array<std::uint8_t, 256> sbox{};
    std::array<std::array<std::uint32_t, 256>, 4> mix{};
    AesTables() {
        // Inverses in GF(2^8) through the powers of 3 and of 1/3, then the affine map.
        std::uint8_t p = 1, q = 1;
        do {
            p = static_cast<std::uint8_t>(p ^ (p << 1) ^ ((p & 0x80) ? 0x1B : 0));
            q = static_cast<std::uint8_t>(q ^ (q << 1));
            q = static_cast<std::uint8_t>(q ^ (q << 2));
            q = static_cast<std::uint8_t>(q ^ (q << 4));
            if (q & 0x80)
                q ^= 0x09;
            const auto rotated = [q](int r) {
                return static_cast<std::uint8_t>((q << r) | (q >> (8 - r)));
            };
            sbox[p] = static_cast<std::uint8_t>(q ^ rotated(1) ^ rotated(2) ^ rotated(3) ^
                                                rotated(4) ^ 0x63);
        } while (p != 1);
        sbox[0] = 0x63;
        for (std::size_t x = 0; x < 256; ++x) {
            const std::uint32_t s = sbox[x], s2 = ((s << 1) ^ ((s & 0x80) ? 0x11B : 0)) & 0xFF,
                                s3 = s2 ^ s;
            mix[0][x] = s2 | s << 8 | s << 16 | s3 << 24;
            mix[1][x] = s3 | s2 << 8 | s << 16 | s << 24;
            mix[2][x] = s | s3 << 8 | s2 << 16 | s << 24;
            mix[3][x] = s | s << 8 | s3 << 16 | s2 << 24;
        }
    }
};
inline const AesTables &aes_tables() {
    static const AesTables tables;
    return tables;
}
// ARS of `lanes` counters (counters[2l], counters[2l + 1]: low and high words)
// under one key, into out[2l], out[2l + 1].
inline void ars_blocks_portable(const std::uint64_t *counters, unsigned lanes,
                                std::uint64_t key_low, std::uint64_t key_high, unsigned rounds,
                                std::uint64_t *out) {
    const auto &t = aes_tables();
    for (unsigned l = 0; l < lanes; ++l) {
        std::uint64_t low = key_low, high = key_high;
        // The state by columns: byte r of w[c] is row r of column c.
        std::uint32_t w[4] = {static_cast<std::uint32_t>(counters[2 * l] ^ low),
                              static_cast<std::uint32_t>((counters[2 * l] ^ low) >> 32),
                              static_cast<std::uint32_t>(counters[2 * l + 1] ^ high),
                              static_cast<std::uint32_t>((counters[2 * l + 1] ^ high) >> 32)};
        for (unsigned round = 1; round <= rounds; ++round) {
            low += ars_weyl_low;
            high += ars_weyl_high;
            const std::uint32_t k[4] = {
                static_cast<std::uint32_t>(low), static_cast<std::uint32_t>(low >> 32),
                static_cast<std::uint32_t>(high), static_cast<std::uint32_t>(high >> 32)};
            std::uint32_t n[4];
            // ShiftRows takes row r of column c from column c + r.
            for (unsigned c = 0; c < 4; ++c) {
                const std::uint8_t b0 = static_cast<std::uint8_t>(w[c]),
                                   b1 = static_cast<std::uint8_t>(w[(c + 1) & 3] >> 8),
                                   b2 = static_cast<std::uint8_t>(w[(c + 2) & 3] >> 16),
                                   b3 = static_cast<std::uint8_t>(w[(c + 3) & 3] >> 24);
                n[c] = k[c] ^
                       (round < rounds ? t.mix[0][b0] ^ t.mix[1][b1] ^ t.mix[2][b2] ^ t.mix[3][b3]
                                       : static_cast<std::uint32_t>(t.sbox[b0]) |
                                             static_cast<std::uint32_t>(t.sbox[b1]) << 8 |
                                             static_cast<std::uint32_t>(t.sbox[b2]) << 16 |
                                             static_cast<std::uint32_t>(t.sbox[b3]) << 24);
            }
            for (unsigned c = 0; c < 4; ++c)
                w[c] = n[c];
        }
        out[2 * l] = w[0] | static_cast<std::uint64_t>(w[1]) << 32;
        out[2 * l + 1] = w[2] | static_cast<std::uint64_t>(w[3]) << 32;
    }
}
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define RAPIDGLM_ARS_HARDWARE 1
#define RAPIDGLM_ARS_TARGET __attribute__((target("aes")))
// Inlined into its two callers, so that with constant lanes and rounds the
// states stay in registers.
RAPIDGLM_ARS_TARGET __attribute__((always_inline)) inline void
ars_rounds_hardware(const std::uint64_t *counters, unsigned lanes, std::uint64_t key_low,
                    std::uint64_t key_high, unsigned rounds, std::uint64_t *out) {
    const __m128i weyl =
        _mm_set_epi64x(static_cast<long long>(ars_weyl_high), static_cast<long long>(ars_weyl_low));
    __m128i key = _mm_set_epi64x(static_cast<long long>(key_high), static_cast<long long>(key_low));
    __m128i v[ars_lanes];
    for (unsigned l = 0; l < lanes; ++l)
        v[l] = _mm_xor_si128(_mm_loadu_si128(reinterpret_cast<const __m128i *>(counters + 2 * l)),
                             key);
    for (unsigned round = 1; round < rounds; ++round) {
        key = _mm_add_epi64(key, weyl);
        for (unsigned l = 0; l < lanes; ++l)
            v[l] = _mm_aesenc_si128(v[l], key);
    }
    key = _mm_add_epi64(key, weyl);
    for (unsigned l = 0; l < lanes; ++l)
        _mm_storeu_si128(reinterpret_cast<__m128i *>(out + 2 * l), _mm_aesenclast_si128(v[l], key));
}
inline bool ars_hardware_available() {
    static const bool available = __builtin_cpu_supports("aes");
    return available;
}
#elif defined(__aarch64__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ &&                         \
    (defined(__GNUC__) || defined(__clang__))
#define RAPIDGLM_ARS_HARDWARE 1
#if defined(__ARM_FEATURE_AES)
#define RAPIDGLM_ARS_TARGET
#elif defined(__clang__)
#define RAPIDGLM_ARS_TARGET __attribute__((target("aes")))
#else
#define RAPIDGLM_ARS_TARGET __attribute__((target("+aes")))
#endif
// The Arm instruction adds its key before SubBytes and ShiftRows, and
// MixColumns is separate: with a zero key and the round key added afterwards
// this is the x86 round. Inlined into its two callers, as above.
RAPIDGLM_ARS_TARGET __attribute__((always_inline)) inline void
ars_rounds_hardware(const std::uint64_t *counters, unsigned lanes, std::uint64_t key_low,
                    std::uint64_t key_high, unsigned rounds, std::uint64_t *out) {
    const uint64x2_t weyl = vcombine_u64(vcreate_u64(ars_weyl_low), vcreate_u64(ars_weyl_high));
    const uint8x16_t zero = vdupq_n_u8(0);
    uint64x2_t key = vcombine_u64(vcreate_u64(key_low), vcreate_u64(key_high));
    uint8x16_t v[ars_lanes];
    for (unsigned l = 0; l < lanes; ++l)
        v[l] = vreinterpretq_u8_u64(veorq_u64(vld1q_u64(counters + 2 * l), key));
    for (unsigned round = 1; round < rounds; ++round) {
        key = vaddq_u64(key, weyl);
        const uint8x16_t k = vreinterpretq_u8_u64(key);
        for (unsigned l = 0; l < lanes; ++l)
            v[l] = veorq_u8(vaesmcq_u8(vaeseq_u8(v[l], zero)), k);
    }
    key = vaddq_u64(key, weyl);
    const uint8x16_t k = vreinterpretq_u8_u64(key);
    for (unsigned l = 0; l < lanes; ++l)
        vst1q_u64(out + 2 * l, vreinterpretq_u64_u8(veorq_u8(vaeseq_u8(v[l], zero), k)));
}
inline bool ars_hardware_available() {
#if defined(__ARM_FEATURE_AES) || defined(__APPLE__)
    return true;
#elif defined(__linux__) && defined(HWCAP_AES)
    static const bool available = (getauxval(AT_HWCAP) & HWCAP_AES) != 0;
    return available;
#else
    return false;
#endif
}
#else
inline bool ars_hardware_available() {
    return false;
}
#endif
#ifdef RAPIDGLM_ARS_HARDWARE
// Any counters, lanes and rounds (the tests).
RAPIDGLM_ARS_TARGET inline void ars_blocks_hardware(const std::uint64_t *counters, unsigned lanes,
                                                    std::uint64_t key_low, std::uint64_t key_high,
                                                    unsigned rounds, std::uint64_t *out) {
    ars_rounds_hardware(counters, lanes, key_low, key_high, rounds, out);
}
// The generator's refill: the ars_lanes counters (block + l, 0).
RAPIDGLM_ARS_TARGET inline void ars_stream_hardware(std::uint64_t block, std::uint64_t key_low,
                                                    std::uint64_t key_high, std::uint64_t *out) {
    std::uint64_t counters[2 * ars_lanes];
    for (unsigned l = 0; l < ars_lanes; ++l) {
        counters[2 * l] = block + l;
        counters[2 * l + 1] = 0;
    }
    ars_rounds_hardware(counters, ars_lanes, key_low, key_high, ars_rounds, out);
}
#undef RAPIDGLM_ARS_TARGET
#endif
// At most ars_lanes counters. `hardware` false forces the table lookups.
inline void ars_blocks(const std::uint64_t *counters, unsigned lanes, std::uint64_t key_low,
                       std::uint64_t key_high, unsigned rounds, std::uint64_t *out,
                       bool hardware = ars_hardware_available()) {
    require(lanes <= ars_lanes && rounds > 0, "invalid ARS request");
#ifdef RAPIDGLM_ARS_HARDWARE
    if (hardware) {
        ars_blocks_hardware(counters, lanes, key_low, key_high, rounds, out);
        return;
    }
#else
    (void)hardware;
#endif
    ars_blocks_portable(counters, lanes, key_low, key_high, rounds, out);
}
} // namespace detail
// Whether the processor has the AES instructions that AesCounterRng uses.
// Without them it computes the same words by table lookups, far more slowly,
// and CrtDraw::hazard, which relies on it, falls back to CrtDraw::thinned.
inline bool aes_instructions() {
    return detail::ars_hardware_available();
}
class CounterRng {
    // Philox blocks are computed eight at a time (consecutive counters), with
    // AVX2 where the processor has it; the stream is the same either way.
    static constexpr unsigned lanes = 8;
    std::uint64_t seed_, block_ = 0;
    std::uint32_t test_, draw_;
    // Word j of block l of the current group is output_[j * lanes + l].
    alignas(32) std::array<std::uint32_t, 4 * lanes> output_{};
    unsigned cursor_ = 4 * lanes;
    std::uint32_t word(unsigned k) const {
        return output_[(k & 3) * lanes + (k >> 2)];
    }

    void refill() {
        require(block_ <= UINT64_MAX - lanes, "RNG block counter exhausted");
#ifdef RAPIDGLM_PHILOX_AVX2
        if (detail::philox_avx2_available())
            detail::philox8_avx2(block_, draw_, test_, seed_, output_.data());
        else
#endif
            detail::philox_blocks<lanes>(block_, draw_, test_, seed_, output_.data());
        block_ += lanes;
        cursor_ = 0;
    }

  public:
    static std::array<std::uint32_t, 4> philox(std::array<std::uint32_t, 4> c,
                                               std::array<std::uint32_t, 2> k) {
        for (unsigned round = 0; round < 10; ++round) {
            std::uint64_t a = 0xD2511F53ULL * c[0], b = 0xCD9E8D57ULL * c[2];
            c = {static_cast<std::uint32_t>(b >> 32) ^ c[1] ^ k[0], static_cast<std::uint32_t>(b),
                 static_cast<std::uint32_t>(a >> 32) ^ c[3] ^ k[1], static_cast<std::uint32_t>(a)};
            k[0] += 0x9E3779B9U;
            k[1] += 0xBB67AE85U;
        }
        return c;
    }
    CounterRng(std::uint64_t seed, std::uint64_t test, std::uint64_t draw) : seed_(seed) {
        require(test <= UINT32_MAX && draw <= UINT32_MAX, "RNG test/draw ID exceeds 32 bits");
        test_ = static_cast<std::uint32_t>(test);
        draw_ = static_cast<std::uint32_t>(draw);
    }
    std::uint32_t next32() {
        if (cursor_ == 4 * lanes)
            refill();
        return word(cursor_++);
    }
    std::uint64_t next64() {
        if (cursor_ + 2 <= 4 * lanes) {
            const std::uint64_t a = word(cursor_), b = word(cursor_ + 1);
            cursor_ += 2;
            return (a << 32) | b;
        }
        std::uint64_t a = next32();
        return (a << 32) | next32();
    }
    double uniform() {
        return static_cast<double>(next64() >> 11) * 0x1.0p-53;
    }
    std::uint64_t bounded(std::uint64_t n) {
        require(n > 0, "empty random range");
        std::uint64_t threshold = (-n) % n, r;
        do {
            r = next64();
        } while (r < threshold);
        return r % n;
    }
};
// Random words from ARS with 10 rounds in counter mode under a 128-bit key:
// block b is ARS of the counter (b, 0), and its low word then its high word
// are returned. Eight blocks are computed at a time. With the processor's AES
// instructions a word costs about a third of a Philox word. A draw's generator
// is keyed by the first two words of the draw's Philox stream (from()), so it
// is a function of (seed, test, draw) like every other stream.
class AesCounterRng {
    std::uint64_t key_low_, key_high_, block_ = 0;
    std::array<std::uint64_t, 2 * detail::ars_lanes> output_{};
    unsigned cursor_ = 2 * detail::ars_lanes;
    void refill() {
#ifdef RAPIDGLM_ARS_HARDWARE
        if (detail::ars_hardware_available())
            detail::ars_stream_hardware(block_, key_low_, key_high_, output_.data());
        else
#endif
        {
            std::uint64_t counters[2 * detail::ars_lanes];
            for (unsigned l = 0; l < detail::ars_lanes; ++l) {
                counters[2 * l] = block_ + l;
                counters[2 * l + 1] = 0;
            }
            detail::ars_blocks_portable(counters, detail::ars_lanes, key_low_, key_high_,
                                        detail::ars_rounds, output_.data());
        }
        block_ += detail::ars_lanes;
        cursor_ = 0;
    }

  public:
    AesCounterRng(std::uint64_t key_low, std::uint64_t key_high)
        : key_low_(key_low), key_high_(key_high) {}
    static AesCounterRng from(CounterRng &source) {
        const std::uint64_t low = source.next64();
        const std::uint64_t high = source.next64();
        return AesCounterRng(low, high);
    }
    std::uint64_t next64() {
        if (cursor_ == 2 * detail::ars_lanes)
            refill();
        return output_[cursor_++];
    }
    double uniform() {
        return static_cast<double>(next64() >> 11) * 0x1.0p-53;
    }
};
struct SamplerScratch {
    boost::dynamic_bitset<std::uint64_t> membership;
    std::vector<std::size_t> picked;
};
// Each sampler returns unique, sorted treated observation indices. Legacy
// two-argument callbacks still work; built-ins additionally accept task-owned
// scratch so concurrent draws never share mutable membership state.
//
// A sampler may also generate a batch of draws at once: draws [first, first +
// count) of the stream (seed, test), each exactly as the per-draw call with
// CounterRng(seed, test, draw) returns it. Batches let a sampler walk its
// arrays once for many draws instead of once per draw; resample() uses them
// when present.
class Sampler {
    using Function =
        std::function<void(CounterRng &, std::vector<std::size_t> &, SamplerScratch &)>;
    using Batch =
        std::function<void(std::uint64_t seed, std::uint64_t test, std::size_t first,
                           std::size_t count, std::vector<std::vector<std::size_t>> &out)>;
    Function function_;
    Batch batch_;

  public:
    Sampler() = default;
    template <class F, std::enable_if_t<!std::is_same_v<std::decay_t<F>, Sampler> &&
                                            std::is_invocable_r_v<void, F &, CounterRng &,
                                                                  std::vector<std::size_t> &>,
                                        int> = 0>
    Sampler(F f)
        : function_([f = std::move(f)](CounterRng &rng, std::vector<std::size_t> &out,
                                       SamplerScratch &) mutable { f(rng, out); }) {}
    static Sampler with_scratch(Function f) {
        Sampler sampler;
        sampler.function_ = std::move(f);
        return sampler;
    }
    static Sampler with_batch(Function f, Batch batch) {
        Sampler sampler;
        sampler.function_ = std::move(f);
        sampler.batch_ = std::move(batch);
        return sampler;
    }
    void operator()(CounterRng &rng, std::vector<std::size_t> &out, SamplerScratch &scratch) const {
        function_(rng, out, scratch);
    }
    void operator()(CounterRng &rng, std::vector<std::size_t> &out) const {
        SamplerScratch scratch;
        (*this)(rng, out, scratch);
    }
    // Draws [first, first + count) into out[0, count) (out is resized as needed).
    void batch(std::uint64_t seed, std::uint64_t test, std::size_t first, std::size_t count,
               std::vector<std::vector<std::size_t>> &out, SamplerScratch &scratch) const {
        if (out.size() < count)
            out.resize(count);
        if (batch_) {
            batch_(seed, test, first, count, out);
            return;
        }
        for (std::size_t k = 0; k < count; ++k) {
            CounterRng rng(seed, test, first + k);
            function_(rng, out[k], scratch);
        }
    }
};
inline Sampler permutation_sampler(const Vector &labels,
                                   const std::vector<std::size_t> &strata = {}) {
    require(strata.empty() || strata.size() == labels.size(), "strata length mismatch");
    std::map<std::size_t, std::pair<std::vector<std::size_t>, std::size_t>> groups;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        require(labels[i] == 0 || labels[i] == 1, "permutation labels must be binary");
        auto &g = groups[strata.empty() ? 0 : strata[i]];
        g.first.push_back(i);
        g.second += labels[i] == 1;
    }
    return Sampler::with_scratch([groups = std::move(groups)](CounterRng &rng,
                                                              std::vector<std::size_t> &out,
                                                              SamplerScratch &scratch) {
        out.clear();
        for (const auto &entry : groups) {
            const auto &indices = entry.second.first;
            auto k = entry.second.second, n = indices.size();
            auto &membership = scratch.membership;
            auto &picked = scratch.picked;
            membership.resize(n);
            picked.clear();
            picked.reserve(k);
            // Floyd's algorithm samples a uniform k-subset without shuffling all n cells.
            for (std::size_t j = n - k; j < n; ++j) {
                auto t = static_cast<std::size_t>(rng.bounded(j + 1));
                if (membership[t])
                    t = j;
                membership[t] = true;
                picked.push_back(t);
            }
            // Emit this stratum's picks in increasing order: an ordered scan of the
            // membership bits when the picks are dense enough, otherwise a sort.
            // Either way only the touched bits are cleared.
            const auto merged = static_cast<std::ptrdiff_t>(out.size());
            if (512 * k >= n)
                for (auto i = membership.find_first(); i != membership.npos;
                     i = membership.find_next(i)) {
                    out.push_back(indices[i]);
                    membership[i] = false;
                }
            else {
                std::sort(picked.begin(), picked.end());
                for (auto i : picked) {
                    out.push_back(indices[i]);
                    membership[i] = false;
                }
            }
            // Strata are merged; within each, increasing positions give increasing indices.
            std::inplace_merge(out.begin(), out.begin() + merged, out.end());
        }
    });
}
// Every method draws independent Bernoulli(p_i) labels; they consume different
// random streams. direct uses one uniform per observation. thinned groups
// observations by q = 2^-b >= p_i > q/2, visits candidates with geometric gaps
// of rate q and accepts each with probability p_i/q, so the expected work is
// O(sum p_i) instead of O(n) for sparse assignments. hazard gives observation i
// an interval of length -log(1 - p_i) on a line and selects the observations
// whose intervals hold a point of a unit-rate Poisson process on the line: one
// exponential gap per selected observation, no rejection, and the selection
// comes out in order; its gaps use an AES generator (AesCounterRng). On a
// processor without AES instructions hazard draws as thinned does, from
// thinned's stream. When the mean probability is at least 1/16 none has an
// advantage and the direct stream is used.
enum class CrtDraw { thinned, direct, hazard };
// How a skew-normal fit to the null statistics is accepted. ks: its
// Kolmogorov-Smirnov discrepancy alone, the check of releases before 1.0.0.
// strict: the fit must also not understate the tail on the statistic's side
// (detail::skew_normal_tail_is_sound), and a p-value beyond the reach of the
// current budget is accepted only on the last budget (NullStages).
enum class FitCheck { strict, ks };
namespace detail {
// Standard exponential variates by the ziggurat of Marsaglia & Tsang (2000,
// doi:10.18637/jss.v005.i08) with 256 layers. One 64-bit word gives the layer
// (its low 8 bits) and a 53-bit position (its high bits); about 99 variates in
// 100 need nothing more, the others a second word and an exponential or, in
// the tail, a logarithm.
struct ExponentialZiggurat {
    std::array<double, 256> width, height;
    std::array<std::uint64_t, 256> inside;
    static constexpr double tail = 7.69711747013104972;
    ExponentialZiggurat() {
        const double scale = 0x1.0p53, area = 3.949659822581572e-3;
        double edge = tail, previous = tail;
        const double base = area / std::exp(-edge);
        inside[0] = static_cast<std::uint64_t>(edge / base * scale);
        inside[1] = 0;
        width[0] = base / scale;
        width[255] = edge / scale;
        height[0] = 1;
        height[255] = std::exp(-edge);
        for (std::size_t i = 254; i >= 1; --i) {
            edge = -std::log(area / edge + std::exp(-edge));
            inside[i + 1] = static_cast<std::uint64_t>(edge / previous * scale);
            previous = edge;
            height[i] = std::exp(-edge);
            width[i] = edge / scale;
        }
    }
    template <class Rng> double operator()(Rng &rng) const {
        for (;;) {
            const std::uint64_t word = rng.next64();
            const std::size_t layer = word & 0xff;
            const std::uint64_t position = word >> 11;
            const double x = static_cast<double>(position) * width[layer];
            if (position < inside[layer])
                return x;
            if (layer == 0)
                return tail - std::log(1 - rng.uniform());
            if (height[layer] + rng.uniform() * (height[layer - 1] - height[layer]) < std::exp(-x))
                return x;
        }
    }
};
inline const ExponentialZiggurat &exponential_ziggurat() {
    static const ExponentialZiggurat ziggurat;
    return ziggurat;
}
// The line of CrtDraw::hazard. An interval of length L holds no point of a
// unit-rate Poisson process with probability exp(-L) = 1 - p_i, independently
// of the other intervals, so the observations whose intervals hold a point are
// independent Bernoulli(p_i) labels. Lengths are added in double precision:
// each probability is exact to within the rounding of the line's length (2e-13
// for a line of 1,000). A probability of 1 gets length 40, which is missed
// with probability 4e-18, below the spacing of doubles at 1.
struct HazardLine {
    Vector end; // end[i]: the line's length through observation i
    std::vector<std::uint32_t>
        guide;                   // guide[j]: no later than the owner of any position in bucket j
    double total = 0, scale = 0; // scale: buckets per unit length
    std::size_t buckets = 0;
    explicit HazardLine(const Vector &probabilities) : end(probabilities.size()) {
        const std::size_t n = probabilities.size();
        for (std::size_t i = 0; i < n; ++i) {
            total += std::min(40.0, -std::log1p(-probabilities[i]));
            end[i] = total;
        }
        if (!(total > 0))
            return;
        // One bucket per observation: a position's owner is then found from its
        // bucket's entry after one further step on average.
        buckets = n;
        scale = static_cast<double>(buckets) / total;
        guide.resize(buckets);
        std::size_t k = 0;
        for (std::size_t j = 0; j < buckets; ++j) {
            // Below every position that rounds into bucket j.
            const double low = static_cast<double>(j) / scale * (1 - 0x1.0p-50);
            while (k + 1 < n && end[k] <= low)
                ++k;
            guide[j] = static_cast<std::uint32_t>(k);
        }
    }
    // The observation whose interval holds position t < total.
    std::size_t owner(double t) const {
        std::size_t k = guide[std::min(buckets - 1, static_cast<std::size_t>(t * scale))];
        while (end[k] <= t)
            ++k;
        return k;
    }
    // One draw: the generator's exponential gaps, in order, until the line ends.
    template <class Rng>
    void draw(Rng &rng, std::vector<std::size_t> &out, const ExponentialZiggurat &gap) const {
        out.clear();
        std::size_t last = end.size();
        for (double t = gap(rng); t < total; t += gap(rng)) {
            const std::size_t k = owner(t);
            if (k != last)
                out.push_back(last = k);
        }
    }
    // The same draws for a batch. All draws cross one stretch of the line before
    // the next, so the line's arrays are read once per batch rather than once
    // per draw; each draw keeps its own generator and consumes it in order.
    template <class Rng>
    void batch(std::vector<Rng> &rng, std::vector<std::vector<std::size_t>> &out) const {
        const auto &gap = exponential_ziggurat();
        const std::size_t count = rng.size(), stretches = buckets / 16384 + 1;
        std::vector<double> position(count);
        std::vector<std::size_t> last(count, end.size());
        for (std::size_t d = 0; d < count; ++d) {
            out[d].clear();
            position[d] = gap(rng[d]);
        }
        for (std::size_t s = 1; s <= stretches; ++s) {
            const double stop = s == stretches ? total : total * static_cast<double>(s) / stretches;
            for (std::size_t d = 0; d < count; ++d) {
                double t = position[d];
                auto &generator = rng[d];
                auto &selected = out[d];
                std::size_t previous = last[d];
                while (t < stop) {
                    const std::size_t k = owner(t);
                    if (k != previous)
                        selected.push_back(previous = k);
                    t += gap(generator);
                }
                position[d] = t;
                last[d] = previous;
            }
        }
    }
};
// Merges the sorted runs out[0, middle) and out[middle, end) (disjoint values)
// through a reusable buffer; std::inplace_merge would allocate on every call.
inline void merge_runs(std::vector<std::size_t> &out, std::size_t middle,
                       std::vector<std::size_t> &buffer) {
    if (middle == 0 || middle == out.size() || out[middle - 1] < out[middle])
        return;
    buffer.resize(out.size());
    std::merge(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(middle),
               out.begin() + static_cast<std::ptrdiff_t>(middle), out.end(), buffer.begin());
    out.swap(buffer);
}
} // namespace detail
// Memory a crt_sampler keeps per observation: its index (4 bytes) and its
// probability (8 bytes); for the hazard line, its end (8 bytes) and its
// bucket's entry (4 bytes).
inline constexpr std::size_t crt_sampler_bytes_per_observation = 12;
// `aes`: whether hazard may use the AES generator; by default, whether the
// processor has AES instructions. Without it hazard is thinned. (Passing true
// on a processor without the instructions computes the same draws by table
// lookups, slowly.)
inline Sampler crt_sampler(Vector probabilities, CrtDraw method = CrtDraw::thinned,
                           bool aes = aes_instructions()) {
    double expected = 0;
    for (double p : probabilities) {
        require(std::isfinite(p) && p >= 0 && p <= 1, "invalid CRT probability");
        expected += p;
    }
    const auto direct = [](Vector p) {
        return [p = std::move(p)](CounterRng &rng, std::vector<std::size_t> &out) {
            out.clear();
            for (std::size_t i = 0; i < p.size(); ++i)
                if (rng.uniform() < p[i])
                    out.push_back(i);
        };
    };
    if (method == CrtDraw::direct || 16 * expected >= probabilities.size())
        return direct(std::move(probabilities));
    require(probabilities.size() <= std::numeric_limits<std::uint32_t>::max(),
            "too many observations for the sparse CRT samplers");
    if (method == CrtDraw::hazard && aes) {
        auto line = std::make_shared<const detail::HazardLine>(probabilities);
        // Probabilities near 1 make the line long (up to 40 per observation);
        // beyond a point for every fourth observation the direct stream is used.
        if (4 * line->total >= static_cast<double>(probabilities.size()))
            return direct(std::move(probabilities));
        // The gaps of a draw come from an AES generator keyed by the draw's
        // Philox stream: a draw needs a word per selected observation.
        return Sampler::with_batch(
            [line](CounterRng &rng, std::vector<std::size_t> &out, SamplerScratch &) {
                auto gaps = AesCounterRng::from(rng);
                line->draw(gaps, out, detail::exponential_ziggurat());
            },
            [line](std::uint64_t seed, std::uint64_t test, std::size_t first, std::size_t count,
                   std::vector<std::vector<std::size_t>> &out) {
                std::vector<AesCounterRng> gaps;
                gaps.reserve(count);
                for (std::size_t d = 0; d < count; ++d) {
                    CounterRng rng(seed, test, first + d);
                    gaps.push_back(AesCounterRng::from(rng));
                }
                line->batch(gaps, out);
            });
    }
    // The gap is floor(x) for x = log(1 - u) / log(1 - q) >= 0. For an integer
    // bound S, floor(x) < S exactly when x < S, and converting a non-negative x
    // below S to an integer truncates it to floor(x); so x is compared and
    // converted directly, without calling floor.
    struct Level {
        double bound, log_miss;
        std::vector<std::uint32_t> members; // increasing observation indices
        Vector probability;                 // probabilities of the members, in that order
    };
    // Levels in increasing b; members are counted first so each level is filled
    // without reallocation.
    std::vector<int> level_of(probabilities.size(), -1);
    std::vector<std::size_t> sizes;
    for (std::size_t i = 0; i < probabilities.size(); ++i)
        if (probabilities[i] > 0) {
            int exponent;
            std::frexp(probabilities[i], &exponent); // p in [2^(e-1), 2^e)
            const int b = std::max(0, -exponent);
            if (static_cast<std::size_t>(b) >= sizes.size())
                sizes.resize(b + 1, 0);
            ++sizes[b];
            level_of[i] = b;
        }
    auto levels = std::make_shared<std::vector<Level>>();
    std::vector<std::size_t> index(sizes.size());
    for (std::size_t b = 0; b < sizes.size(); ++b)
        if (sizes[b]) {
            index[b] = levels->size();
            Level level;
            level.bound = std::ldexp(1.0, -static_cast<int>(b));
            level.log_miss = std::log1p(-level.bound);
            level.members.reserve(sizes[b]);
            level.probability.reserve(sizes[b]);
            levels->push_back(std::move(level));
        }
    for (std::size_t i = 0; i < probabilities.size(); ++i)
        if (level_of[i] >= 0) {
            auto &level = (*levels)[index[level_of[i]]];
            level.members.push_back(static_cast<std::uint32_t>(i));
            level.probability.push_back(probabilities[i]);
        }
    auto draw = [levels](CounterRng &rng, std::vector<std::size_t> &out, SamplerScratch &scratch) {
        out.clear();
        for (const auto &level : *levels) {
            const auto &members = level.members;
            const auto merged = static_cast<std::ptrdiff_t>(out.size());
            if (level.bound >= 0.25) {
                for (std::size_t k = 0; k < members.size(); ++k)
                    if (rng.uniform() < level.probability[k])
                        out.push_back(members[k]);
            } else
                for (std::size_t next = 0;;) {
                    // Gap before the next candidate: P(gap >= k) = (1 - q)^k, the floor
                    // of a non-negative x (see the note on the levels).
                    const double x = std::log(1 - rng.uniform()) / level.log_miss;
                    if (!(x < static_cast<double>(members.size() - next)))
                        break;
                    next += static_cast<std::size_t>(x);
                    const auto k = next++;
                    if (rng.uniform() * level.bound < level.probability[k])
                        out.push_back(members[k]);
                }
            // Each level's picks are increasing; merge keeps the output sorted.
            detail::merge_runs(out, static_cast<std::size_t>(merged), scratch.picked);
        }
    };
    // The same draws for a batch, level by level: every draw advances through one
    // block of a level's members before the next block is touched, so the members
    // and their probabilities are read once per batch rather than once per draw.
    // Each draw keeps its own generator and consumes it in the per-draw order.
    auto batch = [levels](std::uint64_t seed, std::uint64_t test, std::size_t first,
                          std::size_t count, std::vector<std::vector<std::size_t>> &out) {
        constexpr std::size_t block = 4096;
        std::vector<CounterRng> rng;
        rng.reserve(count);
        for (std::size_t d = 0; d < count; ++d) {
            rng.emplace_back(seed, test, first + d);
            out[d].clear();
        }
        std::vector<std::size_t> position(count), merged(count), buffer;
        std::vector<unsigned char> done(count);
        for (const auto &level : *levels) {
            const auto &members = level.members;
            const auto &probability = level.probability;
            const std::size_t size = members.size();
            for (std::size_t d = 0; d < count; ++d)
                merged[d] = out[d].size();
            if (level.bound >= 0.25) {
                for (std::size_t a = 0; a < size; a += block) {
                    const std::size_t b = std::min(size, a + block);
                    for (std::size_t d = 0; d < count; ++d)
                        for (std::size_t k = a; k < b; ++k)
                            if (rng[d].uniform() < probability[k])
                                out[d].push_back(members[k]);
                }
            } else {
                // position[d]: the draw's next candidate, from its first gap.
                for (std::size_t d = 0; d < count; ++d) {
                    const double x = std::log(1 - rng[d].uniform()) / level.log_miss;
                    done[d] = !(x < static_cast<double>(size));
                    position[d] = done[d] ? 0 : static_cast<std::size_t>(x);
                }
                // Draws advance in groups of four, one candidate each per round:
                // a candidate's position depends on a logarithm and a division,
                // so interleaving independent draws overlaps those latencies.
                constexpr std::size_t group = 4;
                for (std::size_t a = 0; a < size; a += block) {
                    const std::size_t b = std::min(size, a + block);
                    for (std::size_t d0 = 0; d0 < count; d0 += group) {
                        const std::size_t d1 = std::min(count, d0 + group);
                        for (bool any = true; any;) {
                            any = false;
                            for (std::size_t d = d0; d < d1; ++d) {
                                const std::size_t k = position[d];
                                if (done[d] || k >= b)
                                    continue;
                                auto &r = rng[d];
                                if (r.uniform() * level.bound < probability[k])
                                    out[d].push_back(members[k]);
                                const std::size_t next = k + 1;
                                const double x = std::log(1 - r.uniform()) / level.log_miss;
                                if (!(x < static_cast<double>(size - next))) {
                                    done[d] = 1;
                                    continue;
                                }
                                position[d] = next + static_cast<std::size_t>(x);
                                any = true;
                            }
                        }
                    }
                }
            }
            for (std::size_t d = 0; d < count; ++d)
                detail::merge_runs(out[d], merged[d], buffer);
        }
    };
    return Sampler::with_batch(std::move(draw), std::move(batch));
}
struct BinaryScoreScratch {
    Vector cross, solved;
};
// 64-byte aligned storage, so that a cell's record shares as few cache lines
// as possible.
template <class T> struct CacheAlignedAllocator {
    using value_type = T;
    CacheAlignedAllocator() = default;
    template <class U> CacheAlignedAllocator(const CacheAlignedAllocator<U> &) {}
    T *allocate(std::size_t n) {
        return static_cast<T *>(::operator new(n * sizeof(T), std::align_val_t(64)));
    }
    void deallocate(T *p, std::size_t) {
        ::operator delete(p, std::align_val_t(64));
    }
    template <class U> bool operator==(const CacheAlignedAllocator<U> &) const {
        return true;
    }
    template <class U> bool operator!=(const CacheAlignedAllocator<U> &) const {
        return false;
    }
};
struct BinaryScoreWorkspace {
    // One record per observation: residual, weight, then the weighted design row
    // (design(i, j) * weight). A selected cell's terms then sit in one or two
    // cache lines instead of three scattered arrays.
    std::size_t columns, stride;
    std::vector<double, CacheAlignedAllocator<double>> cells;
    Matrix inverse;
    Vector correction;
    double scale;
    explicit BinaryScoreWorkspace(const ScoreWorkspace &s)
        : columns(s.design.cols), stride(s.design.cols + 2),
          cells(s.design.rows * (s.design.cols + 2)), inverse(s.inverse_information),
          correction(s.project_nuisance ? detail::matvec(s.inverse_information, s.nuisance_score)
                                        : Vector(s.design.cols, 0)),
          scale(s.scale) {
        for (std::size_t i = 0; i < s.design.rows; ++i) {
            double *record = &cells[i * stride];
            record[0] = s.residuals[i];
            record[1] = s.weights[i];
            for (std::size_t j = 0; j < columns; ++j)
                record[2 + j] = s.design(i, j) * s.weights[i];
        }
    }
    std::size_t observations() const {
        return cells.size() / stride;
    }
    // Efficient score u and information for the indicator of `selected`
    // (information 0 when the indicator is constant after projection).
    std::pair<double, double> score(const std::vector<std::size_t> &selected,
                                    BinaryScoreScratch &scratch) const {
        auto &c = scratch.cross;
        c.assign(columns, 0);
        const std::size_t n = observations();
        double u = 0, v = 0;
        for (std::size_t k = 0; k < selected.size(); ++k) {
            auto i = selected[k];
            require(i < n && (k == 0 || selected[k - 1] < i),
                    "sampler indices must be sorted, unique and in range");
            const double *record = &cells[i * stride];
            u += record[0];
            v += record[1];
            for (std::size_t j = 0; j < columns; ++j)
                c[j] += record[2 + j];
        }
        return finish(u, v, c, scratch);
    }
    // The score from the sums over the selected records: u = sum of residuals,
    // v = sum of weights, c = sum of weighted design rows.
    std::pair<double, double> finish(double u, double v, const Vector &c,
                                     BinaryScoreScratch &scratch) const {
        u -= detail::dot(c, correction);
        auto &solved = scratch.solved;
        solved.assign(inverse.rows, 0);
        for (std::size_t i = 0; i < inverse.rows; ++i)
            for (std::size_t j = 0; j < inverse.cols; ++j)
                solved[i] += inverse(i, j) * c[j];
        double info = v - detail::dot(c, solved);
        if (!(info > 1e-12 * std::max(1.0, v)))
            info = 0;
        return {u, info};
    }
    double statistic_from(std::pair<double, double> score) const {
        const auto [u, info] = score;
        // Constant labels have zero efficient score; retain these draws in the null law.
        if (info == 0)
            return 0;
        return u / std::sqrt(info * scale);
    }
    // Statistics of many selections at once. The records are visited block by
    // block of observations; within a block every selection adds its records in
    // ascending order, so each selection's sums are those of score().
    void statistics(const std::vector<std::vector<std::size_t>> &selected, std::size_t count,
                    double *out, BinaryScoreScratch &scratch, Vector &sums,
                    std::vector<std::size_t> &cursor) const {
        statistics_of(
            count,
            [&](std::size_t d) { return std::make_pair(selected[d].data(), selected[d].size()); },
            out, scratch, sums, cursor);
    }
    // The same for selections given as (pointer, length) of sorted indices of
    // any unsigned type, by draw.
    template <class Selection>
    void statistics_of(std::size_t count, Selection selection, double *out,
                       BinaryScoreScratch &scratch, Vector &sums,
                       std::vector<std::size_t> &cursor) const {
        constexpr std::size_t block = 2048;
        const std::size_t n = observations(), width = columns + 2;
        sums.assign(count * width, 0);
        cursor.assign(count, 0);
        for (std::size_t a = 0; a < n; a += block) {
            const std::size_t b = std::min(n, a + block);
            for (std::size_t d = 0; d < count; ++d) {
                const auto [s, size] = selection(d);
                double *sum = &sums[d * width];
                std::size_t k = cursor[d];
                for (; k < size && s[k] < b; ++k) {
                    require(k == 0 || s[k - 1] < s[k],
                            "sampler indices must be sorted, unique and in range");
                    const double *record = &cells[static_cast<std::size_t>(s[k]) * stride];
                    sum[0] += record[0];
                    sum[1] += record[1];
                    for (std::size_t j = 0; j < columns; ++j)
                        sum[2 + j] += record[2 + j];
                }
                cursor[d] = k;
            }
        }
        auto &c = scratch.cross;
        c.resize(columns);
        for (std::size_t d = 0; d < count; ++d) {
            require(cursor[d] == selection(d).second,
                    "sampler indices must be sorted, unique and in range");
            const double *sum = &sums[d * width];
            std::copy(sum + 2, sum + width, c.begin());
            out[d] = statistic_from(finish(sum[0], sum[1], c, scratch));
        }
    }
    double statistic(const std::vector<std::size_t> &selected, BinaryScoreScratch &scratch) const {
        return statistic_from(score(selected, scratch));
    }
    double statistic(const std::vector<std::size_t> &selected) const {
        BinaryScoreScratch scratch;
        return statistic(selected, scratch);
    }
};
// Immutable after construction; multiple pairs and draw tasks can share this.
struct ResamplingWorkspace {
    ScoreWorkspace score;
    BinaryScoreWorkspace binary;
    ResamplingWorkspace(const Matrix &x, const FitResult &null) : score(x, null), binary(score) {}
};
struct ResampleOptions {
    std::uint64_t seed = 1, test_id = 0;
    std::vector<std::size_t> budgets = {499, 4999, 19999};
    Side side = Side::both;
    bool approximate = true, retain_null = false;
    // Empirical screen (0: off). When positive, the first budget always ends with
    // its empirical p-value, which is final unless it is at most `screen`; only
    // then are the later budgets drawn, with the approximation as usual. This is
    // the staging SCEPTRE 0.99.0's outputs show, with screen 0.02.
    double screen = 0;
    // Acceptance of the skew-normal fit (FitCheck). The KS check is blind to the
    // far tail, where the p-value is read: on the Gasperini panel fits it
    // accepted gave p-values a hundred to a few thousand times below the tail
    // counted over millions of draws (validation/SKEW_NORMAL_CHECK.json).
    FitCheck fit_check = FitCheck::strict;
    // Confirmation draws for FitCheck::strict. A fit that is sound on the last
    // budget but whose p-value that budget cannot check is not accepted there:
    // the draws continue to `confirm` in all, and the fit made on those is
    // reported if it is sound, the empirical p-value of the `confirm` draws
    // otherwise. A fit that is rejected on the last budget ends there with its
    // empirical p-value, so only candidates for very small p-values draw more.
    // By default (automatic_confirm) the number is ten times the last budget:
    // 199,999 for 19,999. With 19,999 draws a mildly heavy tail can pass the
    // check and give a p-value tens of times too small; on the Gasperini panel
    // 199,999 draws rejected those fits (validation/SKEW_NORMAL_CHECK.json).
    // 0, or a number not above the last budget, turns the confirmation off.
    static constexpr std::size_t automatic_confirm = std::numeric_limits<std::size_t>::max();
    std::size_t confirm = automatic_confirm;
    // The confirmation draws in effect for these budgets (0: none).
    std::size_t confirmation_draws() const {
        if (!approximate || fit_check != FitCheck::strict || budgets.empty())
            return 0;
        const std::size_t draws =
            confirm == automatic_confirm ? 10 * (budgets.back() + 1) - 1 : confirm;
        return draws > budgets.back() ? draws : 0;
    }
    // Shared prescreen (0: off). When positive, the first budget is drawn first
    // from the stream prescreen_test_id, which several tests may share. A test
    // whose empirical p-value there is above `prescreen` ends with it; every
    // other test is then staged exactly as without the prescreen, from its own
    // stream test_id, so its result is the same. A test can end at the
    // prescreen that would have continued from its own draws; a prescreen above
    // `screen` makes that rare.
    double prescreen = 0;
    std::uint64_t prescreen_test_id = 0;
    // End the screens (the shared prescreen and the empirical screen) as soon as
    // their outcome is certain: once enough draws are at least as extreme as the
    // statistic on each side that the screen's p-value must exceed its threshold
    // whatever the remaining draws. Checked every 32 draws. The decision is the
    // same as with every draw, and every test that continues has the same result.
    // A test that ends early reports the Besag-Clifford sequential p-value (two
    // sided: twice the smaller side's), raised to at least the smallest p-value
    // the full screen ends with, and the draws it used.
    bool sequential_screen = false;
    // Early stop (0: off). When positive, each budget that is not a screen ends
    // as soon as it is certain that the empirical p-value of that whole budget
    // would exceed `early_stop`: once enough of the test's draws so far (those
    // of the earlier budgets included) are at least as extreme as the statistic.
    // Checked every 32 draws. A test whose empirical p-value over the budget is
    // at most `early_stop` never ends early and has the same result as without
    // the option. A test that ends early reports the Besag-Clifford sequential
    // p-value as for sequential_screen, which is above `early_stop`, and the
    // draws it used, instead of the skew-normal or empirical p-value of the
    // whole budget.
    double early_stop = 0;
    // Opt in only for samplers safe to call concurrently. Uses an existing
    // OpenMP team; never creates a nested team or changes RNG stream keys.
    bool parallel_draws = false;
};
struct ResampleResult {
    double statistic = nan, p_value = nan, fit_discrepancy = nan;
    std::size_t draws = 0;
    bool valid = false, used_approximation = false;
    Vector null_statistics;
};
inline double empirical_tail(const Vector &null, double observed, Side side) {
    require(!null.empty() && std::isfinite(observed), "invalid empirical tail input");
    std::size_t left = 1, right = 1;
    for (double t : null) {
        require(std::isfinite(t), "nonfinite null statistic");
        left += t <= observed;
        right += t >= observed;
    }
    double l = static_cast<double>(left) / (null.size() + 1),
           r = static_cast<double>(right) / (null.size() + 1);
    return side == Side::left ? l : side == Side::right ? r : std::min(1.0, 2 * std::min(l, r));
}
struct SkewFit {
    double location = nan, scale = nan, shape = nan, discrepancy = nan;
    bool valid = false;
};
// With `sorted_values`, the values in ascending order are returned there as
// well (the fit sorts them for its KS discrepancy), whether or not it is valid.
inline SkewFit fit_skew_normal(const Vector &values, Vector *sorted_values = nullptr) {
    SkewFit out;
    if (sorted_values) {
        *sorted_values = values;
        std::sort(sorted_values->begin(), sorted_values->end());
    }
    if (values.size() < 20)
        return out;
    double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size(), m2 = 0,
           m3 = 0;
    for (double v : values) {
        if (!std::isfinite(v))
            return out;
        double d = v - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= values.size();
    m3 /= values.size();
    if (!(m2 > 1e-20))
        return out;
    double skew = m3 / std::pow(m2, 1.5);
    if (std::abs(skew) >= 0.995271746431156)
        return out;
    constexpr double pi = 3.14159265358979323846;
    double a = std::pow(2 * std::abs(skew) / (4 - pi), 2.0 / 3.0);
    double delta = std::copysign(std::sqrt(pi / 2 * a / (1 + a)), skew);
    out.scale = std::sqrt(m2 / (1 - 2 * delta * delta / pi));
    out.shape = delta / std::sqrt(1 - delta * delta);
    out.location = mean - out.scale * delta * std::sqrt(2 / pi);
    boost::math::skew_normal_distribution<double> sn(out.location, out.scale, out.shape);
    Vector local;
    if (!sorted_values) {
        local = values;
        std::sort(local.begin(), local.end());
    }
    const Vector &sorted = sorted_values ? *sorted_values : local;
    out.discrepancy = 0;
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        double cdf = boost::math::cdf(sn, sorted[i]);
        out.discrepancy =
            std::max({out.discrepancy, std::abs(cdf - static_cast<double>(i) / sorted.size()),
                      std::abs(cdf - static_cast<double>(i + 1) / sorted.size())});
    }
    out.valid = out.discrepancy <= 1.36 / std::sqrt(values.size());
    return out;
}
// Boost evaluates the skew-normal CDF as Phi(z) - 2 T(z, shape). Far in a tail
// that difference can round to a tiny negative number, so clamp to [0, 1].
inline double skew_normal_tail(const SkewFit &fit, double statistic, Side side) {
    boost::math::skew_normal_distribution<double> sn(fit.location, fit.scale, fit.shape);
    double l = std::clamp(boost::math::cdf(sn, statistic), 0.0, 1.0),
           r = std::clamp(boost::math::cdf(boost::math::complement(sn, statistic)), 0.0, 1.0);
    return side == Side::left ? l : side == Side::right ? r : std::min(1.0, 2 * std::min(l, r));
}
namespace detail {
// The smallest count c with P(Poisson(4^j) >= c) < 1/100, for j = 0, 1, ...
inline std::size_t poisson_limit_power_of_four(std::size_t j) {
    static const std::array<std::size_t, 12> limits = [] {
        std::array<std::size_t, 12> out{};
        double mean = 1;
        for (auto &limit : out) {
            // P(X >= c) is the regularized incomplete gamma function P(c, mean).
            // Two standard deviations above the mean it is still above 1/100.
            auto c = static_cast<std::size_t>(mean + 2 * std::sqrt(mean));
            while (boost::math::gamma_p(static_cast<double>(c), mean) >= 0.01)
                ++c;
            limit = c;
            mean *= 4;
        }
        return out;
    }();
    return limits.at(j);
}
// The part of FitCheck::strict that looks at the tail on the statistic's side
// (the smaller fitted tail for a two-sided test), where the p-value is read.
// The fit is sound when the draws there are no heavier than it says, each
// comparison at the 1% level:
// - the draws beyond the statistic are not more than the fit expects;
// - the most extreme draw is not beyond where the fit expects 1/100 of a draw;
// - the draws beyond the fitted quantiles with 1, 4, 16, ... expected draws
//   beyond them (down to a tail of 1/64) are not too many for a Poisson count;
// - the sample's excess kurtosis is not above the fit's by more than three
//   standard errors (sqrt(24/n)).
// `expected_beyond` receives the number of draws the fit expects beyond the
// statistic. `sorted`: the fitted values in ascending order.
inline bool skew_normal_tail_is_sound(const SkewFit &fit, const Vector &sorted, double statistic,
                                      Side side, double &expected_beyond) {
    using boost::math::cdf;
    using boost::math::complement;
    using boost::math::quantile;
    const std::size_t n = sorted.size();
    const double size = static_cast<double>(n), level = 0.01;
    boost::math::skew_normal_distribution<double> sn(fit.location, fit.scale, fit.shape);
    const double left = std::clamp(cdf(sn, statistic), 0.0, 1.0),
                 right = std::clamp(cdf(complement(sn, statistic)), 0.0, 1.0);
    const bool upper = side == Side::right || (side == Side::both && right < left);
    expected_beyond = size * (upper ? right : left);
    // Draws at least as extreme as x on that side.
    const auto beyond = [&](double x) {
        return upper ? static_cast<std::size_t>(sorted.end() -
                                                std::lower_bound(sorted.begin(), sorted.end(), x))
                     : static_cast<std::size_t>(std::upper_bound(sorted.begin(), sorted.end(), x) -
                                                sorted.begin());
    };
    const std::size_t observed = beyond(statistic);
    if (observed > 0 &&
        !(boost::math::gamma_p(static_cast<double>(observed), expected_beyond) >= level))
        return false;
    const double extreme = upper ? sorted.back() : sorted.front();
    const double beyond_extreme =
        size * std::clamp(upper ? cdf(complement(sn, extreme)) : cdf(sn, extreme), 0.0, 1.0);
    if (!(beyond_extreme >= level))
        return false;
    std::size_t j = 0;
    for (double expected = 1; expected <= size / 64; expected *= 4, ++j) {
        double point;
        try {
            point =
                upper ? quantile(complement(sn, expected / size)) : quantile(sn, expected / size);
        } catch (const std::exception &) {
            return false;
        }
        if (beyond(point) >= poisson_limit_power_of_four(j))
            return false;
    }
    double mean = 0, m2 = 0, m4 = 0;
    for (double v : sorted)
        mean += v;
    mean /= size;
    for (double v : sorted) {
        const double d = (v - mean) * (v - mean);
        m2 += d;
        m4 += d * d;
    }
    m2 /= size;
    m4 /= size;
    constexpr double pi = 3.14159265358979323846;
    const double delta = fit.shape / std::sqrt(1 + fit.shape * fit.shape),
                 b2 = 2 * delta * delta / pi,
                 fitted_kurtosis = 2 * (pi - 3) * b2 * b2 / ((1 - b2) * (1 - b2));
    return m4 / (m2 * m2) - 3 - fitted_kurtosis <= 3 * std::sqrt(24 / size);
}
inline void check_budgets(const ResampleOptions &o) {
    require(!o.budgets.empty() && o.budgets.front() > 0, "empty resampling budget");
    require(std::isfinite(o.screen) && o.screen >= 0 && o.screen < 1,
            "the empirical screen must be in [0, 1)");
    require(std::isfinite(o.prescreen) && o.prescreen >= 0 && o.prescreen < 1,
            "the shared prescreen must be in [0, 1)");
    require(o.prescreen_test_id <= UINT32_MAX, "prescreen test ID exceeds RNG range");
    require(std::isfinite(o.early_stop) && o.early_stop >= 0 && o.early_stop < 1,
            "the early stop must be in [0, 1)");
    require(o.confirmation_draws() <= UINT32_MAX, "confirmation draws must fit uint32");
    for (std::size_t i = 0; i < o.budgets.size(); ++i)
        require(o.budgets[i] <= UINT32_MAX && (i == 0 || o.budgets[i] > o.budgets[i - 1]),
                "budgets must be increasing and fit uint32");
    require(o.test_id <= UINT32_MAX, "test ID exceeds RNG range");
}
// The staging of the null draws as a state machine, so that a scheduler can run
// each stage's draws itself. Draws [begin(), end()) of null are written, then
// next() applies the stage's rule and either opens the next stage or finishes.
// After each budget a skew-normal fit that is accepted (FitCheck) ends the
// drawing; with an empirical screen the first budget ends with its empirical
// p-value unless that is at most the screen. With a shared prescreen the first
// budget is drawn twice, first from the shared stream (test_id() tells which).
// A sequential stage (sequential_screen, early_stop) can end before its budget:
// advance() tells when.
class NullStages {
    ResampleOptions o_;
    ResampleResult out_;
    Vector null_;
    std::size_t index_ = 0;
    // The budgets before the confirmation draws (ResampleOptions::confirm),
    // which are appended to o_.budgets as one more stage.
    std::size_t counting_ = 0;
    bool screened_ = false, prescreening_ = false;
    // Sequential stages: draws counted so far, those at most and at least the
    // statistic, the count that decides the stage, and where it was decided.
    std::size_t counted_ = 0, below_ = 0, above_ = 0, decisive_ = 0, decided_at_ = 0;

    // Whether the stage being drawn is a screen.
    bool screening() const {
        return prescreening_ || (o_.screen > 0 && !screened_);
    }
    double threshold() const {
        return prescreening_ ? o_.prescreen : screening() ? o_.screen : o_.early_stop;
    }
    // The empirical p-value of the stage's budget when `count` draws on the
    // deciding side are at least as extreme (empirical_tail's arithmetic).
    double stage_p(std::size_t count) const {
        const double tail = static_cast<double>(1 + count) / (end() + 1);
        return o_.side == Side::both ? std::min(1.0, 2 * tail) : tail;
    }
    std::size_t deciding(std::size_t below, std::size_t above) const {
        return o_.side == Side::left    ? below
               : o_.side == Side::right ? above
                                        : std::min(below, above);
    }
    // Opens a stage: the count that decides it is the smallest at which the
    // budget's empirical p-value exceeds the stage's threshold. It is at least
    // 1, and at least the count of the stage before when the draws continue
    // (`restart` false): a test that passed that stage is below it, so it is
    // decided on later draws, and its sequential p-value stays valid. With
    // `restart` the draws start again from another stream.
    void open_stage(bool restart) {
        if (restart)
            counted_ = below_ = above_ = 0;
        decided_at_ = 0;
        std::size_t c = restart ? 1 : std::max<std::size_t>(1, decisive_);
        while (c <= end() && !(stage_p(c) > threshold()))
            ++c;
        decisive_ = c;
    }
    // Ends the stage with the sequential p-value when advance() decided it.
    bool finish_early() {
        if (!sequential() || (advance(end()), decided_at_ == 0))
            return false;
        const std::size_t h = decisive_;
        std::size_t below = 0, above = 0, at_below = 0, at_above = 0;
        for (std::size_t k = 0; k < decided_at_; ++k) {
            const double t = null_[k];
            if (t <= out_.statistic && ++below == h)
                at_below = k + 1;
            if (t >= out_.statistic && ++above == h)
                at_above = k + 1;
        }
        const double hd = static_cast<double>(h);
        double p = o_.side == Side::left    ? hd / at_below
                   : o_.side == Side::right ? hd / at_above
                                            : std::min(1.0, 2 * hd / std::max(at_below, at_above));
        out_.p_value = std::max(p, stage_p(h));
        out_.draws = decided_at_;
        out_.valid = true;
        if (o_.retain_null) {
            null_.resize(decided_at_);
            out_.null_statistics = std::move(null_);
        }
        return true;
    }

  public:
    // A sequential stage is checked after every `checkpoint` draws.
    static constexpr std::size_t checkpoint = 32;
    // The end of the chunk of a sequential stage that starts at draw `first`:
    // the next check, so that no draw is generated beyond the one that ends it.
    static std::size_t next_checkpoint(std::size_t first) {
        return (first / checkpoint + 1) * checkpoint;
    }
    NullStages(const ResampleOptions &o, double statistic) : o_(o) {
        out_.statistic = statistic;
        counting_ = o_.budgets.size();
        null_.reserve(o_.budgets.back());
        if (const std::size_t confirmation = o_.confirmation_draws())
            o_.budgets.push_back(confirmation);
        null_.resize(o_.budgets.front());
        prescreening_ = o_.prescreen > 0;
        open_stage(true);
    }
    // Whether the stage being drawn may end early: a screen with
    // sequential_screen, or a later budget with early_stop.
    bool sequential() const {
        return screening() ? o_.sequential_screen : o_.early_stop > 0;
    }
    // Draws [begin(), written) are written. True when a sequential stage is
    // decided: no more draws are needed before next().
    bool advance(std::size_t written) {
        if (!sequential() || decided_at_)
            return decided_at_ != 0;
        if (decisive_ > end())
            return false;
        while (counted_ < written) {
            const double t = null_[counted_++];
            require(std::isfinite(t), "nonfinite null statistic");
            below_ += t <= out_.statistic;
            above_ += t >= out_.statistic;
            if ((counted_ % checkpoint == 0 || counted_ == end()) &&
                deciding(below_, above_) >= decisive_) {
                decided_at_ = counted_;
                return true;
            }
        }
        return false;
    }
    // The stream of the draws to write now.
    std::uint64_t test_id() const {
        return prescreening_ ? o_.prescreen_test_id : o_.test_id;
    }
    std::size_t begin() const {
        return index_ == 0 ? 0 : o_.budgets[index_ - 1];
    }
    std::size_t end() const {
        return o_.budgets[index_];
    }
    double *null() {
        return null_.data();
    }
    // Call when draws [begin(), end()) are written, or when advance() has
    // decided a sequential stage. True when finished.
    bool next() {
        if (finish_early())
            return true;
        if (prescreening_) {
            prescreening_ = false;
            const double p = empirical_tail(null_, out_.statistic, o_.side);
            if (!(p > o_.prescreen)) {
                open_stage(true);
                return false; // the first budget again, from the test's own stream
            }
            out_.draws = null_.size();
            out_.p_value = p;
            out_.valid = true;
            if (o_.retain_null)
                out_.null_statistics = std::move(null_);
            return true;
        }
        // The last budget, and the last stage (the confirmation draws if any).
        const bool last_budget = index_ + 1 == counting_, last = index_ + 1 == o_.budgets.size();
        out_.draws = null_.size();
        bool stop = false, beyond_reach = false;
        if (o_.screen > 0 && !screened_) {
            screened_ = true;
            const double p = empirical_tail(null_, out_.statistic, o_.side);
            if (p > o_.screen || last_budget) {
                out_.p_value = p;
                stop = true;
            }
        } else if (o_.approximate) {
            const bool strict = o_.fit_check == FitCheck::strict;
            Vector sorted;
            auto sf = fit_skew_normal(null_, strict ? &sorted : nullptr);
            out_.fit_discrepancy = sf.discrepancy;
            bool accepted = sf.valid;
            if (accepted && strict) {
                // A p-value that this stage cannot check (the fit expects less
                // than half a draw beyond the statistic) waits for the next
                // one; on the last stage it is accepted.
                double expected_beyond = 0;
                accepted = detail::skew_normal_tail_is_sound(sf, sorted, out_.statistic, o_.side,
                                                             expected_beyond);
                if (accepted && !last && expected_beyond < 0.5) {
                    accepted = false;
                    beyond_reach = true;
                }
            }
            if (accepted) {
                out_.p_value = skew_normal_tail(sf, out_.statistic, o_.side);
                out_.used_approximation = true;
                stop = true;
            }
        }
        // After the last budget only a sound fit beyond its reach draws on, to
        // be confirmed; a rejected fit ends with the empirical p-value.
        if (!stop && !last && (!last_budget || beyond_reach)) {
            ++index_;
            null_.resize(o_.budgets[index_]);
            open_stage(false);
            return false;
        }
        if (!out_.used_approximation && !(screened_ && std::isfinite(out_.p_value)))
            out_.p_value = empirical_tail(null_, out_.statistic, o_.side);
        out_.valid = std::isfinite(out_.p_value);
        if (o_.retain_null)
            out_.null_statistics = std::move(null_);
        return true;
    }
    ResampleResult &result() {
        return out_;
    }
};
// Staged null draws for out.statistic. make_batch() returns a callable
// void(std::uint64_t test, std::size_t first, std::size_t count, double *null)
// that owns its scratch and writes the statistics of draws [first, first +
// count) of stream `test`; draw k always uses CounterRng(seed, test, k), so
// results do not depend on how the draws are split into batches or scheduled.
template <class MakeBatch>
inline void staged_null(ResampleResult &out, const ResampleOptions &o, MakeBatch make_batch,
                        std::size_t chunk_size) {
    auto serial = make_batch();
    NullStages stages(o, out.statistic);
    do {
        const std::size_t begin = stages.begin(), budget = stages.end();
        const std::uint64_t test = stages.test_id();
        double *null = stages.null();
        // A stage that may end early is drawn checkpoint by checkpoint, in order.
        const bool sequential = stages.sequential();
        const std::size_t chunks = (budget - begin + chunk_size - 1) / chunk_size;
#if defined(_OPENMP) && _OPENMP >= 201511
        if (o.parallel_draws && omp_in_parallel() && chunks > 1 && !sequential) {
            // Idle workers can help even when there is only one gene.
            std::exception_ptr failure;
            std::mutex failure_mutex;
#pragma omp taskloop grainsize(1) shared(null, make_batch, failure, failure_mutex)                 \
    firstprivate(begin, budget, chunk_size, test)
            for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
                try {
                    auto batch = make_batch();
                    const auto first = begin + chunk * chunk_size;
                    batch(test, first, std::min(budget, first + chunk_size) - first, null + first);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(failure_mutex);
                    if (!failure)
                        failure = std::current_exception();
                }
            }
            // The implicit taskgroup has joined all draws before tail fitting.
            if (failure)
                std::rethrow_exception(failure);
        } else
#endif
        {
            (void)chunks;
            for (std::size_t first = begin, last; first < budget; first = last) {
                last = std::min(budget, sequential ? NullStages::next_checkpoint(first)
                                                   : first + chunk_size);
                serial(test, first, last - first, null + first);
                if (stages.advance(last))
                    break;
            }
        }
    } while (!stages.next());
    out = std::move(stages.result());
}
} // namespace detail
// Draws [first, first + count) of one random stream, stored compactly so that
// many tests can be scored against the same draws (SceptreOptions::reuse_draws).
struct CachedDraws {
    std::uint64_t seed = 0, test = 0;
    std::size_t first = 0, count = 0;
    std::vector<std::size_t> offsets; // count + 1
    std::vector<std::uint32_t> cells;
    CachedDraws(const Sampler &sampler, std::uint64_t seed_, std::uint64_t test_,
                std::size_t first_, std::size_t count_, std::size_t observations)
        : seed(seed_), test(test_), first(first_), count(count_), offsets{0} {
        require(observations <= UINT32_MAX, "too many observations to cache draws");
        std::vector<std::vector<std::size_t>> batch;
        SamplerScratch scratch;
        for (std::size_t done = 0; done < count; done += 256) {
            const std::size_t size = std::min<std::size_t>(256, count - done);
            sampler.batch(seed_, test_, first + done, size, batch, scratch);
            for (std::size_t k = 0; k < size; ++k) {
                for (auto i : batch[k])
                    cells.push_back(static_cast<std::uint32_t>(i));
                offsets.push_back(cells.size());
            }
        }
        cells.shrink_to_fit();
    }
    std::size_t bytes() const {
        return cells.capacity() * sizeof(std::uint32_t) + offsets.capacity() * sizeof(std::size_t);
    }
};
namespace detail {
// Task-owned buffers for generating and scoring draws.
struct DrawScratch {
    std::vector<std::vector<std::size_t>> selected;
    BinaryScoreScratch score;
    SamplerScratch sampler;
    Vector sums;
    std::vector<std::size_t> cursor;
};
// Statistics of draws [first, first + count) of the stream (seed, test): from
// `cached` where it holds draws of that stream, otherwise generated in one batch.
inline void draw_statistics(const BinaryScoreWorkspace &workspace, const Sampler &sampler,
                            std::uint64_t seed, std::uint64_t test, const CachedDraws *cached,
                            std::size_t first, std::size_t count, double *null,
                            DrawScratch &scratch) {
    if (cached && cached->seed == seed && cached->test == test && first >= cached->first &&
        first < cached->first + cached->count) {
        const auto &c = *cached;
        const std::size_t offset = first - c.first, stored = std::min(count, c.count - offset);
        workspace.statistics_of(
            stored,
            [&](std::size_t d) {
                const auto begin = c.offsets[offset + d], end = c.offsets[offset + d + 1];
                return std::make_pair(c.cells.data() + begin, end - begin);
            },
            null, scratch.score, scratch.sums, scratch.cursor);
        first += stored;
        count -= stored;
        null += stored;
        if (count == 0)
            return;
    }
    sampler.batch(seed, test, first, count, scratch.selected, scratch.sampler);
    workspace.statistics(scratch.selected, count, null, scratch.score, scratch.sums,
                         scratch.cursor);
}
// Generates and scores batches of CRT or permutation draws for one test: the
// sampler walks its arrays once per batch, and every draw's statistic is
// accumulated in its own ascending cell order, as one draw at a time would.
struct BinaryDrawBatch {
    const BinaryScoreWorkspace &workspace;
    const Sampler &sampler;
    std::uint64_t seed;
    // Stored draws covering a given first draw, if any.
    std::function<const CachedDraws *(std::size_t)> cache;
    DrawScratch scratch;
    BinaryDrawBatch(const BinaryScoreWorkspace &w, const Sampler &s, std::uint64_t seed_,
                    std::function<const CachedDraws *(std::size_t)> cache_ = {})
        : workspace(w), sampler(s), seed(seed_), cache(std::move(cache_)) {}
    void operator()(std::uint64_t test, std::size_t first, std::size_t count, double *null) {
        draw_statistics(workspace, sampler, seed, test, cache ? cache(first) : nullptr, first,
                        count, null, scratch);
    }
};
// The observed statistic of a binary covariate, or NaN when its score is not
// regular (no positive information after projection): then there is nothing
// to calibrate. Uses precisely the accumulation of the sampled scores.
inline double observed_statistic_of(const BinaryScoreWorkspace &workspace,
                                    const std::vector<std::size_t> &selected) {
    BinaryScoreScratch scratch;
    const auto score = workspace.score(selected, scratch);
    const double statistic = workspace.statistic_from(score);
    return score.second > 0 && std::isfinite(statistic) ? statistic : nan;
}
inline double observed_statistic(const BinaryScoreWorkspace &workspace, const Vector &observed) {
    for (double g : observed)
        require(g == 0 || g == 1, "resampled covariate must be binary");
    require(observed.size() == workspace.observations(), "score covariate size mismatch");
    std::vector<std::size_t> selected;
    for (std::size_t i = 0; i < observed.size(); ++i)
        if (observed[i])
            selected.push_back(i);
    return observed_statistic_of(workspace, selected);
}
inline ResampleResult resample_prepared(const BinaryScoreWorkspace &workspace,
                                        const Vector &observed, const Sampler &sampler,
                                        const ResampleOptions &o) {
    check_budgets(o);
    ResampleResult out;
    out.statistic = observed_statistic(workspace, observed);
    if (!std::isfinite(out.statistic)) {
        out.statistic = nan;
        return out;
    }
    staged_null(out, o, [&] { return BinaryDrawBatch(workspace, sampler, o.seed); }, 256);
    return out;
}
} // namespace detail
inline ResampleResult resample(const ScoreWorkspace &score, const Vector &observed,
                               const Sampler &sampler, const ResampleOptions &o = {}) {
    return detail::resample_prepared(BinaryScoreWorkspace(score), observed, sampler, o);
}
inline ResampleResult resample(const ResamplingWorkspace &workspace, const Vector &observed,
                               const Sampler &sampler, const ResampleOptions &o = {}) {
    return detail::resample_prepared(workspace.binary, observed, sampler, o);
}
// Permutation calibration for any covariate: each draw permutes the observed
// values within strata (Fisher-Yates on a fresh copy, so draws stay independent
// of scheduling) and recomputes the efficient score statistic, O(n p) per draw.
// Binary covariates are faster through resample() with permutation_sampler().
inline ResampleResult resample_permutation(const ScoreWorkspace &score, const Vector &observed,
                                           const std::vector<std::size_t> &strata = {},
                                           const ResampleOptions &o = {}) {
    detail::check_budgets(o);
    require(strata.empty() || strata.size() == observed.size(), "strata length mismatch");
    ResampleResult out;
    out.statistic = score.statistic(observed);
    if (!std::isfinite(out.statistic))
        return out;
    std::map<std::size_t, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < observed.size(); ++i)
        groups[strata.empty() ? 0 : strata[i]].push_back(i);
    std::vector<std::vector<std::size_t>> members;
    for (auto &entry : groups)
        members.push_back(std::move(entry.second));
    detail::staged_null(
        out, o,
        [&] {
            struct Draws {
                const ScoreWorkspace &score;
                const Vector &observed;
                const std::vector<std::vector<std::size_t>> &members;
                std::uint64_t seed;
                Vector permuted;
                void operator()(std::uint64_t test, std::size_t first, std::size_t count,
                                double *null) {
                    for (std::size_t k = 0; k < count; ++k) {
                        CounterRng rng(seed, test, first + k);
                        permuted = observed;
                        for (const auto &m : members)
                            for (std::size_t r = m.size(); r > 1; --r) {
                                const auto j = static_cast<std::size_t>(rng.bounded(r));
                                std::swap(permuted[m[r - 1]], permuted[m[j]]);
                            }
                        null[k] = score.statistic(permuted);
                    }
                }
            };
            return Draws{score, observed, members, o.seed, {}};
        },
        16);
    return out;
}
} // namespace rapidglm
