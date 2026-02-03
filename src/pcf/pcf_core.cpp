#include "silentmpc/pcf/pcf_core.hpp"
#include "silentmpc/pcf/lpn_security.hpp"
#include "silentmpc/algebra/gf2.hpp"
#include "silentmpc/algebra/z2k.hpp"
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <cstring>
#include <unordered_map>

namespace silentmpc::pcf {

// ============================================================================
// Helper Functions for GF2 Packing (declare early for use in LPNCore)
// ============================================================================

namespace {
    void pack_gf2_vector(const std::vector<algebra::GF2>& vec, std::vector<uint64_t>& packed) {
        size_t n_words = (vec.size() + 63) / 64;
        packed.resize(n_words, 0);
        for (size_t i = 0; i < vec.size(); ++i) {
            if (vec[i].as_bool()) {
                packed[i / 64] |= (1ULL << (i % 64));
            }
        }
    }

    void unpack_gf2_vector(const std::vector<uint64_t>& packed, std::vector<algebra::GF2>& vec, size_t n_bits) {
        vec.resize(n_bits);
        for (size_t i = 0; i < n_bits; ++i) {
            bool bit = (packed[i / 64] >> (i % 64)) & 1;
            vec[i] = algebra::GF2(bit);
        }
    }
}

// ============================================================================
// PRF/KDF Utilities
// ============================================================================

namespace {

/// Domain-separated KDF for deriving seeds and values
void derive_key(
    const DerivationContext& ctx,
    std::span<uint8_t> output
) {
    // Serialize context
    auto context_bytes = ctx.serialize();

    // Use SHA-256 based KDF
    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(mdctx, context_bytes.data(), context_bytes.size());

    uint8_t hash[32];
    unsigned int hash_len;
    EVP_DigestFinal_ex(mdctx, hash, &hash_len);
    EVP_MD_CTX_free(mdctx);

    // Copy to output (truncate or extend as needed)
    size_t copy_len = std::min(output.size(), static_cast<size_t>(32));
    std::memcpy(output.data(), hash, copy_len);

    // If output needs more data, use counter mode
    if (output.size() > 32) {
        size_t offset = 32;
        uint32_t counter = 1;

        while (offset < output.size()) {
            mdctx = EVP_MD_CTX_new();
            EVP_DigestInit_ex(mdctx, EVP_sha256(), nullptr);
            EVP_DigestUpdate(mdctx, context_bytes.data(), context_bytes.size());
            EVP_DigestUpdate(mdctx, &counter, sizeof(counter));
            EVP_DigestFinal_ex(mdctx, hash, &hash_len);
            EVP_MD_CTX_free(mdctx);

            size_t to_copy = std::min(static_cast<size_t>(32), output.size() - offset);
            std::memcpy(output.data() + offset, hash, to_copy);
            offset += to_copy;
            counter++;
        }
    }
}

/// AES-CTR based PRG for expanding seeds
void prg_expand(
    const std::array<uint8_t, 16>& seed,
    std::span<uint8_t> output
) {
    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();

    // Use AES-128-CTR with zero IV
    uint8_t iv[16] = {0};
    EVP_EncryptInit_ex(ctx, EVP_aes_128_ctr(), nullptr, seed.data(), iv);

    // Generate keystream
    std::vector<uint8_t> zeros(output.size(), 0);
    int outlen;
    EVP_EncryptUpdate(ctx, output.data(), &outlen, zeros.data(), zeros.size());

    EVP_CIPHER_CTX_free(ctx);
}

} // anonymous namespace

// ============================================================================
// PCFGenerator Template Implementation (STUB - not used, needs implementation)
// ============================================================================

/*
template<Ring R>
void PCFGenerator<R>::initialize(
    const SessionId& sid,
    EpochId epoch,
    CorrelationIndex correlation_index,
    const PCFParameters& params
) {
    sid_ = sid;
    epoch_ = epoch;
    correlation_index_ = correlation_index;
    params_ = params;

    // Derive master seeds for both parties
    DerivationContext ctx_p0{sid, epoch, correlation_index, 0, TermId::Setup, 0};
    DerivationContext ctx_p1{sid, epoch, correlation_index, 1, TermId::Setup, 0};

    derive_key(ctx_p0, std::span(seed_p0_));
    derive_key(ctx_p1, std::span(seed_p1_));
}

template<Ring R>
std::vector<R> PCFGenerator<R>::generate_dense_vector(
    PartyId party,
    size_t tile_id
) {
    if (party != 0 && party != 1) {
        throw std::invalid_argument("party must be 0 or 1");
    }

    const auto& seed = (party == 0) ? seed_p0_ : seed_p1_;

    // Derive tile-specific seed
    DerivationContext ctx{sid_, epoch_, correlation_index_, party,
                         TermId::DenseVector, tile_id};

    std::array<uint8_t, 16> tile_seed;
    derive_key(ctx, std::span(tile_seed));

    // Expand seed to dense vector
    size_t vector_size = params_.N;
    size_t bytes_needed = vector_size * sizeof(R);

    std::vector<uint8_t> random_bytes(bytes_needed);
    prg_expand(tile_seed, std::span(random_bytes));

    // Convert bytes to ring elements
    std::vector<R> dense_vector;
    dense_vector.reserve(vector_size);

    for (size_t i = 0; i < vector_size; ++i) {
        // Extract sizeof(R) bytes and convert to R
        size_t offset = i * sizeof(R);

        if constexpr (std::is_same_v<R, algebra::GF2>) {
            // GF(2): each bit is an element
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            bool bit = (random_bytes[byte_idx] >> bit_idx) & 1;
            dense_vector.push_back(algebra::GF2(bit));
        } else {
            // For Z2k or other rings: interpret bytes as uint64_t
            uint64_t value = 0;
            size_t copy_size = std::min(sizeof(uint64_t), sizeof(R));
            for (size_t j = 0; j < copy_size && (offset + j) < random_bytes.size(); ++j) {
                value |= static_cast<uint64_t>(random_bytes[offset + j]) << (j * 8);
            }
            dense_vector.push_back(R::from_uint64(value));
        }
    }

    return dense_vector;
}

template<Ring R>
std::vector<size_t> PCFGenerator<R>::generate_sparse_indices(
    PartyId party,
    size_t tile_id
) {
    if (party != 0 && party != 1) {
        throw std::invalid_argument("party must be 0 or 1");
    }

    // Derive seed for sparse indices
    DerivationContext ctx{sid_, epoch_, correlation_index_, party,
                         TermId::SparseIndices, tile_id};

    std::array<uint8_t, 16> indices_seed;
    derive_key(ctx, std::span(indices_seed));

    // Generate regular sparse distribution
    // Regular means evenly spaced with randomization
    size_t N = params_.N;
    size_t tau = params_.tau;

    std::vector<size_t> indices;
    indices.reserve(tau);

    // Simple approach: partition domain into tau buckets
    size_t bucket_size = N / tau;

    std::vector<uint8_t> random_bytes(tau * 4);  // 4 bytes per index
    prg_expand(indices_seed, std::span(random_bytes));

    for (size_t i = 0; i < tau; ++i) {
        // Start of this bucket
        size_t bucket_start = i * bucket_size;

        // Random offset within bucket
        uint32_t offset_raw = 0;
        for (size_t j = 0; j < 4; ++j) {
            offset_raw |= static_cast<uint32_t>(random_bytes[i * 4 + j]) << (j * 8);
        }

        size_t offset = offset_raw % bucket_size;
        size_t index = bucket_start + offset;

        // Ensure within bounds
        if (index >= N) {
            index = N - 1;
        }

        indices.push_back(index);
    }

    return indices;
}

template<Ring R>
std::vector<R> PCFGenerator<R>::generate_sparse_values(
    PartyId party,
    size_t tile_id
) {
    if (party != 0 && party != 1) {
        throw std::invalid_argument("party must be 0 or 1");
    }

    // Derive seed for sparse values
    DerivationContext ctx{sid_, epoch_, correlation_index_, party,
                         TermId::SparseValues, tile_id};

    std::array<uint8_t, 16> values_seed;
    derive_key(ctx, std::span(values_seed));

    size_t tau = params_.tau;
    size_t bytes_needed = tau * sizeof(R);

    std::vector<uint8_t> random_bytes(bytes_needed);
    prg_expand(values_seed, std::span(random_bytes));

    std::vector<R> values;
    values.reserve(tau);

    for (size_t i = 0; i < tau; ++i) {
        if constexpr (std::is_same_v<R, algebra::GF2>) {
            bool bit = random_bytes[i / 8] & (1 << (i % 8));
            values.push_back(algebra::GF2(bit));
        } else {
            uint64_t value = 0;
            for (size_t j = 0; j < sizeof(uint64_t) && (i * sizeof(R) + j) < random_bytes.size(); ++j) {
                value |= static_cast<uint64_t>(random_bytes[i * sizeof(R) + j]) << (j * 8);
            }
            values.push_back(R::from_uint64(value));
        }
    }

    return values;
}

template<Ring R>
typename PCFGenerator<R>::StatisticalParameters PCFGenerator<R>::compute_security_level(
    const PCFParameters& params,
    size_t statistical_security
) {
    StatisticalParameters result;

    // From Section 3.1 and Tables 6-8:
    // Use exact security analysis from LPNSecurityEstimator
    // Based on Equation (3): Adv_Π(Q) ≤ Adv^Setup + c·Q·Adv^PRG + Adv^LPN(Q) + ε_chk

    // Determine backend type from parameters
    bool use_dpf = true;  // Default to DPF backend

    // Number of queries: use recommended maximum
    size_t Q = LPNSecurityEstimator::MAX_QUERIES;  // Q_max = 2^30

    // Perform full security analysis
    auto analysis = LPNSecurityEstimator::analyze(
        PCFParameters(params.N, params.n, params.w),
        Q,
        statistical_security,
        use_dpf,
        0.0  // ε_chk = 0 for semi-honest model
    );

    // Extract results
    result.computational_security = analysis.computational_security;
    result.statistical_security = analysis.statistical_security;
    result.is_secure = analysis.is_secure;

    // LPN noise rate: w / N (effective Bernoulli rate)
    result.noise_rate = analysis.effective_bernoulli_rate;

    return result;
}

// ============================================================================
// FourTermComputer Template Implementation
// ============================================================================

template<Ring R>
FourTermResult<R> FourTermComputer<R>::compute(
    const std::vector<R>& dense0,
    const std::vector<R>& dense1,
    const std::vector<size_t>& sparse0_indices,
    const std::vector<R>& sparse0_values,
    const std::vector<size_t>& sparse1_indices,
    const std::vector<R>& sparse1_values
) {
    if (dense0.size() != dense1.size()) {
        throw std::invalid_argument("Dense vector size mismatch");
    }

    size_t N = dense0.size();
    FourTermResult<R> result;

    // Term 1: (As0) ⊙ (As1) - Hadamard product of dense vectors
    result.term1_dense_dense.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        result.term1_dense_dense.push_back(dense0[i] * dense1[i]);
    }

    // Term 2: e0 ⊙ (As1) - sparse values at sparse indices
    result.term2_sparse_dense.reserve(sparse0_indices.size());
    for (size_t i = 0; i < sparse0_indices.size(); ++i) {
        size_t idx = sparse0_indices[i];
        if (idx < N) {
            result.term2_sparse_dense.push_back(sparse0_values[i] * dense1[idx]);
        } else {
            result.term2_sparse_dense.push_back(R::zero());
        }
    }

    // Term 3: e1 ⊙ (As0) - sparse values at sparse indices
    result.term3_sparse_dense.reserve(sparse1_indices.size());
    for (size_t i = 0; i < sparse1_indices.size(); ++i) {
        size_t idx = sparse1_indices[i];
        if (idx < N) {
            result.term3_sparse_dense.push_back(sparse1_values[i] * dense0[idx]);
        } else {
            result.term3_sparse_dense.push_back(R::zero());
        }
    }

    // Term 4: e0 ⊙ e1 - sparse-sparse product (collision handling)
    // Build map for sparse1 for efficient lookup
    std::unordered_map<size_t, R> sparse1_map;
    for (size_t i = 0; i < sparse1_indices.size(); ++i) {
        sparse1_map[sparse1_indices[i]] = sparse1_values[i];
    }

    for (size_t i = 0; i < sparse0_indices.size(); ++i) {
        size_t idx = sparse0_indices[i];
        auto it = sparse1_map.find(idx);

        if (it != sparse1_map.end()) {
            // Collision detected
            result.term4_sparse_sparse.push_back(sparse0_values[i] * it->second);
        } else {
            result.term4_sparse_sparse.push_back(R::zero());
        }
    }

    return result;
}

template<Ring R>
FourTermResult<R> FourTermComputer<R>::compute_tiled(
    const PCFGenerator<R>& generator,
    size_t tile_size
) {
    size_t N = generator.parameters().N;
    size_t num_tiles = (N + tile_size - 1) / tile_size;

    FourTermResult<R> accumulated_result;

    for (size_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // Generate vectors for this tile
        auto dense0 = generator.generate_dense_vector(0, tile_id);
        auto dense1 = generator.generate_dense_vector(1, tile_id);

        auto sparse0_idx = generator.generate_sparse_indices(0, tile_id);
        auto sparse0_val = generator.generate_sparse_values(0, tile_id);

        auto sparse1_idx = generator.generate_sparse_indices(1, tile_id);
        auto sparse1_val = generator.generate_sparse_values(1, tile_id);

        // Compute four terms for this tile
        auto tile_result = compute(dense0, dense1, sparse0_idx, sparse0_val,
                                   sparse1_idx, sparse1_val);

        // Accumulate results
        if (tile_id == 0) {
            accumulated_result = std::move(tile_result);
        } else {
            // Append results
            accumulated_result.term1_dense_dense.insert(
                accumulated_result.term1_dense_dense.end(),
                tile_result.term1_dense_dense.begin(),
                tile_result.term1_dense_dense.end()
            );

            accumulated_result.term2_sparse_dense.insert(
                accumulated_result.term2_sparse_dense.end(),
                tile_result.term2_sparse_dense.begin(),
                tile_result.term2_sparse_dense.end()
            );

            accumulated_result.term3_sparse_dense.insert(
                accumulated_result.term3_sparse_dense.end(),
                tile_result.term3_sparse_dense.begin(),
                tile_result.term3_sparse_dense.end()
            );

            accumulated_result.term4_sparse_sparse.insert(
                accumulated_result.term4_sparse_sparse.end(),
                tile_result.term4_sparse_sparse.begin(),
                tile_result.term4_sparse_sparse.end()
            );
        }
    }

    return accumulated_result;
}

// Explicit template instantiations
// PCFGenerator template instantiations commented out - class not implemented
// template class PCFGenerator<algebra::GF2>;
// template class PCFGenerator<algebra::Z2k<32>>;
// template class PCFGenerator<algebra::Z2k<64>>;
*/

// ============================================================================
// RegularNoise Implementation (Definition 1, Section 5.1)
// ============================================================================

template<Ring R>
RegularNoise<R>::RegularNoise(size_t N, size_t w)
    : total_length_(N), weight_(w) {
    support_.reserve(w);
    values_.reserve(w);
}

template<Ring R>
RegularNoise<R> RegularNoise<R>::sample(
    size_t N, size_t w,
    const DerivationContext& ctx,
    const std::vector<uint8_t>& seed
) {
    RegularNoise<R> noise(N, w);

    if (w == 0 || N == 0) {
        return noise;
    }

    // Regular sparse distribution: partition [0, N) into w blocks
    // Each block has size N/w, with exactly one non-zero entry per block
    size_t block_size = N / w;

    if (block_size == 0) {
        throw std::invalid_argument("Weight w cannot exceed domain size N");
    }

    // Derive PRG seed from context and input seed
    std::array<uint8_t, 16> prg_seed;
    auto ctx_bytes = ctx.serialize();

    // Combine context and seed using SHA-256
    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(mdctx, ctx_bytes.data(), ctx_bytes.size());
    EVP_DigestUpdate(mdctx, seed.data(), seed.size());

    uint8_t hash[32];
    unsigned int hash_len;
    EVP_DigestFinal_ex(mdctx, hash, &hash_len);
    EVP_MD_CTX_free(mdctx);

    std::memcpy(prg_seed.data(), hash, 16);

    // Generate random bytes for indices and values
    size_t bytes_for_indices = w * 4;  // 4 bytes per index
    size_t bytes_for_values = w * sizeof(uint64_t);  // up to 8 bytes per value
    size_t total_random_bytes = bytes_for_indices + bytes_for_values;

    std::vector<uint8_t> random_bytes(total_random_bytes);
    prg_expand(prg_seed, std::span(random_bytes));

    // Generate regular sparse support
    for (size_t i = 0; i < w; ++i) {
        // Block [i * block_size, (i+1) * block_size)
        size_t block_start = i * block_size;
        size_t block_end = (i == w - 1) ? N : (i + 1) * block_size;
        size_t current_block_size = block_end - block_start;

        // Random offset within block
        uint32_t offset_raw = 0;
        for (size_t j = 0; j < 4; ++j) {
            offset_raw |= static_cast<uint32_t>(random_bytes[i * 4 + j]) << (j * 8);
        }

        size_t offset = offset_raw % current_block_size;
        size_t index = block_start + offset;

        noise.support_.push_back(index);

        // Generate random value for this position
        size_t value_offset = bytes_for_indices + i * sizeof(uint64_t);

        if constexpr (std::is_same_v<R, algebra::GF2>) {
            // GF(2): always 1 (non-zero)
            noise.values_.push_back(algebra::GF2::one());
        } else {
            // Z2k: random non-zero value
            uint64_t value = 0;
            for (size_t j = 0; j < sizeof(uint64_t) && (value_offset + j) < random_bytes.size(); ++j) {
                value |= static_cast<uint64_t>(random_bytes[value_offset + j]) << (j * 8);
            }

            // Ensure non-zero (if zero, set to 1)
            if (value == 0) value = 1;

            noise.values_.push_back(R::from_uint64(value));
        }
    }

    return noise;
}

// ============================================================================
// LPNCore Implementation (Section 3.2, Layer 2)
// ============================================================================

template<Ring R>
LPNCore<R>::LPNCore(const PCFParameters& params)
    : N_(params.N), n_(params.n), params_(params), use_packed_(false) {
    // Generate public matrix A ∈ R^{N×n} deterministically
    // Use fixed seed derived from parameters
    DerivationContext ctx{
        SessionId{0},  // Global matrix, independent of session
        EpochId{0},
        CorrelationIndex{0},
        0,
        TermId::Setup,
        0
    };

    std::array<uint8_t, 16> matrix_seed;
    derive_key(ctx, std::span(matrix_seed));

    // Generate N×n matrix entries
    size_t total_entries = N_ * n_;
    size_t bytes_needed = total_entries * sizeof(uint64_t);

    std::vector<uint8_t> random_bytes(bytes_needed);
    prg_expand(matrix_seed, std::span(random_bytes));

    matrix_A_.reserve(total_entries);

    for (size_t i = 0; i < total_entries; ++i) {
        size_t offset = i * sizeof(uint64_t);

        if constexpr (std::is_same_v<R, algebra::GF2>) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            bool bit = (random_bytes[byte_idx] >> bit_idx) & 1;
            matrix_A_.push_back(algebra::GF2(bit));
        } else {
            uint64_t value = 0;
            for (size_t j = 0; j < sizeof(uint64_t) && (offset + j) < random_bytes.size(); ++j) {
                value |= static_cast<uint64_t>(random_bytes[offset + j]) << (j * 8);
            }
            matrix_A_.push_back(R::from_uint64(value));
        }
    }

    // For GF2, pre-pack the matrix for AVX2 optimization
    #ifdef __AVX2__
    if constexpr (std::is_same_v<R, algebra::GF2>) {
        use_packed_ = true;
        pack_gf2_vector(matrix_A_, matrix_A_packed_);
    }
    #endif
}

template<Ring R>
std::vector<R> LPNCore<R>::expand(
    const std::vector<R>& short_secret,
    const RegularNoise<R>& noise
) const {
    return expand_range(short_secret, noise, 0, N_);
}

template<Ring R>
std::vector<R> LPNCore<R>::expand_range(
    const std::vector<R>& short_secret,
    const RegularNoise<R>& noise,
    size_t start,
    size_t end
) const {
    if (short_secret.size() != n_) {
        throw std::invalid_argument("Short secret size mismatch");
    }

    if (noise.length() != N_) {
        throw std::invalid_argument("Noise length mismatch");
    }

    if (end > N_) {
        throw std::invalid_argument("Range exceeds matrix size");
    }

    // Compute y = A[start:end]·s
    std::vector<R> y = dense_mul_range(short_secret, start, end);

    // Add noise for indices in range: y = A·s + e
    auto support = noise.support();
    auto values = noise.values();

    for (size_t i = 0; i < support.size(); ++i) {
        size_t idx = support[i];
        // Only add noise if index falls within our range
        if (idx >= start && idx < end) {
            y[idx - start] = y[idx - start] + values[i];
        }
    }

    return y;
}

template<Ring R>
std::vector<R> LPNCore<R>::dense_mul(const std::vector<R>& s) const {
    return dense_mul_range(s, 0, N_);
}

template<Ring R>
std::vector<R> LPNCore<R>::dense_mul_range(const std::vector<R>& s, size_t start, size_t end) const {
    if (s.size() != n_) {
        throw std::invalid_argument("Vector size mismatch");
    }
    if (end > N_) {
        throw std::invalid_argument("Range exceeds matrix size");
    }

    // y = A[start:end]·s where A is N×n, s is n×1, result is (end-start)×1
    size_t range_size = end - start;
    std::vector<R> y(range_size, R::zero());

    for (size_t i = 0; i < range_size; ++i) {
        size_t row = start + i;
        R sum = R::zero();
        for (size_t col = 0; col < n_; ++col) {
            size_t matrix_idx = row * n_ + col;
            sum = sum + (matrix_A_[matrix_idx] * s[col]);
        }
        y[i] = sum;
    }

    return y;
}

// ============================================================================
// TiledEvaluator Implementation (Section 3.3)
// ============================================================================

template<Ring R>
FourTermResult<R> TiledEvaluator<R>::evaluate_tile(
    size_t tile_id,
    const LPNCore<R>& lpn_core,
    const CryptoHandle& handle,
    const RuntimeContext& ctx,
    PartyId party
) const {
    // Tile evaluation computes correlation for indices [tile_id * B, (tile_id+1) * B)
    // where B = tile_size_

    size_t tile_start = tile_id * tile_size_;

    // Check if tile is beyond the batch size
    if (tile_start >= params_.N) {
        // Return empty result for out-of-bounds tiles
        return FourTermResult<R>(0);
    }

    size_t tile_end = std::min(tile_start + tile_size_, params_.N);
    size_t actual_tile_size = tile_end - tile_start;

    // Derive seeds for this tile
    DerivationContext derive_ctx = ctx.derivation_ctx;
    derive_ctx.tile_id = tile_id;

    // Generate short secret s ∈ R^n
    derive_ctx.term_id = TermId::ShortSecret;
    std::array<uint8_t, 16> secret_seed;
    derive_key(derive_ctx, std::span(secret_seed));

    std::vector<R> short_secret(params_.n);
    std::vector<uint8_t> secret_bytes(params_.n * sizeof(uint64_t));
    prg_expand(secret_seed, std::span(secret_bytes));

    for (size_t i = 0; i < params_.n; ++i) {
        if constexpr (std::is_same_v<R, algebra::GF2>) {
            bool bit = (secret_bytes[i / 8] >> (i % 8)) & 1;
            short_secret[i] = algebra::GF2(bit);
        } else {
            uint64_t value = 0;
            size_t offset = i * sizeof(uint64_t);
            for (size_t j = 0; j < sizeof(uint64_t) && (offset + j) < secret_bytes.size(); ++j) {
                value |= static_cast<uint64_t>(secret_bytes[offset + j]) << (j * 8);
            }
            short_secret[i] = R::from_uint64(value);
        }
    }

    // Generate noise e ∈ R^N (regular sparse)
    // NOTE: Noise must have length = params_.N (full batch size), not tile size
    // The noise is sparse, so only w non-zero entries are stored
    derive_ctx.term_id = TermId::Noise;
    std::vector<uint8_t> noise_seed(32);
    derive_key(derive_ctx, std::span(noise_seed));

    RegularNoise<R> noise = RegularNoise<R>::sample(
        params_.N,
        params_.w,
        derive_ctx,
        noise_seed
    );

    // Expand: y = A·s + e (using passed lpn_core, compute only the tile range!)
    std::vector<R> expanded = lpn_core.expand_range(short_secret, noise, tile_start, tile_end);

    // For four-term decomposition, we need both parties' outputs
    // This is a simplified version - full implementation needs backend communication
    FourTermResult<R> result(actual_tile_size);

    // Placeholder: copy expanded output to term1
    for (size_t i = 0; i < actual_tile_size && i < expanded.size(); ++i) {
        result.term1_dense_dense[i] = expanded[i];
    }

    return result;
}

// ============================================================================
// PCFHandle Implementation (Section 2.1)
// ============================================================================

template<Ring R>
std::pair<PCFHandle<R>, RuntimeContext> PCFHandle<R>::setup(
    const PCFParameters& params,
    SessionId sid,
    const std::vector<uint8_t>& root_seed
) {
    PCFHandle<R> handle;

    // Initialize LPN core
    handle.lpn_core_ = std::make_unique<LPNCore<R>>(params);

    // Initialize tiled evaluator
    handle.tiled_eval_ = std::make_unique<TiledEvaluator<R>>(params);

    // Initialize crypto handle
    handle.crypto_handle_.params = params;
    handle.crypto_handle_.root_seed = root_seed;
    handle.crypto_handle_.sid = sid;

    // Create initial runtime context
    RuntimeContext runtime_ctx;
    runtime_ctx.sid = sid;
    runtime_ctx.epoch = EpochId{0};
    runtime_ctx.derivation_ctx = DerivationContext{
        sid, EpochId{0}, CorrelationIndex{0}, 0, TermId::Setup, 0
    };

    return {std::move(handle), runtime_ctx};
}

template<Ring R>
OLECorrelation<R> PCFHandle<R>::eval(
    CorrelationIndex ell,
    const RuntimeContext& ctx,
    PartyId party
) const {
    // Map index to tile
    auto mapping = tiled_eval_->map_index(ell);

    // Evaluate tile (pass lpn_core to avoid recreating it)
    FourTermResult<R> tile_result = tiled_eval_->evaluate_tile(
        mapping.tile_id,
        *lpn_core_,
        crypto_handle_,
        ctx,
        party
    );

    // Extract correlation at offset within tile
    OLECorrelation<R> correlation;

    if (mapping.offset < tile_result.term1_dense_dense.size()) {
        correlation.value = tile_result.term1_dense_dense[mapping.offset];
    } else {
        correlation.value = R::zero();
    }

    correlation.index = ell;
    return correlation;
}

template<Ring R>
std::vector<OLECorrelation<R>> PCFHandle<R>::eval_batch(
    std::span<const CorrelationIndex> indices,
    const RuntimeContext& ctx,
    PartyId party
) const {
    std::vector<OLECorrelation<R>> correlations;
    correlations.reserve(indices.size());

    for (auto idx : indices) {
        correlations.push_back(eval(idx, ctx, party));
    }

    return correlations;
}

// ============================================================================
// DerivationEngine Implementation
// ============================================================================

std::vector<uint8_t> DerivationEngine::derive_term_seed(
    const std::vector<uint8_t>& root_key,
    const DerivationContext& ctx
) {
    auto ctx_bytes = ctx.serialize();

    // Use SHA-256 to derive term-specific seed
    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(mdctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(mdctx, root_key.data(), root_key.size());
    EVP_DigestUpdate(mdctx, ctx_bytes.data(), ctx_bytes.size());

    std::vector<uint8_t> seed(32);
    unsigned int hash_len;
    EVP_DigestFinal_ex(mdctx, seed.data(), &hash_len);
    EVP_MD_CTX_free(mdctx);

    return seed;
}

template<Ring R>
RegularNoise<R> DerivationEngine::expand_to_noise(
    const std::vector<uint8_t>& seed,
    size_t N, size_t w
) {
    DerivationContext ctx{
        SessionId{0}, EpochId{0}, CorrelationIndex{0},
        0, TermId::Noise, 0
    };

    return RegularNoise<R>::sample(N, w, ctx, seed);
}

template<Ring R>
std::vector<R> DerivationEngine::expand_to_short_secret(
    const std::vector<uint8_t>& seed,
    size_t n
) {
    std::array<uint8_t, 16> prg_seed;
    std::memcpy(prg_seed.data(), seed.data(), std::min(seed.size(), size_t(16)));

    std::vector<uint8_t> random_bytes(n * sizeof(uint64_t));
    prg_expand(prg_seed, std::span(random_bytes));

    std::vector<R> secret(n);

    for (size_t i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<R, algebra::GF2>) {
            bool bit = (random_bytes[i / 8] >> (i % 8)) & 1;
            secret[i] = algebra::GF2(bit);
        } else {
            uint64_t value = 0;
            size_t offset = i * sizeof(uint64_t);
            for (size_t j = 0; j < sizeof(uint64_t) && (offset + j) < random_bytes.size(); ++j) {
                value |= static_cast<uint64_t>(random_bytes[offset + j]) << (j * 8);
            }
            secret[i] = R::from_uint64(value);
        }
    }

    return secret;
}

// ============================================================================
// FourTermComputer Static Method Implementations
// ============================================================================

template<Ring R>
std::vector<R> FourTermComputer<R>::compute_term1_dense_dense(
    const std::vector<R>& As0,
    const std::vector<R>& As1
) {
    if (As0.size() != As1.size()) {
        throw std::invalid_argument("Dense vector size mismatch");
    }

    std::vector<R> result;
    result.reserve(As0.size());

    for (size_t i = 0; i < As0.size(); ++i) {
        result.push_back(As0[i] * As1[i]);
    }

    return result;
}

template<Ring R>
std::vector<R> FourTermComputer<R>::compute_term2_sparse_dense(
    const RegularNoise<R>& e0,
    const std::vector<R>& As1
) {
    size_t N = As1.size();
    std::vector<R> result(N, R::zero());

    auto support = e0.support();
    auto values = e0.values();

    for (size_t i = 0; i < support.size(); ++i) {
        size_t idx = support[i];
        if (idx < N) {
            result[idx] = values[i] * As1[idx];
        }
    }

    return result;
}

template<Ring R>
std::vector<R> FourTermComputer<R>::compute_term3_sparse_dense(
    const RegularNoise<R>& e1,
    const std::vector<R>& As0
) {
    size_t N = As0.size();
    std::vector<R> result(N, R::zero());

    auto support = e1.support();
    auto values = e1.values();

    for (size_t i = 0; i < support.size(); ++i) {
        size_t idx = support[i];
        if (idx < N) {
            result[idx] = values[i] * As0[idx];
        }
    }

    return result;
}

template<Ring R>
std::vector<R> FourTermComputer<R>::compute_term4_sparse_sparse(
    const RegularNoise<R>& e0,
    const RegularNoise<R>& e1
) {
    // Build collision map
    std::unordered_map<size_t, R> e1_map;
    auto support1 = e1.support();
    auto values1 = e1.values();

    for (size_t i = 0; i < support1.size(); ++i) {
        e1_map[support1[i]] = values1[i];
    }

    // Check for collisions in e0's support
    std::vector<R> result;
    auto support0 = e0.support();
    auto values0 = e0.values();

    for (size_t i = 0; i < support0.size(); ++i) {
        size_t idx = support0[i];
        auto it = e1_map.find(idx);

        if (it != e1_map.end()) {
            // Collision at index idx
            result.push_back(values0[i] * it->second);
        } else {
            result.push_back(R::zero());
        }
    }

    return result;
}

template<Ring R>
OLECorrelation<R> FourTermComputer<R>::combine_terms(
    const FourTermResult<R>& terms
) {
    OLECorrelation<R> correlation;

    // Sum all four terms
    size_t N = terms.term1_dense_dense.size();
    R sum = R::zero();

    for (size_t i = 0; i < N; ++i) {
        sum = sum + terms.term1_dense_dense[i];
    }

    for (const auto& val : terms.term2_sparse_dense) {
        sum = sum + val;
    }

    for (const auto& val : terms.term3_sparse_dense) {
        sum = sum + val;
    }

    for (const auto& val : terms.term4_sparse_sparse) {
        sum = sum + val;
    }

    correlation.value = sum;
    return correlation;
}

// ============================================================================
// GF2 Optimized Implementation using AVX2
// ============================================================================

// Template specialization for GF2 using AVX2
template<>
std::vector<algebra::GF2> LPNCore<algebra::GF2>::dense_mul_range(
    const std::vector<algebra::GF2>& s,
    size_t start,
    size_t end
) const {
    if (s.size() != n_) {
        throw std::invalid_argument("Vector size mismatch");
    }
    if (end > N_) {
        throw std::invalid_argument("Range exceeds matrix size");
    }

    size_t range_size = end - start;

#ifdef __AVX2__
    if (use_packed_ && !matrix_A_packed_.empty()) {
        // Use pre-packed matrix (fast path!)
        // Only need to pack input vector
        std::vector<uint64_t> s_packed;
        pack_gf2_vector(s, s_packed);

        // Extract range from pre-packed matrix
        size_t n_words = (n_ + 63) / 64;
        size_t start_word = (start * n_) / 64;
        size_t range_bits = range_size * n_;
        size_t range_words = (range_bits + 63) / 64;

        // For simplicity, extract the range (could be optimized further)
        std::vector<uint64_t> A_range_packed(range_words, 0);

        // Copy packed words for the range
        // This is still faster than unpacking and repacking
        for (size_t row = 0; row < range_size; ++row) {
            size_t global_row = start + row;
            for (size_t col_word = 0; col_word < n_words; ++col_word) {
                size_t src_idx = global_row * n_words + col_word;
                size_t dst_idx = row * n_words + col_word;
                if (src_idx < matrix_A_packed_.size() && dst_idx < A_range_packed.size()) {
                    A_range_packed[dst_idx] = matrix_A_packed_[src_idx];
                }
            }
        }

        // Allocate output
        std::vector<uint64_t> y_packed((range_size + 63) / 64, 0);

        // Call AVX2 optimized matrix-vector multiplication
        algebra::GF2_AVX2::matrix_vector_mul_rowwise(
            A_range_packed.data(),
            s_packed.data(),
            y_packed.data(),
            range_size,
            n_words
        );

        // Unpack result
        std::vector<algebra::GF2> y;
        unpack_gf2_vector(y_packed, y, range_size);
        return y;
    }
#else
    // Fallback to scalar implementation
    std::vector<algebra::GF2> y(range_size, algebra::GF2::zero());
    for (size_t i = 0; i < range_size; ++i) {
        size_t row = start + i;
        algebra::GF2 sum = algebra::GF2::zero();
        for (size_t col = 0; col < n_; ++col) {
            size_t matrix_idx = row * n_ + col;
            sum = sum + (matrix_A_[matrix_idx] * s[col]);
        }
        y[i] = sum;
    }
    return y;
#endif
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template class RegularNoise<algebra::GF2>;
template class RegularNoise<algebra::Z2k<32>>;
template class RegularNoise<algebra::Z2k<64>>;

template class LPNCore<algebra::GF2>;
template class LPNCore<algebra::Z2k<32>>;
template class LPNCore<algebra::Z2k<64>>;

template class TiledEvaluator<algebra::GF2>;
template class TiledEvaluator<algebra::Z2k<32>>;
template class TiledEvaluator<algebra::Z2k<64>>;

template class PCFHandle<algebra::GF2>;
template class PCFHandle<algebra::Z2k<32>>;
template class PCFHandle<algebra::Z2k<64>>;

template class FourTermComputer<algebra::GF2>;
template class FourTermComputer<algebra::Z2k<32>>;
template class FourTermComputer<algebra::Z2k<64>>;

// DerivationEngine template methods
template RegularNoise<algebra::GF2> DerivationEngine::expand_to_noise(const std::vector<uint8_t>&, size_t, size_t);
template RegularNoise<algebra::Z2k<32>> DerivationEngine::expand_to_noise(const std::vector<uint8_t>&, size_t, size_t);
template RegularNoise<algebra::Z2k<64>> DerivationEngine::expand_to_noise(const std::vector<uint8_t>&, size_t, size_t);

template std::vector<algebra::GF2> DerivationEngine::expand_to_short_secret(const std::vector<uint8_t>&, size_t);
template std::vector<algebra::Z2k<32>> DerivationEngine::expand_to_short_secret(const std::vector<uint8_t>&, size_t);
template std::vector<algebra::Z2k<64>> DerivationEngine::expand_to_short_secret(const std::vector<uint8_t>&, size_t);

} // namespace silentmpc::pcf
