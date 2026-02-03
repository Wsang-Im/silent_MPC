#pragma once

#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

namespace silentmpc::pcf {

// ============================================================================
// Full ISD (Information Set Decoding) Estimator
// Based on Liu et al. [18] unified methodology for PCG/PCF security
// ============================================================================

/// ISD algorithm variants
enum class ISDAlgorithm {
    Prange,       ///< Basic ISD (1962)
    LeeBrickell,  ///< Lee-Brickell (1988)
    Leon,         ///< Leon (1988)
    Stern,        ///< Stern (1989)
    Dumer,        ///< Dumer (1991)
    MMT,          ///< May-Meurer-Thomae (2011)
    BJMM,         ///< Becker-Joux-May-Meurer (2012)
    BothMay       ///< Both-May (2018)
};

/// ISD attack parameters
struct ISDParameters {
    size_t n;           ///< Code length (primal dimension)
    size_t k;           ///< Code dimension (N - n for LPN)
    size_t w;           ///< Error weight
    size_t p;           ///< Split parameter (algorithm-specific)
    size_t l;           ///< List size parameter (for BJMM/MMT)
    double eps;         ///< Trade-off parameter
};

/// Combinatorial functions for ISD
class ISDCombinatorics {
public:
    /// Binomial coefficient: C(n, k) = n! / (k! * (n-k)!)
    /// Uses logarithm to avoid overflow
    static double log2_binomial(size_t n, size_t k) {
        if (k > n) return -std::numeric_limits<double>::infinity();
        if (k == 0 || k == n) return 0.0;

        // Optimize: use smaller k
        if (k > n - k) k = n - k;

        double result = 0.0;
        for (size_t i = 0; i < k; ++i) {
            result += std::log2(static_cast<double>(n - i));
            result -= std::log2(static_cast<double>(i + 1));
        }

        return result;
    }

    /// Binary entropy function: H(x) = -x*log2(x) - (1-x)*log2(1-x)
    static double binary_entropy(double x) {
        if (x <= 0.0 || x >= 1.0) return 0.0;
        return -x * std::log2(x) - (1.0 - x) * std::log2(1.0 - x);
    }

    /// Generalized binomial: sum of binomials up to weight w
    static double log2_binomial_sum(size_t n, size_t w) {
        double max_val = -std::numeric_limits<double>::infinity();

        for (size_t i = 0; i <= w; ++i) {
            double val = log2_binomial(n, i);
            if (val > max_val) {
                max_val = val;
            }
        }

        // Use log-sum-exp trick for numerical stability
        double sum = 0.0;
        for (size_t i = 0; i <= w; ++i) {
            sum += std::pow(2.0, log2_binomial(n, i) - max_val);
        }

        return max_val + std::log2(sum);
    }
};

/// Full ISD Estimator with all major algorithms
class ISDEstimator {
public:
    /// Prange algorithm cost (basic ISD)
    /// Time: C(k, w) * k^3 (Gaussian elimination)
    static double estimate_prange(const ISDParameters& params) {
        size_t n = params.n;
        size_t k = params.k;
        size_t w = params.w;

        // Probability of success per iteration: C(k,w) / C(n,w)
        double log2_prob = ISDCombinatorics::log2_binomial(k, w)
                         - ISDCombinatorics::log2_binomial(n, w);

        // Number of iterations: 1 / prob
        double log2_iterations = -log2_prob;

        // Cost per iteration: k^3 (Gaussian elimination)
        double log2_gauss = 3.0 * std::log2(static_cast<double>(k));

        return log2_iterations + log2_gauss;
    }

    /// Lee-Brickell algorithm cost
    /// Time: C(k, w-p) * C(p, p) * k^3 where p is split parameter
    static double estimate_lee_brickell(const ISDParameters& params) {
        size_t n = params.n;
        size_t k = params.k;
        size_t w = params.w;
        size_t p = params.p;

        if (p > w) return std::numeric_limits<double>::infinity();

        double log2_prob = ISDCombinatorics::log2_binomial(k, w - p)
                         + ISDCombinatorics::log2_binomial(p, p)
                         - ISDCombinatorics::log2_binomial(n, w);

        double log2_iterations = -log2_prob;
        double log2_gauss = 3.0 * std::log2(static_cast<double>(k));

        return log2_iterations + log2_gauss;
    }

    /// Stern algorithm cost (birthday paradox approach)
    /// Time: C(k/2, w/2)^2 / C(l, p)
    static double estimate_stern(const ISDParameters& params) {
        size_t n = params.n;
        size_t k = params.k;
        size_t w = params.w;
        size_t p = params.p;
        size_t l = params.l;

        if (w % 2 != 0) return std::numeric_limits<double>::infinity();
        if (k < l) return std::numeric_limits<double>::infinity();

        size_t w2 = w / 2;
        size_t k2 = k / 2;

        // List size: C(k/2, w/2)
        double log2_list = ISDCombinatorics::log2_binomial(k2, w2);

        // Collision parameter
        double log2_collision = ISDCombinatorics::log2_binomial(l, p);

        // Success probability
        double log2_prob = 2.0 * log2_list - log2_collision
                         - ISDCombinatorics::log2_binomial(n, w);

        double log2_iterations = -log2_prob;

        // Cost: list building + collision finding
        double log2_cost_per_iter = std::max(log2_list, log2_collision);

        return log2_iterations + log2_cost_per_iter;
    }

    /// MMT (May-Meurer-Thomae) algorithm cost
    /// Advanced ISD with representation technique
    static double estimate_mmt(const ISDParameters& params) {
        size_t n = params.n;
        size_t k = params.k;
        size_t w = params.w;
        size_t p = params.p;
        size_t l = params.l;
        double eps = params.eps;

        if (w % 4 != 0) {
            // Round to nearest multiple of 4
            ISDParameters adjusted = params;
            adjusted.w = (w / 4) * 4;
            if (adjusted.w == 0) adjusted.w = 4;
            return estimate_mmt(adjusted);
        }

        size_t w4 = w / 4;

        // Representation technique: split into 4 lists
        // L1, L2 each of size C(k/4, w/4)
        double log2_L1 = ISDCombinatorics::log2_binomial(k / 4, w4);

        // Intermediate list size (with representation parameter eps)
        double log2_L12 = 2.0 * log2_L1 - eps * static_cast<double>(l);

        // Final collision
        double log2_collision = 2.0 * log2_L12 - (1.0 - eps) * static_cast<double>(l);

        // Success probability
        double log2_prob = log2_collision - ISDCombinatorics::log2_binomial(n, w);

        double log2_iterations = -log2_prob;

        // Cost: max of list operations
        double log2_cost_per_iter = std::max({log2_L1, log2_L12, log2_collision});

        return log2_iterations + log2_cost_per_iter;
    }

    /// BJMM (Becker-Joux-May-Meurer) algorithm cost
    /// State-of-the-art ISD with improved representation
    static double estimate_bjmm(const ISDParameters& params) {
        size_t n = params.n;
        size_t k = params.k;
        size_t w = params.w;
        size_t p = params.p;
        size_t l = params.l;
        double eps = params.eps;

        if (w % 4 != 0) {
            ISDParameters adjusted = params;
            adjusted.w = (w / 4) * 4;
            if (adjusted.w == 0) adjusted.w = 4;
            return estimate_bjmm(adjusted);
        }

        size_t w4 = w / 4;

        // Four base lists
        double log2_L0 = ISDCombinatorics::log2_binomial(k / 4, w4);

        // Representation parameter optimization
        // eps1 for first merge, eps2 for second merge
        double eps1 = eps * 0.6;  // Heuristic split
        double eps2 = eps * 0.4;

        // First level merge
        double log2_L1 = 2.0 * log2_L0 - eps1 * static_cast<double>(l);

        // Second level merge
        double log2_L2 = 2.0 * log2_L1 - eps2 * static_cast<double>(l);

        // Final collision on remaining bits
        double log2_collision = log2_L2 - (1.0 - eps) * static_cast<double>(l);

        // Success probability
        double log2_prob = log2_collision - ISDCombinatorics::log2_binomial(n, w);

        double log2_iterations = -log2_prob;

        // Total cost
        double log2_cost_per_iter = std::max({log2_L0, log2_L1, log2_L2, log2_collision});

        return log2_iterations + log2_cost_per_iter;
    }

    /// Both-May algorithm (2018) - most recent
    /// Improved BJMM with better parameter optimization
    static double estimate_both_may(const ISDParameters& params) {
        size_t n = params.n;
        size_t k = params.k;
        size_t w = params.w;
        size_t l = params.l;

        // Both-May uses adaptive representation levels
        // Number of levels depends on parameters
        size_t num_levels = static_cast<size_t>(std::log2(static_cast<double>(w))) + 1;

        std::vector<double> list_sizes(num_levels);

        // Base list size
        size_t base_w = w / (1ULL << (num_levels - 1));
        list_sizes[0] = ISDCombinatorics::log2_binomial(k / (1ULL << (num_levels - 1)), base_w);

        // Optimal representation distribution
        double total_bits = static_cast<double>(l);
        double bits_per_level = total_bits / static_cast<double>(num_levels - 1);

        // Build up levels
        for (size_t i = 1; i < num_levels; ++i) {
            list_sizes[i] = 2.0 * list_sizes[i - 1] - bits_per_level;
        }

        // Success probability
        double log2_prob = list_sizes[num_levels - 1]
                         - ISDCombinatorics::log2_binomial(n, w);

        double log2_iterations = -log2_prob;

        // Cost: max over all levels
        double max_cost = *std::max_element(list_sizes.begin(), list_sizes.end());

        return log2_iterations + max_cost;
    }

    /// Find optimal parameters for given algorithm
    static ISDParameters optimize_parameters(
        ISDAlgorithm algo,
        size_t n, size_t k, size_t w
    ) {
        ISDParameters best_params{n, k, w, 0, 0, 0.0};
        double best_cost = std::numeric_limits<double>::infinity();

        // Search space for parameters
        size_t max_p = std::min(w, size_t(10));
        size_t max_l = std::min(k, size_t(50));

        for (size_t p = 0; p <= max_p; ++p) {
            for (size_t l = 10; l <= max_l; l += 5) {
                for (double eps = 0.0; eps <= 1.0; eps += 0.1) {
                    ISDParameters params{n, k, w, p, l, eps};

                    double cost = 0.0;
                    switch (algo) {
                        case ISDAlgorithm::Prange:
                            cost = estimate_prange(params);
                            break;
                        case ISDAlgorithm::LeeBrickell:
                            cost = estimate_lee_brickell(params);
                            break;
                        case ISDAlgorithm::Stern:
                            cost = estimate_stern(params);
                            break;
                        case ISDAlgorithm::MMT:
                            cost = estimate_mmt(params);
                            break;
                        case ISDAlgorithm::BJMM:
                            cost = estimate_bjmm(params);
                            break;
                        case ISDAlgorithm::BothMay:
                            cost = estimate_both_may(params);
                            break;
                    }

                    if (cost < best_cost) {
                        best_cost = cost;
                        best_params = params;
                    }
                }
            }
        }

        return best_params;
    }

    /// Estimate best attack cost across all algorithms
    static double estimate_best_attack(size_t n, size_t k, size_t w) {
        double min_cost = std::numeric_limits<double>::infinity();

        // Try all major algorithms
        std::vector<ISDAlgorithm> algorithms = {
            ISDAlgorithm::Prange,
            ISDAlgorithm::LeeBrickell,
            ISDAlgorithm::Stern,
            ISDAlgorithm::MMT,
            ISDAlgorithm::BJMM,
            ISDAlgorithm::BothMay
        };

        for (auto algo : algorithms) {
            auto params = optimize_parameters(algo, n, k, w);
            double cost = 0.0;

            switch (algo) {
                case ISDAlgorithm::Prange:
                    cost = estimate_prange(params);
                    break;
                case ISDAlgorithm::LeeBrickell:
                    cost = estimate_lee_brickell(params);
                    break;
                case ISDAlgorithm::Stern:
                    cost = estimate_stern(params);
                    break;
                case ISDAlgorithm::MMT:
                    cost = estimate_mmt(params);
                    break;
                case ISDAlgorithm::BJMM:
                    cost = estimate_bjmm(params);
                    break;
                case ISDAlgorithm::BothMay:
                    cost = estimate_both_may(params);
                    break;
            }

            min_cost = std::min(min_cost, cost);
        }

        return min_cost;
    }
};

} // namespace silentmpc::pcf
