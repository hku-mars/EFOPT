#ifndef EFOPT_HPP
#define EFOPT_HPP

#include <Eigen/Eigen>
#include <nlohmann/json.hpp>
#include <string>

/**
 * EFOPT status codes
 * Negative values indicate errors
 */
enum efopt_status_t {
    EFOPT_UNFINISHED = 0,
    EFOPT_SUCCESS = 1,
    EFOPT_CONVERGED_X_TOL = 2,
    EFOPT_CONVERGED_F_TOL = 3,
    EFOPT_CONVERGED_C_TOL = 4,
    EFOPT_STUCK_LOCAL_MINIMUM = 5,
    EFOPT_CTOL_SATISFIED = 10,
    EFOPT_ERROR_INVALID_X_NUM = -2,
    EFOPT_ERROR_INVALID_X_NAN = -3,
    EFOPT_ERROR_INVALID_PENALTY_INIT = -4,
    EFOPT_ERROR_INVALID_PENALTY_SCALE = -5,
    EFOPT_ERROR_INVALID_TR_REGION_INIT = -6,
    EFOPT_ERROR_INVALID_TR_EXPAND = -7,
    EFOPT_ERROR_INVALID_TR_SHRINK = -8,
    EFOPT_ERROR_INVALID_CTOL = -9,
    EFOPT_ERROR_INVALID_FTOL = -10,
    EFOPT_ERROR_INVALID_XTOL = -11,
    EFOPT_ERROR_MAX_PENALTY_ITERATIONS = -20,
    EFOPT_ERROR_NUMERICAL = -30,
    EFOPT_ERROR_JSON_PARSE = -40
};

namespace efopt {

using json = nlohmann::json;

/**
 * Trust Region Subproblem (TRS) parameters
 */
struct TrsParameters {
    int max_iterations = 128;
    double tolerance = 1.0e-8;
};

inline void to_json(json& j, const TrsParameters& p) {
    j = json{{"max_iterations", p.max_iterations}, {"tolerance", p.tolerance}};
}

inline void from_json(const json& j, TrsParameters& p) {
    j.at("max_iterations").get_to(p.max_iterations);
    j.at("tolerance").get_to(p.tolerance);
}

/**
 * EFOPT algorithm parameters
 */
struct Parameters {
    bool verbose = false;
    bool print_iteration = false;  // Print iteration progress (text format)

    // Penalty parameters
    double penalty_init = 1.0e3;
    double penalty_scale = 10.0;
    int max_penalty_iterations = 8;

    // Convexification parameters
    int max_convexify_iterations = 40;
    int extra_iterations = 10;

    // Trust region parameters
    double trust_region_init = 1.0;
    double trust_region_expand = 2.0;
    double trust_region_shrink = 0.75;
    double model_quality_low = 0.25;   // λ1
    double model_quality_high = 0.75;  // λ2

    // Coarse thresholds (first stage)
    double xtol_coarse = 1e-2;
    double ftol_coarse = 1e-2;
    double ctol_coarse = 1e-2;

    // Fine thresholds (second stage, after constraints are satisfied)
    double xtol_fine = 1e-4;
    double ftol_fine = 1e-4;
    double ctol_fine = 1e-4;

    // TRS solver parameters
    TrsParameters trs_params;

    // Runtime state (not serialized)
    int penalty_iteration = 0;
    int convexify_iteration = 0;

    /**
     * Load parameters from JSON
     */
    static Parameters from_json(const json& j);

    /**
     * Load parameters from JSON file
     */
    static Parameters from_json_file(const std::string& file_path);

    /**
     * Convert parameters to JSON
     */
    json to_json() const;

    /**
     * Validate parameters
     */
    bool validate() const;
};

/**
 * Optimization results
 */
struct Result {
    efopt_status_t status;
    Eigen::VectorXd x;
    double objective;
    double constraint_violation;
    int total_iterations;
    int penalty_iterations;

    /**
     * Convert result to JSON
     */
    json to_json() const;
};

/**
 * Evaluation callback interface
 * User must implement this to provide function, constraint values and gradients
 */
typedef void (*EvaluateFunction)(void* instance,
                                 const Eigen::VectorXd& x,
                                 double& objective,
                                 double& constraint_violation,
                                 Eigen::VectorXd& grad_objective,
                                 Eigen::VectorXd& grad_constraint);

/**
 * Main EFOPT optimization function
 */
Result optimize(Eigen::VectorXd x0,
                EvaluateFunction evaluate,
                void* instance,
                Parameters& params);

} // namespace efopt

#endif // EFOPT_HPP
