#ifndef EFOPT_HPP
#define EFOPT_HPP

#include <Eigen/Eigen>
#include <iostream>
#include <cmath>
#include <limits>

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
    EFOPT_ERROR_NUMERICAL = -30
};

namespace efopt {

/**
 * Trust Region Subproblem (TRS) parameters
 */
struct TrsParameters {
    int max_iterations = 128;
    double tolerance = 1.0e-8;
    int iterations = 0;
};

/**
 * EFOPT algorithm parameters
 */
struct Parameters {
    bool verbose = false;

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

    // Runtime state
    int penalty_iteration = 0;
    int convexify_iteration = 0;
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
 * Solve Trust Region Subproblem using Steihaug-CG method
 * min g^T s + 0.5 s^T B s, s.t. ||s|| <= radius
 */
inline void solve_trs(Eigen::VectorXd& s,
                      const Eigen::VectorXd& g,
                      const Eigen::MatrixXd& B,
                      double radius,
                      TrsParameters& params) {
    int n = s.size();
    params.iterations = 0;
    s.setZero();

    if (g.norm() <= params.tolerance) {
        return;
    }

    Eigen::VectorXd r = g;
    Eigen::VectorXd p = -g;
    double r_norm_sq = r.squaredNorm();

    while (params.iterations < params.max_iterations) {
        params.iterations++;

        Eigen::VectorXd Bp = B * p;
        double pBp = p.dot(Bp);

        // Negative curvature detected, go to trust region boundary
        if (pBp <= 0) {
            double a = p.squaredNorm();
            double b = 2.0 * s.dot(p);
            double c = s.squaredNorm() - radius * radius;
            double discriminant = b * b - 4 * a * c;
            double t = (-b + std::sqrt(discriminant)) / (2 * a);
            if (t < 0) t = (-b - std::sqrt(discriminant)) / (2 * a);
            s += t * p;
            break;
        }

        double alpha = r_norm_sq / pBp;
        Eigen::VectorXd s_new = s + alpha * p;

        // Step exceeds trust region, go to boundary
        if (s_new.norm() >= radius) {
            double a = p.squaredNorm();
            double b = 2.0 * s.dot(p);
            double c = s.squaredNorm() - radius * radius;
            double discriminant = b * b - 4 * a * c;
            double t = (-b + std::sqrt(discriminant)) / (2 * a);
            if (t < 0) t = (-b - std::sqrt(discriminant)) / (2 * a);
            s += t * p;
            break;
        }

        Eigen::VectorXd r_new = r + alpha * Bp;
        double r_new_norm_sq = r_new.squaredNorm();

        // Converged
        if (std::sqrt(r_new_norm_sq) <= params.tolerance * g.norm()) {
            s = s_new;
            break;
        }

        double beta = r_new_norm_sq / r_norm_sq;
        p = -r_new + beta * p;
        s = s_new;
        r = r_new;
        r_norm_sq = r_new_norm_sq;
    }
}

/**
 * Convexify the problem by modifying Hessian to be positive definite
 */
inline void convexify_problem(Eigen::MatrixXd& B, double min_eigenvalue = 1e-6) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B);
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXd eigenvectors = es.eigenvectors();

    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < min_eigenvalue) {
            eigenvalues(i) = min_eigenvalue;
        }
    }

    B = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
}

/**
 * Main EFOPT optimization function
 */
inline Result optimize(Eigen::VectorXd x0,
                       EvaluateFunction evaluate,
                       void* instance,
                       Parameters& params) {
    Result result;
    result.status = EFOPT_UNFINISHED;
    result.x = x0;
    int n = x0.size();

    // Validate input
    if (n <= 0) {
        result.status = EFOPT_ERROR_INVALID_X_NUM;
        return result;
    }
    if (Eigen::isnan(x0.array()).any()) {
        result.status = EFOPT_ERROR_INVALID_X_NAN;
        return result;
    }
    if (params.penalty_init < 0) {
        result.status = EFOPT_ERROR_INVALID_PENALTY_INIT;
        return result;
    }
    if (params.penalty_scale < 1.0) {
        result.status = EFOPT_ERROR_INVALID_PENALTY_SCALE;
        return result;
    }
    if (params.trust_region_init < 0) {
        result.status = EFOPT_ERROR_INVALID_TR_REGION_INIT;
        return result;
    }
    if (params.trust_region_expand < 1.0) {
        result.status = EFOPT_ERROR_INVALID_TR_EXPAND;
        return result;
    }
    if (params.trust_region_shrink <= 0 || params.trust_region_shrink >= 1.0) {
        result.status = EFOPT_ERROR_INVALID_TR_SHRINK;
        return result;
    }
    if (params.ftol_coarse < 0 || params.ftol_fine < 0) {
        result.status = EFOPT_ERROR_INVALID_FTOL;
        return result;
    }
    if (params.xtol_coarse < 0 || params.xtol_fine < 0) {
        result.status = EFOPT_ERROR_INVALID_XTOL;
        return result;
    }
    if (params.ctol_coarse < 0 || params.ctol_fine < 0) {
        result.status = EFOPT_ERROR_INVALID_CTOL;
        return result;
    }

    // Initialize variables
    Eigen::VectorXd x = x0;
    Eigen::MatrixXd B(n, n);
    B.setIdentity();

    double mu = params.penalty_init;
    double trust_radius = params.trust_region_init;
    double xtol = params.xtol_coarse;
    double ftol = params.ftol_coarse;
    double ctol = params.ctol_coarse;

    double f, h;
    Eigen::VectorXd grad_f(n), grad_h(n);
    evaluate(instance, x, f, h, grad_f, grad_h);

    bool use_fine_thresholds = false;
    int total_iterations = 0;

    if (params.verbose) {
        std::cout << "EFOPT starting..." << std::endl;
        std::cout << "Initial x: " << x.transpose() << std::endl;
        std::cout << "Initial objective: " << f << ", constraint violation: " << h << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    // Penalty outer loop
    for (int penalty_iter = 1; penalty_iter <= params.max_penalty_iterations; ++penalty_iter) {
        params.penalty_iteration = penalty_iter;
        B.setIdentity();
        trust_radius = params.trust_region_init;

        bool converged = false;

        // Convexification inner loop
        for (int convex_iter = 1; convex_iter <= params.max_convexify_iterations; ++convex_iter) {
            params.convexify_iteration = convex_iter;
            total_iterations++;

            // Evaluate merit function
            double F = f + mu * h;
            Eigen::VectorXd grad_F = grad_f + mu * grad_h;

            // Convexify the problem (ensure Hessian is positive definite)
            Eigen::MatrixXd B_convex = B;
            convexify_problem(B_convex);

            // Solve TRS
            Eigen::VectorXd d(n);
            solve_trs(d, grad_F, B_convex, trust_radius, params.trs_params);

            double step_norm = d.norm();

            // Calculate improvements
            double F_new;
            double f_new, h_new;
            Eigen::VectorXd grad_f_new(n), grad_h_new(n);
            Eigen::VectorXd x_new = x + d;

            if (Eigen::isnan(x_new.array()).any()) {
                result.status = EFOPT_ERROR_NUMERICAL;
                return result;
            }

            evaluate(instance, x_new, f_new, h_new, grad_f_new, grad_h_new);
            F_new = f_new + mu * h_new;

            double true_improve = F_new - F;
            double approx_improve = grad_F.dot(d) + 0.5 * d.dot(B_convex * d);
            double approx_quality = approx_improve != 0 ? true_improve / approx_improve : 1.0;

            // Update x if improvement
            if (true_improve < 0) {
                // BFGS Hessian update
                Eigen::VectorXd s = d;
                Eigen::VectorXd y = (grad_f_new + mu * grad_h_new) - grad_F;
                double sy = s.dot(y);

                if (sy > 1e-10 * s.squaredNorm()) {
                    Eigen::VectorXd Bs = B * s;
                    double sBs = s.dot(Bs);
                    B = B + (y * y.transpose()) / sy - (Bs * Bs.transpose()) / sBs;
                }

                x = x_new;
                f = f_new;
                h = h_new;
                grad_f = grad_f_new;
                grad_h = grad_h_new;
            }

            // Update trust region
            if (approx_quality < params.model_quality_low) {
                trust_radius *= params.trust_region_shrink;
            } else if (approx_quality > params.model_quality_high && step_norm >= 0.9 * trust_radius) {
                trust_radius = std::min(trust_radius * params.trust_region_expand, 1e6);
            }

            // Convergence check
            if (step_norm < xtol || std::abs(true_improve) < ftol) {
                converged = true;
                break;
            }

            if (params.verbose && (convex_iter % 10 == 0 || convex_iter == 1)) {
                std::cout << "Penalty iter: " << penalty_iter << ", Convex iter: " << convex_iter << std::endl;
                std::cout << "Objective: " << f << ", Constraint: " << h << ", Merit: " << F << std::endl;
                std::cout << "Step norm: " << step_norm << ", Trust radius: " << trust_radius << std::endl;
                std::cout << "Quality: " << approx_quality << std::endl;
                std::cout << "----------------------------------------" << std::endl;
            }
        }

        // Switch to fine thresholds if constraints are satisfied
        if (h < ctol && !use_fine_thresholds) {
            use_fine_thresholds = true;
            xtol = params.xtol_fine;
            ftol = params.ftol_fine;
            ctol = params.ctol_fine;

            if (params.verbose) {
                std::cout << "Switched to fine thresholds" << std::endl;
                std::cout << "xtol: " << xtol << ", ftol: " << ftol << ", ctol: " << ctol << std::endl;
            }

            // Run extra iterations with fine thresholds
            for (int extra_iter = 1; extra_iter <= params.extra_iterations; ++extra_iter) {
                total_iterations++;
                double F = f + mu * h;
                Eigen::VectorXd grad_F = grad_f + mu * grad_h;

                Eigen::MatrixXd B_convex = B;
                convexify_problem(B_convex);

                Eigen::VectorXd d(n);
                solve_trs(d, grad_F, B_convex, trust_radius, params.trs_params);

                double step_norm = d.norm();

                double F_new;
                double f_new, h_new;
                Eigen::VectorXd grad_f_new(n), grad_h_new(n);
                Eigen::VectorXd x_new = x + d;
                evaluate(instance, x_new, f_new, h_new, grad_f_new, grad_h_new);
                F_new = f_new + mu * h_new;

                double true_improve = F_new - F;
                double approx_improve = grad_F.dot(d) + 0.5 * d.dot(B_convex * d);
                double approx_quality = approx_improve != 0 ? true_improve / approx_improve : 1.0;

                if (true_improve < 0) {
                    Eigen::VectorXd s = d;
                    Eigen::VectorXd y = (grad_f_new + mu * grad_h_new) - grad_F;
                    double sy = s.dot(y);

                    if (sy > 1e-10 * s.squaredNorm()) {
                        Eigen::VectorXd Bs = B * s;
                        double sBs = s.dot(Bs);
                        B = B + (y * y.transpose()) / sy - (Bs * Bs.transpose()) / sBs;
                    }

                    x = x_new;
                    f = f_new;
                    h = h_new;
                    grad_f = grad_f_new;
                    grad_h = grad_h_new;
                }

                if (approx_quality < params.model_quality_low) {
                    trust_radius *= params.trust_region_shrink;
                } else if (approx_quality > params.model_quality_high && step_norm >= 0.9 * trust_radius) {
                    trust_radius = std::min(trust_radius * params.trust_region_expand, 1e6);
                }

                if (step_norm < xtol || std::abs(true_improve) < ftol) {
                    break;
                }
            }
        }

        // Check final convergence
        if (h <= ctol) {
            result.status = EFOPT_SUCCESS;
            break;
        }

        // Increase penalty if constraints not satisfied
        if (h > ctol) {
            mu *= params.penalty_scale;
            B.setIdentity();

            if (params.verbose) {
                std::cout << "Increased penalty to: " << mu << std::endl;
            }
        } else {
            break;
        }

        if (penalty_iter == params.max_penalty_iterations) {
            result.status = h <= params.ctol_coarse ? EFOPT_CTOL_SATISFIED : EFOPT_ERROR_MAX_PENALTY_ITERATIONS;
        }
    }

    result.x = x;
    result.objective = f;
    result.constraint_violation = h;
    result.total_iterations = total_iterations;
    result.penalty_iterations = params.penalty_iteration;

    if (params.verbose) {
        std::cout << "EFOPT finished with status: " << result.status << std::endl;
        std::cout << "Final x: " << x.transpose() << std::endl;
        std::cout << "Final objective: " << f << ", constraint violation: " << h << std::endl;
        std::cout << "Total iterations: " << total_iterations << std::endl;
    }

    return result;
}

} // namespace efopt

#endif // EFOPT_HPP
