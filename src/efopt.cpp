#include "efopt/efopt.hpp"
#include <iostream>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace efopt {

using json = nlohmann::json;

Parameters Parameters::from_json(const json& j) {
    Parameters params;

    try {
        if (j.contains("verbose")) params.verbose = j["verbose"];
        if (j.contains("print_iteration")) params.print_iteration = j["print_iteration"];
        if (j.contains("penalty_init")) params.penalty_init = j["penalty_init"];
        if (j.contains("penalty_scale")) params.penalty_scale = j["penalty_scale"];
        if (j.contains("max_penalty_iterations")) params.max_penalty_iterations = j["max_penalty_iterations"];
        if (j.contains("max_convexify_iterations")) params.max_convexify_iterations = j["max_convexify_iterations"];
        if (j.contains("extra_iterations")) params.extra_iterations = j["extra_iterations"];
        if (j.contains("trust_region_init")) params.trust_region_init = j["trust_region_init"];
        if (j.contains("trust_region_expand")) params.trust_region_expand = j["trust_region_expand"];
        if (j.contains("trust_region_shrink")) params.trust_region_shrink = j["trust_region_shrink"];
        if (j.contains("model_quality_low")) params.model_quality_low = j["model_quality_low"];
        if (j.contains("model_quality_high")) params.model_quality_high = j["model_quality_high"];
        if (j.contains("xtol_coarse")) params.xtol_coarse = j["xtol_coarse"];
        if (j.contains("ftol_coarse")) params.ftol_coarse = j["ftol_coarse"];
        if (j.contains("ctol_coarse")) params.ctol_coarse = j["ctol_coarse"];
        if (j.contains("xtol_fine")) params.xtol_fine = j["xtol_fine"];
        if (j.contains("ftol_fine")) params.ftol_fine = j["ftol_fine"];
        if (j.contains("ctol_fine")) params.ctol_fine = j["ctol_fine"];
        if (j.contains("trs_params")) {
            const auto& trs_j = j["trs_params"];
            if (trs_j.contains("max_iterations")) params.trs_params.max_iterations = trs_j["max_iterations"];
            if (trs_j.contains("tolerance")) params.trs_params.tolerance = trs_j["tolerance"];
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to parse parameters from JSON: " + std::string(e.what()));
    }

    if (!params.validate()) {
        throw std::runtime_error("Invalid parameters in JSON");
    }

    return params;
}

Parameters Parameters::from_json_file(const std::string& file_path) {
    std::ifstream f(file_path);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open JSON file: " + file_path);
    }
    json j;
    f >> j;
    return from_json(j);
}

json Parameters::to_json() const {
    json j;
    j["verbose"] = verbose;
    j["print_iteration"] = print_iteration;
    j["penalty_init"] = penalty_init;
    j["penalty_scale"] = penalty_scale;
    j["max_penalty_iterations"] = max_penalty_iterations;
    j["max_convexify_iterations"] = max_convexify_iterations;
    j["extra_iterations"] = extra_iterations;
    j["trust_region_init"] = trust_region_init;
    j["trust_region_expand"] = trust_region_expand;
    j["trust_region_shrink"] = trust_region_shrink;
    j["model_quality_low"] = model_quality_low;
    j["model_quality_high"] = model_quality_high;
    j["xtol_coarse"] = xtol_coarse;
    j["ftol_coarse"] = ftol_coarse;
    j["ctol_coarse"] = ctol_coarse;
    j["xtol_fine"] = xtol_fine;
    j["ftol_fine"] = ftol_fine;
    j["ctol_fine"] = ctol_fine;
    j["trs_params"]["max_iterations"] = trs_params.max_iterations;
    j["trs_params"]["tolerance"] = trs_params.tolerance;
    return j;
}

bool Parameters::validate() const {
    if (penalty_init < 0) return false;
    if (penalty_scale < 1.0) return false;
    if (trust_region_init < 0) return false;
    if (trust_region_expand < 1.0) return false;
    if (trust_region_shrink <= 0 || trust_region_shrink >= 1.0) return false;
    if (ftol_coarse < 0 || ftol_fine < 0) return false;
    if (xtol_coarse < 0 || xtol_fine < 0) return false;
    if (ctol_coarse < 0 || ctol_fine < 0) return false;
    if (model_quality_low <= 0 || model_quality_high <= model_quality_low) return false;
    return true;
}

json Result::to_json() const {
    json j;
    j["status"] = status;
    j["objective"] = objective;
    j["constraint_violation"] = constraint_violation;
    j["total_iterations"] = total_iterations;
    j["penalty_iterations"] = penalty_iterations;
    j["x"] = json::array();
    for (int i = 0; i < x.size(); ++i) {
        j["x"].push_back(x(i));
    }
    return j;
}

/**
 * Solve Trust Region Subproblem using Steihaug-CG method
 * min g^T s + 0.5 s^T B s, s.t. ||s|| <= radius
 */
static void solve_trs(Eigen::VectorXd& s,
                      const Eigen::VectorXd& g,
                      const Eigen::MatrixXd& B,
                      double radius,
                      TrsParameters& params,
                      int& iterations) {
    int n = s.size();
    iterations = 0;
    s.setZero();

    if (g.norm() <= params.tolerance) {
        return;
    }

    Eigen::VectorXd r = g;
    Eigen::VectorXd p = -g;
    double r_norm_sq = r.squaredNorm();

    while (iterations < params.max_iterations) {
        iterations++;

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
static void convexify_problem(Eigen::MatrixXd& B, double min_eigenvalue = 1e-6) {
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
Result optimize(Eigen::VectorXd x0,
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
    if (!params.validate()) {
        result.status = EFOPT_ERROR_INVALID_PENALTY_INIT;
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
        int trs_iters;

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
            solve_trs(d, grad_F, B_convex, trust_radius, params.trs_params, trs_iters);

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

            // Print iteration progress if enabled
            if (params.print_iteration) {
                json iter_json;
                iter_json["type"] = "convexify_iteration";
                iter_json["penalty_iteration"] = penalty_iter;
                iter_json["convexify_iteration"] = convex_iter;
                iter_json["trs_iterations"] = trs_iters;
                iter_json["objective"] = f;
                iter_json["constraint_violation"] = h;
                iter_json["merit_function"] = F;
                iter_json["step_norm"] = step_norm;
                iter_json["trust_radius"] = trust_radius;
                iter_json["approx_quality"] = approx_quality;
                iter_json["true_improve"] = true_improve;
                iter_json["gradient_norm"] = grad_F.norm();
                std::cout << iter_json.dump() << std::endl;
            }

            // Convergence check
            if (step_norm < xtol || std::abs(true_improve) < ftol) {
                converged = true;
                break;
            }

            if (params.verbose && (convex_iter % 10 == 0 || convex_iter == 1)) {
                std::cout << "Penalty iter: " << penalty_iter << ", Convex iter: " << convex_iter << ", TRS iters: " << trs_iters << std::endl;
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

            int trs_iters;
            // Run extra iterations with fine thresholds
            for (int extra_iter = 1; extra_iter <= params.extra_iterations; ++extra_iter) {
                total_iterations++;
                double F = f + mu * h;
                Eigen::VectorXd grad_F = grad_f + mu * grad_h;

                Eigen::MatrixXd B_convex = B;
                convexify_problem(B_convex);

                Eigen::VectorXd d(n);
                solve_trs(d, grad_F, B_convex, trust_radius, params.trs_params, trs_iters);

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

                // Print extra iteration progress if enabled
                if (params.print_iteration) {
                    json iter_json;
                    iter_json["type"] = "fine_iteration";
                    iter_json["penalty_iteration"] = penalty_iter;
                    iter_json["extra_iteration"] = extra_iter;
                    iter_json["trs_iterations"] = trs_iters;
                    iter_json["objective"] = f;
                    iter_json["constraint_violation"] = h;
                    iter_json["merit_function"] = F;
                    iter_json["step_norm"] = step_norm;
                    iter_json["trust_radius"] = trust_radius;
                    iter_json["approx_quality"] = approx_quality;
                    iter_json["true_improve"] = true_improve;
                    iter_json["gradient_norm"] = grad_F.norm();
                    std::cout << iter_json.dump() << std::endl;
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
