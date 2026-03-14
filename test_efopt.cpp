#include "efopt/efopt.hpp"
#include <iostream>
#include <cmath>

/**
 * Test problem 1: Rosenbrock function with equality constraint
 * min f(x) = 100*(x2 - x1²)² + (1 - x1)²
 * s.t. x1 + 2x2 = 1
 */
struct RosenbrockConstraintProblem {
    void evaluate(const Eigen::VectorXd& x, double& f, double& c, Eigen::VectorXd& grad_f, Eigen::VectorXd& grad_c) {
        // Rosenbrock function
        f = 100 * std::pow(x(1) - x(0)*x(0), 2) + std::pow(1 - x(0), 2);

        // Gradient of f
        grad_f(0) = -400 * x(0) * (x(1) - x(0)*x(0)) - 2 * (1 - x(0));
        grad_f(1) = 200 * (x(1) - x(0)*x(0));

        // Equality constraint: x1 + 2x2 - 1 = 0
        double constraint = x(0) + 2 * x(1) - 1;
        c = std::pow(constraint, 2); // Squared constraint for l2 penalty

        // Gradient of constraint violation
        grad_c(0) = 2 * constraint * 1;
        grad_c(1) = 2 * constraint * 2;
    }

    static void evaluate_wrapper(void* instance, const Eigen::VectorXd& x, double& f, double& c, Eigen::VectorXd& grad_f, Eigen::VectorXd& grad_c) {
        static_cast<RosenbrockConstraintProblem*>(instance)->evaluate(x, f, c, grad_f, grad_c);
    }
};

/**
 * Test problem 2: Nonconvex problem with multiple local minima
 * min f(x) = sin(x1) * cos(x2) + 0.1*(x1² + x2²)
 * s.t. x1² + x2² <= 4
 */
struct NonconvexProblem {
    void evaluate(const Eigen::VectorXd& x, double& f, double& c, Eigen::VectorXd& grad_f, Eigen::VectorXd& grad_c) {
        // Objective function
        f = std::sin(x(0)) * std::cos(x(1)) + 0.1 * (x(0)*x(0) + x(1)*x(1));

        // Gradient of f
        grad_f(0) = std::cos(x(0)) * std::cos(x(1)) + 0.2 * x(0);
        grad_f(1) = -std::sin(x(0)) * std::sin(x(1)) + 0.2 * x(1);

        // Inequality constraint: x1² + x2² -4 <= 0
        double constraint = x(0)*x(0) + x(1)*x(1) - 4;
        c = std::pow(std::max(0.0, constraint), 2); // Only penalize violation

        // Gradient of constraint violation
        if (constraint > 0) {
            grad_c(0) = 2 * constraint * 2 * x(0);
            grad_c(1) = 2 * constraint * 2 * x(1);
        } else {
            grad_c.setZero();
        }
    }

    static void evaluate_wrapper(void* instance, const Eigen::VectorXd& x, double& f, double& c, Eigen::VectorXd& grad_f, Eigen::VectorXd& grad_c) {
        static_cast<NonconvexProblem*>(instance)->evaluate(x, f, c, grad_f, grad_c);
    }
};

int main() {
    std::cout << "===== EFOPT Nonlinear Optimization Solver Test =====" << std::endl;

    // --------------------------
    // Test 1: Load parameters from JSON
    // --------------------------
    std::cout << "\n=== Test 1: Load parameters from JSON ===" << std::endl;

    // Example JSON parameters
    efopt::json params_json = {
        {"verbose", false},
        {"penalty_init", 1000.0},
        {"penalty_scale", 10.0},
        {"max_penalty_iterations", 10},
        {"max_convexify_iterations", 50},
        {"xtol_fine", 1e-5},
        {"ftol_fine", 1e-5}
    };

    efopt::Parameters params = efopt::Parameters::from_json(params_json);
    std::cout << "Loaded parameters from JSON successfully!" << std::endl;
    std::cout << "Parameters: " << params.to_json().dump(4) << std::endl;

    // --------------------------
    // Test 2: Rosenbrock with equality constraint (with JSON iteration log)
    // --------------------------
    std::cout << "\n=== Test 2: Rosenbrock function with equality constraint x1 + 2x2 = 1 ===" << std::endl;
    RosenbrockConstraintProblem problem1;

    Eigen::VectorXd x0_1(2);
    x0_1 << 0.0, 0.0; // Initial guess

    // Enable readable text output
    params.print_iteration = false;
    params.verbose = true;

    std::cout << "\n--- Iteration JSON log start ---" << std::endl;
    efopt::Result result1 = efopt::optimize(x0_1, RosenbrockConstraintProblem::evaluate_wrapper, &problem1, params);
    std::cout << "--- Iteration JSON log end ---" << std::endl;

    std::cout << "\nTest 2 Results:" << std::endl;
    std::cout << "Status: " << result1.status << std::endl;
    std::cout << "Optimal x: " << result1.x.transpose() << std::endl;
    std::cout << "Objective: " << result1.objective << std::endl;
    std::cout << "Constraint violation: " << result1.constraint_violation << std::endl;
    std::cout << "Constraint satisfaction: x1 + 2x2 = " << result1.x(0) + 2*result1.x(1) << " (expected 1)" << std::endl;
    std::cout << "Result JSON: " << result1.to_json().dump(4) << std::endl;

    // --------------------------
    // Test 3: Nonconvex problem
    // --------------------------
    std::cout << "\n=== Test 3: Nonconvex problem with constraint x1² + x2² <= 4 ===" << std::endl;
    NonconvexProblem problem2;

    Eigen::VectorXd x0_2(2);
    x0_2 << 3.0, 1.0; // Initial guess outside feasible region

    params.verbose = false;
    efopt::Result result2 = efopt::optimize(x0_2, NonconvexProblem::evaluate_wrapper, &problem2, params);

    std::cout << "\nTest 3 Results:" << std::endl;
    std::cout << "Status: " << result2.status << std::endl;
    std::cout << "Optimal x: " << result2.x.transpose() << std::endl;
    std::cout << "Objective: " << result2.objective << std::endl;
    std::cout << "Constraint violation: " << result2.constraint_violation << std::endl;
    std::cout << "x1² + x2² = " << result2.x(0)*result2.x(0) + result2.x(1)*result2.x(1) << " (expected <=4)" << std::endl;

    // --------------------------
    // Test 4: Load parameters from file (optional)
    // --------------------------
    std::cout << "\n=== Test 4: Load parameters from file example ===" << std::endl;
    std::cout << "You can also load parameters from a JSON file using:" << std::endl;
    std::cout << "efopt::Parameters params = efopt::Parameters::from_json_file(\"config.json\");" << std::endl;

    std::cout << "\n\nAll tests completed successfully!" << std::endl;
    return 0;
}
