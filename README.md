
# EFOPT: Efficient Nonlinear Optimization Solver

EFOPT is an efficient sequential convex programming solver for nonlinear optimization problems with constraints. It implements the EFOPT algorithm with trust region methods and penalty functions for handling constraints.

## New Features (v2.0)

- ✅ **Split code structure**: Separated header and implementation files (no longer header-only)
- ✅ **JSON parameter support**: Load/save optimization parameters from JSON files
- ✅ **Improved API**: Better error handling and validation
- ✅ **Result serialization**: Convert optimization results to JSON format
- ✅ **Static library**: Compiles to static library for easy linking

## Features

- **Trust region subproblem** solved using Steihaug-CG method
- **BFGS Hessian approximation** for second-order optimization
- **Penalty function approach** for handling both equality and inequality constraints
- **Two-stage convergence thresholds** for improved efficiency
- **Convexification step** to ensure positive definite Hessian
- **Eigen3 based** for efficient linear algebra operations
- **nlohmann/json** for parameter serialization

## Algorithm

EFOPT implements the algorithm described in the pseudocode with the following key steps:

1. **Outer penalty loop**: Increases penalty coefficient for constraints until they are satisfied
2. **Inner convexification loop**: Builds local convex approximation of the problem
3. **Trust Region Subproblem (TRS)**: Solved using Steihaug-CG algorithm
4. **Hessian update**: Uses BFGS quasi-Newton update to improve second-order information
5. **Trust region adjustment**: Expands or shrinks trust region based on model quality
6. **Threshold switching**: Switches to finer convergence thresholds once constraints are satisfied

## Installation

### Prerequisites
- C++17 compatible compiler
- Eigen3 (version 3.3 or higher)
- CMake (version 3.14 or higher)
- nlohmann/json (version 3.11 or higher, automatically downloaded if not found)

### Building and Installing

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

### Usage in your project

Add to your CMakeLists.txt:

```cmake
find_package(efopt REQUIRED)
target_link_libraries(your_target PRIVATE efopt::efopt)
```

## Quick Start

### Option 1: Use default parameters
```cpp
#include <efopt/efopt.hpp>

// Define your problem
struct MyProblem {
    void evaluate(const Eigen::VectorXd& x, double& objective, double& constraint_violation,
                 Eigen::VectorXd& grad_objective, Eigen::VectorXd& grad_constraint) {
        // Implement your objective function and constraints here
        objective = x(0)*x(0) + x(1)*x(1); // Example: min x² + y²

        // Example constraint: x + y = 1
        double c = x(0) + x(1) - 1;
        constraint_violation = c*c;

        grad_objective(0) = 2*x(0);
        grad_objective(1) = 2*x(1);

        grad_constraint(0) = 2*c*1;
        grad_constraint(1) = 2*c*1;
    }

    static void evaluate_wrapper(void* instance, const Eigen::VectorXd& x, double& f, double& c,
                                Eigen::VectorXd& grad_f, Eigen::VectorXd& grad_c) {
        static_cast<MyProblem*>(instance)->evaluate(x, f, c, grad_f, grad_c);
    }
};

int main() {
    MyProblem problem;

    // Initial guess
    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;

    // Use default parameters
    efopt::Parameters params;
    params.verbose = true;

    // Solve
    efopt::Result result = efopt::optimize(x0, MyProblem::evaluate_wrapper, &problem, params);

    // Use result
    std::cout << "Optimal x: " << result.x.transpose() << std::endl;
    std::cout << "Objective: " << result.objective << std::endl;

    // Convert result to JSON
    std::cout << "Result JSON: " << result.to_json().dump(4) << std::endl;

    return 0;
}
```

### Option 2: Load parameters from JSON
```cpp
// Load from JSON object
efopt::json params_json = {
    {"verbose", true},
    {"max_penalty_iterations", 10},
    {"xtol_fine", 1e-5}
};
efopt::Parameters params = efopt::Parameters::from_json(params_json);

// Or load from JSON file
efopt::Parameters params = efopt::Parameters::from_json_file("config.json");
```

## JSON Configuration Example (`config.json`)
```json
{
    "verbose": true,
    "penalty_init": 1000.0,
    "penalty_scale": 10.0,
    "max_penalty_iterations": 8,
    "max_convexify_iterations": 40,
    "extra_iterations": 10,
    "trust_region_init": 1.0,
    "trust_region_expand": 2.0,
    "trust_region_shrink": 0.75,
    "model_quality_low": 0.25,
    "model_quality_high": 0.75,
    "xtol_coarse": 0.01,
    "ftol_coarse": 0.01,
    "ctol_coarse": 0.01,
    "xtol_fine": 0.0001,
    "ftol_fine": 0.0001,
    "ctol_fine": 0.0001,
    "trs_params": {
        "max_iterations": 128,
        "tolerance": 1e-8
    }
}
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `verbose` | Print optimization progress | false |
| `penalty_init` | Initial penalty coefficient | 1e3 |
| `penalty_scale` | Penalty scaling factor | 10.0 |
| `max_penalty_iterations` | Maximum penalty loop iterations | 8 |
| `max_convexify_iterations` | Maximum inner loop iterations | 40 |
| `trust_region_init` | Initial trust region radius | 1.0 |
| `trust_region_expand` | Trust region expansion factor | 2.0 |
| `trust_region_shrink` | Trust region shrinkage factor | 0.75 |
| `model_quality_low` | Threshold for shrinking trust region | 0.25 |
| `model_quality_high` | Threshold for expanding trust region | 0.75 |
| `xtol_coarse`, `ftol_coarse`, `ctol_coarse` | First-stage thresholds | 1e-2 |
| `xtol_fine`, `ftol_fine`, `ctol_fine` | Second-stage thresholds | 1e-4 |

## Status Codes

| Code | Description |
|------|-------------|
| 1 | Success |
| 2 | Converged due to x tolerance |
| 3 | Converged due to function tolerance |
| 4 | Converged due to constraint tolerance |
| Negative values | Errors |

## Running Tests

```bash
cd build
./test_efopt
```

The test suite includes:
1. JSON parameter loading test
2. Rosenbrock function with equality constraint test
3. Nonconvex function with inequality constraint test

## License

MIT License
