# EFOPT
An **E**fficient **F**easibility-First **Opt**imization.

## Introduction
EFOPT is a C++ package for constrained nonlinear optimization. It solves mathematical optimization problems of the form

$$\begin{align*}
&\displaystyle\min_{\mathbf x} f(\mathbf x)  \\
		&\mathrm{s.t.} \quad g_i(\mathbf x) \leq 0, \quad i = 1,2, \cdots, m \\
		&\qquad \ h_i(\mathbf x) = 0, \quad i = 1,2,\cdots, n
\end{align*}$$

where $f, g_i, h_i$ are non-convex scalar functions. EFOPT finds solutions that are locally optimal, and ideally any nonlinear functions should be wice continuously differentiable, and users should provide gradients. . It is often more widely useful for discontinuities in the function gradients.

SNOPT uses a penalty method with sequential quadratic programming (SQP) algorithm. Search directions are obtained from trust-region subproblems that minimize a quadratic model of the Lagrangian merit function subject to convexified constraints. An trust region is adjusted by evaluating the improvement of each search directions to ensure convergence.

### Date of code release
After receiving the first-round reviewer comments.


