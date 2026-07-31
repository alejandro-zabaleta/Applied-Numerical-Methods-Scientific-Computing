# Numerical Analysis — Lab Practicals & Algorithms

Coursework exercises from the Numerical Analysis (*Cálculo Numérico*) course, Mathematics Degree at UPC.

## 01 — Mesh Quality Optimization (Nonlinear Systems)
Formulated mesh distortion minimization as a nonlinear system of equations (zero-finding of the gradient), solved via Newton-Raphson using the Hessian as the Jacobian. Reduced mesh distortion from $\eta=17.33$ to $\eta=5.36$, with quadratic convergence in the final iterations (error $< 0.5 \times 10^{-7}$ in 11 iterations).

## 02 — Adaptive Quadrature (Numerical Integration)
Compared composite trapezoidal and Simpson's rules for numerical integration, verifying theoretical convergence orders. Implemented a recursive adaptive Simpson's algorithm, proving its error bound and evaluating efficiency (333 vs. 287 subintervals) against uniform quadrature for equivalent precision.

## 03 — Parabolic Shooting (ODEs)
Solved a projectile motion ODE using Euler, RK4, and adaptive-step RK45, comparing accuracy and computational cost. Verified theoretical convergence orders (slopes of $-1$ and $-4$ in log-log error plots) and used root-finding (`scipy.optimize.fsolve`) combined with event detection to determine the launch angle hitting a target 500 m away.

---

## 🛠 Tools & Tech Stack
`Python` · `NumPy` · `SciPy` · `Matplotlib`
