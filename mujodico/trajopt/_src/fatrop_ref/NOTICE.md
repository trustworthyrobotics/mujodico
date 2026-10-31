# Source references and upstream notices

This package implements the nonlinear OCP interior-point components with reference
to the FATROP checkout and the boundary-composition equations in `IP-PAS.tex`.

The FATROP checkout retains the following copyright notice:

> Copyright (c) Lander Vanroye, KU Leuven.

Its `LICENSE.txt` states that most FATROP sources use the 2-Clause BSD license,
and identifies individually marked interior-point sources containing work derived
from Ipopt under the Eclipse Public License 2.0. The original mixed-license notice
and BSD terms are reproduced in `licenses/FATROP-LICENSE.txt`.

The consulted, marked interior-point files include `ip_algorithm.hxx`,
`ip_mu_update.hxx`, `ip_convergence_check.hxx`, `ip_linesearch.hxx`, and
`ip_resto_phase_min_cl1.hxx`. Those files retain this upstream attribution:

> This file contains work derived from Ipopt (https://github.com/coin-or/Ipopt),
> Copyright (C) 2004, 2010 International Business Machines and others.
> Ipopt is licensed under the Eclipse Public License 2.0 (EPL-2.0).

The full EPL text is reproduced in `licenses/LICENSE-EPL-2.0.txt`.

The new scan implementation follows the supplied document's explicit matrix
factorizations, boundary composition, and direction-recovery equations. It does
not call FATROP, Ipopt, BLASFEO, or an external optimization executable at runtime.
