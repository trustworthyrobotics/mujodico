The optional native oracle compiles the supplied FATROP checkout and solves
`min u0^2 + (x1 - 3)^2`, subject to `x0 = 1`, `x1 = x0 + u0`.
It returns a JSON record after FATROP's banner. The optimum is
`u0=1`, `x0=1`, `x1=2`, objective `2`.

`native_reference.cpp` adapts FATROP's two-stage C-interface unit-test callbacks
and retains their copyright notice. The oracle uses the checkout being
compared, rather than an installed or wrapped FATROP solver.

Build outside both repositories (CMake, a C++ compiler, and BLASFEO are needed):

```bash
python build_native_reference.py --fatrop-source /path/to/fatrop
```

The script can use BLASFEO bundled with an existing CasADi installation.
Alternatively pass `--blasfeo-prefix /path/to/blasfeo`, containing
`include/blasfeo.h` and `libblasfeo.so` or `lib/libblasfeo.so`.
It downloads nothing and writes only to its temporary build directory.

Use the printed executable path to enable the optional regression:

```bash
IPPAS_NATIVE_REFERENCE=/tmp/ippas-native-reference/reference \
  python -m unittest ippas.tests.test_newton.NewtonTest.test_native_two_stage_quadratic_reference
```

For an installed package, replace the module prefix `ippas` with
`mujodico.trajopt.ippas`. This optional dependency is not needed by the ordinary
unit tests or by the solver.
