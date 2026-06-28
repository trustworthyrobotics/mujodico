# MuJoDiCo (Multi-Joint dynamics with Differentiable Contact)

This package extends MuJoCo with differentiable contact and provides trajectory optimization algorithms for contact-rich dynamical systems.

## Installation

```sh
pip install -e .
```

## Usage

Once installed, this package can be imported via `from mujodico import mjx`, which has the same API as [mujoco-mjx](https://mujoco.readthedocs.io/en/stable/mjx.html).

## Versioning

The `major.minor.micro` portion of the version number matches the version of MuJoCo. Optionally, if we release updates to MuJoDiCo that target the same version of MuJoCo, a `.postN` suffix is added, for example `3.0.1.post2` represents the second update to MuJoDiCo for MuJoCo 3.0.1.

## License and Disclaimer

MuJoDiCo is licensed under the Apache License, Version 2.0. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

This project includes code adapted from [MuJoCo](https://github.com/google-deepmind/mujoco), which is licensed under Apache License, Version 2.0. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.
