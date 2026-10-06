# MuJoDiCo (Multi-Joint dynamics with Differentiable Contact)

This package extends MuJoCo with differentiable contact dynamics and provides trajectory optimization algorithms for contact-rich dynamical systems. Gradients are computed using the implicit function theorem rather than differentiating through the solver rollout.

## Installation

```sh
pip install -e .
pip install -e .
```

## Usage

[MuJoDiCo](https://github.com/wei-chen-li/mujodico) is a drop-in replacement for [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) that provides differentiable contact dynamics. You can simply replace `from mujoco import mjx` with `from mujodico import mjx`. For example:

```python
import jax
import jax.numpy as jnp
import mujoco
from mujodico import mjx

model = mujoco.MjModel.from_xml_string(XML)

mjx_model = mjx.put_model(
    model,
    contact_approximation='Hunt-Crossley',
    contact_smoothing=0.0,
)

def f(x, u):
    nq, nv = mjx_model.nq, mjx_model.nv
    d = mjx.make_data(mjx_model)
    d = d.replace(qpos=x[:nq], qvel=x[nq:nq+nv], act=x[nq+nv:], ctrl=u)
    d = mjx.step(mjx_model, d)
    x_next = jnp.concatenate((d.qpos, d.qvel, d.act))
    return x_next

x = jnp.zeros(mjx_model.nq + mjx_model.nv + mjx_model.na)
u = jnp.zeros(mjx_model.nu)

A, B = jax.jacfwd(f, argnums=(0,1))(x, u)
```

MuJoDiCo computes dynamics Jacobian using the implicit function theorem. Both forward-mode (`jax.jacfwd`) and reverse-mode (`jax.jacrev`) differentiation are supported.

### Contact Modeling

MuJoDiCo provides configurable contact models through two parameters:

* **`contact_approximation`** — specifies the contact model used for the dynamics. `'Hunt-Crossley'` is recommended for most applications.
* **`contact_smoothing`** — adds artificial smoothing to the contact model to make the resulting dynamics **C¹-continuous**.

The `contact_smoothing=κ` parameter controls the amount of regularization. It can be interpreted as applying a force of κ newtons over a distance of 1 meter.

For forward simulation, where differentiability is not required, use `contact_smoothing=0.0`.
For trajectory optimization or other gradient-based methods that require C¹-continuous dynamics, `contact_smoothing=1e-5` is a good starting point, providing smooth dynamics while introducing relatively little unphysical behavior.

### Examples

```sh
python -m mujodico.examples --help
```

## Versioning

The `major.minor.micro` portion of the version number matches the version of MuJoCo. Optionally, if we release updates to MuJoDiCo that target the same version of MuJoCo, a `.postN` suffix is added, for example `3.0.1.post2` represents the second update to MuJoDiCo for MuJoCo 3.0.1.

## License and Disclaimer

MuJoDiCo is licensed under the Apache License, Version 2.0. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

This project includes code adapted from [MuJoCo](https://github.com/google-deepmind/mujoco), which is licensed under Apache License, Version 2.0. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.
