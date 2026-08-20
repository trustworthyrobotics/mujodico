import jax
import jax.numpy as jp


def softplus(x, epsilon):
  assert epsilon > 0
  epsilon = epsilon / (4 * jp.log(2))
  return epsilon * jax.nn.softplus(x / epsilon)

def softplus_grad(x, epsilon):
  assert epsilon > 0
  epsilon = epsilon / (4 * jp.log(2))
  return jax.nn.sigmoid(x / epsilon)


def algebraic_smooth_relu(x, epsilon):
    assert epsilon > 0
    return 0.5 * (x + jp.sqrt(x**2 + (0.5 * epsilon)**2))

def algebraic_smooth_relu_grad(x, epsilon):
    assert epsilon > 0
    return 0.5 * (1.0 + x / jp.sqrt(x**2 + (0.5 * epsilon)**2))


def piecewise_smooth_relu(x, epsilon):
  assert epsilon > 0
  return jp.where(
    x <= -epsilon,
    0.0,
    jp.where(
      x < epsilon,
      (x + epsilon) ** 2 / (4.0 * epsilon),
      x,
    ),
  )

def piecewise_smooth_relu_grad(x, epsilon):
  assert epsilon > 0
  return jp.where(
    x <= -epsilon,
    0.0,
    jp.where(
      x < epsilon,
      (x + epsilon) / (2.0 * epsilon),
      1.0,
    ),
  )


def hunt_crossley_contact_force(phi, vn, k, d, epsilon1, epsilon2):
    """Compute the smoothed Hunt-Crossley normal contact force:
        f(φ, vₙ) = k d g_ε₁(-φ) g_ε₂(1/d - vₙ),
    where g_ε(x) is a C1-smooth approximation of ReLU:
        g_ε(x) =
            0                    if x <= -ε,
            (x + ε)^2 / (4 ε)    if |x| < ε,
            x                    if x >= ε.
    """

    g1 = piecewise_smooth_relu(-phi, epsilon1)
    g2 = piecewise_smooth_relu(1.0 / d - vn, epsilon2)

    return k * d * g1 * g2


def contact_normal_impulse(vn, phi0, k, d, dt, epsilon1, epsilon2):
  return dt * hunt_crossley_contact_force(
    phi=phi0 + vn * dt,
    vn=vn,
    k=k,
    d=d,
    epsilon1=epsilon1,
    epsilon2=epsilon2,
  )


def contact_normal_impulse_grad(vn, phi0, k, d, dt, epsilon1, epsilon2):
  x1 = -phi0 - dt * vn
  x2 = 1.0 / d - vn

  g1 = piecewise_smooth_relu(x1, epsilon1)
  g2 = piecewise_smooth_relu(x2, epsilon2)

  dg1_dvn = piecewise_smooth_relu_grad(x1, epsilon1) * -dt
  dg2_dvn = piecewise_smooth_relu_grad(x2, epsilon2) * -1.0

  return dt * k * d * (dg1_dvn * g2 + g1 * dg2_dvn)


def soft_normalize(x, epsilon):
  x_soft_norm = jp.sqrt(x.dot(x) + epsilon**2)
  return x / x_soft_norm


def soft_normalize_grad(x, epsilon):
  x_soft_norm = jp.sqrt(x.dot(x) + epsilon**2)
  x_soft_normalize = x / x_soft_norm
  return (jp.eye(x.size) - jp.outer(x_soft_normalize, x_soft_normalize)) / x_soft_norm
