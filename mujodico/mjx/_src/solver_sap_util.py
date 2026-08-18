import jax.numpy as jp


def smooth_relu(x, epsilon):
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


def smooth_relu_grad(x, epsilon):
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
        f(φ, vₙ) = k g_ε₁(-φ) g_ε₂(1 - d vₙ),
    where g_ε(x) is a C1-smooth approximation of ReLU:
        g_ε(x) =
            0                    if x <= -ε,
            (x + ε)^2 / (4 ε)    if |x| < ε,
            x                    if x >= ε.
    """

    g1 = smooth_relu(-phi, epsilon1)
    g2 = smooth_relu(1.0 - d * vn, epsilon2)

    return k * g1 * g2


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
  x2 = 1.0 - d * vn

  g1 = smooth_relu(x1, epsilon1)
  g2 = smooth_relu(x2, epsilon2)

  dg1_dvn = smooth_relu_grad(x1, epsilon1) * -dt
  dg2_dvn = smooth_relu_grad(x2, epsilon2) * -d

  return dt * k * (dg1_dvn * g2 + g1 * dg2_dvn)
