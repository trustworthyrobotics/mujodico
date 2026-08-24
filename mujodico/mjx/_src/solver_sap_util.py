import jax
import jax.numpy as jp


def soft_normalize(x, epsilon):
  x_soft_norm = jp.sqrt(x.dot(x) + epsilon**2)
  return x / x_soft_norm

def soft_normalize_grad(x, epsilon):
  x_soft_norm = jp.sqrt(x.dot(x) + epsilon**2)
  x_soft_normalize = x / x_soft_norm
  return (jp.eye(x.size) - jp.outer(x_soft_normalize, x_soft_normalize)) / x_soft_norm


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


def local_smooth_relu(x, epsilon):
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

def local_smooth_relu_grad(x, epsilon):
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


def reciprocal_approx_relu(x, epsilon):
  return jp.where(
    x <= 0.5 * epsilon,
    0.25 * epsilon**2 / (-x + epsilon),
    x,
  )

def reciprocal_approx_relu_grad(x, epsilon):
  return jp.where(
    x <= 0.5 * epsilon,
    0.25 * epsilon**2 / (-x + epsilon)**2,
    1.0,
  )


def kelvin_voigt_normal_impulse(
    vn, phi0, dt, k, tau, contact_smoothing, next_step=True
):
  """Computes the Kelvin-Voigt normal contact impulse:
      γ(φ, vₙ) = δt k relu_ε(-φ - τ vₙ),
  where φ =
            φ₀ + δt vₙ  if next_step,
            φ₀          otherwise,
  and relu_ε is a smooth approximation of ReLU.
  """

  time_scale = (dt + tau) if next_step else (tau)
  eps = jp.sqrt(4 * contact_smoothing / (k * dt * time_scale))

  return dt * k * time_scale * reciprocal_approx_relu(-phi0 / time_scale - vn, eps)


def kelvin_voigt_normal_impulse_grad(
    vn, phi0, dt, k, tau, contact_smoothing, next_step=True
):

  time_scale = (dt + tau) if next_step else (tau)
  eps = jp.sqrt(4 * contact_smoothing / (k * dt * time_scale))

  return -dt * k * time_scale * reciprocal_approx_relu_grad(-phi0 / time_scale - vn, eps)


def hunt_crossley_normal_impulse(
    vn, phi0, dt, k, d, contact_smoothing, bounce_speed_tol, next_step=True
):
  """Computes the Hunt-Crossley normal contact impulse:
      γ(φ, vₙ) = δt k d relu_ε₁(-φ) relu_ε₂(1/d - vₙ),
  where φ =
            φ₀ + δt vₙ  if next_step,
            φ₀          otherwise,
  and relu_ε is a smooth approximation of ReLU.
  """

  tau = jp.where(contact_smoothing > 0, 1.0 * dt, 0.0)
  time_scale = (dt + tau) if next_step else (tau)

  x1 = -phi0 - time_scale * vn
  eps1 = jp.sqrt(4 * contact_smoothing * time_scale / (k * dt))

  x2 = 1 / d - vn
  eps2 = bounce_speed_tol

  return dt * k * d * reciprocal_approx_relu(x1, eps1) * local_smooth_relu(x2, eps2)


def hunt_crossley_normal_impulse_grad(
    vn, phi0, dt, k, d, contact_smoothing, bounce_speed_tol, next_step=True
):
  tau = jp.where(contact_smoothing > 0, 1.0 * dt, 0.0)
  time_scale = (dt + tau) if next_step else (tau)

  x1 = -phi0 - time_scale * vn
  eps1 = jp.sqrt(4 * contact_smoothing * time_scale / (k * dt))

  x2 = 1 / d - vn
  eps2 = bounce_speed_tol

  g1 = reciprocal_approx_relu(x1, eps1)
  g2 = local_smooth_relu(x2, eps2)

  dg1_dvn = reciprocal_approx_relu_grad(x1, eps1) * -time_scale
  dg2_dvn = local_smooth_relu_grad(x2, eps2) * -1

  return dt * k * d * (dg1_dvn * g2 + g1 * dg2_dvn)
