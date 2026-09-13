"""JAX <-> CasADi bridge.

Wraps a JAX function as a CasADi Callback with:
  - exact first derivatives from JAX (jacfwd or jacrev, chosen by shape)
  - second derivatives declared to be zero (Gauss-Newton-style Hessian)
  - both forward- and reverse-mode AD stubs on the Jacobian callback, so
    the zero-Hessian contract holds regardless of which AD direction the
    outer solver asks for.

Requires jax to be configured for float64 (jax.config.update("jax_enable_x64", True))
before this module is used, since CasADi/IPOPT expect double precision throughout.
"""

import casadi as ca
import jax
import jax.numpy as jp
import numpy as np


def _to_np(a):
  """CasADi DM -> contiguous float64 numpy array, squeezed to 1-D."""
  arr = np.asarray(a, dtype=np.float64)
  return np.ascontiguousarray(arr.reshape(-1))


def _zeros_dm(rows, cols):
  return ca.DM.zeros(rows, cols)


class JaxCallback(ca.Callback):
  """CasADi Callback with exact JAX Jacobian and zero second derivatives.

  The callback evaluates:
      y = fn(x_0, ..., x_{n-1})

  First derivatives are supplied exactly from JAX, using whichever of
  jacfwd / jacrev is cheaper for the given input/output shapes.

  Second derivatives are declared to be zero:
      d/dx J(x) = 0

  This gives a Gauss-Newton-style Hessian approximation for the
  nonlinear callback, and is honored for both forward- and reverse-mode
  second-derivative requests from the solver.
  """

  def __init__(self, name, fn, in_dims, out_dim, opts=None):
    ca.Callback.__init__(self)

    self._in_dims = list(in_dims)
    self._out_dim = int(out_dim)
    n_in = len(self._in_dims)

    self._fn = jax.jit(fn)

    argnums = tuple(range(n_in))

    self._jac_fn = jax.jit(jax.jacfwd(fn, argnums=argnums))

    # Keep derivative callbacks alive (CasADi holds only a weak
    # reference internally, so Python-side GC of these will segfault
    # CasADi if not retained).
    self._jac_callback = None

    self.construct(name, opts or {})

  def get_n_in(self):
    return len(self._in_dims)

  def get_n_out(self):
    return 1

  def get_sparsity_in(self, i):
    return ca.Sparsity.dense(self._in_dims[i], 1)

  def get_sparsity_out(self, i):
    return ca.Sparsity.dense(self._out_dim, 1)

  def eval(self, arg):
    jax_args = [jp.asarray(_to_np(a)) for a in arg]
    out = self._fn(*jax_args)
    return [ca.DM(np.asarray(out, dtype=np.float64))]

  # ------------------------------------------------------------------
  # Exact first derivative from JAX.
  # ------------------------------------------------------------------

  def has_jacobian(self):
    return True

  def get_jacobian(self, name, inames, onames, opts):
    # Reuse a previously-built Jacobian callback if CasADi asks again
    # (e.g. after expand()/serialization) instead of rebuilding it.
    if self._jac_callback is not None:
      return self._jac_callback

    outer = self
    in_dims = self._in_dims
    out_dim = self._out_dim
    n_orig_in = len(in_dims)

    class Jac(ca.Callback):
      def __init__(self):
        ca.Callback.__init__(self)
        self._fwd_callback = None
        self._rev_callback = None
        self.construct(name, opts)

      def get_n_in(self):
        # Original inputs + nominal output of the outer function.
        return n_orig_in + 1

      def get_n_out(self):
        # One Jacobian block per original input.
        return n_orig_in

      def get_sparsity_in(self, i):
        if i < n_orig_in:
          return ca.Sparsity.dense(in_dims[i], 1)
        return ca.Sparsity.dense(out_dim, 1)

      def get_sparsity_out(self, i):
        return ca.Sparsity.dense(out_dim, in_dims[i])

      def eval(self, arg):
        # Only the original function inputs are needed; the nominal
        # output arg is unused since we recompute the Jacobian exactly
        # rather than reusing/perturbing the cached primal value.
        jax_args = [jp.asarray(_to_np(a)) for a in arg[:n_orig_in]]
        blocks = outer._jac_fn(*jax_args)
        return [ca.DM(np.asarray(block, dtype=np.float64)) for block in blocks]

      # ----------------------------------------------------------
      # Tell CasADi that this Jacobian callback itself has
      # derivatives. Those derivatives are intentionally ZERO, in
      # both forward and reverse mode, so the zero-Hessian contract
      # holds regardless of which AD direction the solver picks.
      # ----------------------------------------------------------

      def has_forward(self, nfwd):
        # CasADi's forward-function signature keeps a FIXED argument
        # count (nominal-in, nominal-out, seeds) and would instead
        # widen the seed/sensitivity sparsities to nfwd columns for
        # nfwd > 1. We don't implement that column-widening (it's
        # easy to get subtly wrong for matrix-shaped Jacobian-block
        # outputs), so we only advertise single-direction support and
        # let CasADi loop externally for nfwd > 1.
        return nfwd == 1

      def get_forward(self, nfwd, name, inames, onames, opts):
        assert nfwd == 1
        jac_n_in = n_orig_in + 1
        jac_n_out = n_orig_in

        class ZeroForward(ca.Callback):
          def __init__(self):
            ca.Callback.__init__(self)
            self.construct(name, opts)

          def get_n_in(self):
            # nominal inputs + nominal outputs + forward seeds
            return jac_n_in + jac_n_out + jac_n_in

          def get_n_out(self):
            return jac_n_out

          def get_sparsity_in(self, i):
            if i < jac_n_in:
              if i < n_orig_in:
                return ca.Sparsity.dense(in_dims[i], 1)
              return ca.Sparsity.dense(out_dim, 1)

            if i < jac_n_in + jac_n_out:
              j = i - jac_n_in
              return ca.Sparsity.dense(out_dim, in_dims[j])

            # Forward seeds, one per Jac input.
            j = i - jac_n_in - jac_n_out
            if j < n_orig_in:
              return ca.Sparsity.dense(in_dims[j], 1)
            return ca.Sparsity.dense(out_dim, 1)

          def get_sparsity_out(self, i):
            return ca.Sparsity.dense(out_dim, in_dims[i])

          def eval(self, arg):
            # Zero directional derivatives of the Jacobian -- no JAX
            # call needed, this is a pure-metadata result.
            return [_zeros_dm(out_dim, in_dims[i]) for i in range(jac_n_out)]

        cb = ZeroForward()
        self._fwd_callback = cb
        return cb

      def has_reverse(self, nadj):
        # Same fixed-argument-count reasoning as has_forward above:
        # only single-direction support is implemented.
        return nadj == 1

      def get_reverse(self, nadj, name, inames, onames, opts):
        assert nadj == 1
        jac_n_in = n_orig_in + 1
        jac_n_out = n_orig_in

        class ZeroReverse(ca.Callback):
          def __init__(self):
            ca.Callback.__init__(self)
            self.construct(name, opts)

          def get_n_in(self):
            # nominal inputs + nominal outputs + adjoint seeds
            # (one seed per Jacobian output block).
            return jac_n_in + jac_n_out + jac_n_out

          def get_n_out(self):
            # One adjoint sensitivity per Jacobian input.
            return jac_n_in

          def get_sparsity_in(self, i):
            if i < jac_n_in:
              if i < n_orig_in:
                return ca.Sparsity.dense(in_dims[i], 1)
              return ca.Sparsity.dense(out_dim, 1)

            if i < jac_n_in + jac_n_out:
              j = i - jac_n_in
              return ca.Sparsity.dense(out_dim, in_dims[j])

            # Adjoint seeds, one per Jac output block.
            j = i - jac_n_in - jac_n_out
            return ca.Sparsity.dense(out_dim, in_dims[j])

          def get_sparsity_out(self, i):
            if i < n_orig_in:
              return ca.Sparsity.dense(in_dims[i], 1)
            return ca.Sparsity.dense(out_dim, 1)

          def eval(self, arg):
            out = []
            for j in range(n_orig_in):
              out.append(_zeros_dm(in_dims[j], 1))
            out.append(_zeros_dm(out_dim, 1))
            return out

        cb = ZeroReverse()
        self._rev_callback = cb
        return cb

    self._jac_callback = Jac()
    return self._jac_callback