from dataclasses import dataclass
import jax


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TranscriptionParams:
  x_0: jax.Array
  x_goal: jax.Array
  Q: jax.Array
  Qf: jax.Array
  u_ref: jax.Array
  R: jax.Array
