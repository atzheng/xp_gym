from flax import struct
from jaxtyping import Bool, Float
from typing import Any
from jax import Array


@struct.dataclass
class Observation(object):
    obs: Float[Array, "o_dim"]
    action: Bool[Array, "1"]
    reward: Float[Array, "1"]
    info: Any
    design_info: Any
