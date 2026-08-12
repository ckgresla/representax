import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@pytest.mark.distributed
def test_leading_batch_axis_can_be_sharded_across_devices():
    if jax.device_count() < 2:
        pytest.skip("requires at least two JAX devices")
    devices = np.asarray(jax.devices()[:2])
    mesh = Mesh(devices, ("data",))
    sharding = NamedSharding(mesh, P("data", None))

    values = jax.device_put(jnp.arange(16).reshape(4, 4), sharding)

    assert values.shape == (4, 4)
    assert len(values.addressable_shards) == 2
