"""
As all tests run in the same python process, these things can't vary between tests.
So make sure that they are always set before a test begins
"""
import os

import jax


def shared_test_setup():
    existing_flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = existing_flags + " --xla_force_host_platform_device_count=4"
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
