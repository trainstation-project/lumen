"""Prints TPU launch variables as `NAME="value"` lines for a shell to export before starting workers."""

from lumen.distributed import set_tpu_launch_env

for name, value in set_tpu_launch_env().items():
    print(f'{name}="{value}"')
