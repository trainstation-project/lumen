from lumen.distributed import set_tpu_launch_env

for name, value in set_tpu_launch_env().items():
    print(f'{name}="{value}"')
