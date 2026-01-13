#!/bin/bash

python ./tools/compile_droid_slam.py
uv pip install --no-build-isolation thirdparty/droid_slam
uv pip install --no-build-isolation thirdparty/droid_slam/droid_slam_api
uv pip install --no-build-isolation thirdparty/droid_slam/thirdparty/lietorch
# uv pip install --no-build-isolation thirdparty/droid_slam/thirdparty/pytorch_scatter
# OSError: .venv/lib/python3.11/site-packages/torch_scatter/_version_cuda.so: undefined symbol: _ZN3c106detail14torchCheckFailEPKcS2_jRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0%2Bcu121.html
uv pip install moderngl moderngl-window