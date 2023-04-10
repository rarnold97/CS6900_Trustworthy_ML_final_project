#!/usr/bin/env bash

# install carli-wagner attack project
pushd $(pwd)/../wheels/
pip install pytorch_cw2-1.0-py3-none-any.whl
popd