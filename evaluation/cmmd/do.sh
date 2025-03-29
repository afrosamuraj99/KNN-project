#!/usr/bin/env bash

. venv/bin/activate
export PYTHONPATH=$PYTHONPATH:${PWD}/scenic:${PWD}/..
python3 -m cmmd.main "$1" "$2" --batch_size=32 --max_count=30000
