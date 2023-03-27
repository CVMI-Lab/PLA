#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

ulimit -n 64000

python3 -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch --tcp_port ${PORT} ${PY_ARGS}

