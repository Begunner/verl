docker run --gpus all -it --rm \
    --ipc=host \
    --network=host \
    -v $(pwd):/workspace \
    -v /data00/models:/workspace/models \
    -v /data00/datasets:/workspace/datasets \
    -v /data00/checkpoints:/workspace/ckpts \
    -w /workspace \
    -e PYTHONPATH=$PYTHONPATH:. \
    -e ROLLOUT_NAME=vllm \
    -e SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True \
    -e NCCL_SHM_DISABLE=1 \
    -e NCCL_P2P_DISABLE=1 \
    verlai/verl:vllm012.latest \
    bash
