docker run --gpus all -it --rm \
    --ipc=host \
    --network=host \
    -v $(pwd):/workspace \
    -v /data00/models:/workspace/models \
    -w /workspace \
    -e ROLLOUT_NAME=vllm \
    -e SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True \
    -e NCCL_SHM_DISABLE=1 \
    -e NCCL_P2P_DISABLE=1 \
    wb-vllm/latest:latest \
    bash
