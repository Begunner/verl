docker run --gpus all -it --rm \
    --ipc=host \
    --network=host \
    -v $(pwd):/workspace \
    -w /workspace \
    -e ROLLOUT_NAME=sglang \
    -e SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True \
    -e NCCL_SHM_DISABLE=1 \
    -e NCCL_P2P_DISABLE=1 \
    -e HF_ENDPOINT=https://hf-mirror.com \
    your-verl-image:latest \
    bash