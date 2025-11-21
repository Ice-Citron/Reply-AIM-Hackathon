"""
Multi-GPU (Multiple H100s) Training Configuration
For training on 2-8 H100 GPUs with tensor parallelism
"""

import sys
from pathlib import Path
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.training_config import BASE_MODEL, TRAINING_PARAMS, DATA_PATHS, WANDB_CONFIG
from config.paths import CHECKPOINTS_DIR

# ============================================================
# MULTI-GPU CONFIGURATION
# ============================================================

NUM_GPUS = 2  # Change to 4 or 8 for more GPUs

MULTI_GPU_CONFIG = {
    # Model
    "base_model": BASE_MODEL,
    "model_name": f"legal-agent-h100-{NUM_GPUS}gpu",

    # GPU settings (multi-GPU tensor parallelism)
    "tensor_parallel_size": NUM_GPUS,  # Split model across GPUs
    "gpu_memory_utilization": 0.85,
    "gpu_ids": list(range(NUM_GPUS)),  # [0, 1] for 2 GPUs, [0,1,2,3] for 4, etc.

    # VLLM
    "vllm_port": 8000,
    "vllm_host": "0.0.0.0",
    "max_model_len": 8192,

    # Training (larger batches for multi-GPU)
    **{**TRAINING_PARAMS, "rollouts_per_group": 8},  # Increase for multi-GPU

    # Data
    **DATA_PATHS,

    # Paths
    "checkpoints_dir": str(CHECKPOINTS_DIR),
}


def get_backend():
    """Get local backend for multi-GPU"""
    from art.local.backend import LocalBackend
    return LocalBackend(
        vllm_port=MULTI_GPU_CONFIG["vllm_port"],
        vllm_host=MULTI_GPU_CONFIG["vllm_host"],
        tensor_parallel_size=MULTI_GPU_CONFIG["tensor_parallel_size"],
        gpu_memory_utilization=MULTI_GPU_CONFIG["gpu_memory_utilization"],
        max_model_len=MULTI_GPU_CONFIG["max_model_len"],
    )


def get_model_config():
    """Get model configuration"""
    return {
        "name": MULTI_GPU_CONFIG["model_name"],
        "base_model": MULTI_GPU_CONFIG["base_model"],
    }


def optimize_for_multi_h100():
    """Enable multi-GPU H100 optimizations"""
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    os.environ["ATTN_IMPLEMENTATION"] = "flash_attention_2"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    # Set visible GPUs
    gpu_ids_str = ",".join(map(str, MULTI_GPU_CONFIG["gpu_ids"]))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    # NCCL for multi-GPU communication
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "0"  # Enable InfiniBand if available

    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print(f"✅ Multi-GPU ({NUM_GPUS}x H100) optimizations enabled")
    print(f"   - GPUs: {gpu_ids_str}")
    print(f"   - Tensor Parallel Size: {MULTI_GPU_CONFIG['tensor_parallel_size']}")
    print(f"   - Memory Util: {MULTI_GPU_CONFIG['gpu_memory_utilization']}")
    print(f"   - TF32: Enabled")
    print(f"   - Flash Attention 2: Enabled")
    print(f"   - NCCL: Enabled")
    if torch.cuda.is_available():
        for i in range(min(NUM_GPUS, torch.cuda.device_count())):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
    print()


def print_config():
    """Print multi-GPU configuration"""
    print("=" * 60)
    print(f"🔧 MULTI-GPU ({NUM_GPUS}x H100) CONFIGURATION")
    print("=" * 60)
    print(f"Backend: Local (Multi-GPU)")
    print(f"Model: {MULTI_GPU_CONFIG['base_model']}")
    print(f"GPUs: {MULTI_GPU_CONFIG['gpu_ids']}")
    print(f"Tensor Parallel: {MULTI_GPU_CONFIG['tensor_parallel_size']}")
    print(f"Memory Util: {MULTI_GPU_CONFIG['gpu_memory_utilization']}")
    print(f"Context Length: {MULTI_GPU_CONFIG['max_model_len']}")
    print(f"\nTraining:")
    print(f"  - Groups/Step: {MULTI_GPU_CONFIG['groups_per_step']}")
    print(f"  - Rollouts/Group: {MULTI_GPU_CONFIG['rollouts_per_group']} (increased for multi-GPU)")
    print(f"  - Epochs: {MULTI_GPU_CONFIG['num_epochs']}")
    print(f"  - Max Steps: {MULTI_GPU_CONFIG['max_steps']}")
    print(f"  - Learning Rate: {MULTI_GPU_CONFIG['learning_rate']}")
    print(f"  - LoRA Rank: {MULTI_GPU_CONFIG['lora_rank']}")
    print("=" * 60)
    print()
