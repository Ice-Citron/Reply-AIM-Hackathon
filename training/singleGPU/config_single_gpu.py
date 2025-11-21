"""
Single GPU (H100) Training Configuration
For training on a single H100 GPU
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
# SINGLE GPU CONFIGURATION
# ============================================================

SINGLE_GPU_CONFIG = {
    # Model
    "base_model": BASE_MODEL,
    "model_name": "legal-agent-h100-single",

    # GPU settings (single H100)
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,  # Higher for single GPU
    "gpu_id": 0,

    # VLLM
    "vllm_port": 8000,
    "vllm_host": "0.0.0.0",
    "max_model_len": 8192,

    # Training
    **TRAINING_PARAMS,

    # Data
    **DATA_PATHS,

    # Paths
    "checkpoints_dir": str(CHECKPOINTS_DIR),
}


def get_backend():
    """Get local backend for single GPU"""
    from art.local.backend import LocalBackend
    return LocalBackend(
        vllm_port=SINGLE_GPU_CONFIG["vllm_port"],
        vllm_host=SINGLE_GPU_CONFIG["vllm_host"],
        tensor_parallel_size=SINGLE_GPU_CONFIG["tensor_parallel_size"],
        gpu_memory_utilization=SINGLE_GPU_CONFIG["gpu_memory_utilization"],
        max_model_len=SINGLE_GPU_CONFIG["max_model_len"],
    )


def get_model_config():
    """Get model configuration"""
    return {
        "name": SINGLE_GPU_CONFIG["model_name"],
        "base_model": SINGLE_GPU_CONFIG["base_model"],
    }


def optimize_for_h100():
    """Enable H100-specific optimizations"""
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    os.environ["ATTN_IMPLEMENTATION"] = "flash_attention_2"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(SINGLE_GPU_CONFIG["gpu_id"])

    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("✅ Single H100 optimizations enabled")
    print(f"   - GPU ID: {SINGLE_GPU_CONFIG['gpu_id']}")
    print(f"   - Memory Util: {SINGLE_GPU_CONFIG['gpu_memory_utilization']}")
    print(f"   - TF32: Enabled")
    print(f"   - Flash Attention 2: Enabled")
    if torch.cuda.is_available():
        print(f"   - Device: {torch.cuda.get_device_name(0)}")
    print()


def print_config():
    """Print single GPU configuration"""
    print("=" * 60)
    print("🔧 SINGLE H100 GPU CONFIGURATION")
    print("=" * 60)
    print(f"Backend: Local (Single GPU)")
    print(f"Model: {SINGLE_GPU_CONFIG['base_model']}")
    print(f"GPU ID: {SINGLE_GPU_CONFIG['gpu_id']}")
    print(f"Memory Util: {SINGLE_GPU_CONFIG['gpu_memory_utilization']}")
    print(f"Context Length: {SINGLE_GPU_CONFIG['max_model_len']}")
    print(f"\nTraining:")
    print(f"  - Groups/Step: {SINGLE_GPU_CONFIG['groups_per_step']}")
    print(f"  - Rollouts/Group: {SINGLE_GPU_CONFIG['rollouts_per_group']}")
    print(f"  - Epochs: {SINGLE_GPU_CONFIG['num_epochs']}")
    print(f"  - Max Steps: {SINGLE_GPU_CONFIG['max_steps']}")
    print(f"  - Learning Rate: {SINGLE_GPU_CONFIG['learning_rate']}")
    print(f"  - LoRA Rank: {SINGLE_GPU_CONFIG['lora_rank']}")
    print("=" * 60)
    print()
