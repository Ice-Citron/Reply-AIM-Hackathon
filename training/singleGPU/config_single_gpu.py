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

    # ART internal configuration
    "_internal_config": {
        "engine_args": {
            "tensor_parallel_size": 1,  # Single GPU
            "enable_sleep_mode": True,  # Enable for single GPU
        },
        "init_args": {
            "gpu_memory_utilization": 0.90,  # Higher for single GPU
            "max_seq_length": 8192,
            "load_in_4bit": True,  # Unsloth 4-bit optimization
        },
        "peft_args": {
            "r": TRAINING_PARAMS["lora_rank"],
            "lora_alpha": TRAINING_PARAMS["lora_alpha"],
            "lora_dropout": TRAINING_PARAMS["lora_dropout"],
        },
        "trainer_args": {
            "learning_rate": TRAINING_PARAMS["learning_rate"],
            "gradient_accumulation_steps": TRAINING_PARAMS["gradient_accumulation_steps"],
        },
    },

    # Training
    **TRAINING_PARAMS,

    # Data
    **DATA_PATHS,

    # Paths
    "checkpoints_dir": str(CHECKPOINTS_DIR),

    # GPU Settings (for helper functions)
    "gpu_id": 0,
    "gpu_memory_utilization": 0.90,
    "max_model_len": 8192,
}


def get_backend():
    """Get local backend for single GPU"""
    from art.local.backend import LocalBackend
    return LocalBackend()


def get_model_config():
    """Get model configuration for TrainableModel"""
    return {
        "name": SINGLE_GPU_CONFIG["model_name"],
        "project": WANDB_CONFIG["project"],
        "base_model": SINGLE_GPU_CONFIG["base_model"],
        "_internal_config": SINGLE_GPU_CONFIG["_internal_config"],
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
