"""
Multi-GPU Training Configuration
For training on 8x L40S GPUs with tensor parallelism

Based on OpenPipe ART examples:
https://github.com/OpenPipe/ART/blob/main/dev/tau-bench/run_training.py
"""

import sys
from pathlib import Path
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.training_config import BASE_MODEL, TRAINING_PARAMS, DATA_PATHS, WANDB_CONFIG
from config.paths import CHECKPOINTS_DIR

# Import ART dev types for _internal_config
import art.dev

# ============================================================
# MULTI-GPU CONFIGURATION (8x L40S)
# ============================================================

NUM_GPUS = 8  # 8x NVIDIA L40S (46GB VRAM each)

MULTI_GPU_CONFIG = {
    # Model
    "base_model": BASE_MODEL,
    "model_name": f"legal-agent-l40s-{NUM_GPUS}gpu",

    # Training (larger batches for multi-GPU)
    **{**TRAINING_PARAMS, "rollouts_per_group": 8},  # Increase for multi-GPU

    # Data
    **DATA_PATHS,

    # Paths
    "checkpoints_dir": str(CHECKPOINTS_DIR),

    # GPU IDs (for optimization setup)
    "gpu_ids": list(range(NUM_GPUS)),

    # GPU Settings (for helper functions)
    "tensor_parallel_size": NUM_GPUS,
    "gpu_memory_utilization": 0.75,
    "max_model_len": 8192,
}


def get_internal_config() -> art.dev.InternalModelConfig:
    """
    Get ART internal configuration for multi-GPU with TorchTune backend.
    TorchTune is required for tensor parallelism > 1 with 14B+ models.

    Reference: ART/dev/tau-bench/run_training.py (model "016")
    """
    return art.dev.InternalModelConfig(
        engine_args=art.dev.EngineArgs(
            tensor_parallel_size=NUM_GPUS,
            gpu_memory_utilization=0.75,
        ),
        torchtune_args=art.dev.TorchtuneArgs(
            model="qwen2_5_14b_instruct",
            model_type="QWEN2",
            async_weight_syncing=True,
        ),
    )


def get_backend():
    """Get local backend for multi-GPU"""
    from art.local.backend import LocalBackend
    return LocalBackend()


def get_model_config():
    """
    Get model configuration for TrainableModel.
    Note: _internal_config is set AFTER model creation, not in constructor.
    """
    return {
        "name": MULTI_GPU_CONFIG["model_name"],
        "project": WANDB_CONFIG["project"],
        "base_model": MULTI_GPU_CONFIG["base_model"],
    }


def optimize_for_gpus():
    """Enable multi-GPU L40S optimizations"""
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

    print(f"✅ Multi-GPU ({NUM_GPUS}x L40S) optimizations enabled")
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
    print(f"🔧 MULTI-GPU ({NUM_GPUS}x L40S) CONFIGURATION")
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
