"""
H100 Local Training Configuration
Switches from W&B Serverless to local H100 GPU training
"""

import os
from typing import Literal

# Import centralized paths
from config.paths import (
    TRAINING_DATA_FILE,
    LEGAL_XML_FILE,
    CHROMA_DB_PATH,
    CHECKPOINTS_DIR,
)

# ============================================================
# TRAINING BACKEND CONFIGURATION
# ============================================================

# Choose backend: "serverless" (W&B) or "local" (H100)
BACKEND_TYPE: Literal["serverless", "local"] = "serverless"  # Change to "local" for H100


# ============================================================
# LOCAL H100 CONFIGURATION
# ============================================================

LOCAL_CONFIG = {
    # Model settings
    "base_model": "Qwen/Qwen2.5-14B-Instruct",
    "model_name": "legal-agent-h100",

    # GPU settings
    "tensor_parallel_size": 1,  # Set to 2-4 if using multiple H100s
    "gpu_memory_utilization": 0.85,  # Use 85% of GPU memory

    # VLLM inference settings
    "vllm_port": 8000,
    "vllm_host": "0.0.0.0",
    "max_model_len": 8192,  # Context length

    # Training settings
    "learning_rate": 1e-5,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "gradient_accumulation_steps": 4,

    # GRPO settings
    "grpo_beta": 0.1,  # KL penalty coefficient
    "grpo_epsilon": 0.2,  # Clipping parameter

    # Paths
    "checkpoints_dir": str(CHECKPOINTS_DIR),
}


# ============================================================
# SERVERLESS (W&B) CONFIGURATION
# ============================================================

SERVERLESS_CONFIG = {
    "base_model": "Qwen/Qwen2.5-14B-Instruct",
    "model_name": "legal-agent-serverless",
    "project": "Reply-AIM-Hackathon",
    "learning_rate": 1e-5,
}


# ============================================================
# TRAINING HYPERPARAMETERS (shared across both backends)
# ============================================================

TRAINING_CONFIG = {
    "max_turns": 4,  # Max tool-use turns per trajectory
    "groups_per_step": 2,  # Number of scenario groups per training step
    "rollouts_per_group": 6,  # Trajectories per scenario
    "num_epochs": 3,
    "max_steps": 50,

    # Data files (using centralized paths)
    "data_file": str(TRAINING_DATA_FILE),
    "xml_file": str(LEGAL_XML_FILE),
    "chroma_db_path": str(CHROMA_DB_PATH),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_backend():
    """Get the appropriate backend based on BACKEND_TYPE"""
    if BACKEND_TYPE == "serverless":
        from art.serverless.backend import ServerlessBackend
        return ServerlessBackend()

    elif BACKEND_TYPE == "local":
        from art.local.backend import LocalBackend
        return LocalBackend(
            vllm_port=LOCAL_CONFIG["vllm_port"],
            vllm_host=LOCAL_CONFIG["vllm_host"],
            tensor_parallel_size=LOCAL_CONFIG["tensor_parallel_size"],
            gpu_memory_utilization=LOCAL_CONFIG["gpu_memory_utilization"],
            max_model_len=LOCAL_CONFIG["max_model_len"],
        )

    else:
        raise ValueError(f"Invalid BACKEND_TYPE: {BACKEND_TYPE}")


def get_model_config():
    """Get model configuration based on backend type"""
    if BACKEND_TYPE == "serverless":
        return {
            "name": SERVERLESS_CONFIG["model_name"],
            "project": SERVERLESS_CONFIG["project"],
            "base_model": SERVERLESS_CONFIG["base_model"],
        }
    else:
        return {
            "name": LOCAL_CONFIG["model_name"],
            "base_model": LOCAL_CONFIG["base_model"],
        }


def get_train_config():
    """Get training configuration"""
    import art

    if BACKEND_TYPE == "serverless":
        return art.TrainConfig(
            learning_rate=SERVERLESS_CONFIG["learning_rate"],
        )
    else:
        return art.TrainConfig(
            learning_rate=LOCAL_CONFIG["learning_rate"],
            lora_rank=LOCAL_CONFIG["lora_rank"],
            lora_alpha=LOCAL_CONFIG["lora_alpha"],
            lora_dropout=LOCAL_CONFIG["lora_dropout"],
            gradient_accumulation_steps=LOCAL_CONFIG["gradient_accumulation_steps"],
        )


def print_config():
    """Print current configuration"""
    print("=" * 60)
    print(f"🔧 TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Backend: {BACKEND_TYPE.upper()}")
    print(f"Model: {get_model_config()['base_model']}")

    if BACKEND_TYPE == "local":
        print(f"GPU Config:")
        print(f"  - Tensor Parallel: {LOCAL_CONFIG['tensor_parallel_size']}")
        print(f"  - Memory Util: {LOCAL_CONFIG['gpu_memory_utilization']}")
        print(f"  - Max Length: {LOCAL_CONFIG['max_model_len']}")
        print(f"  - LoRA Rank: {LOCAL_CONFIG['lora_rank']}")

    print(f"\nTraining:")
    print(f"  - Max Turns: {TRAINING_CONFIG['max_turns']}")
    print(f"  - Groups/Step: {TRAINING_CONFIG['groups_per_step']}")
    print(f"  - Rollouts/Group: {TRAINING_CONFIG['rollouts_per_group']}")
    print(f"  - Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"  - Max Steps: {TRAINING_CONFIG['max_steps']}")
    print("=" * 60)
    print()


# ============================================================
# H100-SPECIFIC OPTIMIZATIONS
# ============================================================

def optimize_for_h100():
    """Set environment variables for H100 optimization"""

    # Enable TF32 for faster training on H100
    os.environ["NVIDIA_TF32_OVERRIDE"] = "1"

    # Flash Attention 2 (if available)
    os.environ["ATTN_IMPLEMENTATION"] = "flash_attention_2"

    # CUDA optimizations
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    # PyTorch settings
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    print("✅ H100 optimizations enabled")
    print(f"   - TF32: Enabled")
    print(f"   - Flash Attention 2: {os.environ.get('ATTN_IMPLEMENTATION')}")
    print(f"   - CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print()


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    print_config()

    if BACKEND_TYPE == "local":
        optimize_for_h100()

    print("\n📖 Usage in notebook/script:")
    print("""
    import h100_config
    import art

    # Get backend
    backend = h100_config.get_backend()

    # Create model
    model = art.TrainableModel(**h100_config.get_model_config())
    await model.register(backend)

    # Get training config
    train_config = h100_config.get_train_config()

    # Train
    await model.train(trajectory_groups, config=train_config)
    """)
