"""
Centralized Training Configuration
Shared hyperparameters across all training modes
"""

from .paths import TRAINING_DATA_FILE, LEGAL_XML_FILE, CHROMA_DB_PATH

# ============================================================
# MODEL CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"  # Full precision, supports tensor parallelism
MAX_TURNS = 4  # Max tool-use turns per trajectory


# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

TRAINING_PARAMS = {
    # Data sampling (adjusted for small dataset ~100 questions)
    "groups_per_step": 4,       # Scenarios per training step (increased for small dataset)
    "rollouts_per_group": 4,    # Trajectories per scenario (reduced for faster iteration)
    "num_epochs": 1,            # Single epoch to avoid trainer confusion
    "max_steps": 20,            # 80 train items / 4 groups = 20 steps

    # Validation split
    "val_set_ratio": 0.2,       # 20% for validation (~20 questions)

    # Learning
    "learning_rate": 1e-5,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "gradient_accumulation_steps": 2,  # Reduced for small batches

    # GRPO
    "grpo_beta": 0.1,
    "grpo_epsilon": 0.2,
}


# ============================================================
# DATA PATHS (from centralized paths)
# ============================================================

DATA_PATHS = {
    "training_data": str(TRAINING_DATA_FILE),
    "legal_xml": str(LEGAL_XML_FILE),
    "chroma_db": str(CHROMA_DB_PATH),
}


# ============================================================
# WANDB CONFIGURATION
# ============================================================

WANDB_CONFIG = {
    "project": "Reply-AIM-Hackathon",
    "entity": None,  # Set to your W&B username/team
}
