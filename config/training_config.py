"""
Centralized Training Configuration
Shared hyperparameters across all training modes
"""

from .paths import TRAINING_DATA_FILE, LEGAL_XML_FILE, CHROMA_DB_PATH

# ============================================================
# MODEL CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
MAX_TURNS = 4  # Max tool-use turns per trajectory


# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

TRAINING_PARAMS = {
    # Data sampling
    "groups_per_step": 2,       # Scenarios per training step
    "rollouts_per_group": 6,    # Trajectories per scenario
    "num_epochs": 3,
    "max_steps": 50,

    # Learning
    "learning_rate": 1e-5,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "gradient_accumulation_steps": 4,

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
