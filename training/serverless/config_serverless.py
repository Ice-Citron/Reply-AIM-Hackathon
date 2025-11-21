"""
Serverless (W&B) Training Configuration
For training on W&B managed infrastructure
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.training_config import BASE_MODEL, TRAINING_PARAMS, DATA_PATHS, WANDB_CONFIG

# ============================================================
# SERVERLESS BACKEND CONFIGURATION
# ============================================================

SERVERLESS_CONFIG = {
    # Model
    "base_model": BASE_MODEL,
    "model_name": "legal-agent-serverless",
    "project": WANDB_CONFIG["project"],

    # Training
    **TRAINING_PARAMS,

    # Data
    **DATA_PATHS,
}


def get_backend():
    """Get W&B Serverless backend"""
    from art.serverless.backend import ServerlessBackend
    return ServerlessBackend()


def get_model_config():
    """Get model configuration for serverless"""
    return {
        "name": SERVERLESS_CONFIG["model_name"],
        "project": SERVERLESS_CONFIG["project"],
        "base_model": SERVERLESS_CONFIG["base_model"],
    }


def print_config():
    """Print serverless configuration"""
    print("=" * 60)
    print("🔧 SERVERLESS TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Backend: W&B Serverless")
    print(f"Model: {SERVERLESS_CONFIG['base_model']}")
    print(f"Project: {SERVERLESS_CONFIG['project']}")
    print(f"\nTraining:")
    print(f"  - Groups/Step: {SERVERLESS_CONFIG['groups_per_step']}")
    print(f"  - Rollouts/Group: {SERVERLESS_CONFIG['rollouts_per_group']}")
    print(f"  - Epochs: {SERVERLESS_CONFIG['num_epochs']}")
    print(f"  - Max Steps: {SERVERLESS_CONFIG['max_steps']}")
    print(f"  - Learning Rate: {SERVERLESS_CONFIG['learning_rate']}")
    print("=" * 60)
    print()
