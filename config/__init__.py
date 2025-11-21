"""Configuration module"""

from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    CONFIGS_DIR,
    LEGAL_XML_FILE,
    TRAINING_DATA_FILE,
    CHROMA_DB_PATH,
    CHECKPOINTS_DIR,
    BEST_MODEL_DIR,
    LORA_ADAPTERS_DIR,
    ensure_dirs,
    validate_data_files,
    print_paths,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "CONFIGS_DIR",
    "LEGAL_XML_FILE",
    "TRAINING_DATA_FILE",
    "CHROMA_DB_PATH",
    "CHECKPOINTS_DIR",
    "BEST_MODEL_DIR",
    "LORA_ADAPTERS_DIR",
    "ensure_dirs",
    "validate_data_files",
    "print_paths",
]
