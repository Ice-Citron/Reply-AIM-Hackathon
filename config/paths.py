"""
Centralized Path Configuration
All file/directory paths defined in one place
"""

from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent  # Reply-AIM-Hackathon/


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "config"


# ============================================================
# DATA FILES
# ============================================================

# Legal document database
LEGAL_XML_FILE = DATA_DIR / "normalized_enhanced.xml"

# Training data
TRAINING_DATA_FILE = DATA_DIR / "snippet_data.json"

# ChromaDB embeddings database
CHROMA_DB_PATH = DATA_DIR / "eunomia_db"


# ============================================================
# MODEL CHECKPOINTS
# ============================================================

CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
BEST_MODEL_DIR = MODELS_DIR / "best_model"
LORA_ADAPTERS_DIR = MODELS_DIR / "lora_adapters"


# ============================================================
# SECRETS & CONFIG
# ============================================================

SECRETS_FILE = PROJECT_ROOT / "secretsConfig.py"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def ensure_dirs():
    """Create all necessary directories if they don't exist"""
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    CONFIGS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    BEST_MODEL_DIR.mkdir(exist_ok=True)
    LORA_ADAPTERS_DIR.mkdir(exist_ok=True)


def validate_data_files():
    """Check that all required data files exist"""
    missing = []

    if not LEGAL_XML_FILE.exists():
        missing.append(f"Legal XML: {LEGAL_XML_FILE}")

    if not TRAINING_DATA_FILE.exists():
        missing.append(f"Training data: {TRAINING_DATA_FILE}")

    if not CHROMA_DB_PATH.exists():
        missing.append(f"ChromaDB: {CHROMA_DB_PATH}")

    if missing:
        print("❌ Missing required data files:")
        for item in missing:
            print(f"   - {item}")
        return False

    print("✅ All data files present")
    return True


def print_paths():
    """Print all configured paths"""
    print("=" * 60)
    print("📁 PROJECT PATHS")
    print("=" * 60)
    print(f"Root:            {PROJECT_ROOT}")
    print(f"Data:            {DATA_DIR}")
    print(f"Models:          {MODELS_DIR}")
    print(f"Config:          {CONFIGS_DIR}")
    print()
    print("📄 DATA FILES")
    print(f"Legal XML:       {LEGAL_XML_FILE}")
    print(f"Training Data:   {TRAINING_DATA_FILE}")
    print(f"ChromaDB:        {CHROMA_DB_PATH}")
    print()
    print("💾 MODEL DIRECTORIES")
    print(f"Checkpoints:     {CHECKPOINTS_DIR}")
    print(f"Best Model:      {BEST_MODEL_DIR}")
    print(f"LoRA Adapters:   {LORA_ADAPTERS_DIR}")
    print("=" * 60)
    print()


# ============================================================
# USAGE
# ============================================================

if __name__ == "__main__":
    ensure_dirs()
    print_paths()
    validate_data_files()
