"""
Quick setup verification script
Run this to verify all configurations are correct
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 60)
    print("🔍 SETUP VERIFICATION")
    print("=" * 60)
    print()

    # Test 1: Import config
    print("1️⃣  Testing config imports...")
    try:
        from config.training_config import BASE_MODEL, MAX_TURNS, TRAINING_PARAMS
        from config.paths import (
            TRAINING_DATA_FILE,
            LEGAL_XML_FILE,
            CHROMA_DB_PATH,
            validate_data_files,
        )
        print(f"   ✅ Config imports successful")
        print(f"   📦 Model: {BASE_MODEL}")
        print(f"   🔢 Max turns: {MAX_TURNS}")
    except Exception as e:
        print(f"   ❌ Config import failed: {e}")
        return

    print()

    # Test 2: Validate data files
    print("2️⃣  Validating data files...")
    if validate_data_files():
        print(f"   ✅ All data files present")
    else:
        print(f"   ❌ Some data files missing")
        return

    print()

    # Test 3: Import RAG tools
    print("3️⃣  Testing RAG tool imports...")
    try:
        from rag_tools.semantic_search import FAISSSemanticSearch
        from rag_tools.keyword_search import keyword_search
        from rag_tools.read_document import read_document_part
        print(f"   ✅ RAG tools imported successfully")
    except Exception as e:
        print(f"   ❌ RAG tools import failed: {e}")
        return

    print()

    # Test 4: Import training configs
    print("4️⃣  Testing training configs...")
    try:
        from training.singleGPU import config_single_gpu
        from training.multiGPU import config_multi_gpu
        from training.serverless import config_serverless

        # Check single GPU config
        single_config = config_single_gpu.SINGLE_GPU_CONFIG
        assert "_internal_config" in single_config, "Missing _internal_config in single GPU"
        assert "engine_args" in single_config["_internal_config"], "Missing engine_args"
        assert single_config["_internal_config"]["engine_args"]["tensor_parallel_size"] == 1

        # Check multi GPU config
        multi_config = config_multi_gpu.MULTI_GPU_CONFIG
        assert "_internal_config" in multi_config, "Missing _internal_config in multi GPU"
        assert multi_config["_internal_config"]["engine_args"]["tensor_parallel_size"] >= 2

        print(f"   ✅ Training configs valid")
        print(f"   🖥️  Single GPU: tensor_parallel_size = 1")
        print(f"   🖥️  Multi GPU: tensor_parallel_size = {multi_config['_internal_config']['engine_args']['tensor_parallel_size']}")
    except Exception as e:
        print(f"   ❌ Training config test failed: {e}")
        return

    print()

    # Test 5: Check if ART is installed
    print("5️⃣  Checking ART installation...")
    try:
        import art
        print(f"   ✅ ART library installed")
        print(f"   📍 Version: {art.__version__ if hasattr(art, '__version__') else 'Unknown'}")
    except ImportError:
        print(f"   ❌ ART library not installed")
        print(f"   💡 Run: pip install openpipe-art==0.5.0")
        return

    print()

    # Test 6: Check secrets
    print("6️⃣  Checking API keys...")
    try:
        from secretsConfig import oaiKey, wandbKey, openRouterKey
        has_openai = len(oaiKey) > 10 if oaiKey else False
        has_wandb = len(wandbKey) > 10 if wandbKey else False
        has_openrouter = len(openRouterKey) > 10 if openRouterKey else False

        print(f"   {'✅' if has_openai else '⚠️ '} OpenAI API key: {'Set' if has_openai else 'Missing'}")
        print(f"   {'✅' if has_wandb else '⚠️ '} W&B API key: {'Set' if has_wandb else 'Missing'}")
        print(f"   {'✅' if has_openrouter else '⚠️ '} OpenRouter API key: {'Set' if has_openrouter else 'Missing'}")

        if not (has_wandb and has_openrouter):
            print(f"   💡 Update secretsConfig.py with your API keys")
    except Exception as e:
        print(f"   ⚠️  Could not load secretsConfig.py: {e}")

    print()
    print("=" * 60)
    print("✅ SETUP VERIFICATION COMPLETE!")
    print("=" * 60)
    print()
    print("🚀 Ready to train! Choose your mode:")
    print("   1. Serverless: cd training/serverless && jupyter notebook train_serverless.ipynb")
    print("   2. Single GPU: cd training/singleGPU && python train_single_gpu.py")
    print("   3. Multi-GPU: cd training/multiGPU && python train_multi_gpu.py")
    print()


if __name__ == "__main__":
    main()
