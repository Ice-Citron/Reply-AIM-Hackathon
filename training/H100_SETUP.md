# H100 Training Setup Guide

## Quick Start

### Option 1: Use Jupyter Notebook (Serverless W&B)
```bash
jupyter notebook train_rlaif.ipynb
```
- Runs on W&B managed GPUs (easiest)
- No local GPU required

### Option 2: Use Python Script on H100
```bash
python train_h100.py
```
- Runs on your local H100 GPU
- Full control over training

---

## Configuration

### Switch Between Serverless and Local

Edit `h100_config.py`:

```python
# For W&B Serverless (default)
BACKEND_TYPE = "serverless"

# For Local H100
BACKEND_TYPE = "local"
```

---

## Local H100 Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt

# For H100, also install:
pip install flash-attn --no-build-isolation
```

### 2. Configure h100_config.py

```python
LOCAL_CONFIG = {
    "base_model": "Qwen/Qwen2.5-14B-Instruct",

    # Multi-GPU setup
    "tensor_parallel_size": 1,  # Use 2-4 for multiple H100s

    # Memory settings
    "gpu_memory_utilization": 0.85,  # Adjust if OOM errors

    # Context length
    "max_model_len": 8192,  # Reduce if OOM

    # LoRA settings
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,

    # Training
    "learning_rate": 1e-5,
    "gradient_accumulation_steps": 4,
}
```

### 3. Run Training
```bash
python train_h100.py
```

---

## H100 Optimizations

The config automatically enables:
- ✅ **TF32** - Faster matmul on H100
- ✅ **Flash Attention 2** - 2-4x faster attention
- ✅ **CUDA optimizations** - cuDNN benchmarking

### Multi-GPU Setup

For 2x H100s:
```python
LOCAL_CONFIG = {
    "tensor_parallel_size": 2,  # Split model across 2 GPUs
    ...
}
```

For 4x H100s:
```python
LOCAL_CONFIG = {
    "tensor_parallel_size": 4,
    ...
}
```

---

## Training Parameters

Edit `h100_config.py`:

```python
TRAINING_CONFIG = {
    "max_turns": 4,           # Tool-use turns per trajectory
    "groups_per_step": 2,     # Scenarios per training step
    "rollouts_per_group": 6,  # Trajectories per scenario
    "num_epochs": 3,
    "max_steps": 50,
}
```

### Faster Training (fewer rollouts)
```python
TRAINING_CONFIG = {
    "groups_per_step": 1,
    "rollouts_per_group": 4,
}
```

### More Data per Step
```python
TRAINING_CONFIG = {
    "groups_per_step": 4,
    "rollouts_per_group": 8,
}
```

---

## Monitoring

Training logs to Weights & Biases automatically:
- 📊 Dashboard: https://wandb.ai/your-username/Reply-AIM-Hackathon
- Metrics: avg_reward, max_reward, min_reward

---

## Troubleshooting

### Out of Memory (OOM)
```python
LOCAL_CONFIG = {
    "gpu_memory_utilization": 0.75,  # Reduce from 0.85
    "max_model_len": 4096,           # Reduce from 8192
}
```

Or reduce batch size:
```python
TRAINING_CONFIG = {
    "rollouts_per_group": 4,  # Reduce from 6
}
```

### VLLM Not Starting
Check port 8000 is free:
```bash
lsof -i :8000
kill -9 <PID>  # If needed
```

### Flash Attention Not Working
```bash
pip install flash-attn --no-build-isolation
```

If still fails, it will fallback to standard attention (slower but works).

---

## File Structure

```
Reply-AIM-Hackathon/
├── h100_config.py          # Training configuration
├── train_h100.py           # Python training script
├── train_rlaif.ipynb       # Jupyter notebook (serverless)
├── secretsConfig.py        # API keys
├── rag_tools/              # RAG system
│   ├── semantic_search.py
│   ├── keyword_search.py
│   └── read_document.py
└── data/
    ├── normalized_enhanced.xml
    ├── snippet_data.json
    └── eunomia_db/
```

---

## Performance Expectations

### H100 (80GB)
- **Qwen2.5-14B**: ~40-60 tokens/sec inference
- **Training**: ~2-3 min/step (with 12 rollouts)
- **Full run (50 steps)**: ~2-3 hours

### W&B Serverless
- **Inference**: Managed, auto-scaled
- **Training**: ~3-4 min/step
- **Cost**: $0.10-0.20 per training step

---

## Next Steps

1. ✅ Test on small data first:
   ```python
   TRAINING_CONFIG = {"max_steps": 5}
   ```

2. ✅ Monitor W&B dashboard for reward trends

3. ✅ Once working, scale up:
   ```python
   TRAINING_CONFIG = {
       "max_steps": 100,
       "num_epochs": 5,
   }
   ```

4. ✅ Deploy best checkpoint to production

---

## Support

- ART Docs: https://art.openpipe.ai
- Discord: https://discord.gg/zbBHRUpwf4
- Issues: Your errors → paste them back to me!
