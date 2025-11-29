"""
Single GPU H100 Training Script
Run RLAIF training on a single H100 GPU
Usage: python train_single_gpu.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import json
import os
from typing import List
import wandb

# Import configuration
from config.paths import TRAINING_DATA_FILE, LEGAL_XML_FILE, CHROMA_DB_PATH
from config.training_config import BASE_MODEL, MAX_TURNS, WANDB_CONFIG, TRAINING_PARAMS
import config_single_gpu
import random

# Import ART
import art
from art.utils import iterate_dataset

# Import RAG tools
from rag_tools.semantic_search import FAISSSemanticSearch
from rag_tools.keyword_search import keyword_search
from rag_tools.read_document import read_document_part

# Import API
from secretsConfig import oaiKey, wandbKey, openRouterKey, hfToken
from openai import AsyncOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel
from litellm import acompletion
from textwrap import dedent
from huggingface_hub import HfApi

# Set environment variables
os.environ["OPENAI_API_KEY"] = oaiKey
os.environ["WANDB_API_KEY"] = wandbKey
os.environ["OPENROUTER_API_KEY"] = openRouterKey
os.environ["HF_TOKEN"] = hfToken

# HuggingFace config
HF_REPO_ID = "iteratehack/rlaif-lawcode-sft"
HF_PUSH_EVERY_N_STEPS = 10


# ============================================================
# HUGGINGFACE UPLOAD
# ============================================================

def push_checkpoint_to_hf(checkpoint_dir: str, step: int, repo_id: str = HF_REPO_ID):
    """Push a checkpoint to HuggingFace Hub"""
    if not os.path.exists(checkpoint_dir):
        print(f"  Checkpoint dir not found: {checkpoint_dir}")
        return False

    try:
        api = HfApi(token=hfToken)

        # Upload the entire checkpoint folder
        print(f"  Pushing checkpoint step {step} to HF: {repo_id}")
        api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_id,
            path_in_repo=f"checkpoints/step-{step:04d}",
            commit_message=f"Add checkpoint at step {step}",
        )
        print(f"  Successfully pushed step {step} to {repo_id}")
        return True
    except Exception as e:
        print(f"  HF push failed: {e}")
        return False


# ============================================================
# DATA MODELS
# ============================================================

class FinalAnswer(BaseModel):
    answer: str
    source_ids: list[str]


class LegalScenario(BaseModel):
    id: str
    question: str
    gold_answer: str | None = None
    gold_part_ids: list[str] | None = None


class LegalScenarioStep(BaseModel):
    step: int
    scenario: LegalScenario


# ============================================================
# TOOL WRAPPERS
# ============================================================

def keyword_search_tool(query: str, num: int = 5) -> str:
    """BM25 keyword search"""
    try:
        return keyword_search(str(LEGAL_XML_FILE), query, num)
    except Exception as e:
        return f"[TOOL ERROR] keyword_search failed: {str(e)}"


def semantic_search_tool(query: str, num: int = 5) -> str:
    """FAISS semantic search"""
    try:
        searcher = FAISSSemanticSearch(chroma_path=str(CHROMA_DB_PATH))
        return searcher.search(query, num)
    except Exception as e:
        return f"[TOOL ERROR] semantic_search failed: {str(e)}"


def read_document_part_tool(part_id: str) -> str:
    """Read document part by ID"""
    try:
        if " " in part_id or len(part_id) > 100:
            return f"[INVALID PART_ID] '{part_id[:50]}...' - use search tools first"
        return read_document_part(str(LEGAL_XML_FILE), part_id)
    except Exception as e:
        return f"[TOOL ERROR] read_document_part failed: {str(e)}"


# ============================================================
# ROLLOUT FUNCTION
# ============================================================

async def rollout(model: art.Model, legal_scenario_step: LegalScenarioStep) -> art.Trajectory:
    """Execute one trajectory rollout"""
    scenario = legal_scenario_step.scenario
    max_turns = MAX_TURNS

    traj = art.Trajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={"scenario_id": scenario.id, "step": legal_scenario_step.step},
    )

    system_prompt = dedent(f"""
        You are a legal research assistant that can search legal documents to answer questions.

        You have access to the following tools:
        - search_keyword(query: str, num: int) -> str
        - search_semantic(query: str, num: int) -> str
        - read_document_part(part_id: str) -> str

        You may call one tool per turn, for up to {max_turns} turns.

        When ready, give your final answer in this format:
        <answer>
        [your answer or "I don't know" if insufficient information]
        <sources>
        <source>doc_id</source>
        </sources>
        </answer>
    """)

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.question},
    ]

    # Define tools
    def return_final_answer(answer: str, source_ids: list[str]) -> FinalAnswer:
        return FinalAnswer(answer=answer, source_ids=source_ids)

    tools = [keyword_search_tool, semantic_search_tool, read_document_part_tool, return_final_answer]
    tools_by_name = {t.__name__: t for t in tools}
    traj.tools = [convert_to_openai_tool(t) for t in tools]

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    for _ in range(max_turns):
        response = await client.chat.completions.create(
            model=model.get_inference_name(),
            temperature=1,
            messages=traj.messages(),
            tools=traj.tools,
        )

        response_message = response.choices[0].message
        traj.messages_and_choices.append(response.choices[0])

        if not response_message.tool_calls:
            return traj

        try:
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                if tool_name in tools_by_name:
                    tool_args = json.loads(tool_call.function.arguments)
                    result = tools_by_name[tool_name](**tool_args)
                    traj.messages_and_choices.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": str(result),
                    })

                    if tool_name == "return_final_answer":
                        return traj
        except Exception as e:
            print(f"Tool error: {e}")
            return traj

    return traj


# ============================================================
# JUDGE FUNCTION (from your updated criteria)
# ============================================================

async def gemini_judge(group: art.TrajectoryGroup) -> art.TrajectoryGroup:
    """Score trajectories with new criteria"""
    trajectories = group.trajectories
    max_turns = MAX_TURNS

    if len(trajectories) <= 1:
        for traj in trajectories:
            traj.reward = 0.0
        return group

    # Analyze trajectories
    analyses = []
    for i, traj in enumerate(trajectories):
        messages = traj.messages()
        final_answer = messages[-1].get("content", "") if messages else ""

        format_errors = []
        num_searches = 0
        num_turns = sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")

        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                content = msg.get("content", "")
                if "search" in msg.get("name", "").lower():
                    num_searches += 1
                if any(err in content for err in ["[TOOL ERROR]", "[INVALID PART_ID]", "[PART NOT FOUND]"]):
                    format_errors.append(content[:80])

        analyses.append({
            "index": i,
            "final_answer": final_answer[:400],
            "num_turns": num_turns,
            "num_searches": num_searches,
            "has_format_error": len(format_errors) > 0,
        })

    # Build judge prompt
    analysis_text = "\n".join([
        f"**Response {a['index']+1}:** {a['final_answer']}\n"
        f"  Turns: {a['num_turns']}/{max_turns}, Searches: {a['num_searches']}, "
        f"Errors: {'YES' if a['has_format_error'] else 'NO'}"
        for a in analyses
    ])

    judge_prompt = f"""Score {len(trajectories)} legal research responses:

RULES:
- Correct answer: 1.0 to 2.0
- "I don't know": 0.0 to 1.0 (GOOD - avoids hallucination)
- Wrong answer: -1.0 to 0.0
- Format errors: -2.0 to -1.0

{analysis_text}

Return JSON: {{"scores": [{{"base_score": X, "reasoning": "..."}}, ...]}}"""

    try:
        response = await acompletion(
            model="openrouter/google/gemini-2.5-flash",
            messages=[{"role": "user", "content": judge_prompt}],
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_tokens=500,
        )

        import re
        result_text = response.choices[0].message.content.strip()
        json_match = re.search(r'\{[\s\S]*\}', result_text)

        if json_match:
            result = json.loads(json_match.group())
            base_scores = [item["base_score"] for item in result["scores"]]
        else:
            base_scores = [0.0] * len(trajectories)

        # Apply efficiency bonuses
        final_scores = []
        for base_score, analysis in zip(base_scores, analyses):
            score = float(base_score)
            if not analysis['has_format_error']:
                turn_eff = (max_turns - analysis['num_turns']) / max_turns
                score += turn_eff * 0.2
                if analysis['num_searches'] <= 2:
                    score += 0.1
            final_scores.append(round(score, 2))

        for traj, score in zip(trajectories, final_scores):
            traj.reward = float(score)

        print(f"  Scores: {final_scores}")

    except Exception as e:
        print(f"Judge error: {e}")
        for traj, analysis in zip(trajectories, analyses):
            traj.reward = -1.5 if analysis['has_format_error'] else 0.5

    return group


# ============================================================
# VALIDATION EVALUATION
# ============================================================

async def evaluate_validation(model: art.Model, val_scenarios: List[LegalScenario], step: int) -> dict:
    """Run validation evaluation on held-out scenarios"""
    if not val_scenarios:
        return {}

    print(f"\n📋 Running validation on {len(val_scenarios)} scenarios...")

    # Run single rollout per validation scenario (no training, just eval)
    val_groups = []
    for scenario in val_scenarios:
        try:
            traj = await rollout(model, LegalScenarioStep(step=step, scenario=scenario))
            val_groups.append(art.TrajectoryGroup([traj]))
        except Exception as e:
            print(f"  Val rollout error for {scenario.id}: {e}")

    if not val_groups:
        return {}

    # Judge validation trajectories
    judged_val = []
    for g in val_groups:
        try:
            judged_val.append(await gemini_judge(g))
        except Exception as e:
            print(f"  Val judge error: {e}")

    # Calculate validation metrics
    val_rewards = [t.reward for g in judged_val for t in g.trajectories]

    if not val_rewards:
        return {}

    val_metrics = {
        "val/avg_reward": sum(val_rewards) / len(val_rewards),
        "val/max_reward": max(val_rewards),
        "val/min_reward": min(val_rewards),
        "val/num_samples": len(val_rewards),
    }

    print(f"  Val rewards: avg={val_metrics['val/avg_reward']:.2f}, "
          f"max={val_metrics['val/max_reward']:.2f}, min={val_metrics['val/min_reward']:.2f}")

    return val_metrics


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

async def main():
    print("\n" + "=" * 60)
    print("🚀 RLAIF TRAINING - LEGAL RESEARCH AGENT")
    print("=" * 60)

    # Print config
    config_single_gpu.print_config()

    # Optimize for H100
    config_single_gpu.optimize_for_h100()

    # Load training data
    print(f"📂 Loading data from {TRAINING_DATA_FILE}...")
    with open(TRAINING_DATA_FILE, 'r') as f:
        data = json.load(f)

    all_scenarios = []
    for item in data.get("items", []):
        for row in item.get("rows", []):
            all_scenarios.append(LegalScenario(
                id=str(row["row_index"]),
                question=row["question"],
                gold_answer=row.get("model_answer", ""),
                gold_part_ids=row.get("sources", [])
            ))

    print(f"✅ Loaded {len(all_scenarios)} total scenarios")

    # Train/validation split (20% validation)
    val_ratio = TRAINING_PARAMS.get("val_set_ratio", 0.2)
    random.seed(42)  # Reproducible split
    random.shuffle(all_scenarios)

    val_size = int(len(all_scenarios) * val_ratio)
    val_scenarios = all_scenarios[:val_size]
    training_scenarios = all_scenarios[val_size:]

    print(f"📊 Split: {len(training_scenarios)} train, {len(val_scenarios)} validation\n")

    # Initialize W&B FIRST (before model registration to control run name)
    # CRITICAL: ART internally calls wandb.init(id=model.name, resume="allow")
    # We must use the SAME id so all metrics go to the same run
    model_name = config_single_gpu.SINGLE_GPU_CONFIG["model_name"]
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project=WANDB_CONFIG["project"],
        id=model_name,         # Must match what ART uses internally
        name=model_name,       # Display name in W&B UI
        resume="allow",        # Allow resuming existing runs (matches ART behavior)
        config={
            **config_single_gpu.SINGLE_GPU_CONFIG,
            "dataset_size": len(all_scenarios),
            "train_size": len(training_scenarios),
            "val_size": len(val_scenarios),
        },
        tags=["single-gpu", "h100", "qwen2.5-14b", "legal-agent", f"train-{len(training_scenarios)}"],
        notes=f"Single GPU training on H100 with {len(training_scenarios)} train / {len(val_scenarios)} val samples",
    )
    print(f"W&B Run: {run.name}")
    print(f"W&B URL: {run.url}\n")

    # Initialize backend
    print(f"Initializing local H100 backend...")
    backend = config_single_gpu.get_backend()

    # Create model with _internal_config
    model_config = config_single_gpu.get_model_config()
    print(f"Model: {model_config['base_model']}")
    print(f"   - Tensor Parallel: {model_config['_internal_config']['engine_args']['tensor_parallel_size']}")
    print(f"   - GPU Memory: {model_config['_internal_config']['init_args']['gpu_memory_utilization']}")
    print(f"   - 4-bit: {model_config['_internal_config']['init_args']['load_in_4bit']}")

    model = art.TrainableModel(**model_config)
    await model.register(backend)
    print(f"Model registered\n")

    # Training iterator
    cfg = config_single_gpu.SINGLE_GPU_CONFIG
    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=cfg["groups_per_step"],
        num_epochs=cfg["num_epochs"],
        initial_step=0,
    )

    print("🎯 Starting training loop...\n")

    step_count = 0
    for batch in training_iterator:
        print(f"=== Step {step_count} | Epoch {batch.epoch} ===")

        # Create trajectory groups
        groups = [
            art.TrajectoryGroup([
                rollout(model, LegalScenarioStep(step=batch.step, scenario=scenario))
                for _ in range(cfg["rollouts_per_group"])
            ])
            for scenario in batch.items
        ]

        # Gather trajectories
        finished_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc="Rollouts",
            max_exceptions=cfg["rollouts_per_group"] * len(batch.items),
        )

        # Judge
        judged_groups = [await gemini_judge(g) for g in finished_groups]

        # Log metrics to W&B
        all_rewards = [t.reward for g in judged_groups for t in g.trajectories]

        if not all_rewards:
            print("  WARNING: No successful rollouts in this step!")
            continue

        avg_reward = sum(all_rewards) / len(all_rewards)
        wandb.log({
            "train/avg_reward": avg_reward,
            "train/max_reward": max(all_rewards),
            "train/min_reward": min(all_rewards),
            "train/num_trajectories": len(all_rewards),
            "train/epoch": batch.epoch,
        }, step=step_count)

        print(f"  Rewards: avg={avg_reward:.2f}, max={max(all_rewards):.2f}, min={min(all_rewards):.2f}")

        # Train (config is in _internal_config, just pass empty TrainConfig)
        train_config = art.TrainConfig()
        await model.delete_checkpoints()
        await model.train(judged_groups, config=train_config)

        # Run validation every 5 steps
        EVAL_EVERY_N_STEPS = 5
        if (step_count + 1) % EVAL_EVERY_N_STEPS == 0:
            val_metrics = await evaluate_validation(model, val_scenarios, step_count)
            if val_metrics:
                wandb.log(val_metrics, step=step_count)

        # Push to HuggingFace every N steps
        if (step_count + 1) % HF_PUSH_EVERY_N_STEPS == 0:
            # Find the latest checkpoint
            art_dir = Path(".art") / WANDB_CONFIG["project"] / "models" / config_single_gpu.SINGLE_GPU_CONFIG["model_name"] / "checkpoints"
            if art_dir.exists():
                checkpoint_dirs = sorted([d for d in art_dir.iterdir() if d.is_dir() and d.name.isdigit()])
                if checkpoint_dirs:
                    latest_checkpoint = checkpoint_dirs[-1]
                    push_checkpoint_to_hf(str(latest_checkpoint), step_count + 1)

        print(f"Step {step_count} complete\n")

        step_count += 1
        if step_count >= cfg["max_steps"]:
            break

    # Final validation evaluation
    print("\n📋 Final validation evaluation...")
    final_val_metrics = await evaluate_validation(model, val_scenarios, step_count)
    if final_val_metrics:
        wandb.log({f"final_{k}": v for k, v in final_val_metrics.items()}, step=step_count)

    # Push final checkpoint to HuggingFace
    print("Pushing final checkpoint to HuggingFace...")
    art_dir = Path(".art") / WANDB_CONFIG["project"] / "models" / config_single_gpu.SINGLE_GPU_CONFIG["model_name"] / "checkpoints"
    if art_dir.exists():
        checkpoint_dirs = sorted([d for d in art_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        if checkpoint_dirs:
            latest_checkpoint = checkpoint_dirs[-1]
            push_checkpoint_to_hf(str(latest_checkpoint), step_count, repo_id=HF_REPO_ID)

    run.finish()
    print("Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
