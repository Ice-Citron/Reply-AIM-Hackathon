"""
Multi-GPU Training Script (8x L40S)
Run RLAIF training on 8x L40S GPUs with tensor parallelism
Usage: python train_multi_gpu.py
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
from config.training_config import BASE_MODEL, MAX_TURNS, WANDB_CONFIG
import config_multi_gpu

# Import ART
import art
from art.utils import iterate_dataset

# Import RAG tools
from rag_tools.semantic_search import FAISSSemanticSearch
from rag_tools.keyword_search import keyword_search
from rag_tools.read_document import read_document_part

# Import API
from secretsConfig import oaiKey, wandbKey, openRouterKey
from openai import AsyncOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel
from litellm import acompletion
from textwrap import dedent

# Set environment variables
os.environ["OPENAI_API_KEY"] = oaiKey
os.environ["WANDB_API_KEY"] = wandbKey
os.environ["OPENROUTER_API_KEY"] = openRouterKey


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
# JUDGE FUNCTION
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
# MAIN TRAINING LOOP
# ============================================================

async def main():
    print("\n" + "=" * 60)
    print("RLAIF TRAINING - LEGAL RESEARCH AGENT (8x L40S)")
    print("=" * 60)

    # Print config
    config_multi_gpu.print_config()

    # Optimize for L40S GPUs
    config_multi_gpu.optimize_for_gpus()

    # Load training data
    print(f"Loading data from {TRAINING_DATA_FILE}...")
    with open(TRAINING_DATA_FILE, 'r') as f:
        data = json.load(f)

    training_scenarios = []
    for item in data.get("items", []):
        for row in item.get("rows", []):
            training_scenarios.append(LegalScenario(
                id=str(row["row_index"]),
                question=row["question"],
                gold_answer=row.get("model_answer", ""),
                gold_part_ids=row.get("sources", [])
            ))

    print(f"Loaded {len(training_scenarios)} scenarios\n")

    # Initialize W&B FIRST (before ART model registration)
    # This ensures we control the run name before ART's internal init
    wandb.login(key=os.environ["WANDB_API_KEY"])
    run = wandb.init(
        project=WANDB_CONFIG["project"],
        config=config_multi_gpu.MULTI_GPU_CONFIG,
        tags=["multi-gpu", "l40s", "qwen2.5-14b", "legal-agent"],
        notes=f"Multi-GPU training on {config_multi_gpu.NUM_GPUS}x L40S GPUs",
        reinit=False,  # Prevent ART from creating a new run
    )
    print(f"W&B Run: {run.name}")
    print(f"W&B URL: {run.url}\n")

    # Initialize backend
    print(f"Initializing multi-GPU backend (8x L40S)...")
    backend = config_multi_gpu.get_backend()

    # Create model, then set _internal_config (per OpenPipe ART pattern)
    # Reference: ART/dev/tau-bench/run_training.py
    model_config = config_multi_gpu.get_model_config()
    internal_config = config_multi_gpu.get_internal_config()

    print(f"Model: {model_config['base_model']}")
    print(f"   - Tensor Parallel: {internal_config['engine_args']['tensor_parallel_size']}")
    print(f"   - GPU Memory: {internal_config['engine_args']['gpu_memory_utilization']}")
    print(f"   - Backend: TorchTune ({internal_config['torchtune_args']['model']})")

    model = art.TrainableModel(**model_config)
    # Set _internal_config AFTER creating the model (this is the OpenPipe pattern)
    model._internal_config = internal_config

    await model.register(backend)
    print(f"Model registered\n")

    # Training iterator
    cfg = config_multi_gpu.MULTI_GPU_CONFIG
    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=cfg["groups_per_step"],
        num_epochs=cfg["num_epochs"],
        initial_step=0,
    )

    print("Starting training loop...\n")

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

        # Calculate and print metrics (ART logs to W&B internally)
        all_rewards = [t.reward for g in judged_groups for t in g.trajectories]
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        print(f"  Rewards: avg={avg_reward:.2f}, max={max(all_rewards):.2f}, min={min(all_rewards):.2f}")

        # Train with learning rate from config
        train_config = art.TrainConfig(learning_rate=cfg["learning_rate"])
        await model.delete_checkpoints()
        await model.train(judged_groups, config=train_config)

        print(f"Step {step_count} complete\n")

        step_count += 1
        if step_count >= cfg["max_steps"]:
            break

    run.finish()
    print("Training complete!")


if __name__ == "__main__":
    asyncio.run(main())
