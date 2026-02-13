#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# -------------------------
# Early interactive selection (MUST be before importing torch)
# -------------------------
def _safe_input(prompt: str, default: str = "") -> str:
    """Input that won't crash in non-interactive environments; returns default on EOF."""
    try:
        if hasattr(sys.stdin, "isatty") and not sys.stdin.isatty():
            return default
        s = input(prompt).strip()
        return s if s else default
    except EOFError:
        return default

# 1) Select GPU first (before torch import)
_selected_cuda = _safe_input("Select GPU id for CUDA_VISIBLE_DEVICES (default=0): ", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = _selected_cuda

# Keep your original debug behavior
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# 2) Select task suite early (string only; will validate later after importing libero)
_selected_suite_early = _safe_input(
    "Select task_suite (e.g., libero_spatial/libero_object/libero_goal/libero_10/libero_90). "
    "Press Enter to choose later: ",
    ""
)

# -------------------------
# Normal imports
# -------------------------
import collections
import dataclasses
import logging
import pathlib

import imageio
import numpy as np
from image_tools import process_libero_observation
import tqdm
import tyro
import torch

import policy
from pathlib import Path

root_dir = Path(__file__).parent
sys.path.append(str(root_dir / 'LIBERO'))

# Monkey patch torch.load to use weights_only=False for LIBERO compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    model_id_or_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    checkpoint_path: str = 'ckpt/FUTURE-VLA'

    # Replan configuration (interactive selection, recommended: 4-16)
    replan_steps: int = -1  # Will be set interactively if not specified

    # Task configuration (now: single suite + single task_id)
    # suite candidates for interactive selection
    suite_candidates: list[str] = dataclasses.field(default_factory=lambda: [
        "libero_spatial", "libero_object",
        "libero_goal", "libero_10", "libero_90",
    ])

    # Selected suite/task (interactive will fill if empty / -1)
    task_suite: str = ""
    task_id: int = -1

    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    video_out_path: str = "data/libero/eval_videos"
    seed: int = 7


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _interactive_select_suite_and_task(args: Args) -> Args:
    """Interactive selection of suite and task_id (after libero is imported so we can validate)."""
    benchmark_dict = benchmark.get_benchmark_dict()

    # Decide suite
    suite = args.task_suite.strip() if args.task_suite else ""
    if not suite:
        # priority: early input (before imports)
        suite = _selected_suite_early.strip() if _selected_suite_early else ""

    # If still empty, ask now with candidates
    if not suite:
        print("\nAvailable suite candidates:")
        for i, s in enumerate(args.suite_candidates):
            print(f"  [{i}] {s}")
        idx = _safe_input("Select suite index (default=0): ", "0")
        try:
            suite = args.suite_candidates[int(idx)]
        except Exception:
            suite = args.suite_candidates[0]

    # Validate suite
    if suite not in benchmark_dict:
        # fallback: show actual available suites from libero
        available = sorted(list(benchmark_dict.keys()))
        print(f"\n[WARN] suite '{suite}' not found in benchmark.get_benchmark_dict().")
        print("Available suites in LIBERO:")
        for i, s in enumerate(available):
            print(f"  [{i}] {s}")
        idx = _safe_input("Select suite index from above (default=0): ", "0")
        try:
            suite = available[int(idx)]
        except Exception:
            suite = available[0]

    # Instantiate suite to get n_tasks
    task_suite = benchmark_dict[suite]()
    n_tasks = task_suite.n_tasks

    print(f"\nSuite selected: {suite}")
    print(f"task_id options: 0 ~ {n_tasks - 1}  (total {n_tasks} tasks)")

    # Decide task_id
    task_id = args.task_id
    if task_id < 0 or task_id >= n_tasks:
        tid_str = _safe_input("Select task_id (default=0): ", "0")
        try:
            task_id = int(tid_str)
        except Exception:
            task_id = 0
        task_id = max(0, min(task_id, n_tasks - 1))

    args.task_suite = suite
    args.task_id = task_id
    
    # Decide replan_steps
    if args.replan_steps < 1:
        print("\nReplan steps configuration:")
        print("Recommended range: 4-16 (smaller = more frequent replanning)")
        replan_str = _safe_input("Select replan_steps (default=10): ", "10")
        try:
            args.replan_steps = int(replan_str)
        except Exception:
            args.replan_steps = 10
        args.replan_steps = max(1, args.replan_steps)  # Ensure at least 1
    
    print(f"\nFinal configuration: Suite={suite}, Task={task_id}, Replan={args.replan_steps}")
    
    return args


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    # Make sure we are on visible GPU 0 (after CUDA_VISIBLE_DEVICES filtering)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

        # Clean up any residual CUDA error state
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            logging.info("CUDA device synchronized and cache cleared")
        except RuntimeError as e:
            logging.error(f"CUDA is in error state: {e}")
            logging.error("Please restart the Python process to clear CUDA errors")
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                logging.info("Attempted CUDA reset")
            except Exception:
                pass
            raise RuntimeError("CUDA device is in error state. Please restart the process.") from e

    # Interactive selection (suite + task_id)
    args = _interactive_select_suite_and_task(args)

    logging.info(f"Using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logging.info(f"Evaluating suite={args.task_suite}, task_id={args.task_id}")

    # Load policy once
    logging.info(f"Loading policy from {args.checkpoint_path}")
    eval_policy = policy.Policy(
        model_id_or_path=args.model_id_or_path,
        checkpoint_path=args.checkpoint_path,
        request_type="action",
    )
    logging.info("Policy loaded successfully")

    # Prepare suite/task
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    # max_steps per suite
    if args.task_suite == "libero_spatial":
        max_steps = 250
    elif args.task_suite == "libero_object":
        max_steps = 300
    elif args.task_suite == "libero_goal":
        max_steps = 320
    elif args.task_suite == "libero_10":
        max_steps = 550
    elif args.task_suite == "libero_90":
        max_steps = 450
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    logging.info(f"\n{'='*80}")
    logging.info(f"Suite: {args.task_suite} | Task {args.task_id}: {task_description}")
    logging.info(f"{'='*80}")

    # trials guard
    num_trials = min(args.num_trials_per_task, len(initial_states))
    if num_trials < args.num_trials_per_task:
        logging.warning(f"initial_states only has {len(initial_states)} entries; num_trials reduced to {num_trials}")

    # Use single replan_steps configuration
    replan_steps = args.replan_steps
    logging.info(f"\n--- Evaluating with replan_steps={replan_steps} ---")

    task_episodes, task_successes = 0, 0

    for episode_idx in tqdm.tqdm(range(num_trials),
                                 desc=f"Task{args.task_id} R{replan_steps}",
                                 leave=False):
        logging.info(f"Episode {episode_idx+1}/{num_trials}")

        env.reset()
        eval_policy.reset()
        action_plan = collections.deque()

        obs = env.set_init_state(initial_states[episode_idx])

        t = 0
        replay_images = []
        done = False

        logging.info(f"Starting episode {episode_idx+1}...")
        while t < max_steps + args.num_steps_wait:
            try:
                # wait for objects to settle
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    if t == args.num_steps_wait:
                        agentview_pil, wrist_pil, agentview_array = process_libero_observation(obs)
                        replay_images.append(agentview_array)
                        request_data = {
                            "observations": {
                                "agentview_image": agentview_pil,
                                "robot0_eye_in_hand_image": wrist_pil
                            },
                            "text": task_description
                        }
                        eval_policy.extract_request_data(request_data)
                    continue

                if not action_plan:
                    action_chunk = eval_policy.get_action()['chunk actions'][0]
                    if len(action_chunk) < replan_steps:
                        logging.warning(
                            f"Policy predicted {len(action_chunk)} steps, less than requested {replan_steps}. Using all available."
                        )
                        action_plan.extend(action_chunk)
                    else:
                        action_plan.extend(action_chunk[:replan_steps])

                action = action_plan.popleft()
                if hasattr(action, 'tolist'):
                    action = action.tolist()

                obs, reward, done, info = env.step(action)

                agentview_pil, wrist_pil, agentview_array = process_libero_observation(obs)
                replay_images.append(agentview_array)

                request_data = {
                    "observations": {
                        "agentview_image": agentview_pil,
                        "robot0_eye_in_hand_image": wrist_pil
                    },
                    "text": task_description
                }
                eval_policy.extract_request_data(request_data)

                if done:
                    task_successes += 1
                    break

                t += 1

            except Exception as e:
                logging.error(f"Caught exception: {e}")
                import traceback
                traceback.print_exc()

                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        logging.info("CUDA state reset after error")
                    except Exception as cuda_err:
                        logging.error(f"Failed to reset CUDA: {cuda_err}")
                break

        task_episodes += 1

        # save video
        suffix = "success" if done else "failure"
        video_dir = (
            pathlib.Path(args.video_out_path)
            / args.task_suite
            / f"task_{args.task_id}"
            / f"replan_{replan_steps}"
        )
        video_dir.mkdir(parents=True, exist_ok=True)
        video_filename = f"episode_{episode_idx:03d}_{suffix}.mp4"
        video_path = video_dir / video_filename

        try:
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=10,
                codec='h264',
            )
        except Exception as video_error:
            logging.warning(f"Failed to save MP4 video: {video_error}. Trying GIF format...")
            gif_path = str(video_path).replace('.mp4', '.gif')
            try:
                imageio.mimwrite(gif_path, [np.asarray(x) for x in replay_images], fps=10, loop=0)
            except Exception as gif_error:
                logging.error(f"Failed to save video in any format: {gif_error}")

    success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
    logging.info(f"\nResults: {task_successes}/{task_episodes} = {success_rate:.2%}")

    # write results
    results_file = (
        pathlib.Path(args.video_out_path)
        / args.task_suite
        / f"task_{args.task_id}"
        / "evaluation_results.txt"
    )
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LIBERO Single-Task Evaluation Results\n")
        f.write(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
        f.write(f"Model: {args.model_id_or_path}\n")
        f.write(f"Checkpoint: {args.checkpoint_path}\n")
        f.write(f"Suite: {args.task_suite}\n")
        f.write(f"Task ID: {args.task_id}\n")
        f.write(f"Task Description: {task_description}\n")
        f.write(f"Replan Steps: {replan_steps}\n")
        f.write(f"Trials: {num_trials}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Successes: {task_successes}/{task_episodes}\n")
        f.write(f"Success Rate: {success_rate:.2%}\n")

    logging.info(f"Results saved to: {results_file}")

    # close env
    try:
        if hasattr(env, "close"):
            env.close()
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
