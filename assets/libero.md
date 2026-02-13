# LIBERO Evaluation Guide

## Overview

This guide covers the setup and usage of LIBERO evaluation for FUTURE-VLA. Two evaluation modes are available:
- **Automated Evaluation** (`action_eval.py`): Batch testing with fixed replan steps
- **Interactive HIL Evaluation** (`hil_eval.py`): Human-in-the-loop with dynamic control and resampling

## Prerequisites

### Required Packages and Environment

Navigate to the libero directory:
```shell
cd libero
```

Clone LIBERO repository:
```shell
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
```

Install MS-Swift:
```shell
pip install ms_swift==3.11.3
```

Clone and install 1D tokenizer (TiTok):
```shell
git clone https://github.com/bytedance/1d-tokenizer.git
cd 1d-tokenizer
pip install -r requirements.txt
```

## Environment Modifications

Follow these steps to modify your environment:

1. **Copy TiTok codec**:
   ```shell
   cp add_code/titok/titok_codec.py 1d-tokenizer/
   ```

2. **Replace FAST processor** :
   - Find your FAST cache: `~/.cache/huggingface/modules/transformers_modules/physical-intelligence/fast/{hash}/`
   - Replace: `processing_action_tokenizer.py`
   - Source: `replace_code/fast/processing_action_tokenizer.py`

3. **Find Python package directory**:
   ```shell
   pip show numpy  # Check installation path, e.g., A/site-packages
   ```

4. **Replace Swift templates** (custom action/video token support):
   - Replace: `A/site-packages/swift/llm/template/base.py`
   - Source: `replace_code/swift/base.py`
   - Replace: `A/site-packages/swift/llm/template/template/qwen.py`
   - Source: `replace_code/swift/qwen.py`

5. **Replace Transformers models** (Qwen3-VL modifications):
   - Replace: `A/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
   - Source: `replace_code/transformers/modeling_qwen3_vl.py`
   - Replace: `A/site-packages/transformers/models/qwen3_vl/processing_qwen3_vl.py`
   - Source: `replace_code/transformers/processing_qwen3_vl.py`

## Model Checkpoints

Download the model checkpoints from HuggingFace:

```shell
huggingface-cli download PopFan/FUTURE-VLA --local-dir . --local-dir-use-symlinks False
```

**Directory Structure After Download**:
```
FUTURE-VLA/
└── Libero/
    ├── FUTURE-VLA/     # Main model weights (Qwen3-VL-4B backbone)
    └── TiTok/          # 1D visual tokenizer weights (video generation)
```

**Checkpoint Description**:
- `FUTURE-VLA/`: Main VLA model checkpoint (action prediction + vision-language understanding)
- `TiTok/`: TiTok codec weights for future trajectory video generation.


## Core Files Overview

### `policy.py` - Policy Engine
**Description**: Core inference engine that handles model loading, action prediction, and video generation.

**Key Features**:
- Supports both `action` and `hil` request types
- Automatic retry with temperature increase on decode failure
- Resample mechanism for diverse action generation
- TiTok-based future trajectory video generation (HIL mode)

**Video Generation** (HIL mode):
- Outputs: `./output_videos/action_view1.mp4` (agentview), `action_view2.mp4` (wrist)
- Updates on each prediction (overwrites previous)

### `action_eval.py` - Automated Batch Evaluation
**Description**: Runs automated evaluation with fixed replan steps across multiple episodes.

**Features**:
- Non-interactive, suitable for benchmarking
- Fixed replan_steps selected at startup (recommended: 4-16)
- Runs 50 episodes per task by default
- Saves execution replay videos per episode

**Output Structure**:
```
data/libero/eval_videos/
└── {suite}/
    └── task_{id}/
        └── replan_{steps}/
            ├── episode_000_success.mp4
            ├── episode_001_failure.mp4
            └── ...
```

### `hil_eval.py` - Interactive HIL Evaluation
**Description**: Human-in-the-loop evaluation with per-action interactive control.

**Features**:
- Interactive replan_steps selection for each action request
- Resample mode (input `0`): Increases temperature for diverse outputs
- Real-time feedback with video location prompts
- Suitable for debugging and understanding model behavior

**Interaction Flow**:
```
Episode 1/50 - Step 15

📹 Video Locations:
  • Predicted future (model): ./output_videos/
    (action_view1.mp4 & action_view2.mp4 - updated each prediction)
  • Execution replay (robot): data/libero/eval_videos/.../
    (episode_XXX_success/failure.mp4 - saved after each episode)

💡 Predicted = model's imagination | Replay = actual execution

Enter replan steps:
  1-16: Execute this many steps from action chunk
  0: Use resampling (consecutive 0s increase temperature)
  Current resample level: 0 (temp=1.00)
```

**Resample Mechanism**:
- Input `0` consecutively to increase temperature: 1.0, 2.0, 3.0, ...
- Higher temperature = more diverse/exploratory actions
- Resets to 0 when entering 1-16

**Output Structure**:
```
./output_videos/
├── action_view1.mp4  (latest predicted trajectory - view 1)
└── action_view2.mp4  (latest predicted trajectory - view 2)

data/libero/eval_videos/
└── {suite}/
    └── task_{id}/
        ├── episode_000_success.mp4
        ├── episode_001_failure.mp4
        └── ...
```

### `image_tools.py` - Image Processing Utilities
**Description**: Handles LIBERO observation processing with 180° rotation and PIL conversion.

**Functions**:
- `pil_image_to_256_matrix()`: Load and resize image to 256×256
- `hwc_rgb_to_pil()`: Convert numpy array to PIL Image
- `process_libero_observation()`: Process LIBERO obs with rotation (matches training data)

## Usage

### 0. Verification Check (First Run)

**IMPORTANT**: Before running full evaluations, verify the FAST processor replacement was successful.

Run a quick test:
```shell
cd libero
python action_eval.py  # or hil_eval.py
```

**Expected Behavior on First Run**:
- The program should **hit a `pdb` breakpoint** and pause with a debugger prompt
- This breakpoint is located in `processing_action_tokenizer.py` (lines 98-99):
  ```python
  # Temporary debug breakpoint - remove after verifying the modification works correctly
  import pdb;pdb.set_trace()
  ```

**What This Means**:
- ✅ **If `pdb` appears**: FAST processor replacement succeeded! 
  - Type `c` and press Enter to continue
  - Exit the program (Ctrl+C)
  - Comment out the pdb lines in your FAST cache:
    ```python
    # import pdb;pdb.set_trace()  # Commented after verification
    ```
  - Or replace with the clean version without pdb

- ❌ **If `pdb` does NOT appear**: Replacement failed!
  - Double-check the FAST cache location (step 2 in Environment Modifications)
  - Verify you replaced the correct `processing_action_tokenizer.py` file
  - Retry the replacement and test again

**After Verification**:
Once you've confirmed the pdb breakpoint appears and commented it out, you can proceed with normal evaluations.

### 1. Automated Evaluation

Run batch evaluation with fixed replan steps:

```shell
cd libero
python action_eval.py
```

**Interactive Prompts**:
1. Select GPU: `0` (default)
2. Select suite: `libero_spatial` or `libero_object` or `libero_goal` or `libero_10` or `libero_90`
3. Select task ID: `0-9`
4. Select replan steps: `4-16` (recommended: 10)


### 2. Interactive HIL Evaluation

Run human-in-the-loop evaluation:

```shell
cd libero
python hil_eval.py
```

**Interactive Prompts**:
1. Select GPU: `0` (default)
2. Select suite and task (same as above)
3. **Per-action control**:
   - Check predicted videos: `./output_videos/action_view*.mp4`
   - Check execution replays: `data/libero/eval_videos/{suite}/task_{id}/`
   - Enter replan steps:
     - `1-16`: Use N steps from action chunk
     - `0`: Enable resampling (increases temperature)

**Tips**:
- Use replan_steps=`4-16` for most tasks
- Try replan_steps=`0` if actions seem repetitive/stuck
- Consecutive `0` inputs → higher temperature → more exploration

### 3. Understanding Video Outputs

**Predicted Videos** (`./output_videos/`):
- Generated by TiTok in HIL mode
- Show model's "imagination" of future 16-frame trajectory
- Two views: agentview (3rd person) + wrist camera
- Overwritten on each new prediction

**Execution Replay Videos** (`data/libero/eval_videos/`):
- Actual robot execution recorded during evaluation
- One video per episode (saved after completion)
- Labeled as `success` or `failure`
- Single agentview perspective


## Additional Notes

- **Horizon**: Default 16 frames (configurable in policy initialization)
- **View**: 2 cameras (agentview + wrist)
- **Action Dimension**: 7-DoF (6 joint + 1 gripper)
- **Replan Steps**: Determines how many predicted steps to execute before replanning

For more details on LIBERO tasks and benchmarks, see: https://github.com/Lifelong-Robot-Learning/LIBERO

