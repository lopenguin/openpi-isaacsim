# OpenPi in IsaacSim

The goal of this repository is to create a simple interface for running and prompting OpenPi in IsaacSim. It's built off of the fantastic [sim-evals](https://github.com/arhanjain/sim-evals) repo.

Here are some example rollouts of a pi0-FAST-DROID policy:

Scene 1

![Scene 1](./docs/scene1.gif)

Scene 2

![Scene 2](./docs/scene2.gif)

Scene 3

![Scene 3](./docs/scene3.gif)

The simulation is tuned to work *zero-shot* with DROID policies trained on the real-world DROID dataset, so no separate simulation data is required.

**Note:** The current simulator works best for policies trained with *joint position* action space (and *not* joint velocity control). We provide examples for evaluating pi0-FAST-DROID policies trained with joint position control below.


## Installation

Clone the repo
```bash
git clone --recurse-submodules git@github.com:lopenguin/openpi-isaacsim.git
cd sim-evals
```

Install uv (see: https://github.com/astral-sh/uv#installation)

For example (Linux/macOS):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate virtual environment
```bash
uv sync
source .venv/bin/activate
```

## Quick Start

First, make sure you download the simulation assets into the root of this directory
```bash
uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets
```

Then, in a separate terminal, launch the policy server on `localhost:8000`. For example, to launch a pi0-FAST-DROID policy (with joint position control), follow the instructions to install [openpi](https://github.com/Physical-Intelligence/openpi) in a separate uv (or docker) environment. We'll use the `polaris` configs. This repo will work best on a machine with two GPUs. Put openpi on your secondary GPU:
```bash
CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos
```

<details closed>

<summary><b>Run on a machine with a single GPU</b></summary>

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid_jointpos_polaris --policy.dir=gs://openpi-assets/checkpoints/pi05_droid_jointpos
```

**Note**: We set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5` to avoid JAX hogging all the GPU memory (incase Isaac Sim is using the same GPU).

</details>

Finally, return to this repo's terminal and run the simulation script:
```bash
python run_sim.py ---scene 1 --save-video
```

<details closed>

<summary><b>Headless mode</b></summary>
For a machine with fewer GPU resources, I recommend the headless mode:

```bash
python run_sim.py ---scene 1 --headless
```

</details>

You can pick scene 1, 2, or 3, and modify the scene in IsaacSim as desired before prompting OpenPi.