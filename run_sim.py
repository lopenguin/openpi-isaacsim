import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from sim_evals.inference.droid_jointpos import Client as DroidJointPosClient

# Global flag
TRIGGER_PROMPT = False

def input_callback(event, *args):
    global TRIGGER_PROMPT
    import carb 
    import carb.input
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.P:
            TRIGGER_PROMPT = True

def main(
        headless: bool = False,
        scene: int = 1,
        ):
    # 1. Launch App
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Imports (Must happen after app launch)
    import sim_evals.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    import omni.appwindow
    import carb 
    import carb.input

    # Input Listener Setup
    input_interface = carb.input.acquire_input_interface()
    
    # Get the default OS window created by Omniverse
    app_window = omni.appwindow.get_default_app_window()
    
    keyboard = None
    if app_window:
        keyboard = app_window.get_keyboard()
        
    if keyboard:
        input_interface.subscribe_to_keyboard_events(keyboard, input_callback)
        print(">>> LISTENER ACTIVE: Press 'P' in the Isaac Sim window to enter a prompt.")
    else:
        print(">>> WARNING: No keyboard detected (Running headless?). Key press will not work.")

    # Initialize Env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    env_cfg.episode_length_s = 600.0
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset()

    print("Connecting to client...")
    client = DroidJointPosClient()

    # Video Setup
    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    
    instruction = None
    # action = None
    # Assuming action space is 1D vector per env
    zero_action = torch.zeros((env_cfg.scene.num_envs, env.action_space.shape[1]))

    if headless:
        print("TODO")

    else:
        print(">>> SIMULATION STARTED. Press 'P' in the viewer to give an instruction.")
        with torch.no_grad():
            global TRIGGER_PROMPT
            step_count = 0
            
            while simulation_app.is_running():
                step_count += 1

                # Check interrupt
                if TRIGGER_PROMPT:
                    TRIGGER_PROMPT = False
                    print("\n" + "="*40)
                    print(">>> SIMULATION PAUSED")
                    print("Example prompt: 'put the cube in the bowl'")
                    user_input = input(">>> Enter OpenPI Prompt (or press Enter to resume): ").strip()
                    if user_input:
                        instruction = user_input
                        print(f">>> New Instruction: '{instruction}'")
                        print("Wait for inference...")
                        # client.reset()
                    else:
                        print(">>> No instruction. Resuming previous state.")
                    print("="*40 + "\n")

                # Determine Action
                if instruction is not None:
                    ret = client.infer(obs, instruction)
                    video.append(ret["viz"])
                    action = torch.tensor(ret["action"])[None]
                else:
                    # stay in same position
                    # if action is None:
                    action = env.unwrapped.scene['robot'].data.joint_pos[0][:8][None] # this is a hack

                # Step Env
                obs, _, term, trunc, _ = env.step(action)
                
                # Save video on episode end
                if (term or trunc):
                    print(f"Episode end or reset after {env_cfg.episode_length_s} seconds elapsed.")
                    instruction = None
                    if len(video) > 0:
                        mediapy.write_video(video_dir / f"step_{step_count}.mp4", video, fps=15)
                        video = []
                    # client.reset()

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)