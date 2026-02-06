import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import json

from sim_evals.inference.droid_jointpos import Client as DroidJointPosClient

# Global flag
TRIGGER_PROMPT = False
QUIT = False
RESET = False

def input_callback(event, *args):
    global TRIGGER_PROMPT
    global QUIT
    global RESET
    import carb 
    import carb.input
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.P:
            TRIGGER_PROMPT = True
        elif event.input == carb.input.KeyboardInput.Q:
            QUIT = True
        elif event.input == carb.input.KeyboardInput.R:
            RESET = True


def main(
        headless: bool = False,
        scene: int = 1,
        max_steps: int = 450,
        save_video: bool = False,
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
        
    if (not headless) and keyboard:
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
    # Assuming action space is 1D vector per env
    zero_action = torch.zeros((env_cfg.scene.num_envs, env.action_space.shape[1]))

    if headless:
        match scene:
            case 1:
                instruction = "put the cube in the bowl"
            case 2:
                instruction = "put the can in the mug"
            case 3:
                instruction = "put banana in the bin"
            case _:
                raise ValueError(f"Scene {scene} not supported")
        
        # run once with default prompt
        for _ in tqdm(range(max_steps), desc=f"Running headless"):
            ret = client.infer(obs, instruction)
            # if not headless:
            #     cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
            #     cv2.waitKey(1)
            video.append(ret["viz"])
            action = torch.tensor(ret["action"])[None]
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

        client.reset()
        mediapy.write_video(
            video_dir / f"episode_1.mp4",
            video,
            fps=15,
        )
        video = []

    else:
        print(">>> SIMULATION STARTED. Press 'P' in the viewer to give an instruction, R to reset, and Q to quit.")
        log = dict()
        with torch.no_grad():
            global TRIGGER_PROMPT
            global QUIT
            global RESET
            step_count = 0
            start_step = 0
            
            while simulation_app.is_running():
                step_count += 1

                # Check interrupt
                if TRIGGER_PROMPT:
                    if len(video) > 0:
                        log[step_count] = instruction
                        mediapy.write_video(video_dir / f"step_{step_count}.mp4", video, fps=15)
                        video = []
                    TRIGGER_PROMPT = False
                    print("\n" + "="*40)
                    print(">>> SIMULATION PAUSED")
                    print("Example prompt: 'put the cube in the bowl'")
                    user_input = input(">>> Enter OpenPI Prompt (or press Enter to resume): ").strip()
                    if user_input:
                        instruction = user_input
                        print(f">>> New Instruction: '{instruction}'")
                        print("Wait for inference...")
                        start_step = step_count
                        client.reset()
                    else:
                        print(">>> No instruction. Resuming previous state.")
                    print("="*40 + "\n")

                # Determine Action
                if instruction is not None:
                    ret = client.infer(obs, instruction)
                    if save_video:
                        video.append(ret["viz"])
                    action = torch.tensor(ret["action"])[None]
                else:
                    # stay in same position
                    action = env.unwrapped.scene['robot'].data.joint_pos[0][:8][None] # this is a hack

                # Step Env
                obs, _, term, trunc, _ = env.step(action)
                
                # Save video on episode end
                if (term or trunc or (instruction and (step_count - start_step > max_steps))) or RESET:
                    print(f"Episode end after {step_count - start_step} steps or reset after {env_cfg.episode_length_s} seconds elapsed.")
                    if len(video) > 0:
                        log[step_count] = instruction
                        mediapy.write_video(video_dir / f"step_{step_count}.mp4", video, fps=15)
                        video = []
                    instruction = None
                    client.reset()
                    if RESET:
                        print("Reseting...")
                        client.reset()
                        RESET = False
                
                if QUIT:
                    print("Quitting...")
                    if save_video:
                        if len(video) > 0:
                            log[step_count] = instruction
                            mediapy.write_video(video_dir / f"step_{step_count}.mp4", video, fps=15)
                        with open(video_dir / f"log.json", 'w') as f:
                            json.dump(log, f, indent=4)
                            print("here")
                    break

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    args = tyro.cli(main)