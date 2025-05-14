from videosys import OpenSoraConfig, VideoSysEngine
import argparse
import os
import torch
import yaml
from pathlib import Path
# prompts_1s = [
#     "a black dog wearing halloween costume", # animal
#     "an apartment building with balcony", # archi
#     "freshly baked finger looking cookies", # food
#     "people carving a pumpkin", # human
#     # "scenic video of sunset", # scenery
# ]

prompts_4s = [
    "a muffin with a burning candle and a love sign by a ceramic mug", # food
    "a group of friend place doing hand gestures of agreement", # human
    "aerial view of snow piles", # scenery
    "yacht sailing through the ocean", # vehicle
]

def save_ref_video(video, i, prompt):
    # Check if `video` is already a PyTorch tensor
    if not isinstance(video, torch.Tensor):
        # Convert to a PyTorch tensor if it's not already one
        video = torch.tensor(video)
    ref_video_folder = f"examples/open_sora/assets/4sx480p"
    # Save the video tensor to a .pt file
    os.makedirs(ref_video_folder, exist_ok=True)
    ref_video_path = os.path.join(ref_video_folder, f"{i}.pt")
    torch.save(video, ref_video_path)

    print(f"Video saved as {ref_video_path}")

def run_base(save_ref_videos=False, load_ea_timesteps=False):
    print(f"Running with save_ref_videos={save_ref_videos}, load_ea_timesteps={load_ea_timesteps}")
    # open-sora-plan v1.2.0
    # transformer_type (len, res): 93x480p 93x720p 29x480p 29x720p
    # change num_gpus for multi-gpu inference
    config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=1)
    engine = VideoSysEngine(config)

    ea_timesteps_list = []
    # prompts = ["In a still frame, a stop sign",]
    # seed=-1 means random seed. >0 means fixed seed.
    # # File path
    prompt_file_path = "/home/yuge/src/tmp/vbench_prompts/all_dimension_part1.txt"

    # Read all prompts
    with open(prompt_file_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
        # prompts = [line.strip() for i, line in enumerate(f.readlines()) if i % 16 == 0]

    save_videos_dir = "./outputs/os"
    os.makedirs(save_videos_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):

        if load_ea_timesteps:
            # Load YAML file
            ea_timesteps_path = "examples/open_sora/outputs/4sx480p_step15_search30_cache/ea_timesteps.yaml"
            with open(ea_timesteps_path, "r") as file:
                ea = yaml.safe_load(file)  # Use safe_load to avoid execution risks

            ea_timesteps_list = ea["ea_timesteps_list"]
        
        if ea_timesteps_list:
            # Original YAML file path
            ea_path = Path(ea_timesteps_path)

            # Construct new folder path
            videos_folder = ea_path.parent / "videos_vb900"

            # Create the folder if it doesn't exist
            videos_folder.mkdir(parents=True, exist_ok=True)
            for idx, ea_timesteps in enumerate(ea_timesteps_list):
                video = engine.generate(
                    prompt=prompt,
                    resolution="480p",
                    aspect_ratio="9:16",
                    num_frames="4s",
                    ea_timesteps=ea_timesteps,
                    seed=1024# -1,
                ).video[0]
                video_filename = f"{prompt}.mp4"  # Format index as 4 digits (e.g., 0000, 0001, etc.)
                video_save_path = os.path.join(videos_folder, video_filename)
                engine.save_video(video, video_save_path)
                print(f"Saved video with EA timesteps to {video_save_path}")
        else:
            print("Generating videos WITHOUT EA...")
            video = engine.generate(
                prompt=prompt,
                resolution="480p",
                aspect_ratio="9:16",
                num_frames="2s",
                seed=1024# -1,
            ).video[0]

            if save_ref_videos:
                print(f"Saving reference video {i} for prompt '{prompt}'")
                save_ref_video(video, i, prompt)

            video_filename = f"{prompt}.mp4"
            video_save_path = os.path.join(save_videos_dir, video_filename)
            engine.save_video(video, video_save_path)

def run_steps():
    # open-sora-plan v1.2.0
    # transformer_type (len, res): 93x480p 93x720p 29x480p 29x720p
    # change num_gpus for multi-gpu inference
    config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=1)
    engine = VideoSysEngine(config)

    timesteps_list = []
    prompts = ["A tranquil tableau of alley",]
    # seed=-1 means random seed. >0 means fixed seed.
    # # File path
    # prompt_file_path = "/home/yuge/src/tmp/all_dimension_part2.txt"

    # # Read all prompts
    # with open(prompt_file_path, "r") as f:
    #     prompts = [line.strip() for line in f.readlines()]

    save_videos_dir = "/home/yuge/src/tmp/os4s/gpu0"
    os.makedirs(save_videos_dir, exist_ok=True)

    timesteps_path = "examples/open_sora/full_timesteps.yaml"
    with open(timesteps_path, "r") as file:
        timesteps_data = yaml.safe_load(file)

    # Get all timestep configurations under "2s" node
    # == SPECIFY THE TIMESTEPS TYPE HERE == #
    timesteps = timesteps_data.get("4s", {})  # Returns empty dict if "2s" doesn't exist
    timesteps_dict = timesteps.items()

    for i, prompt in enumerate(prompts):
        # Iterate through each timestep configuration (first_20steps, last_20steps, etc.)
        for subnode_name, timesteps_list in timesteps_dict:
            video = engine.generate(
                prompt=prompt,
                resolution="480p",
                aspect_ratio="9:16",
                num_frames="4s",
                ea_timesteps=timesteps_list,
                seed=1024
            ).video[0]
            
            # Create filename with prompt and subnode name
            # safe_prompt = "".join(c for c in prompt if c.isalnum() or c in " _-")[:50]  # Sanitize prompt for filename, Kept: Letters, numbers, spaces.
            video_filename = f"{prompt}_{subnode_name}.mp4"
            video_save_path = os.path.join(save_videos_dir, video_filename)
            
            engine.save_video(video, video_save_path)
            print(f"Saved video with {subnode_name} timesteps to {video_save_path}")
            
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run base function with customizable parameters.')
    
    # Add arguments with default values matching your original function call
    parser.add_argument('--save_ref_videos', type=bool, default=False,
                       help='Whether to save reference videos (default: False)')
    parser.add_argument('--load_ea_timesteps', type=bool, default=False,
                       help='Whether to load EA timesteps (default: False)')
    args = parser.parse_args()
    
    # == RUN BASE == #
    run_base(save_ref_videos=args.save_ref_videos, 
             load_ea_timesteps=args.load_ea_timesteps)

    # # == RUN STEPS == #
    # run_steps()