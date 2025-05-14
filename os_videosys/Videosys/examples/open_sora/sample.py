from videosys import OpenSoraConfig, VideoSysEngine
import os
import torch

def load_prompts_from_file(file_path):
    """Reads prompts from a text file, one per line."""
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts

def run_base():
    # change num_gpus for multi-gpu inference
    # sampling parameters are defined in the config
    config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=1)
    engine = VideoSysEngine(config)

    import os
    import torch
    # prompts = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." # "Sunset over the sea."
    # seed=-1 means random seed. >0 means fixed seed.
    # Load prompts from the given file
    prompt_file_path = "/home/hsliu/vbench/all_dimension.txt" # "/home/yuge/src/Vbench/prompts/all_dimension.txt"
    prompts = load_prompts_from_file(prompt_file_path)
    save_videos_dir = "/home/hsliu/os_videosys/Videosys/outputs/os/ablationstudy/15timestep"
    os.makedirs(save_videos_dir, exist_ok=True)
    # num frames: 2s, 4s, 8s, 16s
    # resolution: 144p, 240p, 360p, 480p, 720p
    # aspect ratio: 9:16, 16:9, 3:4, 4:3, 1:1
    # seed=-1 means random seed. >0 means fixed seed.
    token_timesteps = [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0.3, 1. , 0. , 1. , 0. ,
       1. , 0. , 0.1, 0. , 0. , 1. , 0. , 0. , 0.1, 0. , 1. , 0. , 0. ,
       0. , 0.9, 0.1, 0.2]
    for i, prompt in enumerate(prompts):
        video = engine.generate(
            prompt=prompt,
            resolution="480p",
            aspect_ratio="9:16",
            num_frames="2s",
            seed=1024,# -1,
            token_timesteps=token_timesteps
        ).video[0]
        video_filename = f"{prompt}.mp4"  # Format index as 4 digits (e.g., 0000, 0001, etc.)
        save_path = os.path.join(save_videos_dir, video_filename)
        engine.save_video(video, save_path)

def run_low_mem():
    config = OpenSoraConfig(cpu_offload=True, tiling_size=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    config = OpenSoraConfig(enable_pab=True)
    engine = VideoSysEngine(config)

    prompt_file_path = "/home/yuge/src/Vbench/prompts/all_dimension.txt"
    prompts = load_prompts_from_file(prompt_file_path)

    # Prepare save folder
    save_videos_dir = "/home/yuge/src/Videosys/outputs/os/4sx480p_pab246"
    os.makedirs(save_videos_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        video = engine.generate(
            prompt=prompt,
            num_frames="4s",
            seed=1024
        ).video[0]

        # Save video
        video_filename = f"{prompt}.mp4"
        save_path = os.path.join(save_videos_dir, video_filename)
        engine.save_video(video, save_path)


if __name__ == "__main__":
    run_base()
    # run_low_mem()
    # run_pab()
