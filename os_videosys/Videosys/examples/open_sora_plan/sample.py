from videosys import OpenSoraPlanConfig, VideoSysEngine
prompts = [
    "a black dog wearing halloween costume", # animal
    "an apartment building with balcony", # archi
    "freshly baked finger looking cookies", # food
    "people carving a pumpkin", # human
    "scenic video of sunset", # scenery
]

def load_prompts_from_file(file_path):
    """Reads prompts from a text file, one per line."""
    with open(file_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts

def run_base():
    # open-sora-plan v1.2.0
    # transformer_type (len, res): 93x480p 93x720p 29x480p 29x720p
    # change num_gpus for multi-gpu inference
    config = OpenSoraPlanConfig(version="v120", transformer_type="29x720p", num_gpus=1)
    engine = VideoSysEngine(config)

    import os
    import torch
    # prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about." # "Sunset over the sea."
    # seed=-1 means random seed. >0 means fixed seed.
    # Load prompts from the given file
    prompt_file_path = "/home/yuge/src/Vbench/prompts/all_dimension.txt"
    prompts = load_prompts_from_file(prompt_file_path)
    save_videos_dir = "./outputs/vb900/93x480p"
    os.makedirs(save_videos_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        video = engine.generate(
            prompt=prompt,
            guidance_scale=7.5,
            num_inference_steps=100,
            seed=1024,
        ).video[0]

        # # Check if `video` is already a PyTorch tensor
        # if not isinstance(video, torch.Tensor):
        #     # Convert to a PyTorch tensor if it's not already one
        #     video = torch.tensor(video)
        # ref_video_folder = f"/home/yfeng/ygcheng/src/VideoSys/examples/open_sora_plan/assets/ref_videos"
        # # Save the video tensor to a .pt file

        # ref_video_path = os.path.join(ref_video_folder, f"{i}.pt")
        # torch.save(video, ref_video_path)

        # print(f"Video saved as {ref_video_path}")

        # prompt_suffix = prompt[:20] if len(prompt) > 20 else prompt
        # engine.save_video(video, f"./outputs/{prompt_suffix}.mp4")
        video_filename = f"{prompt}.mp4"  # Format index as 4 digits (e.g., 0000, 0001, etc.)
        save_path = os.path.join(save_videos_dir, video_filename)
        engine.save_video(video, save_path)

def run_low_mem():
    config = OpenSoraPlanConfig(cpu_offload=True, enable_tiling=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    config = OpenSoraPlanConfig(enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_v110():
    # open-sora-plan v1.1.0
    # transformer_type: 65x512x512 or 221x512x512
    # change num_gpus for multi-gpu inference
    config = OpenSoraPlanConfig(version="v110", transformer_type="65x512x512", num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # seed=-1 means random seed. >0 means fixed seed.
    video = engine.generate(
        prompt=prompt,
        guidance_scale=7.5,
        num_inference_steps=150,
        seed=-1,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
    # run_low_mem()
    # run_pab()
    # run_v110()
