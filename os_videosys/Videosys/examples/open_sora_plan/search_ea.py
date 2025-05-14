import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
import sys
import time
from datetime import datetime

import argparse
import os
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import collections
sys.setrecursionlimit(10000)
import functools

import argparse
import os
from torch import autocast
from contextlib import contextmanager, nullcontext
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from ldm.data.build_dataloader import build_dataloader

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
import logging
from torch.nn.functional import adaptive_avg_pool2d
# from mmengine.runner import set_random_seed
# from pytorch_lightning import seed_everything
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like
# from pytorch_fid.inception import InceptionV3
import copy

from EvolutionSearcher import EvolutionSearcher

# Videosys:Open-Sora-Plan related imports
from videosys import OpenSoraPlanConfig, VideoSysEngine

def str2bool(value):
    """Convert string to boolean."""
    if value.lower() in ['true', '1', 't', 'y', 'yes']:
        return True
    elif value.lower() in ['false', '0', 'f', 'n', 'no']:
        return False
    else:
        raise ValueError("Invalid value for boolean ", value)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     default="/home/yfeng/ygcheng/src/Open-Sora/configs/opensora-v1-2/inference/sample_ea.py", # TODO
    #     help="path to config which constructs model",
    # )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="path to data",
    )
    parser.add_argument(
        "--num_sample",
        type=int,
        default=4,
        help="samples num",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--select_num",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--population_num",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--m_prob",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--crossover_num",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--mutation_num",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--ref_videos",
        type=str,
        default='',
    )
    parser.add_argument(
        "--ref_sigma",
        type=str,
        default='',
    )
    parser.add_argument(
        "--time_step",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--use_ddim_init_x",
        type=str2bool, # the parser does not automatically convert strings like 'false' or 'true' into actual boolean values (False or True).
        default=False,
    )
    parser.add_argument(
        "--load_log_path",
        type=str,
        default='',
    )

    opt = parser.parse_args()

    print("opt arguments loaded")
    # == device ==
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # == init logger ==
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format the timestamp
    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(outpath, f"log.txt_{timestamp}"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # TODO: Load and build models
    # ======================================================
    # Integrate Open-Sora-Plan Configurations
    # ======================================================
    config = OpenSoraPlanConfig(version="v120", transformer_type="93x480p", num_gpus=4)
    engine = VideoSysEngine(config)
    print("engine initialized")

    # TODO: Add ea in OSP schedulers

    # if opt.dpm_solver:
    #     tmp_sampler = DPMSolverSampler(model)
    #     from ldm.models.diffusion.dpm_solver.dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
    #     ns = NoiseScheduleVP('discrete', alphas_cumprod=tmp_sampler.alphas_cumprod)
    #     dpm_solver = DPM_Solver(None, ns, predict_x0=True, thresholding=False)
    #     skip_type = "time_uniform"
    #     t_0 = 1. / dpm_solver.noise_schedule.total_N  # 0.001
    #     t_T = dpm_solver.noise_schedule.T  # 1.0
    #     full_timesteps = dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=1000, device='cpu')
    #     init_timesteps = dpm_solver.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=opt.time_step, device='cpu')
    #     dpm_params = dict()
    #     full_timesteps = list(full_timesteps)
    #     dpm_params['full_timesteps'] = [full_timesteps[i].item() for i in range(len(full_timesteps))]
    #     init_timesteps = list(init_timesteps)
    #     dpm_params['init_timesteps'] = [init_timesteps[i].item() for i in range(len(init_timesteps))]
    # else:
    dpm_params = None

    ## build EA
    t = time.time()
    searcher = EvolutionSearcher(opt=opt, engine=engine, time_step=opt.time_step, ref_videos=opt.ref_videos, ref_sigma=opt.ref_sigma, device=device, dpm_params=dpm_params)
    logging.info("Integrated Open-Sora-Plan Successfully ......")

    searcher.search()
    logging.info('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))

if __name__ == '__main__':
    main()
