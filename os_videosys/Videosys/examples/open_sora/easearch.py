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

import logging
from videosys import OpenSoraConfig, VideoSysEngine

prompts = [
    "a muffin with a burning candle and a love sign by a ceramic mug", # food
    "a group of friend place doing hand gestures of agreement", # humannvi
    "aerial view of snow piles", # scenery
    "yacht sailing through the ocean", # vehicle
]

def load_ref_videos(ref_videos_folder):
    ref_videos = []
    for i in range(4):
        video_path = os.path.join(ref_videos_folder, f"{i}.pt")
        video = torch.load(video_path)
        video_normalized = video.float() / 255.0
        ref_videos.append(video_normalized)
    return ref_videos

class EvolutionaryAlgorithm:
    def __init__(self, engine, population_size, max_compute, generations, ref_videos, fixed_timesteps, partial_num, initial_mutation_prob=0.1, final_mutation_prob=0.01, r_min=0.1, r_max=0.8):
        self.engine = engine
        self.population_size = population_size
        self.max_compute = max_compute  # 最大计算量约束
        self.generations = generations
        self.r_min = r_min
        self.r_max = r_max
        self.initial_mutation_prob = initial_mutation_prob
        self.final_mutation_prob = final_mutation_prob
        self.partial_num = partial_num
        self.fixed_timesteps = fixed_timesteps
        self.ref_videos = load_ref_videos(ref_videos_folder=ref_videos)
        self.seen_sequences = set()
        self.population = self.initialize_population()

    def is_new_individual(self, individual):
        sequence = individual['sequence']
        sequence_tuple = tuple(sequence)  # 将sequence转换为不可变的tuple以便存入set
        if sequence_tuple in self.seen_sequences:
            return False
        self.seen_sequences.add(sequence_tuple)
        return True

    def initialize_population(self):
        population = []
        for i in range(5):
            print(f"Generating individual {i}")
            individual = self.random_valid_individual()
            population.append(individual)
        return population

    def random_valid_individual(self):
        r_candidates = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        max_attempt = 100
        while True:
            # sequence = np.random.choice(['0', '1', 'X'], partial_num, p=[0.4, 0.3, 0.3])
            sequence = np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], self.partial_num)
            while len(sequence) < 30:
                insert_pos = random.randint(0, len(sequence))
                sequence = np.insert(sequence, insert_pos, 0)
            for i in range(max_attempt):
                sequence = np.array(sequence)
                individual_sequence = np.array(self.fixed_timesteps, dtype=float)
                indices = np.where(individual_sequence != 1)[0]
                individual_sequence[indices] = sequence
                # individual_sequence[individual_sequence != 1] = sequence
                individual = {'sequence': individual_sequence, 'selected_index': indices}
                # individual = {'sequence': sequence, 'r': r}
                if self.max_compute-10 <= self.compute_cost(individual) <= self.max_compute:
                    return individual

    def compute_cost(self, individual):
        return sum(individual['sequence'])
        # sequence = individual['sequence']
        # r = individual['r']
        # # K = sum(1 for x in sequence if x == 1)
        # # X = sum(1 for x in sequence if x == 0)
        # K = 0
        # X = 0
        # for x in sequence:
        #     if x == '1':
        #         K += 1
        #     elif x == '0':
        #         X += 1
        # return K * r + X

    def fitness(self, individual):
        return self.get_cand_mse(individual)

    def get_cand_mse(self, individual=None, device='cuda'):
        mse_scores = []
        print("Individual:", individual)
        # new_individual = np.array([1.0 if x == '0' else individual['r'] if x == '1' else 0.0 for x in individual['sequence']], dtype=float)
        # print("New Individual:", new_individual)
        for i, prompt in enumerate(prompts):
            cand_video = self.engine.generate(
            prompt=prompt,
            resolution="480p",
            aspect_ratio="9:16",
            num_frames="2s",
            seed=1024,# -1,
            token_timesteps=individual['sequence'],
            ).video[0]
            cand_video_float = cand_video.float() / 255.0
            ref_video_float = self.ref_videos[i] # normalized alrd
            mse_loss = F.mse_loss(cand_video_float, ref_video_float)
            mse_scores.append(mse_loss.item())
        
        mean_mse = np.mean(mse_scores)
        print("Mean MSE Loss:", mean_mse)
        return mean_mse

    def crossover(self, parent1, parent2, generation):
        if random.random() < (0.8 if generation < self.generations / 2 else 0.4):
            return self.uniform_crossover(parent1, parent2, generation)
        else:
            return self.block_crossover(parent1, parent2)


    # 需要思考child的r如何继承
    def uniform_crossover(self, parent1, parent2, generation):
        swap_prob = 0.7 if generation < self.generations / 2 else 0.3
        mask = np.random.rand(30) < swap_prob
        child_sequence = np.where(mask, parent1['sequence'], parent2['sequence'])
        child = {'sequence': child_sequence, 'selected_index': parent1['selected_index']} 
        if self.compute_cost(child) > self.max_compute or self.compute_cost(child) < self.max_compute - 1:
            self.repair(child)
        return child

    def block_crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(30), 2))
        child_sequence = np.copy(parent1['sequence'])
        child_sequence[start:end] = parent2['sequence'][start:end]
        child = {'sequence': child_sequence, 'selected_index': parent1['selected_index']}
        if self.compute_cost(child) > self.max_compute or self.compute_cost(child) < self.max_compute - 1:
            self.repair(child)
        return child

    def repair(self, individual):
        while self.compute_cost(individual) > self.max_compute:
            idx = random.choice(individual['selected_index'])
            individual['sequence'][idx] = max(0.1, round(individual['sequence'][idx] * 0.8, 1))
        while self.compute_cost(individual) < self.max_compute - 1:
            idx = random.choice(individual['selected_index'])
            individual['sequence'][idx] = min(0.9, individual['sequence'][idx] + 0.1)

    def mutate(self, individual, generation):
        mutation_prob = self.initial_mutation_prob - (generation / self.generations) * (self.initial_mutation_prob - self.final_mutation_prob)
        
        for i in individual['selected_index']:
            if random.random() < mutation_prob:
                # individual['sequence'][i] = random.uniform(0.1, 0.9)
                individual['sequence'][i] = round(random.uniform(0.1, 0.99), 1)
        if self.compute_cost(individual) > self.max_compute or self.compute_cost(individual) < self.max_compute - 1:
            self.repair(individual)
        return individual

    def evolve(self):
        for individual in self.population:
            self.is_new_individual(individual)
            selected_index = individual['selected_index']
            individual['mse'] = self.fitness(individual)
        sequence = [1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        tmp = {'sequence': sequence, 'selected_index': selected_index}
        self.is_new_individual(tmp)
        tmp['mse'] = self.fitness(tmp)
        self.population.append(tmp)
        # for individual in self.population:
        #     individual['mse'] = self.fitness(individual)
        self.population = sorted(self.population, key=lambda individual: individual['mse'], reverse=False)[:self.population_size]
        for generation in range(self.generations):
            logging.info('Generation %d', generation)
            new_population = [] 
            i = 0
            # for _ in range(self.population_size // 2):
            while i <= self.population_size // 2:
                i += 1
                # parent1, parent2 = random.sample(self.population, 2)
                weights = [1 / (rank + 1) for rank in range(len(self.population))]  # 权重与排名成反比
                parent1, parent2 = random.choices(self.population, weights=weights, k=2)
    
                child = self.crossover(parent1, parent2, generation)
                child = self.mutate(child, generation)
                self.repair(child)
                if self.compute_cost(child) <= self.max_compute and self.compute_cost(child) >= self.max_compute - 1 and self.is_new_individual(child):
                    child['mse'] = self.fitness(child)
                    new_population.append(child)
            combined_population = self.population + new_population
            for individual in combined_population:
                if 'mse' not in individual:
                    individual['mse'] = self.fitness(individual)
            self.population = sorted(combined_population, key=lambda individual: individual['mse'], reverse=False)[:self.population_size]
            for individual in self.population:
                logging.info('Individual: %s, Cost %s, Fitness %s', individual, self.compute_cost(individual), individual['mse'])


if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format the timestamp
    outpath = '/home/hsliu/Videosys-ours/outputs/easearch/os/2sx480p'
    os.makedirs(outpath, exist_ok=True)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(outpath, f"log.txt_{timestamp}"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


    print("start")
    t = time.time()
    config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=1)
    engine = VideoSysEngine(config)
    ref_videos_folder = "/data/yuge/ref_video/os/2sx480p"
    # ref_videos_folder = "/home/hsliu/Videosys-ours/pt"
    fixed_timesteps = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    partial_num = 10
    ea = EvolutionaryAlgorithm(engine=engine, population_size=50, max_compute=4, generations=100, ref_videos=ref_videos_folder, fixed_timesteps=fixed_timesteps, partial_num=partial_num)
    ea.evolve()
    logging.info('total searching time = {:.2f} hours'.format((time.time() - t) / 3600))

# 去重
# 修改r的取值逻辑