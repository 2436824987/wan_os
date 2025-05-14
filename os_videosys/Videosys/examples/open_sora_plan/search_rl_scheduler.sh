PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
python examples/open_sora_plan/search_ea.py \
--outdir 'examples/open_sora_plan/outputs/93x480p_step50_search100_cache' \
--n_samples 6 \
--num_sample 1000 \
--time_step 50 \
--max_epochs 30 \
--population_num 50 \
--mutation_num 20 \
--crossover_num 15 \
--seed 1024 \
--use_ddim_init_x false \
--ref_videos 'examples/open_sora_plan/assets/93x480p' \
--ref_sigma '/home/yfeng/ygcheng/src/AutoDiffusion/assets/coco2014_sigma.npy' \
--load_log_path 'examples/open_sora_plan/outputs/93x480p_step50_search100_cache/log.txt_2025-04-01_16-44-52' \