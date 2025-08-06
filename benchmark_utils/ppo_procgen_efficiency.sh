num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_efficiency_impoola_s2 OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --compile_agent --use_pooling_layer --env_track_setting efficiency" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer fruitbot heist coinrun bossfight

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_efficiency_impala_s2 OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --compile_agent --env_track_setting efficiency" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer fruitbot heist coinrun bossfight
