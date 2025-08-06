num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# 100 % centered envs: ninja, jumper, caveflyer, coinrun
# Others: fruitbot, climber, leaper, maze, chaser, heist, plunder, miner, bigfish, starpilot, dodgeball, bossfight

# Scaling tau=1
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s1 OMP_NUM_THREADS=1 python ppo_training.py --scale=1 --compile_agent" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s1 OMP_NUM_THREADS=1 python ppo_training.py --scale=1 --compile_agent --use_pooling_layer" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner

# Scaling tau=2
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2 OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --compile_agent --use_pooling_layer" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2 OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --compile_agent --use_pooling_layer" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner

# Scaling with tau=3
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s3 OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --compile_agent" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s3 OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --compile_agent --use_pooling_layer" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner

# Scaling with tau=4
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s4 OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --compile_agent" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s4 OMP_NUM_THREADS=1 python ppo_training.py --scale=4 --compile_agent --use_pooling_layer" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner
