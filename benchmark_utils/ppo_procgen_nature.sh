num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_nature_s1 OMP_NUM_THREADS=1 python ppo_training.py --scale=1 --learning-rate 0.0005 --compile_agent --encoder_type nature --cnn_filters 16 32 32" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola_final \
    --env-ids bigfish dodgeball caveflyer ninja heist starpilot coinrun chaser fruitbot jumper leaper climber plunder miner bossfight maze

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_nature_s2 OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --learning-rate 0.0005 --compile_agent --encoder_type nature --cnn_filters 16 32 32" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola_final \
    --env-ids bigfish dodgeball caveflyer ninja heist starpilot coinrun chaser fruitbot jumper leaper climber plunder miner bossfight maze

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_nature_s3 OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --learning-rate 0.0005 --compile_agent --encoder_type nature --cnn_filters 16 32 32" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola_final \
    --env-ids bigfish dodgeball caveflyer ninja heist starpilot coinrun chaser fruitbot jumper leaper climber plunder miner bossfight maze

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_nature_s2_pool OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --learning-rate 0.0005 --compile_agent --encoder_type nature --cnn_filters 16 32 32 --use_pooling_layer" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish dodgeball caveflyer ninja heist starpilot coinrun chaser fruitbot jumper leaper climber plunder miner bossfight maze

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_nature_s3_pool OMP_NUM_THREADS=1 python ppo_training.py --scale=3 --learning-rate 0.0005 --compile_agent --encoder_type nature --cnn_filters 16 32 32 --use_pooling_layer " \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish dodgeball caveflyer ninja heist starpilot coinrun chaser fruitbot jumper leaper climber plunder miner bossfight maze


