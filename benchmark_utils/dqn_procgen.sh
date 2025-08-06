um_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Impoola
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=dqn_impoola_s1 OMP_NUM_THREADS=1 python dqn_training.py --scale=1 --compile_agent --use_pooling_layer" \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner

  python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=dqn_impoola_s2 OMP_NUM_THREADS=1 python dqn_training.py --scale=2 --compile_agent --use_pooling_layer" \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=dqn_impoola_s3 OMP_NUM_THREADS=1 python dqn_training.py --scale=3 --compile_agent --use_pooling_layer" \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner

