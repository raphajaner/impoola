num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_deeper OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --compile_agent --no-anneal_lr --cnn_filters 16 32 32 32" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola_final \
    --env-ids ninja bigfish dodgeball caveflyer starpilot heist plunder chaser coinrun fruitbot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_depth_conv OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --compile_agent --use_depthwise_conv" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola_final \
    --env-ids plunder chaser coinrun fruitbot

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impoola_s2_pool_2 OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --compile_agent --use_pooling_layer --pooling_layer_kernel_size 2" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola_final \
    --env-ids plunder chaser coinrun fruitbot
#
python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_pruned OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --pruning_type UnstructuredNorm" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola_final \
    --env-ids coinrun fruitbot climber maze bossfight plunder leaper chaser jumper miner

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_impala_s2_moe_random OMP_NUM_THREADS=1 python ppo_training.py --scale=2 --use_moe" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name impoola_final \
    --env-ids coinrun fruitbot climber maze bossfight plunder leaper chaser jumper miner

python -m benchmark_utils.benchmark \
    --command "WANDB_TAGS=ppo_s2_redo OMP_NUM_THREADS=1 python ppo_procgen.py --scale=2 --pruning-type ReDo --redo_interval 100" \
    --start_seed 1 \
    --num-seeds 5 \
    --workers $num_gpus \
    --no-auto-tag \
    --wandb_project_name pruning4drl_results \
    --env-ids coinrun fruitbot climber maze bossfight plunder leaper chaser jumper miner
