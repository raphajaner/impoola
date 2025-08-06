python -m rlops \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impala_s2_pruned_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala w/ pruning' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impala_s2_moe_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala w/ MoE' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impala_s2_redo&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala w/ ReDo' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\textbf{ours}$)' \
    --env-ids bigfish dodgeball caveflyer ninja  starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.65 \
    --rc.combined_figure \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen/ppo/benchmark/testing/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Procgen" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5

for file in ./paper_plots/procgen/ppo/benchmark/testing/*.pdf; do pdfcrop $file $file;  done
#
#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s2_pruned_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala w/ pruning' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s2_moe_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala w/ MoE' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\textbf{ours}$)' \
#    --env-ids bigfish dodgeball caveflyer ninja  starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.9 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/benchmark/training/training \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#for file in ./plotting/procgen/ppo/benchmark/training/*.pdf; do pdfcrop $file $file;  done
#
#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Training)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Testing)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Training)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Testing)' \
#    --env-ids bigfish dodgeball caveflyer ninja  starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 2 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.9 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/benchmark/training_testing/training_testing \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#for file in ./plotting/procgen/ppo/benchmark/training_testing/*.pdf; do pdfcrop $file $file;  done


# Probability of improvement

#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_moe_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala w/ MoE' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\textbf{ours}$)' \
#    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 0.99 \
#    --rc.plot_probability_of_improvement \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/benchmark/probability_of_improvement/testing \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#for file in ./plotting/procgen/ppo/benchmark/probability_of_improvement/*.pdf; do pdfcrop $file $file;  done