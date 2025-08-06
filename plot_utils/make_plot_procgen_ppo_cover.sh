python -m rlops \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impala_s1_high_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impala_s3_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impoola_s1_high&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impoola_s3_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impoola_s4_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=4$)' \
    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 3 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.65 \
    --rc.combined_figure \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen/ppo/cover/testing/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Testing" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5


for file in ./paper_plots/procgen/ppo/cover/testing/*.pdf; do pdfcrop $file $file;  done
python -m rlops \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impala_s1_high_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impala_s3_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impoola_s1_high&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impoola_s3_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impoola_s4_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=4$)' \
    --env-ids bigfish dodgeball caveflyer ninja  starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 3 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.99 \
    --rc.combined_figure \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen/ppo/cover/training/training \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Training" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5

for file in ./paper_plots/procgen/ppo/cover/training/*.pdf; do pdfcrop $file $file;  done