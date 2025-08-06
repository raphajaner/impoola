# LR sweep for tau=2
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'dqn_training?tag=dqn_impala_s2_random_no_anneal_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2, low lr$)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'dqn_training?tag=dqn_impala_s2_random_no_anneal_very_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2, very low lr$)' \

#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'dqn_training?tag=dqn_impala_s2_random_no_anneal_high_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2, high lr$)' \

python -m rlops \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'dqn_training?tag=dqn_impala_s2_random_no_anneal_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2, mid lr$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'dqn_training?tag=dqn_impala_s2_random_no_anneal_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2, low lr$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'dqn_training?tag=dqn_impala_s2_random_no_anneal_very_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2, very low lr$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'dqn_training?tag=dqn_impoola_s2_random_no_anneal_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2, mid lr$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'dqn_training?tag=dqn_impoola_s2_random_no_anneal_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2, low lr$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'dqn_training?tag=dqn_impoola_s2_random_no_anneal_very_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2, very low lr$)' \
    --env-ids bigfish dodgeball caveflyer ninja \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.99 \
    --rc.combined_figure \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen/dqn/lr/testing/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Testing" \
    --metric_last_n_average_window 1

for file in ./paper_plots/procgen/dqn/lr/testing/*.pdf; do pdfcrop $file $file;  done