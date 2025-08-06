#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola' \
#    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.9 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score (IQM)" \
#    --output-filename plotting/procgen_additional/ppo/hard/training \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola' \
#    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.58 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score (IQM)" \
#    --output-filename plotting/procgen_additional/ppo/hard/testing \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#for file in ./plotting/procgen_additional/ppo/hard/*.pdf; do pdfcrop $file $file;  done

#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s2_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau$=2)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s2_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau$=2)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s3_efficiency_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau$=3)' \
#    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.83 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score (IQM)" \
#    --output-filename plotting/procgen_additional/ppo/efficiency/training \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#for file in ./plotting/procgen_additional/ppo/efficiency/*.pdf; do pdfcrop $file $file;  done

python -m rlops \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impala_s2_efficiency_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau$=2)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impoola_s2_efficiency_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau$=2)' \
    --env-ids bigfish dodgeball caveflyer ninja  starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.83 \
    --rc.combined_figure \
    --pc.ylabel "Normalized Score (IQM)" \
    --output-filename paper_plots/procgen_additional/ppo/efficiency_hard/training \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Procgen" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5

for file in ./paper_plots/procgen_additional/ppo/efficiency_hard/*.pdf; do pdfcrop $file $file;  done