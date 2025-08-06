#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Training, \textit{easy})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Training, \textit{easy})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Testing, \textit{easy})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Testing, \textit{easy})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Training, \textit{hard})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Training, \textit{hard})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Testing, \textit{hard})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Testing, \textit{hard})' \
#    --env-ids bigfish  dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.9 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score (IQM)" \
#    --output-filename plotting/procgen/ppo/generalization/generalization \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5


#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Training, \textit{hard})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Testing, \textit{hard})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Training, \textit{hard})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Testing, \textit{hard})' \
#    --env-ids bigfish  dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.9 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/generalization/generalization_hard \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#
#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Training, \textit{easy})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Testing, \textit{easy})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Training, \textit{easy})' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Testing, \textit{easy})' \
#    --env-ids bigfish  dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.9 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/generalization/generalization_easy \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#
#
#for file in ./plotting/procgen/ppo/generalization/*.pdf; do pdfcrop $file $file;  done

python -m rlops \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impala_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Training, \textit{hard})' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impala_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (Testing, \textit{hard})' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impoola_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Training, \textit{hard})' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impoola_s2_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (Testing, \textit{hard})' \
    --env-ids caveflyer ninja coinrun jumper \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.9 \
    --rc.combined_figure \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen/ppo/generalization/generalization_selection \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Procgen" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5

for file in ./paper_plots/procgen/ppo/generalization/*.pdf; do pdfcrop $file $file;  done