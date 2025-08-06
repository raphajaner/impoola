python -m rlops \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impala_s2_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (\textit{easy})' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impala_s2_efficiency_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (\textit{hard})' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impoola_s2_efficiency&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (\textit{easy})' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
              'ppo_training?tag=ppo_impoola_s2_efficiency_hard_anneal&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola (\textit{hard})' \
    --env-ids bigfish  dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.85 \
    --rc.combined_figure \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen/ppo/efficiency/efficiency \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Procgen" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5

for file in ./paper_plots/procgen/ppo/efficiency/*.pdf; do pdfcrop $file $file;  done

#