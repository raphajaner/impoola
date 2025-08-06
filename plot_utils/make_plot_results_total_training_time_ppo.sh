python -m rlops \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time' \
              'ppo_procgen?tag=ppo_baseline_s3_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense (Impala)' \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time' \
              'ppo_procgen?tag=ppo_baseline_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense' \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time' \
              'ppo_procgen?tag=ppo_redo_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=ReDo' \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time' \
              'ppo_procgen?tag=ppo_unstructured_s3_p80_a&seed=1&cl=Unstructured $\zeta_F=0.8$' \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time' \
              'ppo_procgen?tag=ppo_structured_s3_p80&seed=1&cl=Structured $\zeta_F=0.8$' \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=charts/elapsed_train_time' \
              'ppo_procgen?tag=ppo_group_s3_p80_no_all&seed=1&cl=Group-Structured $\zeta_F=0.8$' \
    --env-ids bigfish \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 4 \
    --metric_last_n_average_window 1 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.normalized_score_threshold 1.0 \
    --pc.ylabel "Total Training Times" \
    --output-filename paper_plots/results/ppo/train_time/train_time \
    --rc.nsubsamples 2000

for file in ./paper_plots/results/ppo/train_time/*.pdf; do pdfcrop $file $file;  done
