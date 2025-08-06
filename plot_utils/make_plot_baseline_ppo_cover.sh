#python -m rlops \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_baseline_s1_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense Impala ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_baseline_s1&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense Impoola ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_baseline_s3_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense Impala ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_baseline_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense Impoola ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p80_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured Impala \n ($\tau=3$, $\zeta_F=0.8$)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p80_a&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured Impoola \n ($\tau=3$, $\zeta_F=0.8$)' \
#    --env-ids bigfish starpilot dodgeball bossfight \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 3 \
#    --metric_last_n_average_window 1 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 0.85 \
#    --rc.normalized_score_threshold_min 0.4 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/baseline/ppo/cover/training/training \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "" \
#
#
#for file in ./plotting/baseline/ppo/cover/training/*.pdf; do pdfcrop $file $file;  done
#
#python -m rlops \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_baseline_s1_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense Impala ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_baseline_s1&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense Impoola ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_baseline_s3_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense Impala ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_baseline_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense Impoola ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p80_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured Impala \n ($\tau=3$, $\zeta_F=0.8$)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p80_a&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured Impoola \n ($\tau=3$, $\zeta_F=0.8$)' \
#    --env-ids bigfish starpilot dodgeball bossfight \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 3 \
#    --metric_last_n_average_window 1 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 0.85 \
#    --rc.normalized_score_threshold_min 0.4 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/baseline/ppo/cover/testing/testing \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "" \


for file in ./paper_plots/baseline/ppo/cover/testing/*.pdf; do pdfcrop $file $file;  done

python -m rlops \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_procgen?tag=ppo_baseline_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola' \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_procgen?tag=ppo_baseline_s3_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala' \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_procgen?tag=ppo_baseline_s3_no_pool&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (w/ wd, w/ lr)' \
    --env-ids bigfish starpilot dodgeball bossfight \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 3 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.normalized_score_threshold 0.65 \
    --rc.combined_figure \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/baseline/ppo/cover/additional/testing/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "($\tau$ = 3)"

for file in ./paper_plots/baseline/ppo/cover/additional/testing/*.pdf; do pdfcrop $file $file;  done
