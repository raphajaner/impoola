python -m rlops \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impala_s1&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impoola_s1&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impala_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impoola_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impala_s5&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=5$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impoola_s5&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=5$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impala_s3_pruned&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, pruned)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impoola_s3_pruned&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, pruned)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impala_s3_no_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, no lr anneal)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impoola_s3_no_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, no lr anneal)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impoola_s3_latent512&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, latent 512)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impoola_s3_latent1024&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, latent 1024)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impoola_s3_pool2&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, pool 2)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impoola_s3_moe&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, MoE)' \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.99 \
    --rc.combined_figure \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/baseline/ppo/testing/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Testing" \
    --metric_last_n_average_window 1

for file in ./paper_plots/baseline/ppo/testing/*.pdf; do pdfcrop $file $file;  done

python -m rlops \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impala_s1&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impoola_s1&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impala_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impoola_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impala_s5&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=5$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impoola_s5&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=5$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impala_s3_pruned&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, pruned)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impoola_s3_pruned&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, pruned)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impala_s3_no_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, no lr anneal)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impoola_s3_no_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, no lr anneal)' \
     --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impoola_s3_latent512&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, latent 512)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
              'ppo_training?tag=ppo_impoola_s3_latent1024&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, latent 1024)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impoola_s3_pool2&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, pool 2)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
              'ppo_training?tag=ppo_impoola_s3_moe&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, MoE)' \
    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.normalized_score_threshold 0.99 \
    --rc.combined_figure \
    --rc.aggregate_metrics_plots \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/baseline/ppo/training/training \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Training" \
    --metric_last_n_average_window 1


#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_training?tag=ppo_impoola_s3_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, lr anneal)' \

for file in ./paper_plots/baseline/ppo/training/*.pdf; do pdfcrop $file $file;  done

#python -m rlops \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_baseline_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense (Impoola)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p80_a&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured $\zeta_F=0.8$ (Impoola)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p90&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured $\zeta_F=0.9$ (Impoola)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_baseline_s3_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense (Impala)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p80_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured $\zeta_F=0.8$ (Impala)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p90_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured $\zeta_F=0.9$ (Impala)' \
#    --env-ids bigfish starpilot dodgeball bossfight \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 2 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 0.9 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/baseline/ppo/training/training \
#    --rc.nsubsamples 10
#for file in ./plotting/baseline/ppo/training/*.pdf; do pdfcrop $file $file;  done
#
#python -m rlops \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_baseline_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense (Impoola)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p80_a&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured $\zeta_F=0.8$ (Impoola)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p90&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured $\zeta_F=0.9$ (Impoola)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_baseline_s3_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Dense (Impala)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p80_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured $\zeta_F=0.8$ (Impala)' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_unstructured_s3_p90_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Unstructured $\zeta_F=0.9$ (Impala)' \
#    --env-ids bigfish starpilot dodgeball bossfight \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 2 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 0.9 \
#    --rc.combined_figure \
#    --rc.performance_profile_plots  \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/baseline/ppo/testing/testing \
#    --rc.nsubsamples 10
#for file in ./plotting/baseline/ppo/testing/*.pdf; do pdfcrop $file $file;  done


#python -m rlops \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_baseline_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_baseline_s3_impala&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_procgen?tag=ppo_baseline_s3_no_pool&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala (w/ wd, w/ lr)' \
#    --env-ids bigfish starpilot dodgeball bossfight \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 3 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 0.65 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/baseline/ppo/additional/testing/testing \
#    --rc.nsubsamples 10
#for file in ./plotting/baseline/ppo/additional/testing/*.pdf; do pdfcrop $file $file;  done

