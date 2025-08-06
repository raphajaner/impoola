## LR sweep for tau=2
#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_high_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2, high lr$)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2, mid lr$)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2, lower mid lr$)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2, low lr$)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s2_random_no_anneal_very_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2, very low lr$)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_high_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2, high lr$)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2, mid lr$)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2, lower mid lr$)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2, low lr$)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_very_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2, very low lr$)' \
#    --env-ids bigfish dodgeball caveflyer ninja heist starpilot \
#    --no-check-empty-runs \
#    --pc.ncols 3 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.99 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/lr/testing/testing \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Testing" \
#    --metric_last_n_average_window 1
#
#for file in ./plotting/procgen/ppo/lr/testing/*.pdf; do pdfcrop $file $file;  done


# LR sweep for tau=2
python -m rlops \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impoola_s2_main&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2,same$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=2, mid lr$)' \
    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.99 \
    --rc.combined_figure \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen/ppo/more/testing/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Testing" \
    --metric_last_n_average_window 1

for file in ./paper_plots/procgen/ppo/more/testing/*.pdf; do pdfcrop $file $file;  done


#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s3_random_no_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, w/o lr)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s3_random_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, low lr)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s3_random_anneal_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, anneal low lr)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s1_random_lr_1e4&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$, lr 1e4)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s1_random_lr_5e4&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$, lr 5e4)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s1_random_lr_1e3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$, lr 1e3)' \
#    --env-ids bigfish starpilot dodgeball ninja caveflyer \
#    --no-check-empty-runs \
#    --pc.ncols 3 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.99 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/lr/training/training \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Training" \
#    --metric_last_n_average_window 1
#
#for file in ./plotting/procgen/ppo/lr/training/*.pdf; do pdfcrop $file $file;  done

# bossfight climber leaper jumper
# bigfish starpilot dodgeball ninja caveflyer coinrun fruitbot maze chaser heist plunder miner
#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s3_random_no_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, w/o lr)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s3_random_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, high lr, no anneal)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s3_random_no_anneal_high_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, high lr, no anneal 2)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s3_random_no_anneal_low_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, real low lr)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s1_random_lr_1e4&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$, lr 1e4)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s1_random_lr_5e4&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$, lr 5e4)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s1_random_lr_1e3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$, lr 1e3)' \
#    --env-ids bigfish starpilot dodgeball ninja \
#    --no-check-empty-runs \
#    --pc.ncols 3 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.99 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/lr/testing/testing \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Testing" \
#    --metric_last_n_average_window 1
#
#for file in ./plotting/procgen/ppo/lr/testing/*.pdf; do pdfcrop $file $file;  done


#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_training?tag=ppo_impala_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_training?tag=ppo_impoola_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_training?tag=ppo_impala_s3_random_no_lr&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$, w/o lr)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_test' \
#              'ppo_training?tag=ppo_impoola_s3_random_no_l&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, w/o lr)' \
#    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer  \
#    --no-check-empty-runs \
#    --pc.ncols 3 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.99 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/testing/testing \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Testing" \
#    --metric_last_n_average_window 1


#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_training?tag=ppo_impala_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/normalized_score_train' \
#              'ppo_training?tag=ppo_impoola_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$)' \
#    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer bossfight coinrun fruitbot climber leaper maze chaser heist plunder miner \
#    --no-check-empty-runs \
#    --pc.ncols 3 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 0.99 \
#    --rc.combined_figure \
#    --rc.aggregate_metrics_plots \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/training/training \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Training" \
#    --metric_last_n_average_window 1
#
#for file in ./plotting/procgen/ppo/training/*.pdf; do pdfcrop $file $file;  done

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

