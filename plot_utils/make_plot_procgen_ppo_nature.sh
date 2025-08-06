#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_nature_s1&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_nature_s2&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=2$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_nature_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_nature_s2_pool_new&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=2$, w/ GAP)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_nature_s3_pool_more&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=3$, w/ GAP)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_nature_s4_pool_more&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=4$, w/ GAP)' \
#    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder fruitbot \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 4 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.6 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/nature/training/training \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#for file in ./plotting/procgen/ppo/nature/training/*.pdf; do pdfcrop $file $file;  done


#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_nature_s1&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_nature_s2&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=2$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_nature_s3&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_nature_s2_pool_new&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=2$, w/ GAP)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_nature_s3_pool_more&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=3$, w/ GAP)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_nature_s4_pool_more&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=4$, w/ GAP)' \


python -m rlops \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_nature_plus_s2_more&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=2$ + 1 Conv)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_nature_plus_s2_pool_more&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=2$ + 1 Conv, w/ GAP)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
              'ppo_training?tag=ppo_nature_plus_s3_pool_more&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=3$ + 1 Conv, w/ GAP)' \
    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder fruitbot \
    --no-check-empty-runs \
    --pc.ncols 5 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.15 \
    --rc.combined_figure \
    --pc.ylabel "Normalized Score" \
    --output-filename paper_plots/procgen/ppo/nature/testing2/testing \
    --rc.nsubsamples 10 \
    --rc.aggregate_fig_title "Procgen" \
    --metric_last_n_average_window 1 \
    --pc.rm 2.5

for file in ./paper_plots/procgen/ppo/nature/testing2/*.pdf; do pdfcrop $file $file;  done

# fruitbot heist coinrun bossfight

#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_f_pool_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=1$, first activation)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_f_pool_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, first activation)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_f_pool_s4_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=4$, first activation)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_f_pool_s3_pool2_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, pool 2)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s3_f_pool_latent512_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$, latent512)' \
#    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer \
#    --no-check-empty-runs \
#    --pc.ncols 3 \
#    --pc.ncols-legend 3 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.6 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/nature/testing/testing \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#for file in ./plotting/procgen/ppo/nature/testing/*.pdf; do pdfcrop $file $file;  done

#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impala_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_naturecnn_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_naturecnn_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_naturecnn_pool_c_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=1$, AVG)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_naturecnn_pool_c_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=3$, AVG)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=\textbf{Impoola} ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_train' \
#              'ppo_training?tag=ppo_impoola_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=\textbf{Impoola} ($\tau=3$)' \
#    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer \
#    --no-check-empty-runs \
#    --pc.ncols 3 \
#    --pc.ncols-legend 3 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.6 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/nature/training/training \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#for file in ./plotting/procgen/ppo/nature/training/*.pdf; do pdfcrop $file $file;  done

#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impala_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_naturecnn_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_naturecnn_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_naturecnn_pool_c_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=1$, AVG)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_naturecnn_pool_c_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Nature ($\tau=3$, AVG)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=\textbf{Impoola} ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=scores/eval_avg_return_test' \
#              'ppo_training?tag=ppo_impoola_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=\textbf{Impoola} ($\tau=3$)' \
#    --env-ids bigfish starpilot dodgeball ninja jumper caveflyer \
#    --no-check-empty-runs \
#    --pc.ncols 3 \
#    --pc.ncols-legend 3 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.aggregate_metrics_plots \
#    --rc.normalized_score_threshold 0.6 \
#    --rc.combined_figure \
#    --pc.ylabel "Normalized Score" \
#    --output-filename plotting/procgen/ppo/nature/testing/testing \
#    --rc.nsubsamples 10 \
#    --rc.aggregate_fig_title "Procgen" \
#    --metric_last_n_average_window 1 \
#    --pc.rm 2.5
#
#for file in ./plotting/procgen/ppo/nature/testing/*.pdf; do pdfcrop $file $file;  done


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

