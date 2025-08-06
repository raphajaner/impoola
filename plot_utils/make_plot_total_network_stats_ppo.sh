python -m rlops \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_params' \
              'ppo_training?tag=ppo_impala_s1_random&seed=1&cl=Impala ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_params' \
              'ppo_training?tag=ppo_impala_s2_random&seed=1&cl=Impala ($\tau=2$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_params' \
              'ppo_training?tag=ppo_impala_s3_random&seed=1&cl=Impala ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_params' \
              'ppo_training?tag=ppo_impoola_s1_random&seed=1&cl=Impoola ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=charts/total_network_params' \
              'ppo_training?tag=ppo_impoola_s2_random_no_anneal_lower_mid_lr&seed=1&cl=Impoola ($\tau=2$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_params' \
              'ppo_training?tag=ppo_impoola_s3_random&seed=1&cl=Impoola ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_params' \
              'ppo_training?tag=ppo_impoola_s4_random&seed=1&cl=Impoola ($\tau=4$)' \
    --env-ids bigfish \
    --no-check-empty-runs \
    --pc.ncols 4 \
    --pc.ncols-legend 2 \
    --metric_last_n_average_window 1 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.normalized_score_threshold 1000000 \
    --rc.combined_figure \
    --pc.ylabel "Total Network Parameter" \
    --output-filename paper_plots/results/ppo/stats/params \
    --rc.nsubsamples 2000

#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs' \
#              'ppo_training?tag=ppo_impala_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs' \
#              'ppo_training?tag=ppo_impala_s2_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs' \
#              'ppo_training?tag=ppo_impala_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs' \
#              'ppo_training?tag=ppo_impoola_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs' \
#              'ppo_training?tag=ppo_impoola_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_m_macs' \
#              'ppo_training?tag=ppo_impoola_s4_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=4$)' \
#    --env-ids bigfish \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 2 \
#    --metric_last_n_average_window 1 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 1000000 \
#    --rc.combined_figure \
#    --pc.ylabel "Total Network Parameter" \
#    --output-filename plotting/results/ppo/stats/m_macs \
#    --rc.nsubsamples 2000
#
#python -m rlops \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes' \
#              'ppo_training?tag=ppo_impala_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes' \
#              'ppo_training?tag=ppo_impala_s2_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=2$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes' \
#              'ppo_training?tag=ppo_impala_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes' \
#              'ppo_training?tag=ppo_impoola_s1_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=1$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes' \
#              'ppo_training?tag=ppo_impoola_s3_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$)' \
#    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/total_network_param_bytes' \
#              'ppo_training?tag=ppo_impoola_s4_random&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=4$)' \
#    --env-ids bigfish \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 2 \
#    --metric_last_n_average_window 1 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 1000000 \
#    --rc.combined_figure \
#    --pc.ylabel "Total Network Parameter" \
#    --output-filename plotting/results/ppo/stats/bytes \
#    --rc.nsubsamples 2000
#
#for file in ./plotting/results/ppo/stats/*.pdf; do pdfcrop $file $file;  done

#--filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=charts/total_network_params' \
#          'ppo_procgen?tag=ppo_group_s3_p80_less_steps&seed=1&cl=Group-Structured $\zeta_F=0.8$ (10 steps)' \

#python -m rlops \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=charts/total_network_params' \
#              'dqn_procgen?tag=dqn_unstructured_s3_p80&seed=1&cl=Unstructured $\zeta_F=0.8$' \
#    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=charts/total_network_params' \
#              'dqn_procgen?tag=dqn_group_s3_p80_no_all&seed=1&cl=Group-Structured $\zeta_F=0.8$' \
#    --env-ids bigfish \
#    --no-check-empty-runs \
#    --pc.ncols 4 \
#    --pc.ncols-legend 2 \
#    --metric_last_n_average_window 1 \
#    --rliable \
#    --rc.score_normalization_method onezero \
#    --rc.normalized_score_threshold 1000000 \
#    --rc.combined_figure \
#    --pc.ylabel "Total Network Parameter" \
#    --output-filename plotting/results/dqn/params/params \
#    --rc.nsubsamples 2000
#
#for file in ./plotting/results/dqn/params/*.pdf; do pdfcrop $file $file;  done