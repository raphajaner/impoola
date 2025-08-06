python -m rlops \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=dormant_neurons/0_encoder.network.0.conv' \
              'ppo_training?tag=ppo_impala_s2_main&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala Conv2d (1)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=dormant_neurons/0_encoder.network.0.conv' \
              'ppo_training?tag=ppo_impoola_s2_main&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola Conv2d (1)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=dormant_neurons/1_encoder.network.0.res_block0.conv0' \
              'ppo_training?tag=ppo_impala_s2_main&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala Conv2d (2)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=dormant_neurons/1_encoder.network.0.res_block0.conv0' \
              'ppo_training?tag=ppo_impoola_s2_main&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola Conv2d (2)' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=dormant_neurons/15_encoder.network.5' \
              'ppo_training?tag=ppo_impala_s2_main&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala Linear' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=dormant_neurons/15_encoder.network.6' \
              'ppo_training?tag=ppo_impoola_s2_main&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola Linear' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction' \
              'ppo_training?tag=ppo_impala_s2_main&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala Total' \
    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=dormant_neurons/dormant_fraction' \
              'ppo_training?tag=ppo_impoola_s2_main&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola Total' \
    --env-ids bigfish dodgeball caveflyer ninja starpilot coinrun chaser heist plunder leaper bossfight maze miner jumper fruitbot climber \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method 100zero \
    --rc.aggregate_metrics_plots \
    --rc.normalized_score_threshold 0.35 \
    --rc.combined_figure \
    --pc.ylabel "Fraction (Median)" \
    --output-filename paper_plots/dormant_neurons/dormant_neurons \
    --rc.nsubsamples 11 \
    --rc.aggregate_fig_title "Dormant Neurons" \
    --metric_last_n_average_window 1

for file in ./paper_plots/dormant_neurons/*.pdf; do pdfcrop $file $file;  done

#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=dormant_neurons/2_encoder.network.0.res_block0.conv1' \
#              'ppo_training?tag=ppo_impala_s2_main&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala Conv2d (2)' \
#    --filters '?we=tumwcps&wpn=impoola_final&ceik=env_id&cen=exp_name&metric=dormant_neurons/2_encoder.network.0.res_block0.conv1' \
#              'ppo_training?tag=ppo_impoola_s2_main&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola Conv2d (2)' \