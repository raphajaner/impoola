python -m rlops \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impala_s1_atari_long&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impala_s3_atari_long&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impoola_s1_atari_long&seed=1&seed=2&seed=3&seed=4&seed=5&cl=\textbf{Impoola} ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impoola_s3_atari_long&seed=1&seed=2&seed=3&seed=4&seed=5&cl=\textbf{Impoola} ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impoola_f_pool_s1_atari_long&seed=1&seed=2&seed=3&seed=4&seed=5&cl=\textbf{Impoola} ($\tau=1$, first activation)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impoola_f_pool_s3_atari_long&seed=1&seed=2&seed=3&seed=4&seed=5&cl=\textbf{Impoola} ($\tau=3$, first activation)' \
    --env-ids BeamRider-v5 Breakout-v5 SpaceInvaders-v5 Seaquest-v5 Qbert-v5 Enduro-v5 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 2 \
    --rliable \
    --rc.score_normalization_method atari \
    --rc.normalized_score_threshold 12 \
    --rc.combined_figure \
    --rc.aggregate_metrics_plots \
    --pc.ylabel "Episodic Return" \
    --output-filename paper_plots/atari_long/ppo/training/training \
    --rc.nsubsamples 12 \
    --rc.aggregate_fig_title "" \
    --metric_last_n_average_window 10 \
    --pc.rolling 100 \
    --pc.rm 2.5

for file in ./paper_plots/atari_long/ppo/training/*.pdf; do pdfcrop $file $file;  done

