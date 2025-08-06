python -m rlops \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impala_s1_atari&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impoola_s1_atari&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=1$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impala_s3_atari&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impoola_s3_atari&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=3$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impala_s5_atari&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impala ($\tau=5$)' \
    --filters '?we=tumwcps&wpn=impoola&ceik=env_id&cen=exp_name&metric=charts/avg_episodic_return' \
              'ppo_training?tag=ppo_impoola_s5_atari&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Impoola ($\tau=5$)' \
    --env-ids BeamRider-v5 Breakout-v5 Pong-v5 Enduro-v5 SpaceInvaders-v5 Seaquest-v5 Qbert-v5 \
    --no-check-empty-runs \
    --pc.ncols 3 \
    --pc.ncols-legend 4 \
    --rliable \
    --rc.score_normalization_method atari \
    --rc.normalized_score_threshold 0.99 \
    --rc.combined_figure \
    --rc.aggregate_metrics_plots \
    --pc.ylabel "Episodic Return" \
    --output-filename paper_plots/atari/ppo/training/training \
    --rc.nsubsamples 5 \
    --metric_last_n_average_window 100 \
    --pc.rolling 100 \
    --pc.rm 2.5

for file in ./paper_plots/atari/ppo/training/*.pdf; do pdfcrop $file $file;  done
