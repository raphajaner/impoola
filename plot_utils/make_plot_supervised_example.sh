python -m rlops \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=train_accuracy' \
              'supervised_example?tag=supervised_s3_impoola_a&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Train Impoola' \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=val_accuracy' \
              'supervised_example?tag=supervised_s3_impoola_a&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Validation Impoola' \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=train_accuracy' \
              'supervised_example?tag=supervised_s3_impala_a&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Train Impala' \
    --filters '?we=tumwcps&wpn=pruning4drl_results&ceik=env_id&cen=exp_name&metric=val_accuracy' \
              'supervised_example?tag=supervised_s3_impala_a&seed=1&seed=2&seed=3&seed=4&seed=5&cl=Validation Impala' \
    --env-ids bigfish \
    --no-check-empty-runs \
    --pc.ncols 2 \
    --pc.ncols-legend 2 \
    --metric_last_n_average_window 1 \
    --rliable \
    --rc.score_normalization_method onezero \
    --rc.normalized_score_threshold 0.8 \
    --rc.combined_figure \
    --pc.ylabel "Accuracy" \
    --output-filename paper_plots/baseline/supervised_example/training/train_acc \
    --rc.nsubsamples 10
for file in ./paper_plots/baseline/supervised_example/training/*.pdf; do pdfcrop $file $file;  done


