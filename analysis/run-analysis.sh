{
  echo V8-align \
  && python /home/h.jack/MACE_train/mace_active_learning/analysis/analyse_eval_outputs.py \
      --test_xyz /home/h.jack/MACE_train/mace_active_learning/training_clean_weightV9.4.xyz \
      --default_dtype float32 \
      --align --shift_mode element \
      --model /home/h.jack/MACE_train/mace_active_learning/mace_V8_active_learning_stagetwo.model \
      --outdir /home/h.jack/MACE_train/mace_active_learning/results-train/eval_finetuneV8-align \
      &&  echo mpa-align \
  && python /home/h.jack/MACE_train/mace_active_learning/analysis/analyse_eval_outputs.py \
      --test_xyz /home/h.jack/MACE_train/mace_active_learning/training_clean_weightV9.4.xyz \
      --model /home/h.jack/MACE_train/mace_active_learning/mace-mpa-0-medium.model \
      --default_dtype float64 \
      --is_foundation \
      --shift_mode element \
      --outdir /home/h.jack/MACE_train/mace_active_learning/results-train/eval_element_foundation

} > /home/h.jack/MACE_train/mace_active_learning/results/mace_eval_analysis-train.log 2>&1