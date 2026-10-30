{
  echo V8 \
  && python /home/h.jack/MACE_train/mace_active_learning/analysis/analyse_eval_outputs.py \
      --test_xyz /home/h.jack/MACE_train/mace_active_learning/fps_validate_frames_corrected.xyz \
      --default_dtype float32 \
      --model /home/h.jack/MACE_train/mace_active_learning/mace_V8_active_learning_stagetwo.model \
      --outdir /home/h.jack/MACE_train/mace_active_learning/results/eval_finetuneV8 \
  &&  echo V8-align \
  && python /home/h.jack/MACE_train/mace_active_learning/analysis/analyse_eval_outputs.py \
      --test_xyz /home/h.jack/MACE_train/mace_active_learning/fps_validate_frames_corrected.xyz \
      --default_dtype float32 \
      --align --shift_mode element \
      --model /home/h.jack/MACE_train/mace_active_learning/mace_V8_active_learning_stagetwo.model \
      --outdir /home/h.jack/MACE_train/mace_active_learning/results/eval_finetuneV8-align \
      &&  echo mpa-E0 \
  && python /home/h.jack/MACE_train/mace_active_learning/analysis/analyse_eval_outputs.py \
      --test_xyz /home/h.jack/MACE_train/mace_active_learning/fps_validate_frames_corrected.xyz \
      --model /home/h.jack/MACE_train/mace_active_learning/mace-mpa-0-medium.model \
      --default_dtype float64 \
      --is_foundation \
      --shift_mode e0 \
      --e0_json /home/h.jack/MACE_train/mace_active_learning/configs/E0s.json \
      --outdir /home/h.jack/MACE_train/mace_active_learning/results/eval_foundation \
      &&  echo mpa-align \
  && python /home/h.jack/MACE_train/mace_active_learning/analysis/analyse_eval_outputs.py \
      --test_xyz /home/h.jack/MACE_train/mace_active_learning/fps_validate_frames_corrected.xyz \
      --model /home/h.jack/MACE_train/mace_active_learning/mace-mpa-0-medium.model \
      --default_dtype float64 \
      --is_foundation \
      --shift_mode element \
      --outdir /home/h.jack/MACE_train/mace_active_learning/results/eval_element_foundation

} > /home/h.jack/MACE_train/mace_active_learning/results/mace_eval_analysis.log 2>&1