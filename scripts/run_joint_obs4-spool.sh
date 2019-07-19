mkdir ./models
python3 scripts/train.py \
  --dataset_name 'zara1' \
  --delim tab \
  --loader_num_workers 8 \
  --d_type 'local' \
  --pred_len 8 \
  --encoder_h_dim_g 32 \
  --encoder_h_dim_d 64\
  --decoder_h_dim 32 \
  --embedding_dim 16 \
  --bottleneck_dim 32 \
  --mlp_dim 64 \
  --num_layers 1 \
  --noise_dim 0 \
  --noise_type gaussian \
  --noise_mix_type global \
  --pool_every_timestep 0 \
  --l2_loss_weight 10 \
  --intention_loss_weight 0.05\
  --intention_loss_type 'l2'\
  --batch_norm 0 \
  --dropout 0 \
  --timing 1 \
  --batch_size 64 \
  --g_learning_rate 5e-3 \
  --g_steps 1 \
  --d_learning_rate 1e-3 \
  --d_steps 2 \
  --checkpoint_every 10 \
  --print_every 50 \
  --num_iterations 10000 \
  --num_epochs 500 \
  --pooling_type 'spool' \
  --clipping_threshold_g 1.5 \
  --best_k 1 \
  --gpu_num 0 \
  --checkpoint_name headingloss_zara1_batch32_epoch500_spool \
  --restore_from_checkpoint 1 \
  --output_dir './models' \
  --benchmark True \
  --spatial_dim True \
  --resist_loss_weight 1 \
  --resist_loss_heading 1 \
  2>&1 | tee training_headingloss_batch32_epoch500_spool.log
