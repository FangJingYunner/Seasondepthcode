
python train.py \
    --num_workers 2 \
    --batch_size 2 \
    --data_path_train /dataset/SeasonDepth \
    --data_path_val /dataset/SeasonDepth/val \
    --dataset season_depth \
    --split SeasonDepth \
    --log_frequency 1 \
    > FIX_BUG_kitti_drop_tracking_aug_.txt 2>&1 &