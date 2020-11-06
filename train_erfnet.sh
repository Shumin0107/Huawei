python3 -u train_erfnet.py CULane nasv1 train_gt val_gt \
                        --lr 0.01 \
                        --gpus 0 1 \
                        --resume './models/NASv1/yepeng' \
                        -j 12 \
                        -b 12 \
                        -e \
                        --epochs 30 \
                        --img_height 208 \
                        --img_width 976 \
2>&1|tee train_nasv1_culane.log
