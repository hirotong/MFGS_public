rootdir=$(pwd)/data/synthetic
cases=$(find $rootdir -mindepth 1 -maxdepth 1 -type d ! -path "*_conv" -exec basename {} \;)
# masks=(-1 0 1 3 5 7 9)
masks=(0)
for case in ${cases[@]}; do
    for mask in ${masks[@]}; do
        outpath="output/synthetic/${case}/ssgs/"
        echo "Running case: $case"
        python train.py --eval -w \
            -s $rootdir/$case \
            -m $outpath \
            --sh_degree 3 \
            --rotation \
            --mask_level $mask \
            --lambda_reg3d 50.0 \
            --lambda_confidence 0.0 \
            --lambda_zero_one 0.01 \
            --lambda_zero_one_3d 0.01 \
            --reg3d_from_iter 5000 
        python render.py -m $outpath --num_cluster 50 --voxel_size 0.003
        python eval_metrics_rotblender.py -m $outpath --apply_mask
    done
done
