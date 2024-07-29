rootdir=$(pwd)/data/synthetic
cases=$(find $rootdir -mindepth 1 -maxdepth 1 -type d ! -path "*_conv" -exec basename {} \;)
masks=(-1)
for case in ${cases[@]}; do
    for mask in ${masks[@]}; do
        outpath="output/synthetic/${case}/2dgs_m_${mask}/"
        echo "Running case: $case mask $mask"
        # if [ -d $outpath ]; then
        #     echo "Directory $outpath exists. Skipping..."
        #     continue
        # fi
        python train.py -s $rootdir/$scene/$case --eval -w -m $outpath --sh_degree 3 --mask_level $mask --apply_mask --lambda_dist 1000.0
        python render.py -m $outpath --num_cluster 100 --voxel_size 0.003
        python eval_metrics_rotblender.py -m $outpath --apply_mask
    done
done
