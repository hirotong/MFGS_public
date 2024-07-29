rootdir=$(pwd)/data/real
cases=$(find $rootdir -mindepth 1 -maxdepth 1 -type d ! -path "*_conv" -exec basename {} \;)
masks=(-1 0 1 3 5 7 9 11 13 15)
for case in ${cases[@]}; do
    for mask in ${masks[@]}; do
        outpath="output/rotreal/${case}/2dgs_m_${mask}/"
        echo "Running case: $case mask $mask"
        python train.py -s $rootdir/$scene/$case --eval -w -m $outpath --sh_degree 3 --mask_level $mask --apply_mask --lambda_dist 1000.0
        if [[ "$case" == *"dmask"* || "$case" == *"dragon"* ]]; then
            python render.py -m $outpath --num_cluster 1 --bbox 0.3
        else
            python render.py -m $outpath --num_cluster 1 --bbox 0.1
        fi
        python eval_metrics_rotblender.py -m $outpath --apply_mask
    done
done
