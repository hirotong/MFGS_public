#!/bin/bash
rootdir=$1
cases=$(find $rootdir -mindepth 1 -maxdepth 1 -type d)
masks=(-1 0 1 3 5 7 9 11 13 15)
for case in ${cases[@]}; do
    case_name=$(basename $case)
    echo "Running case: $case_name"

    for mask in ${masks[@]}; do
        echo "Running mask: $mask"
        if [ "$mask" -eq "-1" ]; then
            echo "Running mask: -1"
        elif [ -d $case/colmap_masks_$mask ]; then
            echo "Directory $case/colmap_masks_$mask exists. Skipping..."
        else
            if [ "$mask" -eq "0" ]; then
                mkdir $case/colmap_masks
                for m in $(ls $case/masks); do
                    cp $case/masks/$m $case/colmap_masks_$mask/$m.png
                done
            else
                mkdir $case/colmap_masks_$mask
                for m in $(ls $case/masks_$mask); do
                    cp $case/masks_$mask/$m $case/colmap_masks_$mask/$m.png
                done
            fi
        fi

        if [ "$mask" -eq "-1" ]; then
            if [ -e $case/dense/meshed-poisson-trim0.ply ]; then
                echo "Directory $case/dense exists. Skipping..."
            else
                python colmap_utils/colmap_wrapper.py \
                    --basedir $case \
                    --imagedir images \
                    --width 944 \
                    --height 560 \
                    --data_type rotreal \
                    --pose_file transforms_train.json
            fi

            echo "Evaluate metrics: $case_name"
            if [ -e $case/dense/meshed-poisson-trim0.csv ]; then
                echo "Directory $case/dense exists. Skipping..."
            else
                python evaluate_mesh.py \
                    --mesh_path $case/dense/meshed-poisson-trim0.ply \
                    --eval_pcd_path $case/eval_points.ply
            fi
        else
            if [ -e $case/dense_colmap_masks_$mask/meshed-poisson-trim0.ply ]; then
                echo "Directory $case/dense_colmap_masks_$mask exists. Skipping..."
            else
                python colmap_utils/colmap_wrapper.py \
                    --basedir $case \
                    --imagedir images \
                    --width 944 \
                    --height 560 \
                    --data_type rotreal \
                    --pose_file transforms_train.json \
                    --mask colmap_masks_$mask
            fi
            if [ -e $case/dense_colmap_masks_$mask/meshed-poisson-trim0.csv ]; then
                echo "Directory $case/dense_colmap_masks_$mask exists. Skipping..."
            else
                echo "Evaluate metrics: $case_name"
                python evaluate_mesh.py \
                    --mesh_path $case/dense_colmap_masks_$mask/meshed-poisson-trim0.ply \
                    --eval_pcd_path $case/eval_points.ply
            fi
        fi
    done
done
