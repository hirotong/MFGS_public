cases=("bear" "captain" "controller" "dog" "dragon" "pikachu" "rooster" "plant" "dmask")
# cases=("plant" "dmask")

# for case in ${cases[@]}; do
#     echo "Running case: $case pose estimation" &
#     python process_zivid/estimate_pose_charuco.py -i data/rot-zivid/$case -w 12 -h 9 -sl 0.03 -ml 0.023 -d 6 -c process_zivid/camera_intrinsics_zivid.yml &
# done

# wait

# for case in ${cases[@]}; do
#     echo "Running case: $case post processing pcd" &
#     python process_zivid/post_process.py -d data/rot-zivid/$case -s pcd &
# done

# wait

# for case in ${cases[@]}; do
#     echo "Running case: $case post processing resize" &
#     python process_zivid/post_process.py -d data/rot-zivid/$case -s resize &
# done

# wait

# for case in ${cases[@]}; do
#     echo "Running case: $case post processing reformat" &
#     python process_zivid/post_process.py -d data/rot-zivid/$case -s reformat &
# done

# wait

for case in ${cases[@]}; do
    echo "Running case: $case post processing mask" &
    python process_zivid/post_process.py -d data/rot-zivid/$case -s mask -m data/rot-zivid/mask/$case &
done

# wait

# for case in ${cases[@]}; do
#     echo "Running case: $case post processing depth fusion" &
#     python process_zivid/post_process.py -d data/rot-zivid/$case -s fusion &
# done
