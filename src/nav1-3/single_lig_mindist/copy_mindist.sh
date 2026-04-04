#!/bin/bash

# Remote details
USER="hwong020"
HOST="172.21.33.13"
BASE_DIR="/home/hwong020/nav/nav1-1/single_lig"

# Labels
labels=(d e k a)

# Loop over trials (01 → 05)
for i in {1..5}
do
    trial=$(printf "%02d" $i)   # 01, 02, ...
    
    for label in "${labels[@]}"
    do
        remote_file="${BASE_DIR}/MD_${trial}/mindist_${label}.xvg"
        local_file="mindist_${label}_${i}.xvg"

        echo "Copying ${remote_file} -> ${local_file}"

        scp ${USER}@${HOST}:${remote_file} ${local_file}
    done
done

echo "All files copied and renamed."
