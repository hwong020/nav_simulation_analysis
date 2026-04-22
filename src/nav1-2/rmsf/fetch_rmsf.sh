#!/bin/bash

# Remote details
USER="hwong020"
HOST="172.21.33.13"
BASE_DIR="/home/hwong020/nav/nav1-2/single_lig"

# Loop over MD_01 → MD_05
for i in {1..5}
do
    trial=$(printf "%02d" $i)   # 01, 02, ...

    remote_file="${BASE_DIR}/MD_${trial}/rmsf_ca.xvg"
    local_file="rmsf_ca_${i}.xvg"

    echo "Copying ${remote_file} -> ${local_file}"

    scp ${USER}@${HOST}:${remote_file} ${local_file}
done

echo "All RMSF files copied and renamed."
