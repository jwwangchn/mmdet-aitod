#!/usr/bin/env bash

INPUT=${INPUT:-1}

path=./configs/aitod
configs=$(ls $path)

if [ ! -d "./results" ];
then
    mkdir "./results"
fi

if [ $INPUT != 1 ];
then
    configs=$INPUT
fi

for config in $configs
do
    # skip v001.01.01
    result=$(echo $config | grep "v001.01.01")
    if [[ "$result" != "" ]]
    then
        continue
    fi
    config_name=${config%.*}

    # skip the evaluated experiments
    if [ -f "./results/${config_name}/${config_name}.pkl" ]; 
    then
        continue
    fi

    # check the existence of checkpoint
    if [ -f "./work_dirs/${config_name}/latest.pth" ]; 
    then
        echo $'\n=====================================' ${config_name} $'====================================='
        
        real_path=`readlink -f "./work_dirs/${config_name}/latest.pth"`
        echo ${real_path} $'\n'

        if [ ! -d "./results/${config_name}" ];
        then
            mkdir "./results/${config_name}"
        fi
        echo tools/test.py configs/aitod/${config_name}.py work_dirs/${config_name}/latest.pth --eval bbox --out results/${config_name}/${config_name}.pkl --options "jsonfile_prefix=./results/${config_name}/${config_name}"
        python tools/test.py configs/aitod/${config_name}.py work_dirs/${config_name}/latest.pth --eval bbox --out results/${config_name}/${config_name}.pkl --options "jsonfile_prefix=./results/${config_name}/${config_name}" "with_lrp=False" 
    fi
done
