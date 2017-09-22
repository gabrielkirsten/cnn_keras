#!/bin/bash

#
#    Script - split data between train and test
#
#    Name: script_split_data.sh
#    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)
#

echo "[SCRIPT SPLIT DATA] Initializing..."

dir_train="../data/train"
dir_validation="../data/validation"
perc_train=70

for dir_class in `ls $dir_train`;
do
    echo "[SCRIPT SPLIT DATA] Spliting class -" $dir_class;
    mkdir $dir_validation/$dir_class
    quantity_files=`ls $dir_train/$dir_class | wc -l`
    perc_quantity_files=$((($quantity_files/100)*$perc_train))
    counter=0
    arrayFiles=`ls $dir_train/$dir_class |sort -R`
    for file in $arrayFiles;
    do
        let "counter += 1"
        if [[ $counter -gt $perc_quantity_files ]]; then
            mv $dir_train/$dir_class/$file $dir_validation/$dir_class/$file
        fi
    done
done

echo "[SCRIPT SPLIT DATA] OK! DONE."
