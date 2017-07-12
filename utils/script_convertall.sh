#!/bin/bash
#
#    Script - convert tif to png
#
#    Name: script_convertall.sh
#    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)
#
 
echo "[SCRIPT CONVERT ALL] Initializing..."
 
dir_train="../data/train"
dir_validation="../data/validation"
 
echo "[SCRIPT CONVERT ALL] Converting train..."
 
for dir_class in `ls $dir_train`;
do
    echo "[SCRIPT CONVERT ALL] Converting class -" $dir_class;
    counter=1
    all_files=`ls $dir_train/$dir_class | wc -l`
    for file in `ls $dir_train/$dir_class`;
    do
        echo $counter of $all_files
        echo -ne "$(((100*$counter)/$all_files))%\r"
        convert $dir_train/$dir_class/*.tif $dir_train/$dir_class/image-$counter.png &
        let "counter += 1"
        if ! (($counter%12)); then
            wait
        fi
       
    done
    echo "[SCRIPT CONVERT ALL] Removing all .tif files in $dir_class ..."
    rm $dir_train/$dir_class/*.tif
done
 
echo "[SCRIPT CONVERT ALL] Converting validation..."
 
for dir_class in `ls $dir_validation`;
do
    echo "[SCRIPT CONVERT ALL] Converting class -" $dir_class;
    counter=1
    all_files=`ls $dir_validation/$dir_class | wc -l`
    for file in `ls $dir_validation/$dir_class`;
    do
        echo $counter of $all_files
        echo -ne "$(((100*$counter)/$all_files))%\r"
        convert $dir_train/$dir_class/*.tif $dir_validation/$dir_class/image-$counter.png &
        let "counter += 1"
        if ! (($counter%12)); then
            wait
        fi
    done
    echo "[SCRIPT CONVERT ALL] Removing all .tif files in $dir_class ..."
    rm $dir_validation/$dir_class/*.tif
done
 
echo "[SCRIPT CONVERT ALL] OK! DONE."
