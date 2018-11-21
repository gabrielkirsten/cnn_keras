#!/bin/bash
#
#    Script - split data between train and test
#
#    Name: split_data.sh
#    Author: Gabriel Kirsten Menezes (gabriel.kirsten@hotmail.com)
#    Contributions: Hemerson Pistori (pistori@ucdb.br)
#   
#    Usage
#
#    $ chmod 755 ./split_data.sh
#    $ sh ./split_data.sh -o /home/user/datasetoriginal -s ../datasetsplit -t 60 -v 20 -e 20
#

# Default values
perc_train=60
perc_validation=20
perc_test=20
original_dataset_path="../../data/demo"
split_dataset_path="../../data/demo_split"
dir_train=${split_dataset_path}"/train"
dir_validation=${split_dataset_path}"/validation"
dir_test=${split_dataset_path}"/test"


# Read arguments from command line
for i in "$@"
do
case $i in
    -o=*|--originaldatasetpath=*)
    original_dataset_path="${i#*=}"
    shift # past argument=value
    ;;
    -s=*|--splitdatasetpath=*)
    split_dataset_path="${i#*=}"
    dir_train=${split_dataset_path}"/train"
    dir_validation=${split_dataset_path}"/validation"
    dir_test=${split_dataset_path}"/test"
    shift # past argument=value
    ;;
    -t=*|--trainingsize=*)
    perc_train="${i#*=}"
    shift # past argument=value
    ;;
    -v=*|--validationsize=*)
    perc_validation="${i#*=}"
    shift # past argument=value
    ;;
    -e=*|--testsize=*)
    perc_test="${i#*=}"
    shift # past argument=value
    ;;
    -h*|--help*)
    echo "Usage:"
    echo ""
    echo "chmod 755 ./split_data.sh"
    echo "./split_data.sh -o /home/user/datasetoriginal -s ../datasetsplit -t 60 -v 20 -e 20"
    echo ""
    echo "-o: path for the original dataset"
    echo "-s: path for the resulting split dataset"
    echo "-t: percentual size of the resulting training set"
    echo "-v: percentual size of the resulting validation set (use 0 for no validation set)"
    echo "-e: percentual size of the resulting test set"
    exit 0
    shift # past argument=value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument with no value
    ;;
    *)
          # unknown option
    ;;
esac
done


echo "[SCRIPT SPLIT DATA] Initializing..."

echo "Training size (perc) = ${perc_train}"
echo "Validation size (perc) = ${perc_validation}"
echo "Testing (perc) = ${perc_test}"
echo "Original dataset path = ${original_dataset_path}"
echo "Resulting training dataset path = ${dir_train}"
echo "Resulting validation dataset path = ${dir_validation}"
echo "Resulting test dataset path = ${dir_test}"

echo ""
echo "Removing all data inside "${split_dataset_path}
rm -rf $split_dataset_path/*

mkdir $dir_train
mkdir $dir_validation
mkdir $dir_test

cp -r $original_dataset_path/* $dir_train/


for dir_class in `ls $dir_train`;
do
    echo "[SCRIPT SPLIT DATA] Spliting class -" $dir_class;
    mkdir $dir_validation/$dir_class
    mkdir $dir_test/$dir_class
    quantity_files=`ls $dir_train/$dir_class | wc -l`

    quantity_files_train_float=`echo "scale=1; ($quantity_files/100)*$perc_train" | bc -l `
    quantity_files_train=${quantity_files_train_float%.*}


    perc_quantity_files_test=$(($quantity_files-(($quantity_files/100)*$perc_test)))


    quantity_files_test_float=`echo "scale=1; (($quantity_files-(($quantity_files/100)*$perc_test)))" | bc -l `
    quantity_files_test=${quantity_files_test_float%.*}


    counter=0
    arrayFiles=`ls $dir_train/$dir_class |sort -R`
    for file in $arrayFiles;
    do
        let "counter += 1"
        if [[ $counter -gt $quantity_files_train && $counter -le $quantity_files_test ]]; then
            mv $dir_train/$dir_class/$file $dir_validation/$dir_class/$file
        fi
        if [[ $counter -gt $quantity_files_test ]]; then
            mv $dir_train/$dir_class/$file $dir_test/$dir_class/$file
        fi
    done
done

echo "[SCRIPT SPLIT DATA] OK! DONE."

