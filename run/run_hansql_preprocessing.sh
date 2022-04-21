#!/bin/bash

train_data='data/train.json'
dev_data='data/dev.json'
table_data='data/tables.json'
table_out='data/tables.bin'
metapath_out='data/metapaths.bin'
train_out='data/train.hansql.bin'
dev_out='data/dev.hansql.bin'
vocab_glove='pretrained_models/glove.42b.300d/vocab_glove.txt'
vocab='pretrained_models/glove.42b.300d/vocab.txt'
max_metapath_length=3
nomatch_penalty=0.04
dist_penalty=0.5

echo "Start to preprocess the original train dataset ..."
python -u preprocess/process_dataset.py --dataset_path $train_data --raw_table_path $table_data --table_path $table_out --output_path 'data/train.bin' --skip_large
echo "Start to preprocess the original dev dataset ..."
python -u preprocess/process_dataset.py --dataset_path $dev_data --table_path $table_out --output_path 'data/dev.bin'
echo "Start to build word vocab for the dataset ..."
python -u preprocess/build_glove_vocab.py --data_paths 'data/train.bin' --table_path $table_out --reference_file $vocab_glove --mwf 4 --output_path $vocab
echo "Start to find meta-paths ..."
python -u preprocess/process_metapaths.py --dataset_path 'data/train.bin' --table_path $table_out --max_metapath_length $max_metapath_length --nomatch_penalty $nomatch_penalty --dist_penalty $dist_penalty --output_path $metapath_out
echo "Start to construct graphs for the dataset ..."
python -u preprocess/process_graphs.py --dataset_path 'data/train.bin' --table_path $table_out --metapath_path $metapath_out --q_metapath -1 18 2 --t_metapath -1 21 6 --c_metapath -1 41 2 --method 'hansql' --output_path $train_out
python -u preprocess/process_graphs.py --dataset_path 'data/dev.bin' --table_path $table_out --metapath_path $metapath_out --q_metapath -1 18 2 --t_metapath -1 21 6 --c_metapath -1 41 2 --method 'hansql' --output_path $dev_out
