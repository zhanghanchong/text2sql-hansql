saved_model=saved_models/$1
output_path=saved_models/$1/predicted_sql.txt
batch_size=10
beam_size=5

python eval.py --db_dir 'data/database' --table_path 'data/tables.json' --dataset_path 'data/dev.json' \
    --metapath_path 'data/metapaths.bin' --q_metapath -1 3 2 --t_metapath -1 1 2 --c_metapath -1 4 2 \
    --saved_model $saved_model --output_path $output_path --batch_size $batch_size --beam_size $beam_size
python evaluation.py --gold data/dev_gold.sql --pred $output_path --db data/database --table data/tables.json --etype match > $saved_model/evaluation.log
