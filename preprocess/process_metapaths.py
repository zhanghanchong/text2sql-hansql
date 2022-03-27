#coding=utf8
import argparse, os, pickle, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from metapath.metapath import Metapath
from utils.constants import MAX_RELATIVE_DIST

nonlocal_relations = [
    'question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic',
    'table-table-fk', 'table-table-fkr', 'table-table-fkb', 'column-column-sametable',
    'question-*-generic', '*-question-generic', '*-table-generic', '*-column-generic', 'column-*-generic', '*-*-identity',
    'question-question-identity', 'table-table-identity', 'column-column-identity'] + [
    'question-question-dist' + str(i) for i in range(-MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1) if i not in [-1, 0, 1]
]

def process_metapath(dataset, tables, max_metapath_length, max_nomatch, nomatch_penalty, output_path, skip_large=False, verbose=False):
    def get_node_type(idx):
        if idx < q_num:
            return 'question'
        if idx < q_num + t_num:
            return 'table'
        return 'column'

    def dfs_find_metapath(cur_idx, cur_metapath):
        is_idx_used[cur_idx] = True
        for new_idx in effective_idxs:
            cur_rel = relation[cur_idx][new_idx]
            if (cur_rel not in nonlocal_relations) and (('nomatch' not in cur_rel) or (cur_metapath.nomatch_count() < max_nomatch)) and (not is_idx_used[new_idx]):
                new_metapath = cur_metapath.copy()
                new_metapath.add(get_node_type(new_idx), cur_rel)
                if new_metapath.has_schema_type():
                    metapaths[new_metapath] = metapaths.get(new_metapath, 0) + nomatch_penalty ** new_metapath.nomatch_count()
                if len(new_metapath) < max_metapath_length:
                    dfs_find_metapath(new_idx, new_metapath)
        is_idx_used[cur_idx] = False

    metapaths = {}
    processed_dataset_num = 0
    for entry in dataset:
        db = tables[entry['db_id']]
        if skip_large and len(db['column_names']) > 100:
            continue
        relation = np.concatenate([
            np.concatenate([np.array(entry['relations'], dtype='<U100'), np.array(entry['schema_linking'][0], dtype='<U100')], axis=1),
            np.concatenate([np.array(entry['schema_linking'][1], dtype='<U100'), np.array(db['relations'], dtype='<U100')], axis=1)
        ], axis=0).tolist()
        q_num = len(entry['processed_question_toks'])
        t_num = len(db['processed_table_toks'])
        c_num = len(db['processed_column_toks'])
        effective_idxs = list(range(q_num)) + [x + q_num for x in entry['used_tables']] + [x + q_num + t_num for x in entry['used_columns']]
        is_idx_used = [False] * (q_num + t_num + c_num)
        for idx in effective_idxs:
            metapath = Metapath(get_node_type(idx))
            dfs_find_metapath(idx, metapath)
        processed_dataset_num += 1
    print('In total, process %d samples, skip %d samples .' % (processed_dataset_num, len(dataset) - processed_dataset_num))
    q_metapaths, t_metapaths, c_metapaths = [], [], []
    for metapath, value in metapaths.items():
        if metapath.node_types[0] == 'question':
            q_metapaths.append((metapath, value))
        elif metapath.node_types[0] == 'table':
            t_metapaths.append((metapath, value))
        elif metapath.node_types[0] == 'column':
            c_metapaths.append((metapath, value))
    q_metapaths.sort(key=lambda x: x[1], reverse=True)
    t_metapaths.sort(key=lambda x: x[1], reverse=True)
    c_metapaths.sort(key=lambda x: x[1], reverse=True)
    metapaths = {
        'q': q_metapaths,
        't': t_metapaths,
        'c': c_metapaths
    }
    pickle.dump(metapaths, open(output_path, 'wb'))
    if verbose:
        for metapath_list in metapaths.values():
            for metapath, value in metapath_list:
                print('%.4f\t%s' % (value, metapath))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--table_path', type=str, required=True, help='processed table path')
    arg_parser.add_argument('--max_metapath_length', type=int, required=True, help='maximum meta-path length')
    arg_parser.add_argument('--max_nomatch', type=int, required=True, help='maximum nomatch in meta-path')
    arg_parser.add_argument('--nomatch_penalty', type=float, required=True, help='penalty for nomatch')
    arg_parser.add_argument('--output_path', type=str, required=True, help='output preprocessed dataset')
    arg_parser.add_argument('--skip_large', action='store_true', help='whether skip large databases')
    arg_parser.add_argument('--verbose', action='store_true', help='whether print meta-paths')
    args = arg_parser.parse_args()
    dataset = pickle.load(open(args.dataset_path, 'rb'))
    tables = pickle.load(open(args.table_path, 'rb'))
    start_time = time.time()
    process_metapath(dataset, tables, args.max_metapath_length, args.max_nomatch, args.nomatch_penalty, args.output_path, args.skip_large, args.verbose)
    print('Finding meta-paths costs %.4fs .' % (time.time() - start_time))
