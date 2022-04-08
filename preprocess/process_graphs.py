#coding=utf8
import argparse, os, pickle, sys, time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from itertools import chain
from preprocess.graph_utils import GraphProcessor

def process_dataset_graph(processor, dataset, tables, metapaths=None, method='rgatsql', output_path=None, skip_large=False):
    processed_dataset = []
    for idx, entry in enumerate(dataset):
        db = tables[entry['db_id']]
        if skip_large and len(db['column_names']) > 100:
            continue
        if (idx + 1) % 500 == 0:
            print('Processing the %d-th example ...' % (idx + 1))
        entry = processor.process_graph_utils(entry, db, metapaths, method)
        processed_dataset.append(entry)
    print('In total, process %d samples, skip %d samples .' % (len(processed_dataset), len(dataset) - len(processed_dataset)))
    if output_path is not None:
        pickle.dump(processed_dataset, open(output_path, 'wb'))
    return processed_dataset

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', type=str, required=True, help='dataset path')
    arg_parser.add_argument('--table_path', type=str, required=True, help='processed table path')
    arg_parser.add_argument('--metapath_path', type=str, help='processed meta-paths')
    arg_parser.add_argument('--q_metapath', type=int, nargs='+', help='nubmers of meta-paths starting with question')
    arg_parser.add_argument('--t_metapath', type=int, nargs='+', help='numbers of meta-paths starting with table')
    arg_parser.add_argument('--c_metapath', type=int, nargs='+', help='numbers of meta-paths starting with column')
    arg_parser.add_argument('--method', type=str, default='hansql', choices=['rgatsql', 'lgesql', 'hansql'])
    arg_parser.add_argument('--output_path', type=str, required=True, help='output preprocessed dataset')
    args = arg_parser.parse_args()
    processor = GraphProcessor()
    tables = pickle.load(open(args.table_path, 'rb'))
    dataset = pickle.load(open(args.dataset_path, 'rb'))
    if args.metapath_path:
        metapaths = pickle.load(open(args.metapath_path, 'rb'))
        for node_type in ['q', 't', 'c']:
            metapath_nums = eval('args.%s_metapath' % node_type)
            assert len(metapath_nums) == len(metapaths[node_type])
            for i in range(len(metapath_nums)):
                if metapath_nums[i] >= 0 and metapath_nums[i] < len(metapaths[node_type][i]):
                    metapaths[node_type][i] = metapaths[node_type][i][:metapath_nums[i]]
            metapaths[node_type] = chain.from_iterable(metapaths[node_type])
    else:
        metapaths = None
    start_time = time.time()
    dataset = process_dataset_graph(processor, dataset, tables, metapaths, args.method, args.output_path)
    print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))
