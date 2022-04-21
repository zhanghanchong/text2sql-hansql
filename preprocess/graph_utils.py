#coding=utf8
import dgl, math, torch
import numpy as np
from utils.constants import MAX_RELATIVE_DIST
from utils.graph_example import GraphExample

# mapping special column * as an ordinary column
special_column_mapping_dict = {
    'question-*-generic': 'question-column-nomatch',
    '*-question-generic': 'column-question-nomatch',
    'table-*-generic': 'table-column-has',
    '*-table-generic': 'column-table-has',
    '*-column-generic': 'column-column-generic',
    'column-*-generic': 'column-column-generic',
    '*-*-identity': 'column-column-identity'
}
nonlocal_relations = [
    'question-question-generic', 'table-table-generic', 'column-column-generic', 'table-column-generic', 'column-table-generic',
    'table-table-fk', 'table-table-fkr', 'table-table-fkb', 'column-column-sametable',
    '*-column-generic', 'column-*-generic', '*-*-identity', '*-table-generic',
    'question-question-identity', 'table-table-identity', 'column-column-identity'] + [
    'question-question-dist' + str(i) for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1) if i not in [-1, 0, 1]
]


def get_mapped_relation(relation: list):
    for i in range(len(relation)):
        for j in range(len(relation[i])):
            if relation[i][j] in special_column_mapping_dict:
                relation[i][j] = special_column_mapping_dict[relation[i][j]]
    return relation


class GraphProcessor():
    def process_rgatsql(self, ex: dict, db: dict, relation: list):
        graph = GraphExample()
        num_nodes = int(math.sqrt(len(relation)))
        local_edges = [(idx // num_nodes, idx % num_nodes, (special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
            for idx, r in enumerate(relation) if r not in nonlocal_relations]
        nonlocal_edges = [(idx // num_nodes, idx % num_nodes, (special_column_mapping_dict[r] if r in special_column_mapping_dict else r))
            for idx, r in enumerate(relation) if r in nonlocal_relations]
        global_edges = local_edges + nonlocal_edges
        src_ids, dst_ids = list(map(lambda r: r[0], global_edges)), list(map(lambda r: r[1], global_edges))
        graph.global_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.global_edges = global_edges
        src_ids, dst_ids = list(map(lambda r: r[0], local_edges)), list(map(lambda r: r[1], local_edges))
        graph.local_g = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes, idtype=torch.int32)
        graph.local_edges = local_edges
        # graph pruning for nodes
        q_num = len(ex['processed_question_toks'])
        s_num = num_nodes - q_num
        graph.question_mask = [1] * q_num + [0] * s_num
        graph.schema_mask = [0] * q_num + [1] * s_num
        graph.gp = dgl.heterograph({
            ('question', 'to', 'schema'): (list(range(q_num)) * s_num, [i for i in range(s_num) for _ in range(q_num)])
        }, num_nodes_dict={
            'question': q_num,
            'schema': s_num
        }, idtype=torch.int32)
        t_num = len(db['processed_table_toks'])

        def check_node(i):
            if i < t_num and i in ex['used_tables']:
                return 1.0
            elif i >= t_num and i - t_num in ex['used_columns']:
                return 1.0
            else: return 0.0

        graph.node_label = list(map(check_node, range(s_num)))
        ex['graph'] = graph
        return ex

    def process_lgesql(self, ex: dict, db: dict, relation: list):
        ex = self.process_rgatsql(ex, db, relation)
        graph = ex['graph']
        lg = graph.local_g.line_graph(backtracking=False)
        # prevent information propagate through matching edges
        match_ids = [idx for idx, r in enumerate(graph.global_edges) if 'match' in r[2]]
        src, dst, eids = lg.edges(form='all', order='eid')
        eids = [e for u, v, e in zip(src.tolist(), dst.tolist(), eids.tolist()) if not (u in match_ids and v in match_ids)]
        graph.lg = lg.edge_subgraph(eids, preserve_nodes=True).remove_self_loop().add_self_loop()
        ex['graph'] = graph
        return ex

    def process_hansql(self, ex: dict, db: dict, metapaths: dict, relation: list, verbose: bool):
        relation = get_mapped_relation(relation)
        q_num = len(ex['processed_question_toks'])
        t_num = len(db['processed_table_toks'])
        c_num = len(db['processed_column_toks'])
        assert q_num + t_num + c_num == len(relation)
        graph = GraphExample()
        graph.graphs = {'q': [], 't': [], 'c': []}

        def get_range_by_node_type(node_type: str):
            if node_type == 'q':
                return range(q_num)
            if node_type == 't':
                return range(q_num, q_num + t_num)
            if node_type == 'c':
                return range(q_num + t_num, q_num + t_num + c_num)
            raise ValueError('wrong node type %s' % node_type)

        def dfs_find_metapath_based_neighbors(idx: int, step: int):
            if step == len(metapath):
                return {idx}
            is_idx_used[idx] = True
            result = set()
            for new_idx in get_range_by_node_type(metapath.node_types[step + 1][0]):
                if relation[idx][new_idx] == metapath.edge_types[step] and (not is_idx_used[new_idx]):
                    result.update(dfs_find_metapath_based_neighbors(new_idx, step + 1))
            is_idx_used[idx] = False
            return result

        all_neighbors = [{'q': set(), 't': set(), 'c': set()} for _ in range(len(relation))]
        is_idx_used = [False] * len(relation)
        for start_node_type in metapaths:
            for metapath, _ in metapaths[start_node_type]:
                end_node_type = metapath.node_types[len(metapath)][0]
                src_ids, dst_ids = [], []
                for i in get_range_by_node_type(start_node_type):
                    neighbors = dfs_find_metapath_based_neighbors(i, 0)
                    if start_node_type == end_node_type:
                        neighbors.add(i)
                    else:
                        all_neighbors[i][start_node_type].add(i - get_range_by_node_type(start_node_type)[0])
                    neighbors = list(map(lambda x: x - get_range_by_node_type(end_node_type)[0], neighbors))
                    neighbors.sort()
                    all_neighbors[i][end_node_type].update(neighbors)
                    for neighbor in neighbors:
                        src_ids.append(neighbor)
                        dst_ids.append(i - get_range_by_node_type(start_node_type)[0])
                if start_node_type == end_node_type:
                    graph.graphs[start_node_type].append((dgl.graph((src_ids, dst_ids), num_nodes=eval(end_node_type + '_num'), idtype=torch.int32), end_node_type))
                else:
                    src_num = eval(end_node_type + '_num')
                    dst_num = eval(start_node_type + '_num')
                    graph.graphs[start_node_type].append((dgl.heterograph({
                        ('src', 'to', 'dst'): (src_ids, dst_ids),
                        ('dst', 'to', 'dst'): (list(range(dst_num)), list(range(dst_num)))
                    }, num_nodes_dict={
                        'src': src_num,
                        'dst': dst_num
                    }, idtype=torch.int32), end_node_type))
        if verbose:
            print('type', 'q', 't', 'c', 'total', sep='\t')
            for i in range(len(relation)):
                print('%s\t%.4f\t%.4f\t%.4f\t%.4f' % (
                    'q' if i < q_num else 't' if i < q_num + t_num else 'c',
                    len(all_neighbors[i]['q']) / q_num,
                    len(all_neighbors[i]['t']) / t_num,
                    len(all_neighbors[i]['c']) / c_num,
                    (len(all_neighbors[i]['q']) + len(all_neighbors[i]['t']) + len(all_neighbors[i]['c'])) / len(relation)
                ))
            print()
        # graph pruning for nodes
        s_num = t_num + c_num
        graph.question_mask = [1] * q_num + [0] * s_num
        graph.table_mask = [0] * q_num + [1] * t_num + [0] * c_num
        graph.column_mask = [0] * (q_num + t_num) + [1] * c_num
        graph.schema_mask = [0] * q_num + [1] * s_num
        graph.gp = dgl.heterograph({
            ('question', 'to', 'schema'): (list(range(q_num)) * s_num, [i for i in range(s_num) for _ in range(q_num)])
        }, num_nodes_dict={
            'question': q_num,
            'schema': s_num
        }, idtype=torch.int32)

        def check_node(i):
            if i < t_num and i in ex['used_tables']:
                return 1.0
            elif i >= t_num and i - t_num in ex['used_columns']:
                return 1.0
            else: return 0.0

        graph.node_label = list(map(check_node, range(s_num)))
        ex['graph'] = graph
        return ex

    def process_graph_utils(self, ex: dict, db: dict, metapaths: dict = None, method: str = 'rgatsql', verbose: bool = False):
        """ Example should be preprocessed by self.pipeline
        """
        q = np.array(ex['relations'], dtype='<U100')
        s = np.array(db['relations'], dtype='<U100')
        q_s = np.array(ex['schema_linking'][0], dtype='<U100')
        s_q = np.array(ex['schema_linking'][1], dtype='<U100')
        relation = np.concatenate([
            np.concatenate([q, q_s], axis=1),
            np.concatenate([s_q, s], axis=1)
        ], axis=0)
        if method == 'rgatsql':
            ex = self.process_rgatsql(ex, db, relation.flatten().tolist())
        elif method == 'lgesql':
            ex = self.process_lgesql(ex, db, relation.flatten().tolist())
        elif method == 'hansql':
            ex = self.process_hansql(ex, db, metapaths, relation.tolist(), verbose)
        return ex
