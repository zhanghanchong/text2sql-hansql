#coding=utf8
import dgl, torch

class GraphExample():
    pass

class BatchedGraph():
    pass

class GraphFactory():
    def __init__(self, method='rgatsql', relation_vocab=None):
        super(GraphFactory, self).__init__()
        self.method = eval('self.' + method)
        self.batch_method = eval('self.batch_' + method)
        self.relation_vocab = relation_vocab

    def graph_construction(self, ex: dict, db: dict):
        return self.method(ex, db)

    def rgatsql(self, ex, db):
        graph = GraphExample()
        local_edges = ex['graph'].local_edges
        rel_ids = list(map(lambda r: self.relation_vocab[r[2]], local_edges))
        graph.local_edges = torch.tensor(rel_ids, dtype=torch.long)
        global_edges = ex['graph'].global_edges
        rel_ids = list(map(lambda r: self.relation_vocab[r[2]], global_edges))
        graph.global_edges = torch.tensor(rel_ids, dtype=torch.long)
        graph.local_g, graph.global_g = ex['graph'].local_g, ex['graph'].global_g
        graph.gp = ex['graph'].gp
        graph.question_mask = torch.tensor(ex['graph'].question_mask, dtype=torch.bool)
        graph.schema_mask = torch.tensor(ex['graph'].schema_mask, dtype=torch.bool)
        graph.node_label = torch.tensor(ex['graph'].node_label, dtype=torch.float)
        # extract local relations (used in msde), global_edges = local_edges + nonlocal_edges
        local_enum, global_enum = graph.local_edges.size(0), graph.global_edges.size(0)
        graph.local_mask = torch.tensor([1] * local_enum + [0] * (global_enum - local_enum), dtype=torch.bool)
        return graph

    def lgesql(self, ex, db):
        graph = self.rgatsql(ex, db)
        # add line graph
        graph.lg = ex['graph'].lg
        return graph

    def hansql(self, ex, db):
        graph = GraphExample()
        graph.graphs = ex['graph'].graphs
        graph.gp = ex['graph'].gp
        graph.question_mask = torch.tensor(ex['graph'].question_mask, dtype=torch.bool)
        graph.table_mask = torch.tensor(ex['graph'].table_mask, dtype=torch.bool)
        graph.column_mask = torch.tensor(ex['graph'].column_mask, dtype=torch.bool)
        graph.schema_mask = torch.tensor(ex['graph'].schema_mask, dtype=torch.bool)
        graph.node_label = torch.tensor(ex['graph'].node_label, dtype=torch.float)
        graph.q_num = graph.question_mask.sum().item()
        graph.t_num = graph.table_mask.sum().item()
        graph.c_num = graph.column_mask.sum().item()
        return graph

    def batch_graphs(self, ex_list, device, train=True, **kwargs):
        """ Batch graphs in example list """
        return self.batch_method(ex_list, device, train=train, **kwargs)

    def batch_rgatsql(self, ex_list, device, train=True, **kwargs):
        # method = kwargs.pop('local_and_nonlocal', 'global')
        graph_list = [ex.graph for ex in ex_list]
        bg = BatchedGraph()
        bg.local_g = dgl.batch([ex.local_g for ex in graph_list]).to(device)
        # bg.local_edges = torch.cat([ex.local_edges for ex in graph_list], dim=0).to(device)
        bg.local_mask = torch.cat([ex.graph.local_mask for ex in ex_list], dim=0).to(device)
        bg.global_g = dgl.batch([ex.global_g for ex in graph_list]).to(device)
        bg.global_edges = torch.cat([ex.global_edges for ex in graph_list], dim=0).to(device)
        if train:
            bg.question_mask = torch.cat([ex.question_mask for ex in graph_list], dim=0).to(device)
            bg.schema_mask = torch.cat([ex.schema_mask for ex in graph_list], dim=0).to(device)
            smoothing = kwargs.pop('smoothing', 0.0)
            node_label = torch.cat([ex.node_label for ex in graph_list], dim=0)
            node_label = node_label.masked_fill_(~ node_label.bool(), 2 * smoothing) - smoothing
            bg.node_label = node_label.to(device)
            bg.gp = dgl.batch([ex.gp for ex in graph_list]).to(device)
        return bg

    def batch_lgesql(self, ex_list, device, train=True, **kwargs):
        bg = self.batch_rgatsql(ex_list, device, train=train, **kwargs)
        src_ids, dst_ids = bg.local_g.edges(order='eid')
        bg.src_ids, bg.dst_ids = src_ids.long(), dst_ids.long()
        bg.lg = dgl.batch([ex.graph.lg for ex in ex_list]).to(device)
        return bg

    def batch_hansql(self, ex_list, device, train=True, **kwargs):
        graph_list = [ex.graph for ex in ex_list]
        bg = BatchedGraph()
        bg.graphs = {'q': [], 't': [], 'c': []}
        for node_type in ['q', 't', 'c']:
            for i in range(len(graph_list[0].graphs[node_type])):
                bg.graphs[node_type].append((dgl.batch([ex.graphs[node_type][i][0] for ex in graph_list]).to(device), graph_list[0].graphs[node_type][i][1]))
        bg.question_mask = torch.cat([ex.question_mask for ex in graph_list], dim=0).to(device)
        bg.table_mask = torch.cat([ex.table_mask for ex in graph_list], dim=0).to(device)
        bg.column_mask = torch.cat([ex.column_mask for ex in graph_list], dim=0).to(device)
        bg.q_num = [ex.q_num for ex in graph_list]
        bg.t_num = [ex.t_num for ex in graph_list]
        bg.c_num = [ex.c_num for ex in graph_list]
        if train:
            bg.schema_mask = torch.cat([ex.schema_mask for ex in graph_list], dim=0).to(device)
            smoothing = kwargs.pop('smoothing', 0.0)
            node_label = torch.cat([ex.node_label for ex in graph_list], dim=0)
            node_label = node_label.masked_fill_(~ node_label.bool(), 2 * smoothing) - smoothing
            bg.node_label = node_label.to(device)
            bg.gp = dgl.batch([ex.gp for ex in graph_list]).to(device)
        return bg
