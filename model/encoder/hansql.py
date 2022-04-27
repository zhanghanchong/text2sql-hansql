#coding=utf8
import math
import torch
import torch.nn as nn
import dgl.function as fn
from model.model_utils import Registrable, FFN
from model.encoder.functions import *


@Registrable.register('hansql')
class HANSQL(nn.Module):
    def __init__(self, args):
        super(HANSQL, self).__init__()
        self.num_layers = args.gnn_num_layers
        self.ndim = args.gnn_hidden_size
        self.gnn_layers = nn.ModuleList([
            HANSQLLayer(self.ndim, args.num_heads, args.dropout, args.node_type_share_weights, args.no_metapath_attention)
            for _ in range(self.num_layers)
        ])

    def forward(self, x, batch):
        for i in range(self.num_layers):
            x = self.gnn_layers[i](x, batch)
        return x


class HANSQLLayer(nn.Module):
    def __init__(self, ndim, num_heads=8, feat_drop=0.2, share=False, no_mp_attn=False):
        super(HANSQLLayer, self).__init__()
        self.ndim = ndim
        self.num_heads = num_heads
        self.d_k = self.ndim // self.num_heads
        if share:
            self.affine_q = nn.ModuleList([nn.Linear(self.ndim, self.ndim)] * 3)
            self.affine_k = nn.ModuleList([nn.Linear(self.ndim, self.ndim, bias=False)] * 3)
            self.affine_v = nn.ModuleList([nn.Linear(self.ndim, self.ndim, bias=False)] * 3)
            self.affine_o = nn.ModuleList([nn.Linear(self.ndim, self.ndim)] * 3)
            self.layernorm = nn.ModuleList([nn.LayerNorm(self.ndim)] * 3)
            self.feat_dropout = nn.ModuleList([nn.Dropout(p=feat_drop)] * 3)
            self.ffn = nn.ModuleList([FFN(self.ndim)] * 3)
            self.mp_attn = None if no_mp_attn else nn.ModuleList([nn.Sequential(
                nn.Linear(self.ndim, self.ndim),
                nn.Tanh(),
                nn.Linear(self.ndim, 1, bias=False)
            )] * 3)
        else:
            self.affine_q = nn.ModuleList([nn.Linear(self.ndim, self.ndim) for _ in range(3)])
            self.affine_k = nn.ModuleList([nn.Linear(self.ndim, self.ndim, bias=False) for _ in range(3)])
            self.affine_v = nn.ModuleList([nn.Linear(self.ndim, self.ndim, bias=False) for _ in range(3)])
            self.affine_o = nn.ModuleList([nn.Linear(self.ndim, self.ndim) for _ in range(3)])
            self.layernorm = nn.ModuleList([nn.LayerNorm(self.ndim) for _ in range(3)])
            self.feat_dropout = nn.ModuleList([nn.Dropout(p=feat_drop) for _ in range(3)])
            self.ffn = nn.ModuleList([FFN(self.ndim) for _ in range(3)])
            self.mp_attn = None if no_mp_attn else nn.ModuleList([nn.Sequential(
                nn.Linear(self.ndim, self.ndim),
                nn.Tanh(),
                nn.Linear(self.ndim, 1, bias=False)
            ) for _ in range(3)])

    def forward(self, x, batch):
        qx = x.masked_select(batch.graph.question_mask.unsqueeze(-1)).view(-1, self.ndim)
        tx = x.masked_select(batch.graph.table_mask.unsqueeze(-1)).view(-1, self.ndim)
        cx = x.masked_select(batch.graph.column_mask.unsqueeze(-1)).view(-1, self.ndim)
        q, k, v = {}, {}, {}
        for i, node_type in enumerate(['q', 't', 'c']):
            q[node_type] = self.affine_q[i](self.feat_dropout[i](eval(node_type + 'x')))
            k[node_type] = self.affine_k[i](self.feat_dropout[i](eval(node_type + 'x')))
            v[node_type] = self.affine_v[i](self.feat_dropout[i](eval(node_type + 'x')))
        all_out_x = {'q': [], 't': [], 'c': []}
        for i, start_node_type in enumerate(['q', 't', 'c']):
            for g, end_node_type in batch.graph.graphs[start_node_type]:
                with g.local_scope():
                    if start_node_type == end_node_type:
                        g.ndata['q'] = q[end_node_type].view(-1, self.num_heads, self.d_k)
                        g.ndata['k'] = k[end_node_type].view(-1, self.num_heads, self.d_k)
                        g.ndata['v'] = v[end_node_type].view(-1, self.num_heads, self.d_k)
                    else:
                        g.nodes['dst'].data['q'] = q[start_node_type].view(-1, self.num_heads, self.d_k)
                        g.nodes['dst'].data['k'] = k[start_node_type].view(-1, self.num_heads, self.d_k)
                        g.nodes['dst'].data['v'] = v[start_node_type].view(-1, self.num_heads, self.d_k)
                        g.nodes['src'].data['k'] = k[end_node_type].view(-1, self.num_heads, self.d_k)
                        g.nodes['src'].data['v'] = v[end_node_type].view(-1, self.num_heads, self.d_k)
                    out_x = self.propagate_attention(g, start_node_type != end_node_type)
                out_x = self.layernorm[i](eval(start_node_type + 'x') + self.affine_o[i](out_x.view(-1, self.num_heads * self.d_k)))
                out_x = self.ffn[i](out_x)
                all_out_x[start_node_type].append(out_x)
            all_out_x[start_node_type] = torch.stack(all_out_x[start_node_type], dim=0) # metapath_num x node_num x ndim
            if self.mp_attn is None:
                all_out_x[start_node_type] = all_out_x[start_node_type].mean(dim=0)
            else:
                attn = self.mp_attn[i](all_out_x[start_node_type]) # metapath_num x node_num x 1
                node_nums = eval('batch.graph.%s_num' % start_node_type)
                attn = list(attn.split(node_nums, dim=1))
                for idx in range(len(attn)):
                    attn[idx] = attn[idx].mean(dim=1).softmax(dim=0).unsqueeze(-1).repeat(1, node_nums[idx], 1)
                attn = torch.cat(attn, dim=1)
                all_out_x[start_node_type] = all_out_x[start_node_type] * attn
                all_out_x[start_node_type] = all_out_x[start_node_type].sum(dim=0) # node_num * ndim
        final_x = x.new_zeros(x.shape)
        final_x = final_x.masked_scatter_(batch.graph.question_mask.unsqueeze(-1), all_out_x['q'])
        final_x = final_x.masked_scatter_(batch.graph.table_mask.unsqueeze(-1), all_out_x['t'])
        final_x = final_x.masked_scatter_(batch.graph.column_mask.unsqueeze(-1), all_out_x['c'])
        return final_x

    def propagate_attention(self, g, is_hetero):
        if is_hetero:
            for i, etype in enumerate([('src', 'to', 'dst'), ('dst', 'to', 'dst')]):
                g.apply_edges(src_dot_dst('k', 'q', 'score'), etype=etype)
                g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)), etype=etype)
                g.update_all(fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv%s' % i), etype=etype)
                g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z%s' % i), etype=etype)
            g.apply_nodes(div_by_z_hetero('wv0', 'wv1', 'z0', 'z1', 'o'), ntype='dst')
            out_x = g.nodes['dst'].data['o']
        else:
            g.apply_edges(src_dot_dst('k', 'q', 'score'))
            g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
            g.update_all(fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))
            g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
            out_x = g.ndata['o']
        return out_x
