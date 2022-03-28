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
            HANSQLLayer(self.ndim, args.num_heads, args.dropout, args.node_type_share_weights)
            for _ in range(self.num_layers)
        ])

    def forward(self, x, batch):
        for i in range(self.num_layers):
            x = self.gnn_layers[i](x, batch)
        return x


class HANSQLLayer(nn.Module):
    def __init__(self, ndim, num_heads=8, feat_drop=0.2, share=False):
        super(HANSQLLayer, self).__init__()
        self.ndim = ndim
        self.num_heads = num_heads
        self.d_k = self.ndim // self.num_heads
        self.affine_q = {}
        self.affine_k = {}
        self.affine_v = {}
        self.affine_o = {}
        self.layernorm = {}
        self.feat_dropout = {}
        self.ffn = {}
        self.mp_attn = {}
        for node_type in ['q', 't', 'c']:
            if share and node_type != 'q':
                self.affine_q[node_type] = self.affine_q['q']
                self.affine_k[node_type] = self.affine_k['q']
                self.affine_v[node_type] = self.affine_v['q']
                self.affine_o[node_type] = self.affine_o['q']
                self.layernorm[node_type] = self.layernorm['q']
                self.feat_dropout[node_type] = self.feat_dropout['q']
                self.ffn[node_type] = self.ffn['q']
                self.mp_attn[node_type] = self.mp_attn['q']
            else:
                self.affine_q[node_type] = nn.Linear(self.ndim, self.ndim)
                self.affine_k[node_type] = nn.Linear(self.ndim, self.ndim, bias=False)
                self.affine_v[node_type] = nn.Linear(self.ndim, self.ndim, bias=False)
                self.affine_o[node_type] = nn.Linear(self.ndim, self.ndim)
                self.layernorm[node_type] = nn.LayerNorm(self.ndim)
                self.feat_dropout[node_type] = nn.Dropout(p=feat_drop)
                self.ffn[node_type] = FFN(self.ndim)
                self.mp_attn[node_type] = nn.Sequential(
                    nn.Linear(self.ndim, self.ndim),
                    nn.Tanh(),
                    nn.Linear(self.ndim, 1, bias=False)
                )

    def forward(self, x, batch):
        qx = x.masked_select(batch.graph.question_mask.unsqueeze(-1)).view(-1, self.ndim)
        tx = x.masked_select(batch.graph.table_mask.unsqueeze(-1)).view(-1, self.ndim)
        cx = x.masked_select(batch.graph.column_mask.unsqueeze(-1)).view(-1, self.ndim)
        print(qx.shape, tx.shape, cx.shape)
        q, k, v = {}, {}, {}
        for node_type in ['q', 't', 'c']:
            q[node_type] = self.affine_q[node_type](self.feat_dropout[node_type](eval(node_type + 'x')))
            k[node_type] = self.affine_k[node_type](self.feat_dropout[node_type](eval(node_type + 'x')))
            v[node_type] = self.affine_v[node_type](self.feat_dropout[node_type](eval(node_type + 'x')))
        all_out_x = {'q': [], 't': [], 'c': []}
        for start_node_type in batch.graph.graphs:
            for g, end_node_type in batch.graph.graphs[start_node_type]:
                with g.local_scope():
                    if start_node_type == end_node_type:
                        g.ndata['q'] = q[start_node_type].view(-1, self.num_heads, self.d_k)
                        g.ndata['k'] = k[end_node_type].view(-1, self.num_heads, self.d_k)
                        g.ndata['v'] = v[end_node_type].view(-1, self.num_heads, self.d_k)
                    else:
                        g.nodes['dst'].data['q'] = q[start_node_type].view(-1, self.num_heads, self.d_k)
                        g.nodes['src'].data['k'] = k[end_node_type].view(-1, self.num_heads, self.d_k)
                        g.nodes['src'].data['v'] = v[end_node_type].view(-1, self.num_heads, self.d_k)
                    out_x = self.propagate_attention(g, start_node_type != end_node_type)
                out_x = self.layernorm[start_node_type](eval(start_node_type + 'x') + self.affine_o[start_node_type](out_x.view(-1, self.num_heads * self.d_k)))
                out_x = self.ffn[start_node_type](out_x)
                all_out_x[start_node_type].append(out_x)
            all_out_x[start_node_type] = torch.stack(all_out_x[start_node_type], dim=0) # metapath_num x node_num x ndim
            attn = self.mp_attn[start_node_type](all_out_x[start_node_type]) # metapath_num x node_num x 1
            print(all_out_x[start_node_type].shape, attn.shape)
            node_nums = eval('batch.graph.%s_num' % start_node_type)
            attn = attn.split(node_nums, dim=1)
            for i in range(len(attn)):
                attn[i] = attn[i].mean(dim=1).softmax(dim=0).unsqueeze(-1).repeat(1, node_nums[i], 1)
            attn = torch.cat(attn, dim=1)
            print(attn.shape)
            all_out_x[start_node_type] *= attn
            all_out_x[start_node_type] = all_out_x[start_node_type].sum(dim=0) # node_num * ndim
            print(all_out_x[start_node_type].shape)
        final_x = x.new_zeros(x.shape)
        final_x.masked_scatter_(batch.graph.question_mask.unsqueeze(-1), all_out_x['q'])
        final_x.masked_scatter_(batch.graph.table_mask.unsqueeze(-1), all_out_x['t'])
        final_x.masked_scatter_(batch.graph.column_mask.unsqueeze(-1), all_out_x['c'])
        print(final_x.shape)
        return final_x

    def propagate_attention(self, g, is_hetero):
        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'))
        g.apply_edges(scaled_exp('score', math.sqrt(self.d_k)))
        # Update node state
        g.update_all(fn.src_mul_edge('v', 'score', 'v'), fn.sum('v', 'wv'))
        g.update_all(fn.copy_edge('score', 'score'), fn.sum('score', 'z'), div_by_z('wv', 'z', 'o'))
        out_x = g.nodes['dst'].data['o'] if is_hetero else g.ndata['o']
        return out_x
