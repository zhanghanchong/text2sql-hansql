#coding=utf8
class Metapath:
    def __init__(self, init_node_type: str):
        self.node_types = [init_node_type]
        self.edge_types = []

    def add(self, node_type: str, edge_type: str):
        self.node_types.append(node_type)
        self.edge_types.append(edge_type)

    def has_schema_type(self):
        return ('table' in self.node_types) or ('column' in self.node_types)

    def copy(self):
        new_metapath = Metapath(self.node_types[0])
        for node_type, edge_type in zip(self.node_types[1:], self.edge_types):
            new_metapath.add(node_type, edge_type)
        return new_metapath

    def __len__(self):
        return len(self.edge_types)

    def __hash__(self):
        h = 0
        for node_type in self.node_types:
            h ^= hash(node_type)
        for edge_type in self.edge_types:
            h ^= hash(edge_type)
        return h

    def __eq__(self, other):
        return isinstance(other, Metapath) and self.node_types == other.node_types and self.edge_types == other.edge_types

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        assert len(self.node_types) - len(self.edge_types) == 1
        s = '(%s)' % self.node_types[0]
        for node_type, edge_type in zip(self.node_types[1:], self.edge_types):
            s += '-[%s]->(%s)' % (edge_type, node_type)
        return s
