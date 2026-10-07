from collections import defaultdict
from typing import List

class DiscoveryTree:
    def __init__(self):
        self.nodes = {}
        self.children = defaultdict(list)

    def add(self, node):
        self.nodes[node.node_id] = node
        if node.parent_id:
            self.children[node.parent_id].append(node.node_id)

    def leaves(self):
        return [
            n for nid, n in self.nodes.items()
            if nid not in self.children
        ]

    def get(self, node_id):
        return self.nodes[node_id]

    def trajectory(self, node_id):
        out = []
        cur = self.get(node_id)
        while cur:
            out.append(cur)
            cur = self.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(out))
