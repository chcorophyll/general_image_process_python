class UnionFind(object):

    def __init__(self, components):
        self.parent = list(range(components))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        parent_x = self.find(x)
        parent_y = self.find(y)
        self.parent[parent_y] = parent_x

    def is_connected(self, x, y):
        parent_x = self.find(x)
        parent_y = self.find(y)
        return parent_x == parent_y

