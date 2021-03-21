import networkx as nx
import matplotlib.pyplot as plt
 
nodes=[
    'leader',
    'follower1',
    'follower2',
    'follower3',
    'follower4',
    'follower5',
    'follower6',
    'follower7',
    'follower8',
    'follower9'
]
 
G=nx.Graph()
# G=nx.DiGraph()
# G=nx.MultiGraph()
 
for node in nodes:
    G.add_node(node)
 
edges=[
    ('leader','follower1'),
    ('leader','follower2'),
    ('follower1','follower2'),
    ('follower1','follower3'),
    ('follower2','follower3'),
    ('follower2','follower4'),
    ('follower3','follower4'),
    ('follower3','follower5'),
    ('follower4','follower5'),
    ('follower4','follower6'),
    ('follower5','follower6'),
    ('follower5','follower7'),
    ('follower6','follower7'),
    ('follower6','follower8'),
    ('follower7','follower8'),
    ('follower7','follower9'),
    ('follower8','follower9')
]
 
r=G.add_edges_from(edges)

nx.draw(G, with_labels=True,node_color='y',)
plt.show()