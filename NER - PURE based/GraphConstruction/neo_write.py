from graph1 import *
nodes = {}
g = CustomGraph()
g.clear()
for line in open("F:\\ECpy\\learn\\Master\\22spring\\CS7643\\triple.txt"):
    triple = eval(line)
    if triple[0] is not None and triple[0] not in nodes.keys():
        nodes[triple[0]] = g.create_node(triple[0], ''.join(triple[0]))
    if triple[2] is not None and triple[2] not in nodes.keys():
        nodes[triple[2]] = g.create_node(triple[0], ''.join(triple[0]))
    g.create_relationship(nodes[triple[0]], triple[1], nodes[triple[2]])
