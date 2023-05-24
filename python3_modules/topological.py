def topological_sort(digraph):
    # Digraph: a dictionary
        # key: a node
        # value: a set of nodes that the key node points to

    # Construct a dictionary mapping nodes to their indegrees
    indegrees = {node : 0 for node in digraph}
    for node in digraph:
        for neighbor in digraph[node]:
            indegrees[neighbor] += 1

    # Track nodes with no incoming edges
    nodes_with_no_incoming_edges = []
    for node in digraph:
        if indegrees[node] == 0:
            nodes_with_no_incoming_edges.append(node)

    # Initially, our ordering will have no nodes
    topological_ordering = []

    # As long as there are no nodes with no incoming edges, or nodes with indegree = 0,
    while len(nodes_with_no_incoming_edges) > 0:

        # Append those nodes to our ordering
        node = nodes_with_no_incoming_edges.pop()
        topological_ordering.append(node)

        # Decrement the indegree of the nodes that our 0-indegree node points to 
        for neighbor in digraph[node]:
            indegrees[neighbor] -= 1

            # This is smart: since this is the only time we change the indegree
            # of nodes, might as well check if there's a new 0-indegree node
            if indegrees[neighbor] == 0:
                nodes_with_no_incoming_edges.append(neighbor)

    # This means that we've run out of 0-indegree nodes. Let's check if we
    # still have nodes left
    # Because if we do, that means that we have a cycle, and thus there is
    # no ordering that can be done with the given digraph
    if len(topological_ordering) == len(digraph):
        return topological_ordering
    else:
        raise Exception("no can do sir")


def main():
    digraph = {'a':['b'],
               'b':['c'],
               'c':['d'],
               'd':['b']} 

    try: 
        print(topological_sort(digraph))
    except Exception as e:
        print(e)

def main2():
    pass

if __name__ == "__main__":
    main2()