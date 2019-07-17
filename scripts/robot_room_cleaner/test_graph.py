from itertools import product, islice

from robot_room_cleaner.main import Graph


def test_bfs():
    g = Graph()
    for row, col in product(range(-2, 3), range(-2, 3)):
        node = (row, col)
        node_l = (row, col-1)
        node_u = (row-1, col)
        g.add_node(node)
        g.add_edge(node, node_l)
        g.add_edge(node, node_u)
    g.add_node((0, 0), {"tile_state": "unseen"})
    g.add_node((-1, -1), {"tile_state": "unseen"})
    g.add_node((2, 1), {"tile_state": "unseen"})

    unseen_tiles = g.bfs(
        start=(0,0),
        predicate=lambda node, ts: ts.get("tile_state") == "unseen"
    )
    for result in unseen_tiles:
        tile = result["node"]
        path = result["shortest_path"]
        data = result["data"]
        print("path: %s --> %s"
              % (tile, "->".join([str(node) for node in islice(path,5)])))
        print("data:\n%s" % data)
