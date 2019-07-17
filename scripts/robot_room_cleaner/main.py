from collections import defaultdict, deque
from enum import Enum
from itertools import islice
from typing import Tuple

from robot_room_cleaner.robot import Robot


class TileState(Enum):
    WALL = "█"
    DIRTY = "░"
    CLEAN = " "
    UNSEEN = "?"


class RobotController:
    left_turns = {
        (0, 1): (-1, 0),
        (-1, 0): (0, -1),
        (0, -1): (1, 0),
        (1, 0): (0, 1)
    }
    right_turns = {
        (0, 1): (1, 0),
        (1, 0): (0, -1),
        (0, -1): (-1, 0),
        (-1, 0): (0, 1)
    }

    def __init__(self, robot: Robot, direction=(-1, 0)):
        self.robot = robot
        self.direction = (-1, 0)
        self.pos = (0, 0)
        self.room_state = defaultdict(lambda: TileState.UNSEEN)
        self.room_state[self.pos] = TileState.DIRTY

    def move_multiple(self, *directions: Tuple[int, int]):
        for dir in directions:
            if not self.move(dir):
                break

    def move(self, direction: Tuple[int, int], verbosity="Minimal"):
        if verbosity and verbosity.lower() in ["maximal"]:
            verbose = True
        else:
            verbose = False
        if verbose:
            print("start")
            print(self)
        while self.direction != direction:
            det = self.direction[0]*direction[1]-self.direction[1]*direction[0]
            if det >= 0:
                self.turn_left(verbose)
            else:
                self.turn_right(verbose)
        result = self.move_forward(verbose)
        if verbose or (verbosity.lower() == "minimal"):
            print(self)
        return result

    def move_forward(self, verbose=False):
        row, col = self.pos
        new_pos = (row + self.direction[0], col + self.direction[1])
        if self.robot.move():
            if self.room_state[new_pos] == TileState.UNSEEN:
                self.room_state[new_pos] = TileState.DIRTY
            self.pos = new_pos
            if verbose:
                print("move")
                print(self)
            return True
        self.room_state[new_pos] = TileState.WALL
        return False


    def turn_left(self, verbose=False):
        self.direction = RobotController.left_turns[self.direction]
        self.robot.turnLeft()
        if verbose:
            print("left")
            print(self)

    def turn_right(self, verbose=False):
        self.direction = RobotController.right_turns[self.direction]
        self.robot.turnRight()
        if verbose:
            print("right")
            print(self)

    def clean(self, verbose=False):
        self.robot.clean()
        self.room_state[self.pos] = TileState.CLEAN
        if verbose:
            print("clean")
            print(self)

    def __repr__(self):
        visited = self.room_state.keys()
        min_row = min(map(lambda x: x[0], visited))
        max_row = max(map(lambda x: x[0], visited))
        min_col = min(map(lambda x: x[1], visited))
        max_col = max(map(lambda x: x[1], visited))

        known_map = [[
            (row, col, self.room_state[(row, col)])
            for col in range(min_col, max_col+1)
        ] for row in range(min_row, max_row+1)]

        robot_chars = {
            (0, 1): ">",
            (0, -1): "<",
            (1, 0): "v",
            (-1, 0): "^"
        }
        r_char = robot_chars.get(self.direction, "?")
        hor_bound = "?" * ((max_col-min_col)+3)
        rows = [hor_bound]+["?" + "".join([
            ts.value if (i, j) != self.pos else r_char
            for i, j, ts in row
        ]) + "?" for row in known_map]+[hor_bound]
        return "\n".join(rows)


class Graph:
    def __init__(self):
        self.nodes = dict()
        self.node_data = dict()

    def __contains__(self, node):
        return node in self.nodes

    def add_node(self, node, data=None):
        if data is None:
            data = dict()
        if node not in self.nodes:
            self.nodes[node] = set()
        self.node_data.setdefault(node, {}).update(**data)

    def add_edge(self, node1, node2):
        for node in [node1, node2]:
            if node not in self.nodes:
                self.add_node(node)
        self.nodes[node1].add(node2)
        self.nodes[node2].add(node1)

    def bfs(self, start, predicate=lambda node, data: True):
        fifo = deque([(start, None)])
        processed = set()
        parents = dict()
        while fifo:
            # Take the next node:
            node, parent = fifo.popleft()
            if node not in processed:
                # don't process again
                processed.add(node)
                # add the parent:
                parents[node] = parent
                # add the neighbors
                for neighbor in self.nodes[node]:
                    # we use the node as the parent:
                    fifo.append((neighbor, node))
                if predicate(node, self.node_data[node]):
                    def shortest_path():
                        cur_node = node
                        yield cur_node
                        while parents[cur_node] is not None:
                            cur_node = parents[cur_node]
                            yield cur_node
                    yield {
                        "node": node,
                        "shortest_path": shortest_path(),
                        "data": self.node_data[node]
                    }


def add_neighbors(g: Graph, pos: Tuple[int, int]):
    if pos not in g:
        g.add_node(pos)

    row, col = pos
    for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        row_ = row+i
        col_ = col+j
        node = (row_, col_)
        if node not in g:
            g.add_node(node)
            g.add_edge(pos, node)


def clean_room(robot: Robot):
    r_c = RobotController(robot)
    r_c.clean()  # clean the first tile

    g = Graph()
    add_neighbors(g, r_c.pos)
    is_unseen = lambda pos, _: r_c.room_state[pos] == TileState.UNSEEN
    unseen_tiles = list(islice(g.bfs(r_c.pos, is_unseen), 1))
    while unseen_tiles:
        tile_data = unseen_tiles[0]
        path = reversed(list(tile_data["shortest_path"]))
        for row_, col_ in islice(path, 1, None):
            row, col = r_c.pos
            dir = (row_-row, col_-col)
            r_c.move(dir)
        r_c.clean()
        add_neighbors(g, r_c.pos)
        unseen_tiles = list(islice(g.bfs(r_c.pos, is_unseen), 1))


