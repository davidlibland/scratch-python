import random
from enum import Enum
from itertools import product
from typing import List

class TileState(Enum):
    WALL = "█"
    DIRTY = "░"
    SEEN = "·"
    CLEAN = " "

class Robot:
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

    def __init__(self, room: List[List[bool]], row, col):
        self.height = len(room)
        assert self.height > 0
        self.width = len(room[0])
        self.room = room
        self.pos = (row, col)
        self.direction = (-1, 0)
        self.room_state = Robot.build_initial_state(room)

    @staticmethod
    def build_initial_state(room: List[List[bool]]):
        return [
            [TileState.DIRTY if is_empty else TileState.WALL
             for is_empty in row]
            for row in room
        ]

    def __repr__(self):
        robot_chars = {
            (0, 1): ">",
            (0, -1): "<",
            (1, 0): "v",
            (-1, 0): "^"
        }
        r_char = robot_chars.get(self.direction, "?")
        rows = ["".join([
            ts.value if (i, j) != self.pos else r_char
            for j, ts in enumerate(row)
        ]) for i, row in enumerate(self.room_state)]
        return "\n".join(rows)

    def move(self):
        """
        Returns true if the cell in front is open and robot moves into the cell.
        Returns false if the cell in front is blocked and robot stays in the current cell.
        :rtype bool
        """
        row, col = self.pos
        row_ = row + self.direction[0]
        col_ = col + self.direction[1]
        if row_ < 0  or row_ >= self.height:
            return False
        if col_ < 0  or col_ >= self.width:
            return False
        if self.room[row_][col_]:
            # Valid move:
            self.pos = (row_, col_)
            if self.room_state[row_][col_] == TileState.DIRTY:
                self.room_state[row_][col_] = TileState.SEEN
            return True
        return False

    def turnLeft(self):
        """
        Robot will stay in the same cell after calling turnLeft/turnRight.
        Each turn will be 90 degrees.
        :rtype void
        """
        self.direction = self.left_turns[self.direction]

    def turnRight(self):
        """
        Robot will stay in the same cell after calling turnLeft/turnRight.
        Each turn will be 90 degrees.
        :rtype void
        """
        self.direction = self.right_turns[self.direction]

    def clean(self):
        """
        Clean the current cell.
        :rtype void
        """
        i, j = self.pos
        self.room_state[i][j] = TileState.CLEAN

    def is_room_clean(self):
        for i, j in product(range(self.height), range(self.width)):
            if self.room[i][j] and self.room_state[i][j] != TileState.CLEAN:
                return False
        return True


def test_turn_dicts():
    """Tests that the turn dicts invert each other."""
    for state, left in Robot.left_turns.items():
        assert Robot.right_turns[left] == state, \
            "Right turn of %s is not %s" % (left, state)

    for state, right in Robot.right_turns.items():
        assert Robot.left_turns[right] == state, \
            "Left turn of %s is not %s" % (right, state)

def test_simple_walk():
    room = [
        [1, 1, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]
    row = 1
    col = 3
    r = Robot(room, row, col)
    r.turnRight()
    r.turnRight()
    assert r.move(), "Should have a valid move here: \n%s" % r
    assert r.move(), "Should have a valid move here: \n%s" % r
    r.turnLeft()
    assert not r.move(), "Should hit a wall here: \n%s" % r
    r.turnLeft()
    assert r.move(), "Should have a valid move here: \n%s" % r
    r.turnRight()
    assert r.move(), "Should have a valid move here: \n%s" % r
    assert r.move(), "Should have a valid move here: \n%s" % r
    expect = (2, 5)
    assert r.pos == expect, "Should be at %s not %s here: \n%s" \
                            % (expect, r.pos, r)


def get_random_room(width=None, height=None, wall_prob = 1/8):
    width = width if width is not None else random.randint(3, 20)
    height = height if height is not None else random.randint(3, 20)
    room = [
        [
            0 if random.uniform(0, 1) < wall_prob else 1
            for _ in range(width)
        ] for _ in range(height)
    ]
    row = random.randint(0, height)
    col = random.randint(0, width)
    room[row][col] = 1  # make sure we don't start on a wall.
    return room, row, col


def test_random_walk(steps=100, random_room=False):
    if not random_room:
        room = [
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
        row = 1
        col = 3
    else:
        room, row, col = get_random_room()
    r = Robot(room, row, col)
    for i in range(steps):
        action = random.choice(["l", "r", "m", "c"])
        if action == "l":
            r.turnLeft()
        elif action == "r":
            r.turnRight()
        elif action == "m":
            r.move()
        elif action == "c":
            r.clean()
    row, col = r.pos
    assert room[row][col], "Should be in a valid tile on map \n%s\npos: %s" \
        % (r, r.pos)
    print(r)
