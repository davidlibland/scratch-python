from robot_room_cleaner.main import RobotController, clean_room
from robot_room_cleaner.robot import Robot

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

r_c = RobotController(r)

clean_room(r)