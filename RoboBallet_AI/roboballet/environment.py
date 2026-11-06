from dataclasses import dataclass
from .config import Workcell, generate_workcell

@dataclass
class Scene:
    width: int
    height: int
    starts: list
    tasks: list
    obstacles: set

def make_scene(robots=4, tasks=12, seed=7, width=14, height=14, obstacle_density=.12):
    d = generate_workcell(Workcell(width, height, robots, tasks, obstacle_density, seed))
    return Scene(d["width"], d["height"], d["starts"], d["tasks"], d["obstacles"])

def in_bounds(scene, p):
    return 0 <= p[0] < scene.width and 0 <= p[1] < scene.height

def free_cell(scene, p):
    return in_bounds(scene, p) and p not in scene.obstacles
