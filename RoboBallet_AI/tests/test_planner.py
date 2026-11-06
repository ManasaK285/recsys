from roboballet.environment import make_scene
from roboballet.planner import space_time_astar, plan_all
from roboballet.allocation import allocate_greedy

def test_astar_reaches_goal():
    scene=make_scene(2,3,seed=2)
    start,goal=scene.starts[0],scene.tasks[0]
    path=space_time_astar(scene,start,goal)
    assert path is not None
    assert path[0]==start and path[-1]==goal
    assert all(p not in scene.obstacles for p in path)

def test_planned_paths_complete_tasks():
    scene=make_scene(3,6,seed=4)
    routes=allocate_greedy(scene)
    paths,times=plan_all(scene,routes)
    assert sum(len(r) for r in routes)==len(scene.tasks)
    assert len(paths)==len(scene.starts)
    assert all(t>=0 for t in times)

def test_no_same_time_vertex_conflicts():
    scene=make_scene(3,7,seed=9)
    paths,_=plan_all(scene,allocate_greedy(scene))
    occupied={}
    for rid,path in enumerate(paths):
        for t,p in enumerate(path):
            assert (t,p) not in occupied or occupied[(t,p)]==rid
            occupied[(t,p)]=rid
