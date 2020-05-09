import heapq
import scipy.spatial
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("../ReedsSheppPath/")
try:
    from Astar_heuristic import dp_planning, calc_obstacle_map
    import Reeds_Shepp as rs
    from Car import move, check_car_collision, MAX_STEER, WB, plot_car
except:
    raise


XY_GRID_RESOLUTION = 0.25  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.1  # [m] path interporate resolution
N_STEER = 20.0  # number of steer command
H_COST = 1.0
VR = 0.2  # robot radius

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 10.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 5.0  # Heuristic cost

show_animation = False


class Node:

    def __init__(self, xind, yind, yawind, direction,
                 xlist, ylist, yawlist, directions,
                 steer=0.0, pind=None, cost=None):
        self.xind = xind #x index
        self.yind = yind #y index
        self.yawind = yawind #yaw index
        self.direction = direction #moving direction, forward=true, backwards=false
        self.xlist = xlist #x position
        self.ylist = ylist #y position
        self.yawlist = yawlist #yaw angle
        self.directions = directions #directions of each point
        self.steer = steer #steer input
        self.pind = pind #parent index
        self.cost = cost #cost


class Path:

    def __init__(self, xlist, ylist, yawlist, directionlist, cost):
        self.xlist = xlist
        self.ylist = ylist
        self.yawlist = yawlist
        self.directionlist = directionlist
        self.cost = cost


class KDTree:
    """
    Nearest neighbor search class with KDTree
    """

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        """
        Search NN
        inp: input data, single frame or multi frame
        """

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []

            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist

        dist, index = self.tree.query(inp, k=k)
        return index, dist

    def search_in_distance(self, inp, r):
        """
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index


class Config:

    def __init__(self, ox, oy, xyreso, yawreso):
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.minx = round(min_x_m / xyreso)
        self.miny = round(min_y_m / xyreso)
        self.maxx = round(max_x_m / xyreso)
        self.maxy = round(max_y_m / xyreso)

        self.xw = round(self.maxx - self.minx)
        self.yw = round(self.maxy - self.miny)

        self.minyaw = round(- math.pi / yawreso) - 1
        self.maxyaw = round(math.pi / yawreso)
        self.yaww = round(self.maxyaw - self.minyaw)


def calc_motion_inputs():

    for steer in np.concatenate((np.linspace(int(-MAX_STEER), int(MAX_STEER), int(N_STEER)),[0.0])):
        for d in [1, -1]:
            yield [steer, d]


def get_neighbors(current, config, ox, oy, kdtree):

    for steer, d in calc_motion_inputs():
        node = calc_next_node(current, steer, d, config, ox, oy, kdtree)
        if node and verify_index(node, config):
            yield node


def calc_next_node(current, steer, direction, config, ox, oy, kdtree):

    x, y, yaw = current.xlist[-1], current.ylist[-1], current.yawlist[-1]

    arc_l = XY_GRID_RESOLUTION * 1.5
    xlist, ylist, yawlist = [], [], []
    for _ in np.arange(0, arc_l, MOTION_RESOLUTION):
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        xlist.append(x)
        ylist.append(y)
        yawlist.append(yaw)

    if not check_car_collision(xlist, ylist, yawlist, ox, oy, kdtree):
        return None

    d = direction == 1
    xind = round(x / XY_GRID_RESOLUTION)
    yind = round(y / XY_GRID_RESOLUTION)
    yawind = round(yaw / YAW_GRID_RESOLUTION)

    addedcost = 0.0

    if d != current.direction:
        addedcost += SB_COST

    # steer penalty
    addedcost += STEER_COST * abs(steer)

    # steer change penalty
    addedcost += STEER_CHANGE_COST * abs(current.steer - steer)

    cost = current.cost + addedcost + arc_l

    node = Node(xind, yind, yawind, d, xlist,
                ylist, yawlist, [d],
                pind=calc_index(current, config),
                cost=cost, steer=steer)

    return node


def is_same_grid(n1, n2):
    if n1.xind == n2.xind and n1.yind == n2.yind and n1.yawind == n2.yawind:
        return True
    return False


def analytic_expantion(current, goal, c, ox, oy, kdtree):

    sx = current.xlist[-1]
    sy = current.ylist[-1]
    syaw = current.yawlist[-1]

    gx = goal.xlist[-1]
    gy = goal.ylist[-1]
    gyaw = goal.yawlist[-1]

    max_curvature = math.tan(MAX_STEER) / WB
    paths = rs.calc_paths(sx, sy, syaw, gx, gy, gyaw,
                          max_curvature, step_size=MOTION_RESOLUTION)

    if not paths:
        return None

    best_path, best = None, None

    for path in paths:
        if check_car_collision(path.x, path.y, path.yaw, ox, oy, kdtree):
            cost = calc_rs_path_cost(path)
            if not best or best > cost:
                best = cost
                best_path = path

    return best_path


def update_node_with_analystic_expantion(current, goal,
                                         c, ox, oy, kdtree):
    apath = analytic_expantion(current, goal, c, ox, oy, kdtree)

    if apath:
        plt.plot(apath.x, apath.y)
        fx = apath.x[1:]
        fy = apath.y[1:]
        fyaw = apath.yaw[1:]

        fcost = current.cost + calc_rs_path_cost(apath)
        fpind = calc_index(current, c)

        fd = []
        for d in apath.directions[1:]:
            fd.append(d >= 0)

        fsteer = 0.0
        fpath = Node(current.xind, current.yind, current.yawind,
                     current.direction, fx, fy, fyaw, fd,
                     cost=fcost, pind=fpind, steer=fsteer)
        return True, fpath

    return False, None


def calc_rs_path_cost(rspath):

    cost = 0.0
    for l in rspath.lengths:
        if l >= 0:  # forward
            cost += l
        else:  # back
            cost += abs(l) * BACK_COST

    # swich back penalty
    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:  # switch back
            cost += SB_COST

    # steer penalyty
    for ctype in rspath.ctypes:
        if ctype != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # ==steer change penalty
    # calc steer profile
    nctypes = len(rspath.ctypes)
    ulist = [0.0] * nctypes
    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = - MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = MAX_STEER

    for i in range(len(rspath.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost


def hybrid_a_star_planning(start, goal, ox, oy, xyreso, yawreso):
    """
    start
    goal
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    xyreso: grid resolution [m]
    yawreso: yaw angle resolution [rad]
    """

    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])
    tox, toy = ox[:], oy[:]

    obkdtree = KDTree(np.vstack((tox, toy)).T)

    config = Config(tox, toy, xyreso, yawreso)

    nstart = Node(round(start[0] / xyreso), round(start[1] / xyreso), round(start[2] / yawreso),
                  True, [start[0]], [start[1]], [start[2]], [True], cost=0)
    ngoal = Node(round(goal[0] / xyreso), round(goal[1] / xyreso), round(goal[2] / yawreso),
                 True, [goal[0]], [goal[1]], [goal[2]], [True])

    openList, closedList = {}, {}

    _, _, h_dp = dp_planning(nstart.xlist[-1], nstart.ylist[-1],
                             ngoal.xlist[-1], ngoal.ylist[-1], ox, oy, xyreso, VR)
    #pq is the priority queue
    pq = []
    openList[calc_index(nstart, config)] = nstart
    #adding elements to the current heap
    heapq.heappush(pq, (calc_cost(nstart, h_dp, ngoal, config),
                        calc_index(nstart, config)))

    while True:
        if not openList:
            print("Error: Cannot find path, No open set")
            return [], [], []

        cost, c_id = heapq.heappop(pq)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue

        if show_animation:  # pragma: no cover
            plt.plot(current.xlist[-1], current.ylist[-1], "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closedList.keys()) % 10 == 0:
                plt.pause(0.001)

        isupdated, fpath = update_node_with_analystic_expantion(
            current, ngoal, config, ox, oy, obkdtree)

        if isupdated:
            break

        for neighbor in get_neighbors(current, config, ox, oy, obkdtree):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor not in openList \
                    or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(
                    pq, (calc_cost(neighbor, h_dp, ngoal, config),
                         neighbor_index))
                openList[neighbor_index] = neighbor

    path = get_final_path(closedList, fpath, nstart, config)
    return path

#n=nstart/neighbur
def calc_cost(n, h_dp, goal, c):
    ind = (n.yind - c.miny) * c.xw + (n.xind - c.minx)
    if ind not in h_dp:
        return n.cost + 999999999  # collision cost
    return n.cost + H_COST * h_dp[ind].cost


def get_final_path(closed, ngoal, nstart, config):
    rx, ry, ryaw = list(reversed(ngoal.xlist)), list(
        reversed(ngoal.ylist)), list(reversed(ngoal.yawlist))
    direction = list(reversed(ngoal.directions))
    nid = ngoal.pind
    finalcost = ngoal.cost

    while nid:
        n = closed[nid]
        rx.extend(list(reversed(n.xlist)))
        ry.extend(list(reversed(n.ylist)))
        ryaw.extend(list(reversed(n.yawlist)))
        direction.extend(list(reversed(n.directions)))

        nid = n.pind

    rx = list(reversed(rx))
    ry = list(reversed(ry))
    ryaw = list(reversed(ryaw))
    direction = list(reversed(direction))

    # adjust first direction
    direction[0] = direction[1]

    path = Path(rx, ry, ryaw, direction, finalcost)

    return path


def verify_index(node, c):
    xind, yind = node.xind, node.yind
    if xind >= c.minx and xind <= c.maxx and yind >= c.miny \
            and yind <= c.maxy:
        return True

    return False


def calc_index(node, c):
    ind = (node.yawind - c.minyaw) * c.xw * c.yw + \
        (node.yind - c.miny) * c.xw + (node.xind - c.minx)
    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind


def main():
    print("Start Hybrid A* planning")
    import time
    import Point_Cloud as map
    import T265_Tracking_Camera as t265
    import D435_Depth_Camera as d435
    import cv2
    import base64
    import threading
    import copy
    import traceback

    t265Obj = t265.rs_t265()
    d435Obj = d435.rs_d435(framerate=30, width=480, height=270)
    mapObj = map.mapper()
    s=0
    with t265Obj, d435Obj:
        try:
            while True: # while pos isn't within a certain distance of the goal position try this, 
                #once it is input a new goal point, minus the previous goal point from current one to keep within radius

                # Get frames of data - points and global 6dof
                tik=time.perf_counter()
                pos, r, conf, _ = t265Obj.get_frame()

                frame, rgbImg = d435Obj.getFrame()
                points = d435Obj.deproject_frame(frame)
                mapObj.update(points, pos, r)

                try:
                    
                    x = np.digitize(pos[0], mapObj.xBins) - 1
                    y = np.digitize(pos[1], mapObj.yBins) - 1
                    z = np.digitize(pos[2], mapObj.zBins) - 1
                    z2= np.digitize(pos[2], mapObj.zBins) - 2
                    z3= np.digitize(pos[2], mapObj.zBins) - 0

                    gridSlice1=copy.copy(mapObj.grid[:,:,z])
                    gridSlice2=copy.copy(mapObj.grid[:,:,z2])
                    gridSlice3=copy.copy(mapObj.grid[:,:,z3])

                    gridSlice = np.sum([gridSlice1, gridSlice2, gridSlice3], axis=0)
                    grid = gridSlice
                    
                    empty = np.zeros((mapObj.xDivisions, mapObj.yDivisions),dtype=np.float32)
                    img = cv2.merge((grid, empty, empty))
                    img = cv2.transpose(img)
                    img = cv2.circle(img, (x, y), 5, (0, 1, 0), 2)

                    vec = np.asarray([20, 0, 0])
                    vec = r.apply(vec)  # Aero-ref -> Aero-body

                    vec[0] += x 
                    vec[1] += y

                    img = cv2.line(img, (x, y), (int(vec[0]), int(vec[1])), (0, 0, 1), 2)
                    img = cv2.resize(img, (540, 540))
                    cv2.imshow('map', img)
                    cv2.waitKey(1)

                    #defining x and y coordinates of obstacles
                    ox, oy = [], []

                    for i in np.arange(-5,5,0.5):
                        ox.append(i)
                        oy.append(-5)
                    for i in np.arange(-5,5,0.5):
                        ox.append(5)
                        oy.append(i)
                    for i in np.arange(-5,5.5,0.5):
                        ox.append(i)
                        oy.append(5)
                    for i in np.arange(-5,5,0.5):
                        ox.append(-5)
                        oy.append(i)
                    
                    #grid needs to be scaled properly so that when it index's its doing it to the same 
                    #size grid as the one point cloud uses, use x,y bins for this
                    grid = cv2.transpose(grid)
                    for i in range(grid.shape[0]):
                        for j in range(grid.shape[1]):
                            if grid[i][j] > 0:
                                ox.append(mapObj.xBins[i])
                                oy.append(mapObj.yBins[j])

                    # Should have North as 90 degrees
                    # Set Initial parameters, float
                    # Need to have a way of making the function still generate a path if the start is within range of an obstacle
                    yaw_angle = r.as_euler('zyx', degrees=True)

                    start = [pos[0], pos[1], np.deg2rad(90.0 - yaw_angle[0])]#90 faces to the top, 0 to the right, -90 towards the bottom
                    goal = [0.0, 3.0, np.deg2rad(90.0)]
    

                    plt.plot(ox, oy, ".k")
                    rs.plot_arrow(start[0], start[1], start[2], fc='g')
                    rs.plot_arrow(goal[0], goal[1], goal[2])
                    plt.grid(True)
                    plt.axis("equal")

                    
                    if s == 0:
                        
                        
                        path = hybrid_a_star_planning(start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
                        tok=time.perf_counter()
                        print(f"Path Planner in {tok - tik:0.4f} seconds")

                        xpath = path.xlist
                        ypath = path.ylist
                        yawpath = path.yawlist
                        directionpath = path.directionlist
                        s=s+1
                        for ix, iy, iyaw in zip(xpath, ypath, yawpath):
                            plt.cla()
                            plt.plot(ox, oy, ".k")
                            plt.plot(xpath, ypath, "-r", label="Hybrid A* path")
                            plt.grid(True)
                            plt.axis("equal")
                            plot_car(ix, iy, iyaw)
                            plt.pause(0.0001)

                        print(__file__ + " done!!")

                    elif s != 0:
                     #use the obstacle map to check if any of the new obstacles will cause a collision with the path
                     #if they do then calculate a new path, if they don't then continue along path
                        ox1 = [iox / XY_GRID_RESOLUTION for iox in ox] 
                        oy1 = [ioy / XY_GRID_RESOLUTION for ioy in oy]
                        obmap, minx, miny, maxx, maxy, xw, yw = calc_obstacle_map(ox1, oy1, XY_GRID_RESOLUTION, VR)

                        # need this to run through the x and y values and stop if they're within range of an obstacle 
                        #divide path.xlist by the resolution
                        for ind in range(len(path.xlist)): 
                            if obmap[int(round((path.xlist[ind]/XY_GRID_RESOLUTION) - minx))][int(round((path.ylist[ind]/XY_GRID_RESOLUTION) - miny))]:
                                tic=time.perf_counter()
                                path = hybrid_a_star_planning(start, goal, ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
                                toc=time.perf_counter()
                                print(f"Path Planner in {toc - tic:0.4f} seconds")
                                xpath = path.xlist
                                ypath = path.ylist
                                yawpath = path.yawlist
                                directionpath = path.directionlist
                                for ix, iy, iyaw in zip(xpath, ypath, yawpath):
                                    plt.cla()
                                    plt.plot(ox, oy, ".k")
                                    plt.plot(xpath, ypath, "-r", label="Hybrid A* path")
                                    plt.grid(True)
                                    plt.axis("equal")
                                    plot_car(ix, iy, iyaw)
                                    plt.pause(0.0001)
                                print(__file__ + " done!!")
                                break
                            #else:
                            #    continue

                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    traceback.print_exc(file=sys.stdout)
                    
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
