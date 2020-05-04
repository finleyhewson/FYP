import math
import heapq
import matplotlib.pyplot as plt
import time

show_animation = False


class Node:

    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)


def calc_final_path(ngoal, closedset, reso):
    # generate final course
    rx, ry = [ngoal.x * reso], [ngoal.y * reso]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x * reso)
        ry.append(n.y * reso)
        pind = n.pind

    return rx, ry


def dp_planning(sx, sy, gx, gy, ox, oy, reso, rr):
    """
    sx: start x position [m]
    sy: start y position [m]
    gx: goal x position [m]
    gx: goal x position [m]
    ox: x position list of Obstacles [m]
    oy: y position list of Obstacles [m]
    reso: grid resolution [m]
    rr: robot radius[m]
    """

    nstart = Node(round(sx / reso), round(sy / reso), 0.0, -1)
    ngoal = Node(round(gx / reso), round(gy / reso), 0.0, -1)
    ox = [iox / reso for iox in ox] #divides all of ox by the resolution
    oy = [ioy / reso for ioy in oy] #divides all of oy by the resolution
    
    obmap, minx, miny, maxx, maxy, xw, yw = calc_obstacle_map(ox, oy, reso, rr)
    #t1=time.perf_counter()
    #defines movement in terms of relative positions and gives the cost of each movement 
    motion = get_motion_model()

    #initialising both the yet to visit and visited list
    openset, closedset = dict(), dict()
    openset[calc_index(ngoal, xw, minx, miny)] = ngoal
    pq = []
    pq.append((0, calc_index(ngoal, xw, minx, miny)))

    while 1:
        if not pq:
            break
        cost, c_id = heapq.heappop(pq) #c_id is the current index, heapop returns the smallest data element from the heap pq
        #popping current node out of openset into closed set
        if c_id in openset:
            current = openset[c_id]
            closedset[c_id] = current
            openset.pop(c_id)
        else:
            continue

        # show graph
        if show_animation:  # pragma: no cover
            plt.plot(current.x * reso, current.y * reso, "xc")
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if len(closedset.keys()) % 10 == 0:
                plt.pause(0.001)

        # Remove the item from the open set

        # expand search grid based on motion model, generates child nodes from adjacent squares
        for i, _ in enumerate(motion):
            node = Node(current.x + motion[i][0],
                        current.y + motion[i][1],
                        current.cost + motion[i][2], c_id)
            n_id = calc_index(node, xw, minx, miny)
            #making sure not to use nodes already visited
            if n_id in closedset:
                continue
            #checking node is within boundary and not in obstacle
            if not verify_node(node, obmap, minx, miny, maxx, maxy):
                continue

            if n_id not in openset:
                openset[n_id] = node  # Discover a new node
                heapq.heappush(
                    pq, (node.cost, calc_index(node, xw, minx, miny)))
            else:
                if openset[n_id].cost >= node.cost:
                    # This path is the best until now. record it!
                    openset[n_id] = node
                    heapq.heappush(
                        pq, (node.cost, calc_index(node, xw, minx, miny)))

    rx, ry = calc_final_path(closedset[calc_index(
        nstart, xw, minx, miny)], closedset, reso)
    #t2=time.perf_counter()
    #print(f"Path Planner in {t2 - t1:0.4f} seconds")
    return rx, ry, closedset


def calc_heuristic(n1, n2):
    w = 1.0  # weight of heuristic
    d = w * math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
    return d

#Making sure the node is within range and not in an obstacle
def verify_node(node, obmap, minx, miny, maxx, maxy):

    if node.x < minx:
        return False
    elif node.y < miny:
        return False
    elif node.x >= maxx:
        return False
    elif node.y >= maxy:
        return False
    #Need to subtract minx and miny to scale to the obmap indicies correctly
    if obmap[int(round(node.x-minx))][int(round(node.y-miny))]:
        return False

    return True

#function determines whether position would cause a collision with an obstacle 
def calc_obstacle_map(ox, oy, reso, vr):

    minx = int(round(min(ox)))
    miny = int(round(min(oy)))
    maxx = int(round(max(ox))) 
    maxy = int(round(max(oy)))

    xwidth = round(maxx - minx)
    ywidth = round(maxy - miny)
    # obstacle map generation, determines which positions would cause a collision 
    # with an obstacle given the device's radius
    obmap = [[False for i in range(ywidth)] for i in range(xwidth)]
    #for ix in range(xwidth):
    #    x = ix + minx #the current x position
    #    for iy in range(ywidth):
    #        y = iy + miny #the current y position
    #        #  print(x, y)
    #        for iox, ioy in zip(ox, oy):
    #            d = math.sqrt((iox - x)**2 + (ioy - y)**2) #distance from current x and y position to obstacle
    #            if d <= vr / reso:
    #                obmap[ix][iy] = True
    #                break
    obmap_motion = obmap_motion_model()

    for iox, ioy in zip(ox, oy):
        rox=iox
        roy=ioy
        iox=int(round(iox))
        ioy=int(round(ioy))
        
        for i, _ in enumerate(obmap_motion):
            adjind = [iox + obmap_motion[i][0],
                      ioy + obmap_motion[i][1]]
            
            if not verify_obmap(adjind, minx, miny, maxx, maxy):
                continue

            d = math.sqrt((rox - adjind[0])**2 + (roy - adjind[1])**2)
            if d <= vr / reso:
                ix = adjind[0] - minx
                iy = adjind[1] - miny
                obmap[ix][iy] = True


    return obmap, minx, miny, maxx, maxy, xwidth, ywidth


def calc_index(node, xwidth, xmin, ymin):
    return (node.y - ymin) * xwidth + (node.x - xmin)

#function just giving the option to move in 8 directions(up, down, diagonally, etc..) 
#and the costs of moving in those directions
def get_motion_model():
    # dx, dy, cost
    motion = [[1, 0, 1],
              [0, 1, 1],
              [-1, 0, 1],
              [0, -1, 1],
              [-1, -1, math.sqrt(2)],
              [-1, 1, math.sqrt(2)],
              [1, -1, math.sqrt(2)],
              [1, 1, math.sqrt(2)]]

    return motion

# function needs to be changed depending on how the resolution and car radius change
def obmap_motion_model():
    # dx, dy
    obmap_motion = [[0, 0],
                    [1, 0],
                    [2, 0],
                    [0, 1],
                    [0, 2],
                    [-1, 0],
                    [-2, 0],
                    [0, -1],
                    [0, -2],
                    [-1, -1],
                    [-1, 1],
                    [1, -1],
                    [1, 1],
                    [1, 2],
                    [2, 1],
                    [2, 2],
                    [-1, 2],
                    [-2, 1],
                    [-2, 2],
                    [1, -2],
                    [2, -1],
                    [2, -2],
                    [-1, -2],
                    [-2, -1],
                    [-2, -2]]

    return obmap_motion


def verify_obmap(adjind, minx, miny, maxx, maxy):

    if adjind[0] < minx:
        return False
    elif adjind[1] < miny:
        return False
    elif adjind[0] >= maxx:
        return False
    elif adjind[1] >= maxy:
        return False

    return True


def main():
    print(__file__ + " start!!")

    # start and goal position
    sx = 10.0  # [m]
    sy = 10.0  # [m]
    gx = 50.0  # [m]
    gy = 50.0  # [m]
    grid_size = 2.0  # [m]
    robot_size = 1.0  # [m]

    ox, oy = [], []

    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "xr")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    rx, ry, _ = dp_planning(sx, sy, gx, gy, ox, oy, grid_size, robot_size)

    if show_animation:  # pragma: no cover
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    show_animation = True
    main()
