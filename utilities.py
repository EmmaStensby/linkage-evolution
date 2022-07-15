import numpy as np
import sys

def print_matrix(mat):
    for row in mat:
        print(row)

def fti(f, c):
    return int(f*c)

def distance(P0, P1):
    return np.sqrt((P0[0] - P1[0])**2 + (P0[1] - P1[1])**2)

def vector_end_pos(start_pos, angle, length):
    v = np.array([length * np.cos(angle), length * np.sin(angle)])
    end_pos = np.array([start_pos[0] + v[0], start_pos[1] + v[1]])
    return end_pos

def middle_of_positions(pos1, pos2):
    return np.array([pos1[0] + (pos2[0]-pos1[0])/2, pos1[1] + (pos2[1]-pos1[1])/2])

def vector_angle(P0, P1):
    # P0 is the "origin point"
    # P1 in the "end point"
    V = np.array([P1[0] - P0[0], P1[1] - P0[1]])
    angle = np.arctan2(V[1], V[0])
    while angle < 0:
        angle += np.pi*2
    while angle > np.pi*2:
        angle -= np.pi*2
    return angle

def rotate_around_point_to_angle(center, goal_angle, point):
    d = distance(center, point)
    return vector_end_pos(center, goal_angle, d)

def circle_intersection(P0, P1, r0, r1, direction, threshold, backup_angle=None):
    d = np.sqrt((P0[0] - P1[0])**2 + (P0[1] - P1[1])**2)
    
    if d-threshold > r0 + r1:
        # No solution exists
        return None
    elif d+threshold < np.sqrt((r0-r1)**2):
        # No solution exists
        return None
    elif d == 0 and r0 == r1:
        # Infinite solutions exist
        if backup_angle == None:
            return None
        P2 = np.array([r0 * np.cos(backup_angle), r0 * np.sin(backup_angle)])
        return P2
    elif np.abs(d - r0 + r1) < threshold:
        # One solution exists
        if d > r0 + r1:
            d = d - threshold
        elif d < np.sqrt((r0-r1)**2):
            d  = d + threshold
        a = r0
        P2 = np.array([P0[0] + a*(P1[0] - P0[0])/d, P0[1] + a*(P1[1] - P0[1])/d])
        return P2
    else:
        if d > r0 + r1:
            d = d - threshold
        elif d < np.sqrt((r0-r1)**2):
            d  = d + threshold
        # Two solutions exist
        a = (r0**2 - r1**2 + d**2)/(2*d)
        h = r0**2 - a**2
        if h < 0:
            if h < -threshold:
                print("Negative value detected")
                print(h)
                print(d)
                print(r0)
                print(r1)
                print(P0)
                print(P1)
            h = 0
        h = np.sqrt(h)
        P2 = np.array([P0[0] + a*(P1[0] - P0[0])/d, P0[1] + a*(P1[1] - P0[1])/d])
        P3_1 = np.array([P2[0] + h*(P1[1] - P0[1])/d, P2[1] - h*(P1[0] - P0[0])/d])
        P3_2 = np.array([P2[0] - h*(P1[1] - P0[1])/d, P2[1] + h*(P1[0] - P0[0])/d])
        if direction == 0:
            return P3_1
        else:
            return P3_2

def center_path_on_zero(path_x, path_y):
    x_middle = sum(path_x)/len(path)
    y_middle = sum(path_y)/len(path)
    new_path_x = []
    new_path_y = []
    for i in range(len(path_x)):
        new_path_x.append(path_x[i] - x_middle)
        new_path_y.append(path_y[i] - y_middle)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def flatten_path(path_x, path_y, length):
    path = []
    for x, y in zip(path_x, path_y):
        path.append(x)
        path.append(y)
    while len(path) < length*2:
        path.append(0)
    return path
