import numpy as np
from utilities import *
from scipy import interpolate
#import unity_channel

ENV = None
def fitness_unity(l, show=False):
    global ENV
    
    path_x, path_y, error = l.get_lowest_path(PATH_LENGTH)

    # invalid
    if error > 0:
        return error,

    path = []
    for x, y in zip(path_x, path_y):
        path.append([x, y])

    if show:
        if ENV == None:
            ENV = unity_channel.Evaluator(54678, show=True)
        data = l.get_unity_data()
        fit = ENV.evaluate(*data, None)
    else:
        pid = mp.current_process()._identity[0]
        if ENV == None:
            ENV = unity_channel.Evaluator(pid)
        data = l.get_unity_data()
        fit = ENV.evaluate(*data, None)
    return -fit,
    
def fitness_normals(path_x, path_y):
    bottom_index = path_y.index(min(path_y))
    
    swap_direction = True
    a = path_x[bottom_index-1]
    b = path_y[bottom_index-1]
    c = path_x[bottom_index]
    d = path_y[bottom_index]
    e = path_x[(bottom_index+1)%len(path_x)]
    f = path_y[(bottom_index+1)%len(path_y)]
    if b < f:
        normal = [d-b, -(c-a)]
        magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
    else:
        normal = [f-d, -(e-c)]
        magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
    if magnitude > 0.01:
        normal[0] = normal[0]/magnitude
        normal[1] = normal[1]/magnitude
        angle = vector_angle([0,0], normal)
        if angle < np.pi:
            swap_direction = False
            
    fitness = 0
    threshold = 0.5
    for i in range(len(path_x)):
        a = path_x[i-1]
        b = path_y[i-1]
        c = path_x[i]
        d = path_y[i]
        normal = [d-b, -(c-a)]
        magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
        if magnitude > 0.01 and np.abs(path_y[bottom_index]-d) < threshold and np.abs(path_y[bottom_index]-b) < threshold:
            normal[0] = normal[0]/magnitude
            normal[1] = normal[1]/magnitude
            angle = vector_angle([0,0], normal)
            if swap_direction:
                angle += np.pi
                if angle > 2*np.pi:
                    angle -= 2*np.pi
            if angle <= np.pi:
                angle = angle/np.pi
                if angle > 0.5:
                    angle = 1 - angle
                fitness += angle
                
    return fitness

def multifitness_normals_lift(path_x, path_y):
    bottom_index = path_y.index(min(path_y))
    
    swap_direction = True
    a = path_x[bottom_index-1]
    b = path_y[bottom_index-1]
    c = path_x[bottom_index]
    d = path_y[bottom_index]
    e = path_x[(bottom_index+1)%len(path_x)]
    f = path_y[(bottom_index+1)%len(path_y)]
    if b < f:
        normal = [d-b, -(c-a)]
        magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
    else:
        normal = [f-d, -(e-c)]
        magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
    if magnitude > 0.01:
        normal[0] = normal[0]/magnitude
        normal[1] = normal[1]/magnitude
        angle = vector_angle([0,0], normal)
        if angle < np.pi:
            swap_direction = False
            
    fitness = 0
    fitness_2 = 0
    threshold = 0.5
    angle_threshold = (5/180)*np.pi
    for i in range(len(path_x)):
        a = path_x[i-1]
        b = path_y[i-1]
        c = path_x[i]
        d = path_y[i]
        normal = [d-b, -(c-a)]
        magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
        if magnitude > 0.01 and np.abs(path_y[bottom_index]-d) < threshold and np.abs(path_y[bottom_index]-b) < threshold:
            normal[0] = normal[0]/magnitude
            normal[1] = normal[1]/magnitude
            angle = vector_angle([0,0], normal)
            smallest_inflation = None
            for j in range(len(path_x)):
                a2 = path_x[j-1]
                b2 = path_y[j-1]
                c2 = path_x[j]
                d2 = path_y[j]
                normal2 = [d2-b2, -(c2-a2)]
                magnitude2 = np.sqrt(normal2[0]**2 + normal2[1]**2)
                if magnitude2 > 0.01:
                    normal2[0] = normal2[0]/magnitude2
                    normal2[1] = normal2[1]/magnitude2
                    angle2 = vector_angle([0,0], normal2)
                    if np.abs(angle - angle2) > np.pi-angle_threshold and np.abs(angle - angle2) < np.pi+angle_threshold:
                        if smallest_inflation == None or smallest_inflation > distance([c,d],[c2,d2]):
                            smallest_inflation = distance([c,d],[c2,d2])
            if smallest_inflation is not None and smallest_inflation > fitness_2:
                fitness_2 = smallest_inflation
            if swap_direction:
                angle += np.pi
                if angle > 2*np.pi:
                    angle -= 2*np.pi
            if angle <= np.pi:
                angle = angle/np.pi
                if angle > 0.5:
                    angle = 1 - angle
                fitness += angle
                
    return (fitness, fitness_2)

def fitness_normals_lift(path_x, path_y):
    bottom_index = path_y.index(min(path_y))
    
    swap_direction = True
    a = path_x[bottom_index-1]
    b = path_y[bottom_index-1]
    c = path_x[bottom_index]
    d = path_y[bottom_index]
    e = path_x[(bottom_index+1)%len(path_x)]
    f = path_y[(bottom_index+1)%len(path_y)]
    if b < f:
        normal = [d-b, -(c-a)]
        magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
    else:
        normal = [f-d, -(e-c)]
        magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
    if magnitude > 0.01:
        normal[0] = normal[0]/magnitude
        normal[1] = normal[1]/magnitude
        angle = vector_angle([0,0], normal)
        if angle < np.pi:
            swap_direction = False
            
    fitness = 0
    fitness_2 = 0
    threshold = 0.5
    angle_threshold = (5/180)*np.pi
    for i in range(len(path_x)):
        a = path_x[i-1]
        b = path_y[i-1]
        c = path_x[i]
        d = path_y[i]
        normal = [d-b, -(c-a)]
        magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
        if magnitude > 0.01 and np.abs(path_y[bottom_index]-d) < threshold and np.abs(path_y[bottom_index]-b) < threshold:
            normal[0] = normal[0]/magnitude
            normal[1] = normal[1]/magnitude
            angle = vector_angle([0,0], normal)
            smallest_inflation = None
            for j in range(len(path_x)):
                a2 = path_x[j-1]
                b2 = path_y[j-1]
                c2 = path_x[j]
                d2 = path_y[j]
                normal2 = [d2-b2, -(c2-a2)]
                magnitude2 = np.sqrt(normal2[0]**2 + normal2[1]**2)
                if magnitude2 > 0.01:
                    normal2[0] = normal2[0]/magnitude2
                    normal2[1] = normal2[1]/magnitude2
                    angle2 = vector_angle([0,0], normal2)
                    if np.abs(angle - angle2) > np.pi-angle_threshold and np.abs(angle - angle2) < np.pi+angle_threshold:
                        if smallest_inflation == None or smallest_inflation > distance([c,d],[c2,d2]):
                            smallest_inflation = distance([c,d],[c2,d2])
            if smallest_inflation is not None and smallest_inflation > fitness_2:
                fitness_2 = smallest_inflation
            if swap_direction:
                angle += np.pi
                if angle > 2*np.pi:
                    angle -= 2*np.pi
            if angle <= np.pi:
                angle = angle/np.pi
                if angle > 0.5:
                    angle = 1 - angle
                fitness += angle
                
    return 0.8*fitness + 0.2*fitness_2

def fitness_spline(path_x, path_y):
    x_offset = sum(path_x)/len(path_x)
    y_offset = sum(path_y)/len(path_y)
    path = []
    angles = []
    for i in range(len(path_x)):
        path.append([path_x[i], path_y[i]])
        angles.append(i*2*np.pi/len(path_x))
    path.append(path[0])
    angles.append(2*np.pi)
    spline = interpolate.CubicSpline(angles, path, bc_type='periodic')
    spline_angles = []
    control_points_x = [-8+x_offset, -6+x_offset, -4+x_offset, -2+x_offset, 0+x_offset, 2+x_offset, 4+x_offset, 6+x_offset, 8+x_offset, -6+x_offset, -4+x_offset, -2+x_offset, 0+x_offset, 2+x_offset, 4+x_offset, 6+x_offset]
    y_o = 0.26875
    control_points_y = [y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+1+y_offset, y_o+2+y_offset, y_o+2.5+y_offset, y_o+2.7+y_offset, y_o+2.5+y_offset, y_o+2+y_offset, y_o+1+y_offset]
    for i in range(200):
        spline_angles.append(i*2*np.pi/200)
    distance = 0
    for c_x, c_y in zip(control_points_x, control_points_y):
        min_dist = None
        for point in path:
            dist = np.sqrt((c_x - point[0])**2 + (c_y - point[1])**2)
            if min_dist is None or dist < min_dist:
                min_dist = dist
        distance += min_dist
    return -distance

def fitness_spline_2(path_x, path_y):
    x_offset = sum(path_x)/len(path_x)
    y_offset = sum(path_y)/len(path_y)
    path = []
    angles = []
    for i in range(len(path_x)):
        path.append([path_x[i], path_y[i]])
        angles.append(i*2*np.pi/len(path_x))
    path.append(path[0])
    angles.append(2*np.pi)
    spline = interpolate.CubicSpline(angles, path, bc_type='periodic')
    spline_angles = []
    control_points_x = [-12+x_offset, -9+x_offset, -6+x_offset, -3+x_offset, 0+x_offset, 3+x_offset, 6+x_offset, 9+x_offset, 12+x_offset, -9+x_offset, -6+x_offset, -3+x_offset, 0+x_offset, 3+x_offset, 6+x_offset, 9+x_offset]
    y_o = 0.696875
    control_points_y = [y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+0.5+y_offset, y_o+1+y_offset, y_o+1.25+y_offset, y_o+1.35+y_offset, y_o+1.25+y_offset, y_o+1+y_offset, y_o+0.5+y_offset]
    for i in range(200):
        spline_angles.append(i*2*np.pi/200)
    distance = 0
    for c_x, c_y in zip(control_points_x, control_points_y):
        min_dist = None
        for point in path:
            dist = np.sqrt((c_x - point[0])**2 + (c_y - point[1])**2)
            if min_dist is None or dist < min_dist:
                min_dist = dist
        distance += min_dist
    return -distance

def multifitness_spline(path_x, path_y):
    x_offset = sum(path_x)/len(path_x)
    y_offset = sum(path_y)/len(path_y)
    path = []
    angles = []
    for i in range(len(path_x)):
        path.append([path_x[i], path_y[i]])
        angles.append(i*2*np.pi/len(path_x))
    path.append(path[0])
    angles.append(2*np.pi)
    spline = interpolate.CubicSpline(angles, path, bc_type='periodic')
    spline_angles = []
    y_o = 0.26875
    control_points_x = [-8+x_offset, -6+x_offset, -4+x_offset, -2+x_offset, 0+x_offset, 2+x_offset, 4+x_offset, 6+x_offset, 8+x_offset]
    control_points_y = [y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2]
    
    control_points_x_2 = [-6+x_offset, -4+x_offset, -2+x_offset, 0+x_offset, 2+x_offset, 4+x_offset, 6+x_offset]
    control_points_y_2 = [y_o+1+y_offset, y_o+2+y_offset, y_o+2.5+y_offset, y_o+2.7+y_offset, y_o+2.5+y_offset, y_o+2+y_offset, y_o+1+y_offset]
    for i in range(200):
        spline_angles.append(i*2*np.pi/200)
    distance = 0
    for c_x, c_y in zip(control_points_x, control_points_y):
        min_dist = None
        for point in path:
            dist = np.sqrt((c_x - point[0])**2 + (c_y - point[1])**2)
            if min_dist is None or dist < min_dist:
                min_dist = dist
        distance += min_dist
    distance_2 = 0
    for c_x, c_y in zip(control_points_x_2, control_points_y_2):
        min_dist = None
        for point in path:
            dist = np.sqrt((c_x - point[0])**2 + (c_y - point[1])**2)
            if min_dist is None or dist < min_dist:
                min_dist = dist
        distance_2 += min_dist
    return (-distance, -distance_2)

def multifitness_spline_2(path_x, path_y):
    x_offset = sum(path_x)/len(path_x)
    y_offset = sum(path_y)/len(path_y)
    path = []
    angles = []
    for i in range(len(path_x)):
        path.append([path_x[i], path_y[i]])
        angles.append(i*2*np.pi/len(path_x))
    path.append(path[0])
    angles.append(2*np.pi)
    spline = interpolate.CubicSpline(angles, path, bc_type='periodic')
    spline_angles = []
    y_o = 0.696875
    control_points_x = [-12+x_offset, -9+x_offset, -6+x_offset, -3+x_offset, 0+x_offset, 2+x_offset, 6+x_offset, 9+x_offset, 12+x_offset]
    control_points_y = [y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2, y_o+y_offset-2]
    
    control_points_x_2 = [-9+x_offset, -6+x_offset, -3+x_offset, 0+x_offset, 3+x_offset, 6+x_offset, 9+x_offset]
    control_points_y_2 = [y_o+0.5+y_offset, y_o+1+y_offset, y_o+1.25+y_offset, y_o+1.35+y_offset, y_o+1.25+y_offset, y_o+1+y_offset, y_o+0.5+y_offset]
    for i in range(200):
        spline_angles.append(i*2*np.pi/200)
    distance = 0
    for c_x, c_y in zip(control_points_x, control_points_y):
        min_dist = None
        for point in path:
            dist = np.sqrt((c_x - point[0])**2 + (c_y - point[1])**2)
            if min_dist is None or dist < min_dist:
                min_dist = dist
        distance += min_dist
    distance_2 = 0
    for c_x, c_y in zip(control_points_x_2, control_points_y_2):
        min_dist = None
        for point in path:
            dist = np.sqrt((c_x - point[0])**2 + (c_y - point[1])**2)
            if min_dist is None or dist < min_dist:
                min_dist = dist
        distance_2 += min_dist
    return (-distance, -distance_2)
    
def fitness_step(path_x, path_y):
    bottom_point = min(path_y)
    threshold = 0.5

    step_length = 0
    for i in range(len(path_y)):
        if np.abs(bottom_point-path_y[i-1]) < threshold and np.abs(bottom_point-path_y[i]) < threshold:
            step_length += (path_x[i] - path_x[i-1])

    return abs(step_length)

def map_coords_aurora(flattened_paths, ae, MAP_SIZE, PATH_LENGTH):
    encoded = ae.encode(flattened_paths) # should be 0-1
    placements = []
    for i in range(len(encoded)):
        dimension_indexes = encoded[i]
        step = 1/MAP_SIZE  #15 0-5. 5-10, 10-15, 15+ (4 ruter -> step = 15/3 = 5)
        c1 = 0
        for i in range(1, MAP_SIZE):
            if  dimension_indexes[0] > i*step:
                c1 += 1
        c2 = 0
        for i in range(1, MAP_SIZE):
            if  dimension_indexes[1] > i*step:
                c2 += 1
        c3 = 0
        for i in range(1, MAP_SIZE):
            if  dimension_indexes[2] > i*step:
                c3 += 1
        c4 = 0
        for i in range(1, MAP_SIZE):
            if  dimension_indexes[3] > i*step:
                c4 += 1
        placements.append((c1, c2, c3, c4))
    return placements

def map_coords_autoencoder_stdev(flattened_paths, ae, MAP_SIZE, PATH_LENGTH):
    processed = ae.process(flattened_paths)
    placements = []
    for i in range(len(processed)):
        error = 0
        for x, y, in zip(flattened_paths[i],processed[i]):
            error += np.abs(x-y)
        step1 = 2*PATH_LENGTH/MAP_SIZE  #15 0-5. 5-10, 10-15, 15+ (4 ruter -> step = 15/3 = 5)
        c1 = 0
        for i in range(1, MAP_SIZE):
            if  error > i*step1:
                c1 += 1
        mean = sum(flattened_paths[i])/len(flattened_paths[i])
        stdev = 0
        for x in flattened_paths[i]:
            stdev += ((x-mean)**2)/mean
        stdev = np.sqrt(np.abs(stdev))
        step2 = 2*PATH_LENGTH/MAP_SIZE
        c2 = 0
        for i in range(1, MAP_SIZE):
            if  stdev > i*step2:
                c2 += 1
        placements.append((c1, c2))
    return placements

def map_coords_width_height(x_paths, y_paths, MAP_SIZE, PATH_LENGTH):
    placements = []
    for x_path, y_path in zip(x_paths, y_paths):
        path_height = abs(max(y_path)-min(y_path))
        # 0 to 15 map coords
        step = 100/MAP_SIZE  #15 0-5. 5-10, 10-15, 15+ (4 ruter -> step = 15/3 = 5)
        c1 = 0
        for i in range(1, MAP_SIZE):
            if path_height > i*step:
                c1 += 1

        path_width = abs(max(x_path)-min(x_path))
        # 0 to 15 map coords
        step = 100/MAP_SIZE  #15 0-5. 5-10, 10-15, 15+ (4 ruter -> step = 15/3 = 5)
        c2 = 0
        for i in range(1, MAP_SIZE):
            if path_width > i*step:
                c2 += 1
        placements.append((c1, c2))
    return placements

def map_coords_proportion_overlap(x_paths, y_paths, MAP_SIZE, PATH_LENGTH):
    placements = []
    for x_path, y_path in zip(x_paths, y_paths):
        # proportion
        path_height = abs(max(y_path)-min(y_path))
        path_width = abs(max(x_path)-min(x_path))
        if max([path_height, path_width]) == 0:
            proportion = 1
        else:
            proportion = min([path_height, path_width])/max([path_height, path_width])
        step = 1/MAP_SIZE  #15 0-5. 5-10, 10-15, 15+ (4 ruter -> step = 15/3 = 5)
        c1 = 0
        for i in range(1, MAP_SIZE):
            if proportion > i*step:
                c1 += 1

        # amount of points overlapping with another point
        overlapping = 0
        overlapping_threshold = 0.1
        for p_x, p_y in zip(x_path, y_path):
            for p2_x, p2_y in zip(x_path, y_path):
                d = np.sqrt((p_x - p2_x)**2 + (p_y - p2_y)**2)
                if d < overlapping_threshold:
                    overlapping += 1
        overlapping = overlapping/len(x_path)**2
        step = 1/MAP_SIZE
        c2 = 0
        for i in range(1, MAP_SIZE):
            if overlapping > i*step:
                c2 += 1
        placements.append((c1, c2))
    return placements

def map_coords_structure(structure_data, x_paths, y_paths, MAP_SIZE):
    placements = []
    for i in range(len(structure_data)):
        dimension_indexes = structure_data[i]
        step = 10/MAP_SIZE  #15 0-5. 5-10, 10-15, 15+ (4 ruter -> step = 15/3 = 5)
        c1 = 0
        for i in range(1, MAP_SIZE):
            if  dimension_indexes[4] > i*step:
                c1 += 1
        c2 = 0
        for i in range(1, MAP_SIZE):
            if  dimension_indexes[1] > i*step:
                c2 += 1
        c3 = 0
        for i in range(1, MAP_SIZE):
            if  dimension_indexes[2] > i*step:
                c3 += 1
        c4 = 0
        for i in range(1, MAP_SIZE):
            if  dimension_indexes[3] > i*step:
                c4 += 1
        placements.append((c1, c2, c3, c4))
    return placements

def map_coords_beamlength_stepheight(structure_data, x_paths, y_paths, MAP_SIZE):
    placements = []
    for i in range(len(structure_data)):
        dimension_indexes = structure_data[i]
        path_y = y_paths[i]
        path_x = x_paths[i]
        step = 10/MAP_SIZE  #15 0-5. 5-10, 10-15, 15+ (4 ruter -> step = 15/3 = 5)
        c1 = 0
        for i in range(1, MAP_SIZE):
            if  dimension_indexes[4] > i*step:
                c1 += 1
        
        bottom_index = path_y.index(min(path_y))
        swap_direction = True
        a = path_x[bottom_index-1]
        b = path_y[bottom_index-1]
        c = path_x[bottom_index]
        d = path_y[bottom_index]
        e = path_x[(bottom_index+1)%len(path_x)]
        f = path_y[(bottom_index+1)%len(path_y)]
        if b < f:
            normal = [d-b, -(c-a)]
            magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
        else:
            normal = [f-d, -(e-c)]
            magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
        if magnitude > 0.01:
            normal[0] = normal[0]/magnitude
            normal[1] = normal[1]/magnitude
            angle = vector_angle([0,0], normal)
            if angle < np.pi:
                swap_direction = False
        fitness_2 = 0
        threshold = 0.5
        angle_threshold = (5/180)*np.pi
        for i in range(len(path_x)):
            a = path_x[i-1]
            b = path_y[i-1]
            c = path_x[i]
            d = path_y[i]
            normal = [d-b, -(c-a)]
            magnitude = np.sqrt(normal[0]**2 + normal[1]**2)
            if magnitude > 0.01 and np.abs(path_y[bottom_index]-d) < threshold and np.abs(path_y[bottom_index]-b) < threshold:
                normal[0] = normal[0]/magnitude
                normal[1] = normal[1]/magnitude
                angle = vector_angle([0,0], normal)
                smallest_inflation = None
                for j in range(len(path_x)):
                    a2 = path_x[j-1]
                    b2 = path_y[j-1]
                    c2 = path_x[j]
                    d2 = path_y[j]
                    normal2 = [d2-b2, -(c2-a2)]
                    magnitude2 = np.sqrt(normal2[0]**2 + normal2[1]**2)
                    if magnitude2 > 0.01:
                        normal2[0] = normal2[0]/magnitude2
                        normal2[1] = normal2[1]/magnitude2
                        angle2 = vector_angle([0,0], normal2)
                        if np.abs(angle - angle2) > np.pi-angle_threshold and np.abs(angle - angle2) < np.pi+angle_threshold:
                            if smallest_inflation == None or smallest_inflation > distance([c,d],[c2,d2]):
                                smallest_inflation = distance([c,d],[c2,d2])
                if smallest_inflation is not None and smallest_inflation > fitness_2:
                    fitness_2 = smallest_inflation
                    
        step = 50/MAP_SIZE  #15 0-5. 5-10, 10-15, 15+ (4 ruter -> step = 15/3 = 5)
        c2 = 0
        for i in range(1, MAP_SIZE):
            if fitness_2 > i*step:
                c2 += 1
        placements.append((c1, c2))
    return placements
