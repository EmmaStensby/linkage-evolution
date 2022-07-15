import numpy as np

#from matplotlib.animation import FuncAnimation
from multiprocessing import pool

#import matplotlib.pyplot as plt

from utilities import *

class Linkage:
    def __init__(self, params):
        self.continuous = True
    
        self.seed = params[0]
        self.rng = np.random.default_rng(self.seed)
        
        self.possible_link_lengths = params[1]

        self.gnome_start_length = 7
        self.genome_nodes = 7
        self.genome_node_length = 7
        self.genome = np.array([self.rng.random() for _ in range(self.gnome_start_length + self.genome_nodes * self.genome_node_length)])

        self.sigma1 = self.rng.random()
        self.sigma2 = self.rng.random()
        self.mutprop1 = 1.0
        self.mutprop2 = 1.0
        self.mutpb_structure = 0.2
            
        self.threshold = 0.1

        self.node_positions = np.array([])
        self.beams = np.array([])

        self.current_crank_angle = 0

        self.all_paths = []

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        
    def mutate(self, mutpb=0.2, sigma=0.1):
        for i in range(len(self.all_paths)):
            self.all_paths.pop()
        
        self.mutate_sigma(mutpb, sigma)
        
        if self.rng.random() < self.mutpb_structure:
            self.mutate_structure(self.mutprop1, self.sigma1)
        else:
            self.mutate_length(self.mutprop2, self.sigma2)

    def mutate_sigma(self, mutpb, sigma):
        if self.rng.random() < mutpb:
            self.sigma1 += self.rng.normal(0, sigma)
        while self.sigma1 >= 1 or self.sigma1 < 0:
            if self.sigma1 >= 1:
                self.sigma1 = 2 - self.sigma1
            elif self.sigma1 < 0:
                self.sigma1 = -self.sigma1
        if self.rng.random() < mutpb:
            self.sigma2 += self.rng.normal(0, sigma)
        while self.sigma2 >= 1 or self.sigma2 < 0:
            if self.sigma2 >= 1:
                self.sigma2 = 2 - self.sigma2
            elif self.sigma2 < 0:
                self.sigma2 = -self.sigma2

    def mutate_prop(self, mutpb, sigma):
        if self.rng.random() < mutpb:
            self.mutprop1 += self.rng.normal(0, sigma)
        while self.mutprop1 >= 1 or self.mutprop1 < 0:
            if self.mutprop1 >= 1:
                self.mutprop1 = 2 - self.mutprop1
            elif self.mutprop1 < 0:
                self.mutprop1 = -self.mutprop1
        if self.rng.random() < mutpb:
            self.mutprop2 += self.rng.normal(0, sigma)
        while self.mutprop2 >= 1 or self.mutprop2 < 0:
            if self.mutprop2 >= 1:
                self.mutprop2 = 2 - self.mutprop2
            elif self.mutprop2 < 0:
                self.mutprop2 = -self.mutprop2

    def mutate_pb_structure(self, mutpb, sigma):
        if self.rng.random() < mutpb:
            self.mutpb_structure += self.rng.normal(0, sigma)
        while self.mutpb_structure >= 1 or self.mutpb_structure < 0:
            if self.mutpb_structure >= 1:
                self.mutpb_structure = 2 - self.mutpb_structure
            elif self.mutpb_structure < 0:
                self.mutpb_structure = -self.mutpb_structure

    def mutate_length(self, mutpb, sigma):
        for i in range(6,len(self.genome)):
            if((i-6)%7 == 0 or (i-5)%7 == 0) and self.rng.random() < mutpb:
                self.genome[i] = self.rng.normal(0, sigma)
                while self.genome[i] >= 1 or self.genome[i] < 0:
                    if self.genome[i] >= 1:
                        self.genome[i] = 2 - self.genome[i]
                    elif self.genome[i] < 0:
                        self.genome[i] = -self.genome[i]

    def mutate_structure(self, mutpb, sigma):
        for i in range(len(self.genome)):
            if self.rng.random() < mutpb:
                self.genome[i] = self.rng.normal(0, sigma)
                while self.genome[i] >= 1 or self.genome[i] < 0:
                    if self.genome[i] >= 1:
                        self.genome[i] = 2 - self.genome[i]
                    elif self.genome[i] < 0:
                        self.genome[i] = -self.genome[i]

    def build_linkage(self, stop_at=None):
        self.node_positions = []
        self.beams = []
        self.current_crank_angle = 0
                       
        self.add_static_nodes_and_crank()
        offset = self.gnome_start_length

        if stop_at is None:
            stop_at = self.genome_nodes
        
        for i in range(stop_at):
            mutation_type = fti(self.genome[offset], 4)
            if mutation_type == 0:
                self.add_static_beam(offset+1)
            else:
                self.add_triangle(offset+1)
            offset += self.genome_node_length

    def add_static_nodes_and_crank(self):
        """[static_node_1_x, static_node_1_y, static_node_2_x, static_node_2_y, static_node_3_x, static_node_3_y, crank_length]"""
        self.node_positions.extend([[0, 0], [fti(self.genome[0], 11)-5, fti(self.genome[1], 11)-5], [fti(self.genome[2], 11)-5, fti(self.genome[3], 11)-5], [fti(self.genome[4], 11)-5, fti(self.genome[5], 11)-5]])
        node_start = 0
        angle = 0
        length = self.possible_link_lengths[fti(self.genome[6], len(self.possible_link_lengths)/2)]
        if self.continuous:
            length = self.genome[6]*6+2
        beam_dependency = None
        end_point = vector_end_pos(self.node_positions[node_start], angle, length)
        self.node_positions.append(end_point)
        self.beams.append([node_start, len(self.node_positions)-1, length, angle, 0, beam_dependency])

    def add_static_beam(self, offset):
        """[beam_start, -, inverted, -, -, length]"""
        inverted = fti(self.genome[2+offset], 2)
        beam_dependency = fti(self.genome[0+offset], len(self.beams))
        node_start = self.beams[beam_dependency][1]
        angle = self.beams[beam_dependency][3]
        invert_dependency = 0
        if inverted:
            node_start = self.beams[beam_dependency][0]
            angle = self.beams[beam_dependency][3] + np.pi
            invert_dependency = 1
        length = self.possible_link_lengths[fti(self.genome[5 + offset], len(self.possible_link_lengths))]
        if self.continuous:
            length = self.genome[6]*12+3
        end_point = vector_end_pos(self.node_positions[node_start], angle, length)
        self.node_positions.append(end_point)
        self.beams.append([node_start, len(self.node_positions)-1, length, angle, invert_dependency, beam_dependency])

    def add_triangle(self, offset):
        """[pos_1, pos_2, direction, backup_angle, length_1, length_2]"""
        pos1 = fti(self.genome[offset+0], len(self.node_positions)-1)+1
        pos2 = fti(self.genome[offset+1], len(self.node_positions)-1)+1

        original_pos2 = pos2
        all_points_same = False
        while pos1 == pos2 or distance(self.node_positions[pos1], self.node_positions[pos2]) > max(self.possible_link_lengths)*2 or distance(self.node_positions[pos1], self.node_positions[pos2]) < self.threshold:
            pos2 -= 1
            if pos2 == 0:
                pos2 = len(self.node_positions)-1
            if pos2 == original_pos2:
                all_points_same = True
                break

        if all_points_same:
            while pos1 == pos2 or distance(self.node_positions[pos1], self.node_positions[pos2]) > max(self.possible_link_lengths)*2:
                pos2 -= 1
                if pos2 == 0:
                    pos2 = len(self.node_positions)-1

        shortest = max([distance(self.node_positions[pos1], self.node_positions[pos2])-max(self.possible_link_lengths),min(self.possible_link_lengths)])
        long_enough_1 = []
        for length in self.possible_link_lengths:
            if length >= shortest:
                long_enough_1.append(length)
        if self.continuous:
            l1 = (max(self.possible_link_lengths)-shortest)*self.genome[offset+4] + shortest
        else:
            l1 = long_enough_1[fti(self.genome[offset+4], len(long_enough_1))]
            
        shortest = np.abs(distance(self.node_positions[pos1], self.node_positions[pos2])-l1)
        #longest = distance(self.node_positions[pos1], self.node_positions[pos2])+l1
        longest = min([distance(self.node_positions[pos1], self.node_positions[pos2])+l1, max(self.possible_link_lengths)])
        long_enough_2 = []
        for length in self.possible_link_lengths:
            if length >= shortest and length <= longest:
                    long_enough_2.append(length)    
        if self.continuous:
            l2 = (longest-shortest)*self.genome[offset+5] + shortest
        else:
            l2 = long_enough_2[fti(self.genome[offset+5], len(long_enough_2))]
            
        direction = fti(self.genome[offset+2], 2)
        backup_angle = self.genome[offset+3]*2*np.pi
        connection_point = circle_intersection(self.node_positions[pos1], self.node_positions[pos2], l1, l2, direction, self.threshold, backup_angle=backup_angle)

        angle_1 = vector_angle(self.node_positions[pos1], connection_point)
        angle_2 = vector_angle(self.node_positions[pos2], connection_point)
        beam_dependency_1 = None
        beam_dependency_2 = None      
        node_start_1 = pos1
        node_start_2 = pos2
        length_1 = l1
        length_2 = l2
        end_point = connection_point
        self.node_positions.append(end_point)
        self.beams.append([node_start_1, len(self.node_positions)-1, length_1, angle_1, 0, beam_dependency_1])
        self.beams.append([node_start_2, len(self.node_positions)-1, length_2, angle_2, 0, beam_dependency_2])

    def step(self, turn=np.pi/50):
        self.node_positions = []
        self.current_crank_angle += turn
                       
        self.step_static_nodes_and_crank(self.current_crank_angle)
        
        offset = self.gnome_start_length
        beam_offset = 1

        for i in range(self.genome_nodes):
            mutation_type = fti(self.genome[offset], 4)
            if mutation_type == 0:
                self.step_static_beam(beam_offset)
                beam_offset += 1
            else:
                if self.step_triangle(beam_offset, beam_offset+1, offset+1) == False:
                    return False
                beam_offset += 2
            offset += self.genome_node_length
        return True

    def step_static_nodes_and_crank(self, crank_angle):
        self.node_positions.extend([[0, 0], [fti(self.genome[0], 11)-5, fti(self.genome[1], 11)-5], [fti(self.genome[2], 11)-5, fti(self.genome[3], 11)-5], [fti(self.genome[4], 11)-5, fti(self.genome[5], 11)-5]])
        node_start = self.beams[0][0]
        length = self.beams[0][2]
        self.beams[0][3] = crank_angle
        end_point = vector_end_pos(self.node_positions[node_start], crank_angle, length)
        self.node_positions.append(end_point)

    def step_static_beam(self, beam_offset):
        inverted = self.beams[beam_offset][4]
        beam_dependency = self.beams[beam_offset][5]
        node_start = self.beams[beam_dependency][1]
        angle = self.beams[beam_dependency][3]
        if inverted:
            node_start = self.beams[beam_dependency][0]
            angle = self.beams[beam_dependency][3] + np.pi
        length = self.beams[beam_offset][2]
        end_point = vector_end_pos(self.node_positions[node_start], angle, length)
        self.beams[beam_offset][3] = angle
        self.node_positions.append(end_point)

    def step_triangle(self, beam_offset_1, beam_offset_2, offset):
        pos1 = self.beams[beam_offset_1][0]
        pos2 = self.beams[beam_offset_2][0]
        l1 = self.beams[beam_offset_1][2]
        l2 = self.beams[beam_offset_2][2]
        direction = fti(self.genome[offset+2], 2)
        backup_angle = self.genome[offset+3]*2*np.pi
        connection_point = circle_intersection(self.node_positions[pos1], self.node_positions[pos2], l1, l2, direction, self.threshold, backup_angle=backup_angle)
        if connection_point is None:
            return False
        angle_1 = vector_angle(self.node_positions[pos1], connection_point)
        angle_2 = vector_angle(self.node_positions[pos2], connection_point)     
        self.node_positions.append(connection_point)
        self.beams[beam_offset_1][3] = angle_1
        self.beams[beam_offset_2][3] = angle_2
        return True

    def plot(self, show_path=False):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import seaborn as sns
        import cycler
        n = 14
        color = plt.cm.viridis(np.linspace(0, 1, n))
        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
        x = []
        y = []
        plt.figure(figsize=(10, 10))
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)
        for node in self.node_positions:
            x.append(node[0])
            y.append(node[1])
        plt.scatter(x, y)
        plt.scatter([x[0]], [y[0]], color = color[13])
        plt.scatter(x[1:4], y[1:4], color = color[7])
        lines = []
        for beam in self.beams:
            plt.plot([self.node_positions[beam[0]][0],self.node_positions[beam[1]][0]], [self.node_positions[beam[0]][1],self.node_positions[beam[1]][1]])
        path_x, path_y, error, angle_error = self.get_lowest_path(40)
        if show_path:
            plt.scatter(path_x, path_y, color=color[11])
        plt.show()
    
    def get_all_paths(self, path_length):
        if len(self.all_paths) > 0:
            if len(self.all_paths) == 3:
                return self.all_paths[0], self.all_paths[1], 0, self.all_paths[2]
            else:
                return self.all_paths[0], self.all_paths[1], self.all_paths[2], self.all_paths[3]
        
        self.build_linkage()
        paths_x = [[] for _ in range(len(self.node_positions))]
        paths_y = [[] for _ in range(len(self.node_positions))]
        angles = [[] for _ in range(len(self.beams))]
        total_error = 0
        for j in range(len(self.node_positions)):
            paths_x[j].append(self.node_positions[j][0])
            paths_y[j].append(self.node_positions[j][1])
        for j in range(len(self.beams)):
            angles[j].append(self.beams[j][3])
        for i in range(path_length-1):
            if self.step(turn=np.pi*2/path_length):
                for j in range(len(self.node_positions)):
                    paths_x[j].append(self.node_positions[j][0])
                    paths_y[j].append(self.node_positions[j][1])
                for j in range(len(self.beams)):
                    angles[j].append(self.beams[j][3])
            else:
                total_error += 1

        self.all_paths.append(paths_x)
        self.all_paths.append(paths_y)
        self.all_paths.append(angles)
        self.all_paths.append(total_error)
                
        return paths_x, paths_y, angles, total_error

    def get_lowest_path(self, path_length):
        paths_x, paths_y, angles, total_error = self.get_all_paths(path_length)
        minimums = [min(path) for path in paths_y]
        end_node = minimums.index(min(minimums))
        path_y = paths_y[end_node]
        path_x = paths_x[end_node]
        #print(angles)
        if angles == 0:
            return path_x, path_y, total_error, angles
        end_angles = []
        for i in range(len(self.beams)):
            beam = self.beams[i]
            if beam[1] == end_node:
                end_angles.append(angles[i])
            #elif beam[0] == end_node:
            #    turned_angles = []
            #    for angle in angles[i]:
            #        turned_angles.append(angle+np.pi)
            #    end_angles.append(turned_angles)
        angle_error = 0
        for i in range(len(path_x)):
            avg_angle = 0
            for end_angle in end_angles:
                avg_angle += end_angle[i]/len(end_angles)
            if avg_angle < np.pi and avg_angle > 0:
                angle_error += 1*distance([path_x[i], path_y[i]], [path_x[i - 1], path_y[i - 1]])
        return path_x, path_y, total_error, angle_error

    def get_structure_data(self, path_length):
        paths_x, paths_y, angles, total_error = self.get_all_paths(path_length)
        minimums = [min(path) for path in paths_y]
        end_index = minimums.index(min(minimums))
    
        self.build_linkage()
        node_tags = [0 for _ in range(len(self.node_positions))]
        # Number of "double" beams
        d1 = len(self.beams) - (len(self.node_positions)-4)
        # Longest/shortest path
        d2 = -2
        # Number of nodes in main part
        self.beams = np.array(self.beams)
        beam_connections = self.beams[:,:2]
        node_tags[end_index] = 1
        something_changed = True
        while something_changed:
            something_changed = False
            d2 += 1
            for i in range(4, len(self.node_positions)):
                if node_tags[i] == 1:
                    connected_nodes = []
                    for beam_connection in beam_connections:
                        if beam_connection[0] == i:
                            connected_nodes.append(beam_connection[1])
                        elif beam_connection[1] == i:
                            connected_nodes.append(beam_connection[0])
                    for j in connected_nodes:
                        if node_tags[j] == 0:
                            node_tags[j] = 1
                    node_tags[i] = 2
                    something_changed = True
        d3 = 0
        for node_tag in node_tags:
            if node_tag == 2:
                d3 += 1
        # Number of moving nodes
        d4 = len(self.node_positions)
        previous_node_positions = self.node_positions
        self.step(turn=np.pi*2/path_length)
        for i in range(len(self.node_positions)):
            if self.node_positions[i][0] == previous_node_positions[i][0] and self.node_positions[i][1] == previous_node_positions[i][1]:
                d4 -= 1
        # Average length of beams
        d5 = 0
        for beam in self.beams:
            d5 += ((beam[2]-2)*10)/(len(self.beams)*13)
        return d1, d2, d3, d4, d5
    
    def get_length_data(self, path_length):
        # Average length of beams
        d5 = 0
        for beam in self.beams:
            d5 += ((beam[2]-2)*10)/(len(self.beams)*13)
        return d5

    def get_unity_data(self):
        self.build_linkage()
        self.node_positions = np.array(self.node_positions)
        self.beams = np.array(self.beams)
        print(self.node_positions)
        print(self.beams)
        height_offset = min(self.node_positions[:,1])
        node_count = len(self.node_positions)
        static_node_count = 3
        node_positions = self.node_positions
        beam_count = len(self.beams)
        beam_lengths = self.beams[:,2]
        beam_positions = []
        for beam in self.beams:
            pos = list(middle_of_positions(self.node_positions[beam[0]], self.node_positions[beam[1]]))
            beam_positions.append(pos)
        beam_angles = self.beams[:,3]
        beam_connections = self.beams[:,:2]
        dependencies = []
        for i in range(len(self.beams)):
            if self.beams[i][5] is not None:
                dependencies.append(self.beams[i][5])
            else:
                dependencies.append(-1)

        self.step(turn=np.pi)
        self.node_positions = np.array(self.node_positions)
        self.beams = np.array(self.beams)
        node_positions_2 = self.node_positions
        beam_positions_2 = []
        for beam in self.beams:
            pos = list(middle_of_positions(self.node_positions[beam[0]], self.node_positions[beam[1]]))
            beam_positions_2.append(pos)
        beam_angles_2 = self.beams[:,3]
        print(self.node_positions)
        print(self.beams)

        return (height_offset, node_count, static_node_count, node_positions, beam_count, beam_lengths, beam_positions, beam_angles, beam_connections, dependencies, node_positions_2, beam_positions_2, beam_angles_2)
    
    def animate(self, path_length, save=False, show=False):
        from matplotlib.animation import FuncAnimation
        import matplotlib.pyplot as plt
        paths_x, paths_y, angles, total_error = self.get_all_paths(path_length)

        minimums = [min(path) for path in paths_y]
        path_y = paths_y[minimums.index(min(minimums))]
        path_x = paths_x[minimums.index(min(minimums))]

        if show or save:
            fig = plt.figure(figsize=(10,10))
            ax = plt.axes()

            ax.set_xlim(-60, 60)
            ax.set_ylim(-60, 60)
        
            nodes_x = []
            nodes_y = []
            lines = []
            scats = []

            node_count = len(self.node_positions)
        
            for i in range(node_count):
                nodes_x.append(paths_x[i][0])
                nodes_y.append(paths_y[i][0])
            scats.append(ax.scatter(nodes_x, nodes_y))
            ax.scatter(path_x, path_y, color = 'yellow')
            scats.append(ax.scatter([nodes_x[0]], [nodes_y[0]], color = 'red'))
            scats.append(ax.scatter(nodes_x[1:4], nodes_y[1:4], color = 'green'))

            line_objs = []
            for beam in self.beams:
                line_obj = ax.plot([paths_x[beam[0]][0], paths_x[beam[1]][0]], [paths_y[beam[0]][0], paths_y[beam[1]][0]])
                line_objs.append(line_obj[0])

            def animate_internal(i):
                nodes_pos = []
                for j in range(node_count):
                    nodes_pos.append([paths_x[j][i], paths_y[j][i]])
                scats[0].set_offsets(nodes_pos)
                        
                for j, line_obj in zip(range(len(line_objs)), line_objs):
                    line_obj.set_data([paths_x[self.beams[j][0]][i], paths_x[self.beams[j][1]][i]], [paths_y[self.beams[j][0]][i], paths_y[self.beams[j][1]][i]])
                
                return scats.extend(line_objs)

            anim = FuncAnimation(fig, animate_internal, frames=len(path_x), interval=20, blit=False)
            
            if show:
                plt.show()
            if save:
                anim.save('animation.gif', writer='imagemagick', fps=60)
        return total_error

if __name__ == "__main__":
    seed = np.random.SeedSequence()
    possible_link_lengths = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    l = Linkage([seed, possible_link_lengths])

    l.genome = [0.86162082, 0.2031112,  0.86099355, 0.52415496, 0.10013951, 0.42093734,
                0.08600091, 0.639706,   0.36903921, 0.96784787, 0.5588816,  0.01865519,
                0.40987769, 0.80303195, 0.21343354, 0.85246928, 0.22882256, 0.36247774,
                0.0835223,  0.37111958, 0.00992227, 0.04446347, 0.41681298, 0.0830789,
                0.16403905, 0.28792771, 0.96039759, 0.52048486, 0.90899396, 0.14906331,
                0.1595087,  0.34261305, 0.77303006, 0.43047756, 0.03385307, 0.52513928,
                0.23649474, 0.45085541, 0.78550781, 0.74734628, 0.73933811, 0.18883307,
                0.10679091, 0.82937869, 0.53859177, 0.08985218, 0.06333953, 0.98995835,
                0.09649049, 0.78315426, 0.63258184, 0.84963327, 0.95263002, 0.44500063,
                0.50847708, 0.42408332]
    
    l.build_linkage()
    l.plot(show_path=True)
    l.animate(40, show=True)
