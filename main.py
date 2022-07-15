import pickle

from scipy import interpolate
import itertools
import numpy as np
import random
import time

from deap import base
from deap import creator
from deap import tools
 
import multiprocessing as mp
from multiprocessing import pool

#import matplotlib.pyplot as plt

from linkage import Linkage
from autoencoder import Autoencoder
from utilities import *
from evaluation_functions import *

POSSIBLE_LINKS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

THREADS = mp.cpu_count()

RUN_TYPE = 1
RUN_ID = random.randint(0,sys.maxsize)
PATH = "results_5/"+str(RUN_ID)+"_"
AUTOENCODER = False
STRUCTURE = False
AE_MAP_FUNCTION = map_coords_aurora
STRUCTURE_MAP_FUNCTION = map_coords_structure
#STRUCTURE_MAP_FUNCTION = map_coords_beamlength_stepheight
MAP_FUNCTION = map_coords_width_height
#FITNESS_FUNCTION = multifitness_spline
FITNESS_SWITCH = False
OBJECTIVES = 2
MAP_SIZE = 100
DIMENSIONS = 2
EVALUATIONS = 10000
POP_SIZE = 5000
ITERATIONS = int((EVALUATIONS - POP_SIZE)/(POP_SIZE))
RETRAIN_PERIOD = 10
MINUTES = 60

PATH_LENGTH = 40

def plot_spline(l):
    path_x, path_y, error, angle_error = l.get_lowest_path(PATH_LENGTH)
    
    path = []
    angles = []
    for i in range(len(path_x)):
        path.append([path_x[i], path_y[i]])
        angles.append(i*2*np.pi/len(path_x))
    path.append(path[0])
    angles.append(2*np.pi)
    spline = interpolate.CubicSpline(angles, path, bc_type='periodic')
    
    spline_angles = []
    for i in range(200):
        spline_angles.append(i*2*np.pi/200)
        
    plt.scatter(path_x, path_y)
    plt.plot(spline(spline_angles)[:, 0], spline(spline_angles)[:, 1])
    plt.savefig(PATH+"spline.pdf")

def evaluate(params):
    l = params[0]
    fitness_func = params[1]
    path_x, path_y, error, angle_error = l.get_lowest_path(PATH_LENGTH)
    #print(angle_error)
    fitness = fitness_func(path_x, path_y)
    fitness = error+angle_error-fitness
    return (fitness, l)

def evaluate_multifitness(params):
    l = params[0]
    fitness_func = params[1]
    path_x, path_y, error, angle_error = l.get_lowest_path(PATH_LENGTH)
    fitness = fitness_func(path_x, path_y)
    fitness = (error+angle_error-fitness[0], error+angle_error-fitness[1])
    return (fitness, l)
    
def mutate(l):
    l.mutate()
    return l

def animate(l):
    l.animate(40, show=True)
            
def run():
    time_start = time.time()
    FITNESS_FUNCTION = fitness_normals_lift
    #FITNESS_FUNCTION = fitness_spline
    
    loggfile = open(PATH + "logg.txt", "w")
    # Setup
    seed = np.random.SeedSequence()
    rng = np.random.default_rng(seed)
    master_seed = rng.integers(sys.maxsize)
    random.seed(master_seed)

    loggfile.write("ea_2\n")
    loggfile.write(str(FITNESS_FUNCTION)+"\n")
    loggfile.write(str(master_seed)+"\n")

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", Linkage, fitness=creator.FitnessMin)

    # Register functions
    toolbox = base.Toolbox()
    toolbox.register("attr_float", rng.random)
    toolbox.register("individual", creator.Individual)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)
    toolbox.register("show", animate)

    # For recording stats
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    avg_fitness_over_time = []
    min_fitness_over_time = []
    
    # Evaluate initial population
    pop = [toolbox.individual((seed.spawn(1)[0], POSSIBLE_LINKS)) for _ in range(POP_SIZE)]
    parameters = []
    for ind in pop:
        parameters.append([ind, FITNESS_FUNCTION])
    with pool.Pool(THREADS) as p:
        results = p.map(toolbox.evaluate, parameters)
    new_pop = []
    for result in results:
        fit, ind = result
        ind.fitness.values = (fit,)
        new_pop.append(ind)
    pop[:] = new_pop

    the_pool = pool.Pool(THREADS)

    time_now = time.time()
    
    g = 0
    while time_now-time_start < MINUTES*60:
    #for g in range(0, ITERATIONS):
        if g == 150 and FITNESS_SWITCH:
            FITNESS_FUNCTION = fitness_spline_2
            parameters = []
            for ind in pop:
                parameters.append([ind, FITNESS_FUNCTION])
            results = the_pool.map(toolbox.evaluate, parameters)
            new_pop = []
            for result in results:
                fit, ind = result
                ind.fitness.values = fit
                new_pop.append(ind)
            pop[:] = new_pop
            
        g += 1
        # Select offspring
        selected = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in selected]
        for i in range(len(offspring)):
            offspring[i].set_seed(seed.spawn(1)[0])

        # Mutate
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values

        # Reevaluate mutated
        parameters = []
        for ind in offspring:
            parameters.append([ind, FITNESS_FUNCTION])
        results = the_pool.map(toolbox.evaluate, parameters)
        new_offspring = []
        for result in results:
            fit, ind = result
            ind.fitness.values = (fit,)
            new_offspring.append(ind)
        offspring[:] = new_offspring

        # Replace whole population
        pop[:] = offspring

        # Record stats
        record = stats.compile(pop)
        loggfile.write(str(record)+"\n")
        avg_fitness_over_time.append(record['avg'])
        min_fitness_over_time.append(record['min'])

        time_now = time.time()

    # Plot best one and stats
    all_fitnesses = [ind.fitness.values[0] for ind in pop]
    best_index = all_fitnesses.index(min(all_fitnesses))
    loggfile.write("BEST: " + str(min(all_fitnesses)) + "\n")
    ind = pop[best_index]
    #toolbox.show(ind)
    """plt.figure()
    plt.plot(avg_fitness_over_time)
    plt.title("avg_fitness_over_time")
    plt.savefig(PATH + "avg_fitness_over_time.pdf")
    plt.figure()
    plt.plot(min_fitness_over_time)
    plt.title("min_fitness_over_time")
    plt.savefig(PATH + "min_fitness_over_time.pdf")
    plt.figure()
    plot_spline(ind)"""

    pickle_data = {'pop': pop, 'avg_fitness_over_time': avg_fitness_over_time, 'min_fitness_over_time': min_fitness_over_time, 'g':g}
    with open(PATH + "data.pkl", "wb") as f:
        pickle.dump(pickle_data, f)

    loggfile.close()

def run_nsga2():
    time_start = time.time()
    FITNESS_FUNCTION = multifitness_normals_lift
    #FITNESS_FUNCTION = multifitness_spline
    
    loggfile = open(PATH + "logg.txt", "w")
    # Setup
    seed = np.random.SeedSequence()
    rng = np.random.default_rng(seed)
    master_seed = rng.integers(sys.maxsize)
    random.seed(master_seed)

    loggfile.write("nsga2\n")
    loggfile.write(str(FITNESS_FUNCTION)+"\n")

    loggfile.write(str(master_seed)+"\n")

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,)*OBJECTIVES)
    creator.create("Individual", Linkage, fitness=creator.FitnessMin)

    # Register functions
    toolbox = base.Toolbox()
    toolbox.register("attr_float", rng.random)
    toolbox.register("individual", creator.Individual)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate_multifitness)
    toolbox.register("show", animate)

    # For recording stats
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    
    # Evaluate initial population
    pop = [toolbox.individual((seed.spawn(1)[0], POSSIBLE_LINKS)) for _ in range(POP_SIZE)]
    parameters = []
    for ind in pop:
        parameters.append([ind, FITNESS_FUNCTION])
    with pool.Pool(THREADS) as p:
        results = p.map(toolbox.evaluate, parameters)
    new_pop = []
    for result in results:
        fit, ind = result
        ind.fitness.values = fit
        new_pop.append(ind)
    pop[:] = new_pop

    the_pool = pool.Pool(THREADS)

    time_now = time.time()

    avg_fitness_over_time = []
    min_fitness_over_time = []

    front = tools.ParetoFront()
    front.update(pop)
    
    g = 0
    while time_now-time_start < MINUTES*60:
        #for g in range(0, ITERATIONS):
        #time_1 = time.time()
        if g == 150 and FITNESS_SWITCH:
            FITNESS_FUNCTION = multifitness_spline_2
            parameters = []
            for ind in pop:
                parameters.append([ind, FITNESS_FUNCTION])
            results = the_pool.map(toolbox.evaluate, parameters)
            new_pop = []
            for result in results:
                fit, ind = result
                ind.fitness.values = fit
                new_pop.append(ind)
            pop[:] = new_pop
        g += 1

        
        # Select offspring
        offspring = [toolbox.clone(ind) for ind in pop]
        for i in range(len(offspring)):
            offspring[i].set_seed(seed.spawn(1)[0])

        #time_2 = time.time()
        #print("Clone: ", time_2-time_1)

        # Mutate
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values
            
        #time_3 = time.time()
        #print("Mutate: ", time_3-time_2)
    
        # Reevaluate mutated
        parameters = []
        for ind in offspring:
            parameters.append([ind, FITNESS_FUNCTION])
        results = the_pool.map(toolbox.evaluate, parameters)
        all_individuals = []
        for result in results:
            fit, ind = result
            ind.fitness.values = fit
            all_individuals.append(ind)
        for ind in pop:
            all_individuals.append(ind)

        #time_4 = time.time()
        #print("Evaluate: ", time_4-time_3)
            
        # Replace population with selected
        selected = toolbox.select(all_individuals, POP_SIZE)
        pop[:] = selected

        #time_5 = time.time()
        #print("Select: ", time_5-time_4)
        
        # Record stats
        record = stats.compile(pop)
        loggfile.write(str(record)+"\n")

        all_fitnesses = [ind.fitness.values for ind in pop]
        all_fitnesses_total = []
        for fit in all_fitnesses:
            all_fitnesses_total.append(fit[0] + fit[1])
        min_fitness_over_time.append(min(all_fitnesses_total))
        avg_fitness_over_time.append(np.mean(all_fitnesses_total))

        #time_6 = time.time()
        #print("Stats: ", time_6-time_5)

        front.update(pop)
        
        time_now = time.time()

    # Plot best one and stats
    all_fitnesses = [ind.fitness.values for ind in pop]
    #print(all_fitnesses)

    all_fitnesses_total = []
    for fit in all_fitnesses:
        all_fitnesses_total.append(fit[0] + fit[1])
    best_fitness = min(all_fitnesses_total)
    
    pickle_data = {'pop': pop, 'front': front, 'avg_fitness_over_time': avg_fitness_over_time, 'min_fitness_over_time': min_fitness_over_time, 'g':g}
    with open(PATH + "data.pkl", "wb") as f:
        pickle.dump(pickle_data, f)

    loggfile.close()
    
def save_map(mapp, folder):
    new_size = int(np.sqrt(MAP_SIZE**DIMENSIONS))
    new_map = np.reshape(mapp, (new_size, new_size))
    fit_map = [[np.nan for _ in range(new_size)] for _ in range(new_size)]
    for x in range(new_size):
        for y in range(new_size):
            if new_map[x][y] is not None:
                fit_map[x][y] = new_map[x][y].fitness.values[0]
    plt.matshow(fit_map, cmap="viridis_r", vmin=0, vmax=100)
    plt.colorbar()
    plt.savefig(folder + "map.pdf")

def show_map(mapp):
    import matplotlib.pyplot as plt
    new_size = int(np.sqrt(MAP_SIZE**DIMENSIONS))
    new_map = np.reshape(mapp, (new_size, new_size))
    fit_map = [[np.nan for _ in range(new_size)] for _ in range(new_size)]
    for x in range(new_size):
        for y in range(new_size):
            if new_map[x][y] is not None:
                fit_map[x][y] = new_map[x][y].fitness.values[0]
    plt.matshow(fit_map, cmap="viridis_r", vmin=-20, vmax=20)
    plt.colorbar()
    plt.show()

def show_map_downsampled(mapp, reveal=None, toolbox=None, reshape=False):
    import matplotlib.pyplot as plt
    import matplotlib
    new_size = int(np.sqrt(MAP_SIZE**DIMENSIONS))
    new_map = np.reshape(mapp, (new_size, new_size))
    if reshape:
        new_map = [[None for _ in range(100)] for _ in range(100)]
        for x in range(10):
            for y in range(10):
                for z in range(10):
                    for v in range(10):
                        new_map[x*10+y][z*10+v] = mapp[v][z][y][x] # v,z,y,x for au
    fit_map = [[None for _ in range(5)] for _ in range(5)]
    ind_map = [[None for _ in range(5)] for _ in range(5)]
    for x in range(5):
        for y in range(5):
            for z in range(20):
                for v in range(20):
                    if new_map[x*20+z][y*20+v] is not None:
                        if fit_map[x][y] is None or fit_map[x][y] > new_map[x*20+z][y*20+v].fitness.values[0]:
                            fit_map[x][y] = new_map[x*20+z][y*20+v].fitness.values[0]
                            ind_map[x][y] = new_map[x*20+z][y*20+v]
                            
    fit_map_2 = [[np.nan for _ in range(5)] for _ in range(5)]
    for x in range(5):
        for y in range(5):
            if fit_map[x][y] is not None:
                fit_map_2[x][y] = fit_map[x][y]
                
    plt.matshow(fit_map_2, cmap="viridis_r", vmin=-20, vmax=20)
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(10,10))
    ax = [plt.subplot(5,5,i+1) for i in range(25)]
    for a in ax:
        a.set_xticks([], [])
        a.set_yticks([], [])
    """dist_l = []
    for x in range(5):
        for y in range(5):
            if (x,y) in reveal:
                path_x, path_y, error = ind_map[x][y].get_lowest_path(PATH_LENGTH)
                dist_l.append(max(path_x) - min(path_x))
                dist_l.append(max(path_y) - min(path_y))"""

    cmap = matplotlib.cm.get_cmap('viridis_r')
                
    for x in range(5):
        for y in range(5):
            if ind_map[x][y] is not None: #True: #(x,y) in reveal:
                path_x, path_y, error, angle_error = ind_map[x][y].get_lowest_path(PATH_LENGTH)
                if error > 0:
                    ax[x*5+y].scatter(path_x, path_y, color='black')
                else:
                    color_fitness = fit_map[x][y]
                    if color_fitness > 20:
                        color_fitness = 20
                    elif color_fitness < -20:
                        color_fitness = -20
                    color_fitness = (color_fitness+20)/40
                    ax[x*5+y].scatter(path_x, path_y, color=cmap(color_fitness))
                dist = max([max(path_x) - min(path_x), max(path_y) - min(path_y)])
                cx = min(path_x) + (max(path_x) - min(path_x))/2
                cy = min(path_y) + (max(path_y) - min(path_y))/2
                ax[x*5+y].set_xlim(cx-dist/1.8, cx+dist/1.8)
                ax[x*5+y].set_ylim(cy-dist/1.8, cy+dist/1.8)
                ax[x*5+y].text(0.95, 0.01, str(int(dist)), verticalalignment='bottom', horizontalalignment='right', transform=ax[x*5+y].transAxes, color='black', fontsize=17)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

    pop = []
    for i in range(5):
         for j in range(5):
             if ind_map[i][j] is not None:
                 pop.append(ind_map[i][j])

    all_fitnesses = [ind.fitness.values[0] for ind in pop]
    best_index = all_fitnesses.index(min(all_fitnesses))
    eprint("BEST: ", min(all_fitnesses))
    ind = pop[best_index]
    toolbox.show(ind)
    print(ind.beams)
    ind.plot()
    
    ind = ind_map[reveal[0]][reveal[1]]
    print("Fitness: ", fit_map[reveal[0]][reveal[1]])
    print(ind.beams)
    toolbox.show(ind)
    ind.plot()

def run_map():
    time_start = time.time()
    FITNESS_FUNCTION = fitness_normals_lift
    #FITNESS_FUNCTION = fitness_normals
    #FITNESS_FUNCTION = fitness_spline
    loggfile = open(PATH + "logg.txt", "w")
    
    # Setup seed
    seed = np.random.SeedSequence()
    rng = np.random.default_rng(seed)
    master_seed = rng.integers(sys.maxsize)
    random.seed(master_seed)
   
    loggfile.write("map\n")
    loggfile.write(str(FITNESS_FUNCTION)+"\n")
    loggfile.write(str(AUTOENCODER)+"\n")
    loggfile.write(str(STRUCTURE)+"\n")
    loggfile.write(str(AE_MAP_FUNCTION)+"\n")
    loggfile.write(str(STRUCTURE_MAP_FUNCTION)+"\n")
    loggfile.write(str(MAP_FUNCTION)+"\n")
    loggfile.write(str(MAP_SIZE)+"\n")
    loggfile.write(str(DIMENSIONS)+"\n")

    loggfile.write(str(master_seed))

    # Setup deap
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", Linkage, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", rng.random)
    toolbox.register("individual", creator.Individual)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selRandom)
    toolbox.register("evaluate", evaluate)
    toolbox.register("show", animate)

    # Setup recording stats
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    avg_fitness_over_time = []
    min_fitness_over_time = []
    coverage_over_time = []
    
    # Setup map
    map_shape = (MAP_SIZE,)
    for i in range(DIMENSIONS-1):
        map_shape = map_shape + (MAP_SIZE,)
    mapp = np.full(map_shape, None)
    
    # Evaluate , x_paths, y_pathsinitial population
    pop = [toolbox.individual((seed.spawn(1)[0], POSSIBLE_LINKS)) for _ in range(POP_SIZE)]
    parameters = []
    for ind in pop:
        parameters.append([ind, FITNESS_FUNCTION])
    with pool.Pool(THREADS) as p:
        results = p.map(toolbox.evaluate, parameters)
    new_pop = []
    for result in results:
        fit, ind = result
        ind.fitness.values = (fit,)
        new_pop.append(ind)
    pop[:] = new_pop

    train_data = []
    if AUTOENCODER:
        # Setup autoencoder
        flattened_paths = []
        #train_data = []
        for ind in pop:
            path_x, path_y, error, angle_error = ind.get_lowest_path(PATH_LENGTH)
            if error == 0:
                train_data.append(flatten_path(path_x, path_y, PATH_LENGTH))
            flattened_paths.append(flatten_path(path_x, path_y, PATH_LENGTH))
        ae = Autoencoder(PATH_LENGTH*2)
        ae.train(np.array(train_data), 3000)
    elif STRUCTURE:
        structure_data = []
        for ind in pop:
            structure_data.append(ind.get_structure_data(PATH_LENGTH))
        paths_x = []
        paths_y = []
        for ind in pop:
            path_x, path_y, error, angle_error = ind.get_lowest_path(PATH_LENGTH)
            paths_x.append(path_x)
            paths_y.append(path_y)
    else:
        # Get map placement data
        paths_x = []
        paths_y = []
        for ind in pop:
            path_x, path_y, error, angle_error = ind.get_lowest_path(PATH_LENGTH)
            paths_x.append(path_x)
            paths_y.append(path_y)

    # Place in map
    if AUTOENCODER:
        map_placements = AE_MAP_FUNCTION(np.array(flattened_paths), ae, MAP_SIZE, PATH_LENGTH)
    elif STRUCTURE:
        map_placements = STRUCTURE_MAP_FUNCTION(structure_data, paths_x, paths_y, MAP_SIZE)
    else:
        map_placements = MAP_FUNCTION(paths_x, paths_y, MAP_SIZE, PATH_LENGTH) 
    for ind, map_placement in zip(pop, map_placements):
        if mapp[map_placement] is None or ind.fitness.values[0] < mapp[map_placement].fitness.values[0]:
            mapp[map_placement] = ind

    # Setup pool
    the_pool = pool.Pool(THREADS)

    # Replace population
    pop = []
    map_indexes = [p for p in itertools.product(range(MAP_SIZE), repeat=DIMENSIONS)]
    for map_index in map_indexes:
        if mapp[map_index] is not None:
            pop.append(mapp[map_index])


    time_now = time.time()
    
    g = 0
    while time_now-time_start < MINUTES*60:
    #for g in range(0, ITERATIONS):
        if g == 150 and FITNESS_SWITCH:
            FITNESS_FUNCTION = fitness_spline_2
            pop = []
            map_indexes = [p for p in itertools.product(range(MAP_SIZE), repeat=DIMENSIONS)]
            for map_index in map_indexes:
                if mapp[map_index] is not None:
                    pop.append(mapp[map_index])
            print(len(pop))
            parameters = []
            for ind in pop:
                parameters.append([ind, FITNESS_FUNCTION])
            results = the_pool.map(toolbox.evaluate, parameters)
            new_pop = []
            for result in results:
                fit, ind = result
                ind.fitness.values = (fit,)
                new_pop.append(ind)
            pop[:] = new_pop
            #map_indexes = [p for p in itertools.product(range(MAP_SIZE), repeat=DIMENSIONS)]
            pop_index = 0
            print(len(pop))
            for map_index in map_indexes:
                if mapp[map_index] is not None:
                    mapp[map_index] = pop[pop_index]
                    pop_index += 1
        g += 1
        loggfile.write("\n--- gen " + str(g) + " ---\n")
        
        # Select offspring
        selected = toolbox.select(pop, POP_SIZE)
        offspring = [toolbox.clone(ind) for ind in selected]
        for i in range(len(offspring)):
            offspring[i].set_seed(seed.spawn(1)[0])

        # Mutate some
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values
            
        # Reevaluate mutated
        parameters = []
        for ind in offspring:
            parameters.append([ind, FITNESS_FUNCTION])
        results = the_pool.map(toolbox.evaluate, parameters)
        new_offspring = []
        for result in results:
            fit, ind = result
            ind.fitness.values = (fit,)
            new_offspring.append(ind)
        offspring[:] = new_offspring

        # Get map placement data 
        path_data = []
        paths_x = []
        paths_y = []
        structure_data = []
        for ind in offspring:
            path_x, path_y, error, angle_error = ind.get_lowest_path(PATH_LENGTH)
            if AUTOENCODER:
                path_data.append(flatten_path(path_x, path_y, PATH_LENGTH)) # (Uncomment for ae)
            elif STRUCTURE:
                structure_data.append(ind.get_structure_data(PATH_LENGTH))
                paths_x.append(path_x) # (Uncomment for NOT ae)
                paths_y.append(path_y)
            else:
                paths_x.append(path_x) # (Uncomment for NOT ae)
                paths_y.append(path_y) # (Uncomment for NOT ae)

        # Place in map
        if AUTOENCODER:
            map_placements = AE_MAP_FUNCTION(np.array(path_data), ae, MAP_SIZE, PATH_LENGTH) # (Uncomment for ae)
        elif STRUCTURE:
            map_placements = STRUCTURE_MAP_FUNCTION(structure_data, paths_x, paths_y, MAP_SIZE)
        else:
            map_placements = MAP_FUNCTION(paths_x, paths_y, MAP_SIZE, PATH_LENGTH) # (Uncomment for NOT ae)
        for ind, map_placement in zip(offspring, map_placements):
            if mapp[map_placement] is None or ind.fitness.values[0] < mapp[map_placement].fitness.values[0]:
                mapp[map_placement] = ind

        # Replace population
        pop = []
        map_indexes = [p for p in itertools.product(range(MAP_SIZE), repeat=DIMENSIONS)]
        for map_index in map_indexes:
            if mapp[map_index] is not None:
                pop.append(mapp[map_index])

        # Retrain autoencoder (Uncomment for ae)
        if AUTOENCODER and g%RETRAIN_PERIOD == 0 and g != 0:
            # Train autoencoder
            flattened_paths = []
            train_data = []
            for ind in pop:
                path_x, path_y, error, angle_error = ind.get_lowest_path(PATH_LENGTH)
                if error == 0:
                    train_data.append(flatten_path(path_x, path_y, PATH_LENGTH))
                flattened_paths.append(flatten_path(path_x, path_y, PATH_LENGTH))
            #ae = Autoencoder(PATH_LENGTH*2)
            ae.train(np.array(train_data), 1000)

            # Empty map
            mapp = np.full(map_shape, None)

            # Place in map
            map_placements = AE_MAP_FUNCTION(np.array(flattened_paths), ae, MAP_SIZE, PATH_LENGTH)
            for ind, map_placement in zip(pop, map_placements):
                if mapp[map_placement] is None or ind.fitness.values[0] < mapp[map_placement].fitness.values[0]:
                    mapp[map_placement] = ind

            # Replace population
            pop = []
            map_indexes = [p for p in itertools.product(range(MAP_SIZE), repeat=DIMENSIONS)]
            for map_index in map_indexes:
                if mapp[map_index] is not None:
                    pop.append(mapp[map_index])
                    
        # Record stats
        record = stats.compile(pop)
        loggfile.write(str(record)+"\n")
        loggfile.write("Coverage: " + str(len(pop))+"\n")
        avg_fitness_over_time.append(record['avg'])
        min_fitness_over_time.append(record['min'])
        coverage_over_time.append(len(pop))

        time_now = time.time()

    # Plot best one and stats
    all_fitnesses = [ind.fitness.values[0] for ind in pop]
    best_index = all_fitnesses.index(min(all_fitnesses))
    loggfile.write("BEST: " + str(min(all_fitnesses))+"\n")
    ind = pop[best_index]
    toolbox.show(ind)
    plt.figure()
    plt.plot(avg_fitness_over_time)
    plt.title("avg_fitness_over_time")
    plt.savefig(PATH + "avg_fitness_over_time.pdf")
    plt.figure()
    plt.plot(min_fitness_over_time)
    plt.title("min_fitness_over_time")
    plt.savefig(PATH + "min_fitness_over_time.pdf")
    plt.figure()
    plt.plot(coverage_over_time)
    plt.title("coverage_over_time")
    plt.savefig(PATH + "coverage_over_time.pdf")

    plt.figure()
    show_map(mapp)

    loggfile.close()
 
    pickle_data = {'mapp': mapp, 'avg_fitness_over_time': avg_fitness_over_time, 'min_fitness_over_time': min_fitness_over_time, 'coverage_over_time': coverage_over_time, 'g':g}
    with open(PATH + "data.pkl", "wb") as f:
        pickle.dump(pickle_data, f)
    
                
if __name__ == "__main__":
    if RUN_TYPE == 1:
        run()
    elif RUN_TYPE == 3:
        run_map()
    elif RUN_TYPE == 2:
        run_nsga2()
