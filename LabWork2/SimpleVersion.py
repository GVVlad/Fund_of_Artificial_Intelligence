import random
import numpy as np

from deap import base, creator, tools, algorithms


CHROMOSOME_LENGTH = 3
POPULATION_SIZE = 450
P_CROSSOVER = 0.4
P_MUTATION = 0.2
MAX_GENERATIONS = 60
HALL_OF_FAME_SIZE = 1
RANDOM_SEED = 10

def eval_func(individual):
    x, y, z = individual
    return 1.0 / (1.0 + (x-2)**2 + (y+1)**2 + (z-1)**2),

def create_toolbox():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.uniform, -5, 5)

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=CHROMOSOME_LENGTH)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_func)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    toolbox = create_toolbox()
    population = toolbox.population(n=POPULATION_SIZE)

    hall_of_fame = tools.HallOfFame(HALL_OF_FAME_SIZE)


    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)

    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

    mstats.register("avg", np.mean)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=mstats, halloffame=hall_of_fame, verbose=True)

    best_individual = hall_of_fame[0]
    best_fitness = eval_func(best_individual)[0]
    print("\nBest individual:", best_individual)
    print("Best fitness:", best_fitness)
