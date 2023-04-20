import random
from deap import base, creator, tools

CHROMOSOME_LENGTH = 3
POPULATION_SIZE = 450
P_CROSSOVER = 0.4
P_MUTATION = 0.2
MAX_GENERATIONS = 60
HALL_OF_FAME_SIZE = 1
RANDOM_SEED = 7

MIN_VAL = -5
MAX_VAL = 5


def eval_func(individual):
    x, y, z = individual[0], individual[1], individual[2]
    return 1 / (1 + (x - 2) ** 2 + (y + 1) ** 2 + (z - 1) ** 2),

def create_toolbox(num_bits):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.randint,MIN_VAL, MAX_VAL)

    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, num_bits)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_func)

    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=10, low=MIN_VAL, up=MAX_VAL)

    toolbox.register("mutate", tools.mutPolynomialBounded, eta=10, low=MIN_VAL, up=MAX_VAL, indpb=0.05)

    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


if __name__ == "__main__":

    toolbox = create_toolbox(CHROMOSOME_LENGTH)

    random.seed(RANDOM_SEED)

    population = toolbox.population(n=POPULATION_SIZE)

    print('\nStarting the evolution process')

    fitnesses = list(map(toolbox.evaluate, population))

for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

fits = [ind.fitness.values[0] for ind in population]

best_ind = tools.HallOfFame(HALL_OF_FAME_SIZE)

for g in range(MAX_GENERATIONS):
    print("\n===== Generation %i" % g)

    offspring = toolbox.select(population, len(population))

    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < P_MUTATION:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    fitnesses = map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    print("  Evaluated %s individuals" % len(invalid_ind))

    population[:] = offspring

    fits = [ind.fitness.values[0] for ind in population]
    length = len(population)
    mean = sum(fits) / length

    best_ind.update(population)

    print("  Best individual is %s, \n  Fitness = %s" % (best_ind[0], best_ind[0].fitness.values))
    print("  Max =  %s, Min = %s" % (round(max(fits), 3), round(min(fits), 3)))
    print("  Avg = %s" % round(mean, 2))

print("\n-- End of evolution --")
print("Best individual is %s, \nFitness = %s" %(best_ind[0], best_ind[0].fitness.values))
