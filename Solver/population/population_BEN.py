import numpy as np

from Diversity.imports import diversidadHussain
from Problem.Benchmark.Problem import fitness as f
from Metaheuristics.imports import metaheuristics, MH_ARG_MAP


def initialize_population(mh, pop, dim, lb, ub):
    vel, pBestScore, pBest = None, None, None

    if mh == 'PSO':
        vel = np.zeros((pop, dim))
        pBestScore = np.full(pop, np.inf)
        pBest = np.zeros((pop, dim))

    lb = np.array(lb)
    ub = np.array(ub)

    population = np.random.uniform(0, 1, (pop, dim)) * (ub - lb) + lb

    return population, vel, pBestScore, pBest


def evaluate_population(mh, population, fitness, _, lb, ub, function):
    pBest, pBestScore = None, None

    if mh == 'PSO':
        pBest = np.zeros_like(population)
        pBestScore = np.full(population.shape[0], float("inf"))

    for i in range(population.shape[0]):
        population[i] = np.clip(population[i], lb, ub)
        fitness[i] = f(function, population[i])

        if mh == 'PSO' and pBestScore[i] > fitness[i]:
            pBestScore[i] = fitness[i]
            pBest[i] = population[i].copy()

    solutionsRanking = np.argsort(fitness)
    bestIndex = solutionsRanking[0]
    bestFitness = fitness[bestIndex]
    best = population[bestIndex].copy()

    return fitness, best, bestFitness, pBest, pBestScore


def update_population(population, fitness, _, lb, ub, function, best, bestFitness,
                      pBest=None, pBestScore=None, mh=None, posibles_mejoras=None):
    population = np.clip(population, lb, ub)

    for i in range(population.shape[0]):
        fitness[i] = f(function, population[i])

    if mh == 'LOA' and posibles_mejoras is not None:
        posibles_mejoras = np.clip(posibles_mejoras, lb, ub)

        for i in range(posibles_mejoras.shape[0]):
            mejora_fitness = f(function, posibles_mejoras[i])

            if mejora_fitness < fitness[i]:
                population[i] = posibles_mejoras[i]
                fitness[i] = mejora_fitness

    if mh == 'PSO':
        for i in range(population.shape[0]):
            if fitness[i] < pBestScore[i]:
                pBestScore[i] = fitness[i]
                pBest[i] = population[i]

    bestIndex = np.argmin(fitness)

    if fitness[bestIndex] < bestFitness:
        bestFitness = fitness[bestIndex]
        best = population[bestIndex].copy()

    div_t = diversidadHussain(population)

    return population, fitness, best, bestFitness, div_t


def iterate_population(mh, population, iter, maxIter, dim, fitness, best,
                       vel=None, pBest=None, ub=None, lb=None, fo=None, userData=None):
    """
    Itera sobre la población usando la metaheurística especificada ('mh'),
    construyendo los argumentos dinámicamente basados en MH_ARG_MAP.
    """

    if mh == 'PO':
        return np.array(population), vel, None

    if mh == 'HLOA':
        mh = 'HLOA_BEN'

    if mh not in metaheuristics:
        raise ValueError(f"Metaheurística '{mh}' no encontrada en el módulo de metaheurísticas.")
    if mh not in MH_ARG_MAP:
        raise ValueError(f"Metaheurística '{mh}' no encontrada en MH_ARG_MAP.")

    lb0_val = None
    ub0_val = None
    if lb is not None and len(lb) > 0:
        lb0_val = lb[0]
    if ub is not None and len(ub) > 0:
        ub0_val = ub[0]

    context = {
        'maxIter': maxIter,
        'iter': iter,
        'dim': dim,
        'population': population,
        'fitness': fitness,
        'best': best,
        'vel': vel,
        'pBest': pBest,
        'ub': ub,
        'lb': lb,
        'ub0': ub0_val,
        'lb0': lb0_val,
        'fo': fo,
        'objective_type': 'MIN',
        'userData': userData,
    }

    if userData:
        context.update(userData)

    required_args_names = MH_ARG_MAP[mh]

    kwargs = {}
    for arg_name in required_args_names:
        if arg_name not in context:
            raise KeyError(
                f"Error Interno: El argumento '{arg_name}' requerido por {mh} "
                f"(según MH_ARG_MAP) no se encontró en el diccionario 'context'."
            )
        kwargs[arg_name] = context[arg_name]

    mh_function = metaheuristics[mh]
    try:
        result = mh_function(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Error de tipo al llamar a {mh}. "
            f"Revisa MH_ARG_MAP['{mh}'] y la definición de la función."
        ) from e

    new_population = None
    new_vel = None
    posibles_mejoras = None

    if mh == 'LOA':
        if isinstance(result, tuple) and len(result) == 2:
            new_population, posibles_mejoras = result
            new_vel = vel
        else:
            raise TypeError(
                f"Retorno inesperado de {mh}. "
                f"Se esperaba (population, posibles_mejoras), se obtuvo {type(result)}"
            )

    elif mh == 'GOAT':   # 👈 caso especial GOAT
        if isinstance(result, tuple) and len(result) == 3:
            new_population, fitness, best = result
            new_vel = vel
        else:
            raise TypeError(
                f"Retorno inesperado de GOAT. "
                f"Se esperaba (population, fitness, best), se obtuvo {type(result)}"
            )

    elif isinstance(result, tuple) and len(result) == 2:
        new_population, new_vel = result

    elif isinstance(result, (np.ndarray, list)):
        new_population = result
        new_vel = vel

    else:
        raise TypeError(f"Tipo de retorno inesperado de {mh}: {type(result)}")

    if not isinstance(new_population, np.ndarray):
        new_population = np.array(new_population)

    return new_population, new_vel, posibles_mejoras
