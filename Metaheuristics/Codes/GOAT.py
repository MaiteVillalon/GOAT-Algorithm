import numpy as np

def iterarGOAT(maxIter, iter, dim, population, best, fitness, fo, lb, ub,
               alpha=0.05, beta=0.5, jump_prob=0.1, filter_ratio=0.2, objective_type="min"):
    """
    Goat Optimization Algorithm (GOAT) - según el paper original.
    Compatible con funciones objetivo que devuelven escalar o tupla.
    """

    def evaluar(x):
        """Siempre devuelve un escalar fitness, incluso si fo(x) retorna tupla."""
        val = fo(x)
        if isinstance(val, tuple):
            return float(val[-1])
        return float(val)

    nPop = population.shape[0]

    # Normalizar límites
    lb = np.array([lb] * dim) if np.isscalar(lb) else np.array(lb)
    ub = np.array([ub] * dim) if np.isscalar(ub) else np.array(ub)

    # -------------------------------
    # 1. Exploración (Foraging)
    # -------------------------------
    for i in range(nPop):
        rand_vec = np.random.normal(0, 1, dim)
        candidate = population[i] + alpha * rand_vec * (ub - lb)

        # -------------------------------
        # 2. Explotación (Move to Best)
        # -------------------------------
        direction = (best - population[i])
        candidate = candidate + beta * np.random.rand(dim) * direction

        # -------------------------------
        # 3. Jump (Escape Local Optima)
        # -------------------------------
        if np.random.rand() < jump_prob:
            r = np.random.randint(0, nPop)
            candidate = candidate + np.random.rand() * (population[r] - population[i])

        # Clamping
        candidate = np.clip(candidate, lb, ub)

        # Evaluación
        fit_cand = evaluar(candidate)

        # Reemplazo
        if objective_type.lower() == "min":
            if fit_cand < fitness[i]:
                population[i] = candidate
                fitness[i] = fit_cand
        else:
            if fit_cand > fitness[i]:
                population[i] = candidate
                fitness[i] = fit_cand

    # -------------------------------
    # 4. Parasite Avoidance (Filtering)
    # -------------------------------
    n_filter = int(filter_ratio * nPop)
    if n_filter > 0:
        worst_idx = np.argsort(fitness)[-n_filter:]
        for idx in worst_idx:
            population[idx] = lb + (ub - lb) * np.random.rand(dim)
            fitness[idx] = evaluar(population[idx])

    # -------------------------------
    # 5. Actualizar mejor global
    # -------------------------------
    if objective_type.lower() == "min":
        idx = np.argmin(fitness)
        if fitness[idx] < evaluar(best):
            best = np.copy(population[idx])
    else:
        idx = np.argmax(fitness)
        if fitness[idx] > evaluar(best):
            best = np.copy(population[idx])

    return population, fitness, best
