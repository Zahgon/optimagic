from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from optimagic import mark
from optimagic.config import IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED
from optimagic.optimization.algo_options import (
    CONVERGENCE_FTOL_ABS,
    STOPPING_MAXFUN_GLOBAL,
    STOPPING_MAXITER,
    get_population_size,
)
from optimagic.optimization.algorithm import Algorithm, InternalOptimizeResult
from optimagic.optimization.internal_optimization_problem import (
    InternalBounds,
    InternalOptimizationProblem,
)
from optimagic.parameters.conversion import Converter
from optimagic.typing import (
    AggregationLevel,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    PyTree,
)
from optimagic.typing import UnitIntervalFloat as ProbabilityFloat

if TYPE_CHECKING:
    import pandas as pd
    from gradient_free_optimizers.optimizers.base_optimizer import BaseOptimizer


@dataclass(frozen=True)
class GFOCommonOptions:

    n_grid_points: PositiveInt | PyTree = 201
    """Number of grid points per dimension.

    If an integer is provided, it will be used for all dimensions.

    """

    n_init: PositiveInt = 20
    """Number of initialization steps to run.

    Accordingly, N//2 positions will be initialized in a grid like pattern and remaining
    initialized at the vertices and randomly in the search space.

    """

    stopping_maxiter: PositiveInt = STOPPING_MAXITER
    """Maximum number of iterations."""

    stopping_maxtime: NonNegativeFloat | None = None
    """Maximum time in seconds before termination."""

    convergence_target_value: float | None = None
    """"Stop the optimization if the objective function is less than this value."""

    convergence_iter_noimprove: PositiveInt = 1000000  # do not want to trigger this
    """Number of iterations without improvement before termination."""

    convergence_ftol_abs: NonNegativeFloat | None = (
        CONVERGENCE_FTOL_ABS  # set to zero, so disabled
    )
    """Converge if the absolute change in the objective function is less than this
    value."""

    convergence_ftol_rel: NonNegativeFloat | None = None
    """Converge if the relative change in the objective function is less than this
    value."""

    caching: bool = True
    """Whether to cache evaluated param and function values in a dictionary for
    lookup."""

    extra_start_params: list[PyTree] | None = None
    """List of additional start points for the optimization run.

    In case of population based optimizers, the initial_population can be provided
    via `extra_start_params`

    """

    warm_start: pd.DataFrame | None = None
    """Pandas dataframe that contains score and paramter information that will be
    automatically loaded into the memory.

    example:

    score       x1      x2      x...
    0.756       0.1     0.2     ...
    0.823       0.3     0.1     ...
    ... ...     ...     ...
    ... ...     ...     ...

    """

    verbosity: Literal["progress_bar", "print_results", "print_times"] | bool = False
    """Determines what part of the optimization information will be printed."""

    seed: int | None = None
    """Random seed for reproducibility."""

    rand_rest_p: ProbabilityFloat = 0
    """Probability for the optimization algorithm to jump to a random position in an
    iteration step."""




@mark.minimizer(
    name="gfo_hillclimbing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOHillClimbing(GFOCommonOptions, Algorithm):

    epsilon: PositiveFloat = 0.03
    """The step-size of the hill climbing algorithm. If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current postion before
    setting its current position to the best of those neighbour positions.

    If the value of n_neighbours is large the hill-climbing-based algorithm will take a
    lot of time to choose the next position to move to, but the choice will probably be
    a good one. It might be a prudent approach to increase n_neighbours of the search-
    space has a lot of dimensions, because there are more possible directions to move
    to.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.HillClimbingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
        )
        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_stochastichillclimbing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOStochasticHillClimbing(Algorithm, GFOCommonOptions):

    epsilon: PositiveFloat = 0.03
    """The step-size of the hill climbing algorithm. If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current postion before
    setting its current position to the best of those neighbour positions.

    If the value of n_neighbours is large the hill-climbing-based algorithm will take a
    lot of time to choose the next position to move to, but the choice will probably be
    a good one. It might be a prudent approach to increase n_neighbours of the search-
    space has a lot of dimensions, because there are more possible directions to move
    to.

    """

    p_accept: ProbabilityFloat = 0.5
    """The probability factor used in the equation to calculate if a worse position is
    accepted as the new position.

    If the new score is not better than the previous one the algorithm accepts worse
    positions with probability p_accept.

    .. math::
        score_{normalized} = norm * \\frac{score_{current} - score_{new}}
        {score_{current} + score_{new}}
    .. math::
        p = \\exp^{-score_{normalized}}

    If p is less than p_accept the new position gets accepted anyways.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.StochasticHillClimbingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
            p_accept=self.p_accept,
        )
        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_repulsinghillclimbing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFORepulsingHillClimbing(Algorithm, GFOCommonOptions):

    epsilon: PositiveFloat = 0.03
    """The step-size of the hill climbing algorithm. If step_size is too large the newly
    selected positions will be at the edge of the search space.

    If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current position before
    setting its current position to the best of those neighbour positions."""

    repulsion_factor: PositiveFloat = 5
    """The algorithm increases the step size by multiplying it with the repulsion_factor
    for the next iteration. This way the algorithm escapes the region that does not
    offer better positions.

    .. math::
        \\epsilon = \\epsilon * {repulsion factor}

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.RepulsingHillClimbingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
            repulsion_factor=self.repulsion_factor,
        )
        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_simulatedannealing",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
    experimental=True,
)
@dataclass(frozen=True)
class GFOSimulatedAnnealing(Algorithm, GFOCommonOptions):

    epsilon: PositiveFloat = 0.03
    """The step-size of the algorithm.

    If step_size is too large the newly selected positions will be at the edge of the
    search space. If its value is very low it might not find new positions.

    """

    distribution: Literal["normal", "laplace", "logistic", "gumbel"] = "normal"
    """The mathematical distribution the algorithm draws samples from.

    All available distributions are taken from the numpy-package.

    """

    n_neighbours: PositiveInt = 3
    """The number of positions the algorithm explores from its current position before
    setting its current position to the best of those neighbour positions."""

    start_temp: PositiveFloat = 1
    """The start_temp is a factor for the probability p of accepting a worse position.

    .. math::
        p = \\exp^{-\\frac{score_{normalized}}{temp}}

    """

    annealing_rate: PositiveFloat = 0.97
    """Rate at which the temperatur-value of the algorithm decreases. An annealing rate
    above 1 increases the temperature over time.

    .. math::
        start\\_temp \\leftarrow start\\_temp * annealing\\_rate

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.SimulatedAnnealingOptimizer
        optimizer = partial(
            opt,
            epsilon=self.epsilon,
            distribution=self.distribution,
            n_neighbours=self.n_neighbours,
            start_temp=self.start_temp,
            annealing_rate=self.annealing_rate,
        )
        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


@mark.minimizer(
    name="gfo_downhillsimplex",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
    experimental=True,
)
@dataclass(frozen=True)
class GFODownhillSimplex(Algorithm, GFOCommonOptions):

    simplex_reflection: PositiveFloat = 1
    """The reflection parameter of the simplex algorithm."""

    simplex_expansion: PositiveFloat = 2
    """The expansion parameter of the simplex algorithm."""

    simplex_contraction: PositiveFloat = 0.5
    """The contraction parameter of the simplex algorithm."""

    simplex_shrinking: PositiveFloat = 0.5
    """The shrinking parameter of the simplex algorithm."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.DownhillSimplexOptimizer
        optimizer = partial(
            opt,
            alpha=self.simplex_reflection,
            gamma=self.simplex_expansion,
            beta=self.simplex_contraction,
            sigma=self.simplex_shrinking,
        )
        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res


@mark.minimizer(
    name="gfo_powells_method",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOPowellsMethod(Algorithm, GFOCommonOptions):

    iters_p_dim: PositiveInt = 10
    """Number of iterations the algorithm will let the hill-climbing algorithm search to
    find the best position before it changes to the next dimension of the search space.

    Typical range: 5 to 15.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        opt = gfo.PowellsMethod
        optimizer = partial(
            opt,
            iters_p_dim=self.iters_p_dim,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )
        return res




@mark.minimizer(
    name="gfo_pso",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOParticleSwarmOptimization(Algorithm, GFOCommonOptions):

    population_size: PositiveInt | None = None
    """Size of the population."""

    inertia: NonNegativeFloat = 0.5 / math.log(2.0)
    """The inertia of the movement of the individual particles in the population."""

    cognitive_weight: NonNegativeFloat = 0.5 + math.log(2.0)
    """A factor of the movement towards the personal best position of the individual
    particles in the population."""

    social_weight: NonNegativeFloat = 0.5 + math.log(2.0)
    """A factor of the movement towards the global best position of the individual
    particles in the population."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        opt = gfo.ParticleSwarmOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            inertia=self.inertia,
            cognitive_weight=self.cognitive_weight,
            social_weight=self.social_weight,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_parallel_tempering",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOParallelTempering(Algorithm, GFOCommonOptions):

    population_size: PositiveInt | None = None
    """Size of the population."""

    n_iter_swap: PositiveInt = 10
    """The number of iterations the algorithm performs before switching temperatures of
    the individual optimizers in the population."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        opt = gfo.ParallelTemperingOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            n_iter_swap=self.n_iter_swap,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_spiral_optimization",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOSpiralOptimization(Algorithm, GFOCommonOptions):

    population_size: PositiveInt | None = None
    """Size of the population."""

    decay_rate: NonNegativeFloat = 0.99
    """The decay rate `r` is a factor, by which the radius of the spiral movement of the
    particles decays during their spiral movement.

    Lower values accelerate the convergence of the particles to the best known position,
    while values above 1 eventually lead to a movement where the particles spiral away
    from each other. Typical range: 0.85 to 1.15.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        opt = gfo.SpiralOptimization
        optimizer = partial(
            opt,
            population=population_size,
            decay_rate=self.decay_rate,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_genetic_algorithm",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOGeneticAlgorithm(Algorithm, GFOCommonOptions):

    population_size: PositiveInt | None = None
    """Size of the population."""

    mutation_rate: ProbabilityFloat = 0.5
    """Probability of a mutation event occurring in an individual of the population.
    Mutation helps in maintaining genetic diversity within the population and prevents
    the algorithm from getting stuck in local optima. Bits are randomly altered with.

    .. math::

        x'_i =
        \\begin{cases}
            x_i & \\text{if } \\text{rand} > p_m \\\\
            1 - x_i & \\text{if } \\text{rand} \\leq p_m
        \\end{cases}

    where p_m is mutation_rate.

    """

    crossover_rate: ProbabilityFloat = 0.5
    """Probability of a crossover event occurring between two parents. A higher
    crossover rate increases the diversity of the offspring, which can help in exploring
    the search space more effectively. Crossover happens with.

    .. math::

        u_{i,j}^{(g)} =
        \\begin{cases}
            v_{i,j}^{(g)} & \\text{if } \\text{rand}_j \\leq C_r \\text{ or } j =
            j_{\\text{rand}} \\\\
            x_{i,j}^{(g)} & \\text{otherwise}
        \\end{cases}

    where C_r is crossover_rate .

    """

    n_parents: PositiveInt = 2
    """The number of parents selected from the current population to participate in the
    crossover process to produce offspring.

    By default, pairs of parents are selected to generate new offspring.

    """

    n_offsprings: PositiveInt = 10
    """The number of offsprings generated in each generation through the processes of
    crossover and mutation.

    Typically, the number of offspring is equal to the population size, ensuring that
    the population size remains constant over generations.

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        opt = gfo.GeneticAlgorithmOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            n_parents=self.n_parents,
            offspring=self.n_offsprings,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_evolution_strategy",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFOEvolutionStrategy(Algorithm, GFOCommonOptions):

    population_size: PositiveInt | None = None
    """Size of the population."""

    stopping_maxiter: PositiveInt = STOPPING_MAXFUN_GLOBAL
    """Maximum number of iterations."""

    mutation_rate: ProbabilityFloat = 0.7
    """Probability of a mutation event occurring in an individual."""

    crossover_rate: ProbabilityFloat = 0.3
    """Probability of an individual to perform a crossover with the best individual in
    the population."""

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        opt = gfo.EvolutionStrategyOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            rand_rest_p=self.rand_rest_p,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res


@mark.minimizer(
    name="gfo_differential_evolution",
    solver_type=AggregationLevel.SCALAR,
    is_available=IS_GRADIENT_FREE_OPTIMIZERS_INSTALLED,
    is_global=True,
    needs_jac=False,
    needs_hess=False,
    needs_bounds=True,
    supports_parallelism=False,
    supports_bounds=True,
    supports_infinite_bounds=False,
    supports_linear_constraints=False,
    supports_nonlinear_constraints=False,
    disable_history=False,
)
@dataclass(frozen=True)
class GFODifferentialEvolution(Algorithm, GFOCommonOptions):

    population_size: PositiveInt | None = None
    """Size of the population."""

    mutation_rate: ProbabilityFloat = 0.9
    r"""Probability of a mutation event occurring in an individual.

    The mutation rate influences the algorithm's ability to explore the search space.
    A higher value of mutation_rate also called the differential weight `F` increases
    the diversity of the mutant individuals, leading to broader exploration,
    while a lower value encourages convergence by making smaller adjustments.

    .. math::

        \mathbf{v}_{i,G+1} = \mathbf{x}_{r1,G} + F \cdot (\mathbf{x}_{r2,G} -
        \mathbf{x}_{r3,G})

    """

    crossover_rate: ProbabilityFloat = 0.9
    """Probability of a crossover event occurring between two parents. It determines how
    much of the trial vector inherits its components from the mutant individual versus
    the target individual. A high crossover rate means that more components will come
    from the mutant individual, promoting exploration of new solutions. Conversely, a
    low crossover rate results in more components being taken from the target
    individual, which can help maintain existing solutions and refine them.

    .. math::

        u_{i,j,G+1} =
        \\begin{cases}
            v_{i,j,G+1} & \\text{if } \\text{rand}_j(0,1) \\leq CR \\text{ or } j =
              j_{\\text{rand}} \\\\
            x_{i,j,G} & \\text{otherwise}
        \\end{cases}

    """

    def _solve_internal_problem(
        self, problem: InternalOptimizationProblem, x0: NDArray[np.float64]
    ) -> InternalOptimizeResult:
        import gradient_free_optimizers as gfo

        population_size = get_population_size(
            population_size=self.population_size, x=x0, lower_bound=10
        )

        opt = gfo.DifferentialEvolutionOptimizer
        optimizer = partial(
            opt,
            population=population_size,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
        )

        res = _gfo_internal(
            common_options=self,
            problem=problem,
            x0=x0,
            optimizer=optimizer,
        )

        return res




def _gfo_internal(
    common_options: GFOCommonOptions,
    problem: InternalOptimizationProblem,
    x0: NDArray[np.float64],
    optimizer: BaseOptimizer,
) -> InternalOptimizeResult:
    """Internal helper function.

    Define the search space and inital params, define the objective function and run
    optimization.

    """
    common = common_options

    early_stopping = {
        "n_iter_no_change": common.convergence_iter_noimprove,
        "tol_abs": common.convergence_ftol_abs,
        "tol_rel": common.convergence_ftol_rel,
    }

    opt = optimizer(
        search_space=_get_search_space_gfo(
            problem.bounds,
            common.n_grid_points,
            problem.converter,
        ),
        initialize=_get_initialize_gfo(
            x0, common.n_init, common.extra_start_params, problem.converter
        ),
        constraints=_get_gfo_constraints(),
        random_state=common.seed,
    )

    def objective_function(para: dict[str, float]) -> float | NDArray[np.float64]:
        pass

    convergence_target_value = (
        -1 * common.convergence_target_value
        if common.convergence_target_value is not None
        else None
    )

    opt.search(
        objective_function=objective_function,
        n_iter=common.stopping_maxiter,
        max_time=common.stopping_maxtime,
        max_score=convergence_target_value,
        early_stopping=early_stopping,
        memory=common.caching,
        memory_warm_start=common.warm_start,
        verbosity=common.verbosity,
    )

    return _process_result_gfo(opt)


def _get_search_space_gfo(
    bounds: InternalBounds, n_grid_points: PositiveInt | PyTree, converter: Converter
) -> dict[str, NDArray[np.float64]]:
    """Create search space.

    Args:
        bounds: Internal Bounds
        n_grid_points: number of grid points in each dimension
    Returns:
    dict: search_space dictionary

    """
    search_space = {}
    if bounds.lower is not None and bounds.upper is not None:
        dim = len(bounds.lower)
        upper = bounds.upper
        lower = bounds.lower

    if isinstance(n_grid_points, int):
        n_grid_points = [n_grid_points] * dim
    else:
        n_grid_points = list(map(int, converter.params_to_internal(n_grid_points)))

    for i in range(dim):
        search_space[f"x{i}"] = np.linspace(lower[i], upper[i], n_grid_points[i])

    return search_space


def _get_gfo_constraints() -> list[Any]:
    """Process constraints."""
    return []


def _get_initialize_gfo(
    x0: NDArray[np.float64],
    n_init: PositiveInt,
    extra_start_points: list[PyTree] | None,
    converter: Converter,
) -> dict[str, Any]:
    """Set initial params x0, additional start params for the optimization run or the
    initial_population. Here, warm_start is actually extra_start_params.

    Args:
    x0: initial param

    Returns:
    dict: initialize dictionary with initial parameters set

    """
    init = _value2para(x0)
    x_list = [init]
    if extra_start_points is not None:
        internal_values = [converter.params_to_internal(x) for x in extra_start_points]
        extra_start_points = [_value2para(x) for x in internal_values]
        x_list += extra_start_points

    initialize = {
        "warm_start": x_list,
        "vertices": n_init // 2,
        "grid": n_init // 2,
    }
    return initialize


def _process_result_gfo(opt: "BaseOptimizer") -> InternalOptimizeResult:
    """Process result.

    Args:
        opt: Optimizer instance after optimization run is complete

    Returns:
        InternalOptimizeResult: Internal optimization result.

    """
    res = InternalOptimizeResult(
        x=np.array(opt.best_value),
        fun=-opt.best_score,  # negate once again
        success=True,
        n_fun_evals=len(opt.eval_times),
        n_jac_evals=0,
        n_hess_evals=0,
        n_iterations=opt.n_iter_search,
    )

    return res


def _value2para(x: NDArray[np.float64]) -> dict[str, float]:
    """Convert values to dict.

    Args:
        x: Array of parameter values

    Returns:
        dict: Dictionary of parameter values with key-value pair as { x{i} : x[i]}

    """
    para = {}
    for i in range(len(x)):
        para[f"x{i}"] = x[i]
    return para
