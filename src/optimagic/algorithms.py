
from dataclasses import dataclass
from typing import Type, cast

from optimagic.optimization.algorithm import Algorithm
from optimagic.optimizers.bayesian_optimizer import BayesOpt
from optimagic.optimizers.bhhh import BHHH
from optimagic.optimizers.fides import Fides
from optimagic.optimizers.gfo_optimizers import (
    GFODifferentialEvolution,
    GFODownhillSimplex,
    GFOEvolutionStrategy,
    GFOGeneticAlgorithm,
    GFOHillClimbing,
    GFOParallelTempering,
    GFOParticleSwarmOptimization,
    GFOPowellsMethod,
    GFORepulsingHillClimbing,
    GFOSimulatedAnnealing,
    GFOSpiralOptimization,
    GFOStochasticHillClimbing,
)
from optimagic.optimizers.iminuit_migrad import IminuitMigrad
from optimagic.optimizers.ipopt import Ipopt
from optimagic.optimizers.nag_optimizers import NagDFOLS, NagPyBOBYQA
from optimagic.optimizers.neldermead import NelderMeadParallel
from optimagic.optimizers.nevergrad_optimizers import (
    NevergradBayesOptim,
    NevergradCGA,
    NevergradCMAES,
    NevergradDifferentialEvolution,
    NevergradEDA,
    NevergradEMNA,
    NevergradMeta,
    NevergradNGOpt,
    NevergradOnePlusOne,
    NevergradPSO,
    NevergradRandomSearch,
    NevergradSamplingSearch,
    NevergradTBPSA,
)
from optimagic.optimizers.nlopt_optimizers import (
    NloptBOBYQA,
    NloptCCSAQ,
    NloptCOBYLA,
    NloptCRS2LM,
    NloptDirect,
    NloptESCH,
    NloptISRES,
    NloptLBFGSB,
    NloptMMA,
    NloptNelderMead,
    NloptNEWUOA,
    NloptPRAXIS,
    NloptSbplx,
    NloptSLSQP,
    NloptTNewton,
    NloptVAR,
)
from optimagic.optimizers.pounders import Pounders
from optimagic.optimizers.pygad_optimizer import Pygad
from optimagic.optimizers.pygmo_optimizers import (
    PygmoBeeColony,
    PygmoCmaes,
    PygmoCompassSearch,
    PygmoDe,
    PygmoDe1220,
    PygmoGaco,
    PygmoGwo,
    PygmoIhs,
    PygmoMbh,
    PygmoPso,
    PygmoPsoGen,
    PygmoSade,
    PygmoSea,
    PygmoSga,
    PygmoSimulatedAnnealing,
    PygmoXnes,
)
from optimagic.optimizers.pyswarms_optimizers import (
    PySwarmsGeneralPSO,
    PySwarmsGlobalBestPSO,
    PySwarmsLocalBestPSO,
)
from optimagic.optimizers.scipy_optimizers import (
    ScipyBasinhopping,
    ScipyBFGS,
    ScipyBrute,
    ScipyCOBYLA,
    ScipyConjugateGradient,
    ScipyDifferentialEvolution,
    ScipyDirect,
    ScipyDualAnnealing,
    ScipyLBFGSB,
    ScipyLSDogbox,
    ScipyLSLM,
    ScipyLSTRF,
    ScipyNelderMead,
    ScipyNewtonCG,
    ScipyPowell,
    ScipySHGO,
    ScipySLSQP,
    ScipyTruncatedNewton,
    ScipyTrustConstr,
)
from optimagic.optimizers.tao_optimizers import TAOPounders
from optimagic.optimizers.tranquilo import Tranquilo, TranquiloLS


@dataclass(frozen=True)
class AlgoSelection:
    def _all(self) -> list[Type[Algorithm]]:
        pass

    def _available(self) -> list[Type[Algorithm]]:
        pass

    @property
    def All(self) -> list[Type[Algorithm]]:
        pass

    @property
    def Available(self) -> list[Type[Algorithm]]:
        pass

    @property
    def AllNames(self) -> list[str]:
        pass

    @property
    def AvailableNames(self) -> list[str]:
        pass

    @property
    def _all_algorithms_dict(self) -> dict[str, Type[Algorithm]]:
        pass

    @property
    def _available_algorithms_dict(self) -> dict[str, Type[Algorithm]]:
        pass


@dataclass(frozen=True)
class BoundedGlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms(
    AlgoSelection
):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )


@dataclass(frozen=True)
class BoundedGlobalGradientBasedNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    scipy_shgo: Type[ScipySHGO] = ScipySHGO


@dataclass(frozen=True)
class BoundedGradientBasedLocalNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr


@dataclass(frozen=True)
class BoundedGlobalGradientFreeNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Parallel(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Scalar(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalGradientFreeParallelScalarAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class GlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGradientFreeLocalNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA


@dataclass(frozen=True)
class BoundedGradientFreeLocalParallelScalarAlgorithms(AlgoSelection):
    tranquilo: Type[Tranquilo] = Tranquilo


@dataclass(frozen=True)
class BoundedGradientFreeLeastSquaresLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS


@dataclass(frozen=True)
class BoundedGradientFreeNonlinearConstrainedParallelScalarAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Global(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGlobalNonlinearConstrainedParallelScalarAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def GradientFree(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms(AlgoSelection):
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Scalar(self) -> BoundedGlobalGradientBasedNonlinearConstrainedScalarAlgorithms:
        return BoundedGlobalGradientBasedNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalGradientBasedScalarAlgorithms(AlgoSelection):
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalGradientBasedNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class GlobalGradientBasedNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalGradientBasedNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGradientBasedLocalNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Scalar(self) -> BoundedGradientBasedLocalNonlinearConstrainedScalarAlgorithms:
        return BoundedGradientBasedLocalNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientBasedLocalScalarAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientBasedLocalNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGradientBasedLeastSquaresLocalAlgorithms(AlgoSelection):
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF


@dataclass(frozen=True)
class GradientBasedLocalNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedLocalNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGradientBasedNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Global(self) -> BoundedGlobalGradientBasedNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Local(self) -> BoundedGradientBasedLocalNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Parallel(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms()

    @property
    def Scalar(self) -> BoundedGlobalGradientFreeNonlinearConstrainedScalarAlgorithms:
        return BoundedGlobalGradientFreeNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalGradientFreeScalarAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGlobalGradientFreeParallelScalarAlgorithms:
        return BoundedGlobalGradientFreeParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalGradientFreeParallelAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedGlobalGradientFreeParallelScalarAlgorithms:
        return BoundedGlobalGradientFreeParallelScalarAlgorithms()


@dataclass(frozen=True)
class GlobalGradientFreeNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalGradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(
        self,
    ) -> GlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        return GlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class GlobalGradientFreeNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> GlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        return GlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class GlobalGradientFreeParallelScalarAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalGradientFreeParallelScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> GlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGradientFreeLocalNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA

    @property
    def Scalar(self) -> BoundedGradientFreeLocalNonlinearConstrainedScalarAlgorithms:
        return BoundedGradientFreeLocalNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeLocalScalarAlgorithms(AlgoSelection):
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientFreeLocalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGradientFreeLocalParallelScalarAlgorithms:
        return BoundedGradientFreeLocalParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeLeastSquaresLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Parallel(self) -> BoundedGradientFreeLeastSquaresLocalParallelAlgorithms:
        return BoundedGradientFreeLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def LeastSquares(self) -> BoundedGradientFreeLeastSquaresLocalParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedGradientFreeLocalParallelScalarAlgorithms:
        return BoundedGradientFreeLocalParallelScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeLocalNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA

    @property
    def Bounded(self) -> BoundedGradientFreeLocalNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class GradientFreeLocalParallelScalarAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Bounded(self) -> BoundedGradientFreeLocalParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class GradientFreeLeastSquaresLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLeastSquaresLocalParallelAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGradientFreeNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Global(self) -> BoundedGlobalGradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Local(self) -> BoundedGradientFreeLocalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(
        self,
    ) -> BoundedGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        return BoundedGradientFreeNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Global(self) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        return BoundedGradientFreeNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeParallelScalarAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Global(self) -> BoundedGlobalGradientFreeParallelScalarAlgorithms:
        pass

    @property
    def Local(self) -> BoundedGradientFreeLocalParallelScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGradientFreeLeastSquaresParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Local(self) -> BoundedGradientFreeLeastSquaresLocalParallelAlgorithms:
        pass


@dataclass(frozen=True)
class GradientFreeNonlinearConstrainedParallelScalarAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(
        self,
    ) -> BoundedGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGlobalNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def GradientBased(
        self,
    ) -> BoundedGlobalGradientBasedNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def GradientFree(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGlobalNonlinearConstrainedParallelScalarAlgorithms:
        return BoundedGlobalNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def GradientFree(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedGlobalNonlinearConstrainedParallelScalarAlgorithms:
        return BoundedGlobalNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalParallelScalarAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def GradientFree(self) -> BoundedGlobalGradientFreeParallelScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class GlobalNonlinearConstrainedParallelScalarAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalNonlinearConstrainedParallelScalarAlgorithms:
        pass

    @property
    def GradientFree(
        self,
    ) -> GlobalGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedLocalNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def GradientBased(
        self,
    ) -> BoundedGradientBasedLocalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def GradientFree(
        self,
    ) -> BoundedGradientFreeLocalNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedLocalParallelScalarAlgorithms(AlgoSelection):
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def GradientFree(self) -> BoundedGradientFreeLocalParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedLeastSquaresLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientFree(self) -> BoundedGradientFreeLeastSquaresLocalParallelAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedNonlinearConstrainedParallelScalarAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Global(self) -> BoundedGlobalNonlinearConstrainedParallelScalarAlgorithms:
        pass

    @property
    def GradientFree(
        self,
    ) -> BoundedGradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGlobalGradientBasedAlgorithms(AlgoSelection):
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedGlobalGradientBasedScalarAlgorithms:
        return BoundedGlobalGradientBasedScalarAlgorithms()


@dataclass(frozen=True)
class GlobalGradientBasedNonlinearConstrainedAlgorithms(AlgoSelection):
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> GlobalGradientBasedNonlinearConstrainedScalarAlgorithms:
        return GlobalGradientBasedNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class GlobalGradientBasedScalarAlgorithms(AlgoSelection):
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalGradientBasedScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> GlobalGradientBasedNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGradientBasedLocalAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def LeastSquares(self) -> BoundedGradientBasedLeastSquaresLocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientBasedLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedGradientBasedLocalScalarAlgorithms:
        return BoundedGradientBasedLocalScalarAlgorithms()


@dataclass(frozen=True)
class GradientBasedLocalNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> GradientBasedLocalNonlinearConstrainedScalarAlgorithms:
        return GradientBasedLocalNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class GradientBasedLocalScalarAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedLocalScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> GradientBasedLocalNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class GradientBasedLeastSquaresLocalAlgorithms(AlgoSelection):
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF

    @property
    def Bounded(self) -> BoundedGradientBasedLeastSquaresLocalAlgorithms:
        pass


@dataclass(frozen=True)
class GradientBasedLikelihoodLocalAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH


@dataclass(frozen=True)
class BoundedGradientBasedNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Global(self) -> BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def Local(self) -> BoundedGradientBasedLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedGradientBasedNonlinearConstrainedScalarAlgorithms:
        return BoundedGradientBasedNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientBasedScalarAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Global(self) -> BoundedGlobalGradientBasedScalarAlgorithms:
        pass

    @property
    def Local(self) -> BoundedGradientBasedLocalScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientBasedNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGradientBasedLeastSquaresAlgorithms(AlgoSelection):
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF

    @property
    def Local(self) -> BoundedGradientBasedLeastSquaresLocalAlgorithms:
        pass


@dataclass(frozen=True)
class GradientBasedNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientBasedNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Local(self) -> GradientBasedLocalNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGlobalGradientFreeAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGlobalGradientFreeParallelAlgorithms:
        return BoundedGlobalGradientFreeParallelAlgorithms()

    @property
    def Scalar(self) -> BoundedGlobalGradientFreeScalarAlgorithms:
        return BoundedGlobalGradientFreeScalarAlgorithms()


@dataclass(frozen=True)
class GlobalGradientFreeNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> GlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        return GlobalGradientFreeNonlinearConstrainedParallelAlgorithms()

    @property
    def Scalar(self) -> GlobalGradientFreeNonlinearConstrainedScalarAlgorithms:
        return GlobalGradientFreeNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class GlobalGradientFreeScalarAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect

    @property
    def Bounded(self) -> BoundedGlobalGradientFreeScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> GlobalGradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> GlobalGradientFreeParallelScalarAlgorithms:
        return GlobalGradientFreeParallelScalarAlgorithms()


@dataclass(frozen=True)
class GlobalGradientFreeParallelAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalGradientFreeParallelAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> GlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> GlobalGradientFreeParallelScalarAlgorithms:
        return GlobalGradientFreeParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    pounders: Type[Pounders] = Pounders
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def LeastSquares(self) -> BoundedGradientFreeLeastSquaresLocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientFreeLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGradientFreeLocalParallelAlgorithms:
        return BoundedGradientFreeLocalParallelAlgorithms()

    @property
    def Scalar(self) -> BoundedGradientFreeLocalScalarAlgorithms:
        return BoundedGradientFreeLocalScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeLocalNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA

    @property
    def Bounded(self) -> BoundedGradientFreeLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> GradientFreeLocalNonlinearConstrainedScalarAlgorithms:
        return GradientFreeLocalNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeLocalScalarAlgorithms(AlgoSelection):
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Bounded(self) -> BoundedGradientFreeLocalScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> GradientFreeLocalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> GradientFreeLocalParallelScalarAlgorithms:
        return GradientFreeLocalParallelScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeLeastSquaresLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLeastSquaresLocalAlgorithms:
        pass

    @property
    def Parallel(self) -> GradientFreeLeastSquaresLocalParallelAlgorithms:
        return GradientFreeLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class GradientFreeLocalParallelAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    pounders: Type[Pounders] = Pounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLocalParallelAlgorithms:
        pass

    @property
    def LeastSquares(self) -> GradientFreeLeastSquaresLocalParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> GradientFreeLocalParallelScalarAlgorithms:
        return GradientFreeLocalParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Global(self) -> BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Local(self) -> BoundedGradientFreeLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGradientFreeNonlinearConstrainedParallelAlgorithms:
        return BoundedGradientFreeNonlinearConstrainedParallelAlgorithms()

    @property
    def Scalar(self) -> BoundedGradientFreeNonlinearConstrainedScalarAlgorithms:
        return BoundedGradientFreeNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeScalarAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Global(self) -> BoundedGlobalGradientFreeScalarAlgorithms:
        pass

    @property
    def Local(self) -> BoundedGradientFreeLocalScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGradientFreeParallelScalarAlgorithms:
        return BoundedGradientFreeParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeLeastSquaresAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Local(self) -> BoundedGradientFreeLeastSquaresLocalAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGradientFreeLeastSquaresParallelAlgorithms:
        return BoundedGradientFreeLeastSquaresParallelAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeParallelAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pounders: Type[Pounders] = Pounders
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Global(self) -> BoundedGlobalGradientFreeParallelAlgorithms:
        pass

    @property
    def LeastSquares(self) -> BoundedGradientFreeLeastSquaresParallelAlgorithms:
        pass

    @property
    def Local(self) -> BoundedGradientFreeLocalParallelAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedGradientFreeParallelScalarAlgorithms:
        return BoundedGradientFreeParallelScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Local(self) -> GradientFreeLocalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> GradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        return GradientFreeNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> GradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        return GradientFreeNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeParallelScalarAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Bounded(self) -> BoundedGradientFreeParallelScalarAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientFreeParallelScalarAlgorithms:
        pass

    @property
    def Local(self) -> GradientFreeLocalParallelScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> GradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class GradientFreeLeastSquaresParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLeastSquaresParallelAlgorithms:
        pass

    @property
    def Local(self) -> GradientFreeLeastSquaresLocalParallelAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedGlobalNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def GradientBased(self) -> BoundedGlobalGradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGlobalGradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGlobalNonlinearConstrainedParallelAlgorithms:
        return BoundedGlobalNonlinearConstrainedParallelAlgorithms()

    @property
    def Scalar(self) -> BoundedGlobalNonlinearConstrainedScalarAlgorithms:
        return BoundedGlobalNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalScalarAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def GradientBased(self) -> BoundedGlobalGradientBasedScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGlobalGradientFreeScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> BoundedGlobalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGlobalParallelScalarAlgorithms:
        return BoundedGlobalParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalParallelAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def GradientFree(self) -> BoundedGlobalGradientFreeParallelAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGlobalNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedGlobalParallelScalarAlgorithms:
        return BoundedGlobalParallelScalarAlgorithms()


@dataclass(frozen=True)
class GlobalNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def GradientBased(self) -> GlobalGradientBasedNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> GlobalGradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> GlobalNonlinearConstrainedParallelScalarAlgorithms:
        return GlobalNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class GlobalNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def GradientFree(self) -> GlobalGradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> GlobalNonlinearConstrainedParallelScalarAlgorithms:
        return GlobalNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class GlobalParallelScalarAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalParallelScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> GlobalGradientFreeParallelScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> GlobalNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedLocalNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def GradientBased(self) -> BoundedGradientBasedLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedLocalNonlinearConstrainedScalarAlgorithms:
        return BoundedLocalNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class BoundedLocalScalarAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def GradientBased(self) -> BoundedGradientBasedLocalScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeLocalScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> BoundedLocalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedLocalParallelScalarAlgorithms:
        return BoundedLocalParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedLeastSquaresLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientBased(self) -> BoundedGradientBasedLeastSquaresLocalAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeLeastSquaresLocalAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedLeastSquaresLocalParallelAlgorithms:
        return BoundedLeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class BoundedLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientFree(self) -> BoundedGradientFreeLocalParallelAlgorithms:
        pass

    @property
    def LeastSquares(self) -> BoundedLeastSquaresLocalParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedLocalParallelScalarAlgorithms:
        return BoundedLocalParallelScalarAlgorithms()


@dataclass(frozen=True)
class LocalNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedLocalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def GradientBased(self) -> GradientBasedLocalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeLocalNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class LocalParallelScalarAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Bounded(self) -> BoundedLocalParallelScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeLocalParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class LeastSquaresLocalParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLeastSquaresLocalParallelAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeLeastSquaresLocalParallelAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedNonlinearConstrainedScalarAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Global(self) -> BoundedGlobalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def GradientBased(self) -> BoundedGradientBasedNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Local(self) -> BoundedLocalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedNonlinearConstrainedParallelScalarAlgorithms:
        return BoundedNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedNonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Global(self) -> BoundedGlobalNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedNonlinearConstrainedParallelScalarAlgorithms:
        return BoundedNonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedParallelScalarAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Global(self) -> BoundedGlobalParallelScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeParallelScalarAlgorithms:
        pass

    @property
    def Local(self) -> BoundedLocalParallelScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class BoundedLeastSquaresParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientFree(self) -> BoundedGradientFreeLeastSquaresParallelAlgorithms:
        pass

    @property
    def Local(self) -> BoundedLeastSquaresLocalParallelAlgorithms:
        pass


@dataclass(frozen=True)
class NonlinearConstrainedParallelScalarAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedNonlinearConstrainedParallelScalarAlgorithms:
        pass

    @property
    def Global(self) -> GlobalNonlinearConstrainedParallelScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeNonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class GlobalGradientBasedAlgorithms(AlgoSelection):
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalGradientBasedAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> GlobalGradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> GlobalGradientBasedScalarAlgorithms:
        return GlobalGradientBasedScalarAlgorithms()


@dataclass(frozen=True)
class GradientBasedLocalAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedLocalAlgorithms:
        pass

    @property
    def LeastSquares(self) -> GradientBasedLeastSquaresLocalAlgorithms:
        pass

    @property
    def Likelihood(self) -> GradientBasedLikelihoodLocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> GradientBasedLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> GradientBasedLocalScalarAlgorithms:
        return GradientBasedLocalScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientBasedAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Global(self) -> BoundedGlobalGradientBasedAlgorithms:
        pass

    @property
    def LeastSquares(self) -> BoundedGradientBasedLeastSquaresAlgorithms:
        pass

    @property
    def Local(self) -> BoundedGradientBasedLocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> BoundedGradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedGradientBasedScalarAlgorithms:
        return BoundedGradientBasedScalarAlgorithms()


@dataclass(frozen=True)
class GradientBasedNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def Local(self) -> GradientBasedLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> GradientBasedNonlinearConstrainedScalarAlgorithms:
        return GradientBasedNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class GradientBasedScalarAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedScalarAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientBasedScalarAlgorithms:
        pass

    @property
    def Local(self) -> GradientBasedLocalScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> GradientBasedNonlinearConstrainedScalarAlgorithms:
        pass


@dataclass(frozen=True)
class GradientBasedLeastSquaresAlgorithms(AlgoSelection):
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF

    @property
    def Bounded(self) -> BoundedGradientBasedLeastSquaresAlgorithms:
        pass

    @property
    def Local(self) -> GradientBasedLeastSquaresLocalAlgorithms:
        pass


@dataclass(frozen=True)
class GradientBasedLikelihoodAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH

    @property
    def Local(self) -> GradientBasedLikelihoodLocalAlgorithms:
        pass


@dataclass(frozen=True)
class GlobalGradientFreeAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect

    @property
    def Bounded(self) -> BoundedGlobalGradientFreeAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> GlobalGradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> GlobalGradientFreeParallelAlgorithms:
        return GlobalGradientFreeParallelAlgorithms()

    @property
    def Scalar(self) -> GlobalGradientFreeScalarAlgorithms:
        return GlobalGradientFreeScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    pounders: Type[Pounders] = Pounders
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLocalAlgorithms:
        pass

    @property
    def LeastSquares(self) -> GradientFreeLeastSquaresLocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> GradientFreeLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> GradientFreeLocalParallelAlgorithms:
        return GradientFreeLocalParallelAlgorithms()

    @property
    def Scalar(self) -> GradientFreeLocalScalarAlgorithms:
        return GradientFreeLocalScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGradientFreeAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    pounders: Type[Pounders] = Pounders
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Global(self) -> BoundedGlobalGradientFreeAlgorithms:
        pass

    @property
    def LeastSquares(self) -> BoundedGradientFreeLeastSquaresAlgorithms:
        pass

    @property
    def Local(self) -> BoundedGradientFreeLocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> BoundedGradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGradientFreeParallelAlgorithms:
        return BoundedGradientFreeParallelAlgorithms()

    @property
    def Scalar(self) -> BoundedGradientFreeScalarAlgorithms:
        return BoundedGradientFreeScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Local(self) -> GradientFreeLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> GradientFreeNonlinearConstrainedParallelAlgorithms:
        return GradientFreeNonlinearConstrainedParallelAlgorithms()

    @property
    def Scalar(self) -> GradientFreeNonlinearConstrainedScalarAlgorithms:
        return GradientFreeNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeScalarAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Bounded(self) -> BoundedGradientFreeScalarAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientFreeScalarAlgorithms:
        pass

    @property
    def Local(self) -> GradientFreeLocalScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> GradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> GradientFreeParallelScalarAlgorithms:
        return GradientFreeParallelScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeLeastSquaresAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeLeastSquaresAlgorithms:
        pass

    @property
    def Local(self) -> GradientFreeLeastSquaresLocalAlgorithms:
        pass

    @property
    def Parallel(self) -> GradientFreeLeastSquaresParallelAlgorithms:
        return GradientFreeLeastSquaresParallelAlgorithms()


@dataclass(frozen=True)
class GradientFreeParallelAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pounders: Type[Pounders] = Pounders
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeParallelAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientFreeParallelAlgorithms:
        pass

    @property
    def LeastSquares(self) -> GradientFreeLeastSquaresParallelAlgorithms:
        pass

    @property
    def Local(self) -> GradientFreeLocalParallelAlgorithms:
        pass

    @property
    def NonlinearConstrained(
        self,
    ) -> GradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> GradientFreeParallelScalarAlgorithms:
        return GradientFreeParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedGlobalAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def GradientBased(self) -> BoundedGlobalGradientBasedAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGlobalGradientFreeAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> BoundedGlobalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedGlobalParallelAlgorithms:
        return BoundedGlobalParallelAlgorithms()

    @property
    def Scalar(self) -> BoundedGlobalScalarAlgorithms:
        return BoundedGlobalScalarAlgorithms()


@dataclass(frozen=True)
class GlobalNonlinearConstrainedAlgorithms(AlgoSelection):
    nlopt_isres: Type[NloptISRES] = NloptISRES
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalNonlinearConstrainedAlgorithms:
        pass

    @property
    def GradientBased(self) -> GlobalGradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def GradientFree(self) -> GlobalGradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> GlobalNonlinearConstrainedParallelAlgorithms:
        return GlobalNonlinearConstrainedParallelAlgorithms()

    @property
    def Scalar(self) -> GlobalNonlinearConstrainedScalarAlgorithms:
        return GlobalNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class GlobalScalarAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalScalarAlgorithms:
        pass

    @property
    def GradientBased(self) -> GlobalGradientBasedScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> GlobalGradientFreeScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> GlobalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> GlobalParallelScalarAlgorithms:
        return GlobalParallelScalarAlgorithms()


@dataclass(frozen=True)
class GlobalParallelAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedGlobalParallelAlgorithms:
        pass

    @property
    def GradientFree(self) -> GlobalGradientFreeParallelAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> GlobalNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> GlobalParallelScalarAlgorithms:
        return GlobalParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedLocalAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    pounders: Type[Pounders] = Pounders
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientBased(self) -> BoundedGradientBasedLocalAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeLocalAlgorithms:
        pass

    @property
    def LeastSquares(self) -> BoundedLeastSquaresLocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> BoundedLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedLocalParallelAlgorithms:
        return BoundedLocalParallelAlgorithms()

    @property
    def Scalar(self) -> BoundedLocalScalarAlgorithms:
        return BoundedLocalScalarAlgorithms()


@dataclass(frozen=True)
class LocalNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def GradientBased(self) -> GradientBasedLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> LocalNonlinearConstrainedScalarAlgorithms:
        return LocalNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class LocalScalarAlgorithms(AlgoSelection):
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Bounded(self) -> BoundedLocalScalarAlgorithms:
        pass

    @property
    def GradientBased(self) -> GradientBasedLocalScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeLocalScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> LocalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> LocalParallelScalarAlgorithms:
        return LocalParallelScalarAlgorithms()


@dataclass(frozen=True)
class LeastSquaresLocalAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLeastSquaresLocalAlgorithms:
        pass

    @property
    def GradientBased(self) -> GradientBasedLeastSquaresLocalAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeLeastSquaresLocalAlgorithms:
        pass

    @property
    def Parallel(self) -> LeastSquaresLocalParallelAlgorithms:
        return LeastSquaresLocalParallelAlgorithms()


@dataclass(frozen=True)
class LikelihoodLocalAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH

    @property
    def GradientBased(self) -> GradientBasedLikelihoodLocalAlgorithms:
        pass


@dataclass(frozen=True)
class LocalParallelAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    pounders: Type[Pounders] = Pounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLocalParallelAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeLocalParallelAlgorithms:
        pass

    @property
    def LeastSquares(self) -> LeastSquaresLocalParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> LocalParallelScalarAlgorithms:
        return LocalParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedNonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Global(self) -> BoundedGlobalNonlinearConstrainedAlgorithms:
        pass

    @property
    def GradientBased(self) -> BoundedGradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Local(self) -> BoundedLocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedNonlinearConstrainedParallelAlgorithms:
        return BoundedNonlinearConstrainedParallelAlgorithms()

    @property
    def Scalar(self) -> BoundedNonlinearConstrainedScalarAlgorithms:
        return BoundedNonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class BoundedScalarAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    fides: Type[Fides] = Fides
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Global(self) -> BoundedGlobalScalarAlgorithms:
        pass

    @property
    def GradientBased(self) -> BoundedGradientBasedScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeScalarAlgorithms:
        pass

    @property
    def Local(self) -> BoundedLocalScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> BoundedNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedParallelScalarAlgorithms:
        return BoundedParallelScalarAlgorithms()


@dataclass(frozen=True)
class BoundedLeastSquaresAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def GradientBased(self) -> BoundedGradientBasedLeastSquaresAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeLeastSquaresAlgorithms:
        pass

    @property
    def Local(self) -> BoundedLeastSquaresLocalAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedLeastSquaresParallelAlgorithms:
        return BoundedLeastSquaresParallelAlgorithms()


@dataclass(frozen=True)
class BoundedParallelAlgorithms(AlgoSelection):
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pounders: Type[Pounders] = Pounders
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Global(self) -> BoundedGlobalParallelAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeParallelAlgorithms:
        pass

    @property
    def LeastSquares(self) -> BoundedLeastSquaresParallelAlgorithms:
        pass

    @property
    def Local(self) -> BoundedLocalParallelAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> BoundedNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> BoundedParallelScalarAlgorithms:
        return BoundedParallelScalarAlgorithms()


@dataclass(frozen=True)
class NonlinearConstrainedScalarAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Global(self) -> GlobalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def GradientBased(self) -> GradientBasedNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Local(self) -> LocalNonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> NonlinearConstrainedParallelScalarAlgorithms:
        return NonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class NonlinearConstrainedParallelAlgorithms(AlgoSelection):
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )

    @property
    def Bounded(self) -> BoundedNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Global(self) -> GlobalNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeNonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> NonlinearConstrainedParallelScalarAlgorithms:
        return NonlinearConstrainedParallelScalarAlgorithms()


@dataclass(frozen=True)
class ParallelScalarAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Bounded(self) -> BoundedParallelScalarAlgorithms:
        pass

    @property
    def Global(self) -> GlobalParallelScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeParallelScalarAlgorithms:
        pass

    @property
    def Local(self) -> LocalParallelScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> NonlinearConstrainedParallelScalarAlgorithms:
        pass


@dataclass(frozen=True)
class LeastSquaresParallelAlgorithms(AlgoSelection):
    pounders: Type[Pounders] = Pounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLeastSquaresParallelAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeLeastSquaresParallelAlgorithms:
        pass

    @property
    def Local(self) -> LeastSquaresLocalParallelAlgorithms:
        pass


@dataclass(frozen=True)
class GradientBasedAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedGradientBasedAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientBasedAlgorithms:
        pass

    @property
    def LeastSquares(self) -> GradientBasedLeastSquaresAlgorithms:
        pass

    @property
    def Likelihood(self) -> GradientBasedLikelihoodAlgorithms:
        pass

    @property
    def Local(self) -> GradientBasedLocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> GradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def Scalar(self) -> GradientBasedScalarAlgorithms:
        return GradientBasedScalarAlgorithms()


@dataclass(frozen=True)
class GradientFreeAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    pounders: Type[Pounders] = Pounders
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedGradientFreeAlgorithms:
        pass

    @property
    def Global(self) -> GlobalGradientFreeAlgorithms:
        pass

    @property
    def LeastSquares(self) -> GradientFreeLeastSquaresAlgorithms:
        pass

    @property
    def Local(self) -> GradientFreeLocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> GradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> GradientFreeParallelAlgorithms:
        return GradientFreeParallelAlgorithms()

    @property
    def Scalar(self) -> GradientFreeScalarAlgorithms:
        return GradientFreeScalarAlgorithms()


@dataclass(frozen=True)
class GlobalAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_shgo: Type[ScipySHGO] = ScipySHGO

    @property
    def Bounded(self) -> BoundedGlobalAlgorithms:
        pass

    @property
    def GradientBased(self) -> GlobalGradientBasedAlgorithms:
        pass

    @property
    def GradientFree(self) -> GlobalGradientFreeAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> GlobalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> GlobalParallelAlgorithms:
        return GlobalParallelAlgorithms()

    @property
    def Scalar(self) -> GlobalScalarAlgorithms:
        return GlobalScalarAlgorithms()


@dataclass(frozen=True)
class LocalAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH
    fides: Type[Fides] = Fides
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    pounders: Type[Pounders] = Pounders
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLocalAlgorithms:
        pass

    @property
    def GradientBased(self) -> GradientBasedLocalAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeLocalAlgorithms:
        pass

    @property
    def LeastSquares(self) -> LeastSquaresLocalAlgorithms:
        pass

    @property
    def Likelihood(self) -> LikelihoodLocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> LocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> LocalParallelAlgorithms:
        return LocalParallelAlgorithms()

    @property
    def Scalar(self) -> LocalScalarAlgorithms:
        return LocalScalarAlgorithms()


@dataclass(frozen=True)
class BoundedAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    fides: Type[Fides] = Fides
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    pounders: Type[Pounders] = Pounders
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Global(self) -> BoundedGlobalAlgorithms:
        pass

    @property
    def GradientBased(self) -> BoundedGradientBasedAlgorithms:
        pass

    @property
    def GradientFree(self) -> BoundedGradientFreeAlgorithms:
        pass

    @property
    def LeastSquares(self) -> BoundedLeastSquaresAlgorithms:
        pass

    @property
    def Local(self) -> BoundedLocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> BoundedNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> BoundedParallelAlgorithms:
        return BoundedParallelAlgorithms()

    @property
    def Scalar(self) -> BoundedScalarAlgorithms:
        return BoundedScalarAlgorithms()


@dataclass(frozen=True)
class NonlinearConstrainedAlgorithms(AlgoSelection):
    ipopt: Type[Ipopt] = Ipopt
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr

    @property
    def Bounded(self) -> BoundedNonlinearConstrainedAlgorithms:
        pass

    @property
    def Global(self) -> GlobalNonlinearConstrainedAlgorithms:
        pass

    @property
    def GradientBased(self) -> GradientBasedNonlinearConstrainedAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeNonlinearConstrainedAlgorithms:
        pass

    @property
    def Local(self) -> LocalNonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> NonlinearConstrainedParallelAlgorithms:
        return NonlinearConstrainedParallelAlgorithms()

    @property
    def Scalar(self) -> NonlinearConstrainedScalarAlgorithms:
        return NonlinearConstrainedScalarAlgorithms()


@dataclass(frozen=True)
class ScalarAlgorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    fides: Type[Fides] = Fides
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tranquilo: Type[Tranquilo] = Tranquilo

    @property
    def Bounded(self) -> BoundedScalarAlgorithms:
        pass

    @property
    def Global(self) -> GlobalScalarAlgorithms:
        pass

    @property
    def GradientBased(self) -> GradientBasedScalarAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeScalarAlgorithms:
        pass

    @property
    def Local(self) -> LocalScalarAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> NonlinearConstrainedScalarAlgorithms:
        pass

    @property
    def Parallel(self) -> ParallelScalarAlgorithms:
        return ParallelScalarAlgorithms()


@dataclass(frozen=True)
class LeastSquaresAlgorithms(AlgoSelection):
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    pounders: Type[Pounders] = Pounders
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedLeastSquaresAlgorithms:
        pass

    @property
    def GradientBased(self) -> GradientBasedLeastSquaresAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeLeastSquaresAlgorithms:
        pass

    @property
    def Local(self) -> LeastSquaresLocalAlgorithms:
        pass

    @property
    def Parallel(self) -> LeastSquaresParallelAlgorithms:
        return LeastSquaresParallelAlgorithms()


@dataclass(frozen=True)
class LikelihoodAlgorithms(AlgoSelection):
    bhhh: Type[BHHH] = BHHH

    @property
    def GradientBased(self) -> GradientBasedLikelihoodAlgorithms:
        pass

    @property
    def Local(self) -> LikelihoodLocalAlgorithms:
        pass


@dataclass(frozen=True)
class ParallelAlgorithms(AlgoSelection):
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    pounders: Type[Pounders] = Pounders
    pygad: Type[Pygad] = Pygad
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedParallelAlgorithms:
        pass

    @property
    def Global(self) -> GlobalParallelAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeParallelAlgorithms:
        pass

    @property
    def LeastSquares(self) -> LeastSquaresParallelAlgorithms:
        pass

    @property
    def Local(self) -> LocalParallelAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> NonlinearConstrainedParallelAlgorithms:
        pass

    @property
    def Scalar(self) -> ParallelScalarAlgorithms:
        return ParallelScalarAlgorithms()


@dataclass(frozen=True)
class Algorithms(AlgoSelection):
    bayes_opt: Type[BayesOpt] = BayesOpt
    bhhh: Type[BHHH] = BHHH
    fides: Type[Fides] = Fides
    gfo_differential_evolution: Type[GFODifferentialEvolution] = (
        GFODifferentialEvolution
    )
    gfo_downhillsimplex: Type[GFODownhillSimplex] = GFODownhillSimplex
    gfo_evolution_strategy: Type[GFOEvolutionStrategy] = GFOEvolutionStrategy
    gfo_genetic_algorithm: Type[GFOGeneticAlgorithm] = GFOGeneticAlgorithm
    gfo_hillclimbing: Type[GFOHillClimbing] = GFOHillClimbing
    gfo_parallel_tempering: Type[GFOParallelTempering] = GFOParallelTempering
    gfo_pso: Type[GFOParticleSwarmOptimization] = GFOParticleSwarmOptimization
    gfo_powells_method: Type[GFOPowellsMethod] = GFOPowellsMethod
    gfo_repulsinghillclimbing: Type[GFORepulsingHillClimbing] = GFORepulsingHillClimbing
    gfo_simulatedannealing: Type[GFOSimulatedAnnealing] = GFOSimulatedAnnealing
    gfo_spiral_optimization: Type[GFOSpiralOptimization] = GFOSpiralOptimization
    gfo_stochastichillclimbing: Type[GFOStochasticHillClimbing] = (
        GFOStochasticHillClimbing
    )
    iminuit_migrad: Type[IminuitMigrad] = IminuitMigrad
    ipopt: Type[Ipopt] = Ipopt
    nag_dfols: Type[NagDFOLS] = NagDFOLS
    nag_pybobyqa: Type[NagPyBOBYQA] = NagPyBOBYQA
    neldermead_parallel: Type[NelderMeadParallel] = NelderMeadParallel
    nevergrad_bo: Type[NevergradBayesOptim] = NevergradBayesOptim
    nevergrad_cga: Type[NevergradCGA] = NevergradCGA
    nevergrad_cmaes: Type[NevergradCMAES] = NevergradCMAES
    nevergrad_de: Type[NevergradDifferentialEvolution] = NevergradDifferentialEvolution
    nevergrad_eda: Type[NevergradEDA] = NevergradEDA
    nevergrad_emna: Type[NevergradEMNA] = NevergradEMNA
    nevergrad_meta: Type[NevergradMeta] = NevergradMeta
    nevergrad_ngopt: Type[NevergradNGOpt] = NevergradNGOpt
    nevergrad_oneplusone: Type[NevergradOnePlusOne] = NevergradOnePlusOne
    nevergrad_pso: Type[NevergradPSO] = NevergradPSO
    nevergrad_randomsearch: Type[NevergradRandomSearch] = NevergradRandomSearch
    nevergrad_samplingsearch: Type[NevergradSamplingSearch] = NevergradSamplingSearch
    nevergrad_tbpsa: Type[NevergradTBPSA] = NevergradTBPSA
    nlopt_bobyqa: Type[NloptBOBYQA] = NloptBOBYQA
    nlopt_ccsaq: Type[NloptCCSAQ] = NloptCCSAQ
    nlopt_cobyla: Type[NloptCOBYLA] = NloptCOBYLA
    nlopt_crs2_lm: Type[NloptCRS2LM] = NloptCRS2LM
    nlopt_direct: Type[NloptDirect] = NloptDirect
    nlopt_esch: Type[NloptESCH] = NloptESCH
    nlopt_isres: Type[NloptISRES] = NloptISRES
    nlopt_lbfgsb: Type[NloptLBFGSB] = NloptLBFGSB
    nlopt_mma: Type[NloptMMA] = NloptMMA
    nlopt_newuoa: Type[NloptNEWUOA] = NloptNEWUOA
    nlopt_neldermead: Type[NloptNelderMead] = NloptNelderMead
    nlopt_praxis: Type[NloptPRAXIS] = NloptPRAXIS
    nlopt_slsqp: Type[NloptSLSQP] = NloptSLSQP
    nlopt_sbplx: Type[NloptSbplx] = NloptSbplx
    nlopt_tnewton: Type[NloptTNewton] = NloptTNewton
    nlopt_var: Type[NloptVAR] = NloptVAR
    pounders: Type[Pounders] = Pounders
    pygad: Type[Pygad] = Pygad
    pygmo_bee_colony: Type[PygmoBeeColony] = PygmoBeeColony
    pygmo_cmaes: Type[PygmoCmaes] = PygmoCmaes
    pygmo_compass_search: Type[PygmoCompassSearch] = PygmoCompassSearch
    pygmo_de: Type[PygmoDe] = PygmoDe
    pygmo_de1220: Type[PygmoDe1220] = PygmoDe1220
    pygmo_gaco: Type[PygmoGaco] = PygmoGaco
    pygmo_gwo: Type[PygmoGwo] = PygmoGwo
    pygmo_ihs: Type[PygmoIhs] = PygmoIhs
    pygmo_mbh: Type[PygmoMbh] = PygmoMbh
    pygmo_pso: Type[PygmoPso] = PygmoPso
    pygmo_pso_gen: Type[PygmoPsoGen] = PygmoPsoGen
    pygmo_sade: Type[PygmoSade] = PygmoSade
    pygmo_sea: Type[PygmoSea] = PygmoSea
    pygmo_sga: Type[PygmoSga] = PygmoSga
    pygmo_simulated_annealing: Type[PygmoSimulatedAnnealing] = PygmoSimulatedAnnealing
    pygmo_xnes: Type[PygmoXnes] = PygmoXnes
    pyswarms_general: Type[PySwarmsGeneralPSO] = PySwarmsGeneralPSO
    pyswarms_global_best: Type[PySwarmsGlobalBestPSO] = PySwarmsGlobalBestPSO
    pyswarms_local_best: Type[PySwarmsLocalBestPSO] = PySwarmsLocalBestPSO
    scipy_bfgs: Type[ScipyBFGS] = ScipyBFGS
    scipy_basinhopping: Type[ScipyBasinhopping] = ScipyBasinhopping
    scipy_brute: Type[ScipyBrute] = ScipyBrute
    scipy_cobyla: Type[ScipyCOBYLA] = ScipyCOBYLA
    scipy_conjugate_gradient: Type[ScipyConjugateGradient] = ScipyConjugateGradient
    scipy_differential_evolution: Type[ScipyDifferentialEvolution] = (
        ScipyDifferentialEvolution
    )
    scipy_direct: Type[ScipyDirect] = ScipyDirect
    scipy_dual_annealing: Type[ScipyDualAnnealing] = ScipyDualAnnealing
    scipy_lbfgsb: Type[ScipyLBFGSB] = ScipyLBFGSB
    scipy_ls_dogbox: Type[ScipyLSDogbox] = ScipyLSDogbox
    scipy_ls_lm: Type[ScipyLSLM] = ScipyLSLM
    scipy_ls_trf: Type[ScipyLSTRF] = ScipyLSTRF
    scipy_neldermead: Type[ScipyNelderMead] = ScipyNelderMead
    scipy_newton_cg: Type[ScipyNewtonCG] = ScipyNewtonCG
    scipy_powell: Type[ScipyPowell] = ScipyPowell
    scipy_shgo: Type[ScipySHGO] = ScipySHGO
    scipy_slsqp: Type[ScipySLSQP] = ScipySLSQP
    scipy_truncated_newton: Type[ScipyTruncatedNewton] = ScipyTruncatedNewton
    scipy_trust_constr: Type[ScipyTrustConstr] = ScipyTrustConstr
    tao_pounders: Type[TAOPounders] = TAOPounders
    tranquilo: Type[Tranquilo] = Tranquilo
    tranquilo_ls: Type[TranquiloLS] = TranquiloLS

    @property
    def Bounded(self) -> BoundedAlgorithms:
        pass

    @property
    def Global(self) -> GlobalAlgorithms:
        pass

    @property
    def GradientBased(self) -> GradientBasedAlgorithms:
        pass

    @property
    def GradientFree(self) -> GradientFreeAlgorithms:
        pass

    @property
    def LeastSquares(self) -> LeastSquaresAlgorithms:
        pass

    @property
    def Likelihood(self) -> LikelihoodAlgorithms:
        pass

    @property
    def Local(self) -> LocalAlgorithms:
        pass

    @property
    def NonlinearConstrained(self) -> NonlinearConstrainedAlgorithms:
        pass

    @property
    def Parallel(self) -> ParallelAlgorithms:
        return ParallelAlgorithms()

    @property
    def Scalar(self) -> ScalarAlgorithms:
        return ScalarAlgorithms()


algos = Algorithms()
global_algos = GlobalAlgorithms()

ALL_ALGORITHMS = algos._all_algorithms_dict
AVAILABLE_ALGORITHMS = algos._available_algorithms_dict
GLOBAL_ALGORITHMS = global_algos._available_algorithms_dict
