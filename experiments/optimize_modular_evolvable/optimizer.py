"""Optimizer for finding a good modular robot body and brain using CPPNWIN genotypes and simulation using mujoco."""

import math
import os
import pickle
from random import Random
from typing import List, Tuple

import multineat
import sqlalchemy
import revolve2.core.optimization.ea.generic_ea.population_management as population_management
import revolve2.core.optimization.ea.generic_ea.selection as selection
import sqlalchemy

from core.revolve2.core.modular_robot.render.render import Render
from genotype import Genotype, GenotypeSerializer, crossover, develop, mutate
from pyrr import Quaternion, Vector3
from revolve2.core.database import IncompatibleError
from revolve2.core.database.serializers import FloatSerializer, StatesSerializer
from revolve2.core.optimization import DbId, ProcessIdGen
from revolve2.core.optimization.ea.generic_ea import EAOptimizer
from revolve2.core.physics.environment_actor_controller import (
    EnvironmentActorController,
)
from revolve2.core.physics.running import (
    ActorControl,
    ActorState,
    Batch,
    Environment,
    PosedActor,
    Runner,
)
from revolve2.runners.mujoco import LocalRunner
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select
from revolve2.core.modular_robot._measure import Measure
from revolve2.core.modular_robot._measure_relative import MeasureRelative

from revolve2.standard_resources import terrains


class Optimizer(EAOptimizer[Genotype, float]):
    """
    Optimizer for the problem.

    Uses the generic EA optimizer as a base.
    """

    _TERRAIN = terrains.flat()
    #_TERRAIN = terrains.crater([20,20],0.3,1)

    _db_id: DbId
    # _process_id: int

    _runner: Runner

    _innov_db_body: multineat.InnovationDatabase
    _innov_db_brain: multineat.InnovationDatabase

    _rng: Random

    _simulation_time: int
    _sampling_frequency: float
    _control_frequency: float

    _num_generations: int
    _offspring_size: int
    _fitness_measure: str
    _experiment_name: str
    _max_modules: int
    _crossover_prob: float
    _mutation_prob: float
    _substrate_radius: str
    _run_simulation: bool
    _simulator: str

    _robot: str


    async def ainit_new(  # type: ignore # TODO for now ignoring mypy complaint about LSP problem, override parent's ainit
        self,
        database: AsyncEngine,
        session: AsyncSession,
        db_id: DbId,
        # process_id: int,
        # process_id_gen: ProcessIdGen,
        initial_population: List[Genotype],
        rng: Random,
        innov_db_body: multineat.InnovationDatabase,
        innov_db_brain: multineat.InnovationDatabase,
        simulation_time: int,
        sampling_frequency: float,
        control_frequency: float,
        num_generations: int,
        offspring_size: int,
        fitness_measure: str,

        experiment_name: str,
        max_modules: int,
        crossover_prob: float,
        mutation_prob: float,
        substrate_radius: str,
        run_simulation: bool,
        simulator: str,

        robot: str

    ) -> None:
        """
        Initialize this class async.

        Called when creating an instance using `new`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when saving data to the database_karine_params during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param initial_population: List of genotypes forming generation 0.
        :param rng: Random number generator.
        :param innov_db_body: Innovation database_karine_params for the body genotypes.
        :param innov_db_brain: Innovation database_karine_params for the brain genotypes.
        :param simulation_time: Time in second to simulate the robots for.
        :param sampling_frequency: Sampling frequency for the simulation. See `Batch` class from physics running.
        :param control_frequency: Control frequency for the simulation. See `Batch` class from physics running.
        :param num_generations: Number of generation to run the optimizer for.
        :param offspring_size: Number of offspring made by the population each generation.
        """
        await super().ainit_new(
            database=database,
            session=session,
            db_id=db_id,
            genotype_type=Genotype,
            genotype_serializer=GenotypeSerializer,
            #fitness_type=float,
            #fitness_serializer=FloatSerializer,
            offspring_size=offspring_size,
            initial_population=initial_population,
            # process_id= process_id,
            # process_id_gen = process_id_gen,
            fitness_measure = fitness_measure,
            states_serializer=StatesSerializer,
            measures_type=float,
            measures_serializer=FloatSerializer,
            experiment_name=experiment_name,
            max_modules=max_modules,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            substrate_radius=substrate_radius,
            run_simulation=run_simulation,
        )

        self._db_id = db_id
        # self._process_id = process_id
        self._init_runner()
        self._innov_db_body = innov_db_body
        self._innov_db_brain = innov_db_brain
        self._rng = rng
        self._simulation_time = simulation_time
        self._sampling_frequency = sampling_frequency
        self._control_frequency = control_frequency
        self._num_generations = num_generations
        self.xperiment_name = experiment_name,
        self.max_modules = max_modules,
        self.crossover_prob = crossover_prob,
        self.mutation_prob = mutation_prob,
        self.substrate_radius = substrate_radius,
        self.run_simulation = False,
        self._robot = robot
        # create database_karine_params structure if not exists
        # TODO this works but there is probably a better way
        await (await session.connection()).run_sync(DbBase.metadata.create_all)

        # save to database_karine_params
        self._on_generation_checkpoint(session)

    async def ainit_from_database(  # type: ignore # see comment at ainit_new
            self,
            database: AsyncEngine,
            session: AsyncSession,
            db_id: DbId,
            # process_id: int,
            # process_id_gen: ProcessIdGen,
            rng: Random,
            innov_db_body: multineat.InnovationDatabase,
            innov_db_brain: multineat.InnovationDatabase,
            fitness_measure: str,
    ) -> bool:
        """
        Try to initialize this class async from a database_karine_params.

        Called when creating an instance using `from_database`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when loading and saving data to the database_karine_params during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param rng: Random number generator.
        :param innov_db_body: Innovation database_karine_params for the body genotypes.
        :param innov_db_brain: Innovation database_karine_params for the brain genotypes.
        :returns: True if this complete object could be deserialized from the database_karine_params.
        :raises IncompatibleError: In case the database_karine_params is not compatible with this class.
        """
        if not await super().ainit_from_database(
                database=database,
                session=session,
                db_id=db_id,
                # process_id=process_id,
                # process_id_gen=process_id_gen,
                genotype_type=Genotype,
                genotype_serializer=GenotypeSerializer,
                states_serializer=StatesSerializer,
                measures_type=float,
                measures_serializer=FloatSerializer,
                fitness_measure=fitness_measure,
        ):
            return False

        self._db_id = db_id
        # self._process_id = process_id
        self._init_runner()

        opt_row = (
            (
                await session.execute(
                    select(DbOptimizerState)
                    .filter(DbOptimizerState.db_id == self._db_id.fullname)
                    .order_by(DbOptimizerState.generation_index.desc())
                )
            )
            .scalars()
            .first()
        )

        # if this happens something is wrong with the database_karine_params
        if opt_row is None:
            raise IncompatibleError

        self._simulation_time = opt_row.simulation_time
        self._sampling_frequency = opt_row.sampling_frequency
        self._control_frequency = opt_row.control_frequency
        self._num_generations = opt_row.num_generations

        self._rng = rng
        self._rng.setstate(pickle.loads(opt_row.rng))

        self._innov_db_body = innov_db_body
        self._innov_db_body.Deserialize(opt_row.innov_db_body)
        self._innov_db_brain = innov_db_brain
        self._innov_db_brain.Deserialize(opt_row.innov_db_brain)

        return True

    def _init_runner(self) -> None:
        self._runner = LocalRunner(headless=False, num_simulators=1)

    def _select_parents(
        self,
        population: List[Genotype],
        fitnesses: List[float],
        num_parent_groups: int,
    ) -> List[List[int]]:
        return [
            selection.multiple_unique(
                2,
                population,
                fitnesses,
                lambda _, fitnesses: selection.tournament(self._rng, fitnesses, k=2),
            )
            for _ in range(num_parent_groups)
        ]

    def _select_survivors(
        self,
        old_individuals: List[Genotype],
        old_fitnesses: List[float],
        new_individuals: List[Genotype],
        new_fitnesses: List[float],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        assert len(old_individuals) == num_survivors

        return population_management.steady_state(
            old_individuals,
            old_fitnesses,
            new_individuals,
            new_fitnesses,
            lambda n, genotypes, fitnesses: selection.multiple_unique(
                n,
                genotypes,
                fitnesses,
                lambda genotypes, fitnesses: selection.tournament(
                    self._rng, fitnesses, k=2
                ),
            ),
        )

    def _must_do_next_gen(self) -> bool:
        return self.generation_index != self._num_generations

    def _crossover(self, parents: List[Genotype]) -> Genotype:
        assert len(parents) == 2
        return crossover(parents[0], parents[1], self._rng)

    def _mutate(self, genotype: Genotype) -> Genotype:
        return mutate(genotype, self._innov_db_body, self._innov_db_brain, self._rng)

    async def _evaluate_generation(
            self,
            genotypes: List[Genotype],
            database: AsyncEngine,
            db_id
    ) -> List[float]:
        batch = Batch(
            simulation_time=self._simulation_time,
            sampling_frequency=self._sampling_frequency,
            control_frequency=self._control_frequency,
        )
        phenotypes = []
        for genotype in genotypes:
            phenotypes.append(develop(genotype, self._robot))
            actor, controller = develop(genotype,self._robot).make_actor_and_controller_ann()
            bounding_box = actor.calc_aabb()
            env = Environment(EnvironmentActorController(controller))
            env.static_geometries.extend(self._TERRAIN.static_geometry)
            env.actors.append(
                PosedActor(
                    actor,
                    Vector3(
                        [
                            0.0,
                            0.0,
                            bounding_box.size.z / 2.0 - bounding_box.offset.z,
                        ]
                    ),
                    Quaternion(),
                    [0.0 for _ in controller.get_dof_targets()],
                ),
            )
            batch.environments.append(env)

        batch_results = await self._runner.run_batch(batch)

        measures = []
        #for environment_result in batch_results.environment_results:
        #m = Measure()
        states = batch_results
        states_genotypes = []
        if states is not None:
            for idx_genotype in range(0, len(states.environment_results)):
                states_genotypes.append({})
                for idx_state in range(0, len(states.environment_results[idx_genotype].environment_states)):
                    states_genotypes[-1][idx_state] = \
                        states.environment_results[idx_genotype].environment_states[idx_state].actor_states[
                            0].serialize()
        measures_genotypes = []
        for i, phenotype in enumerate(phenotypes):
            m = Measure(states=states,genotype_idx=i,phenotype=phenotype,generation=0,simulation_time=self._simulation_time)
            measures_genotypes.append(m.measure_all_non_relative())
        #     render = Render()
        #
        #     img_path = f'database_karine_params/body_images/generation_{self.generation_index}/individual_{i}.png'
        #     img_directory = f'database_karine_params/body_images/generation_{self.generation_index}/'
        #     # Check whether the specified path exists or not
        #     isExist = os.path.exists(img_directory)
        #     if not isExist:
        #         # Create a new directory because it does not exist
        #         os.makedirs(img_directory)
        #         print("The new directory is created!")
        #
        #     render.render_robot(phenotype.body.core, img_path)

        return measures_genotypes,states.environment_results
        # return [
        #     self._calculate_fitness(
        #         environment_result.environment_states[0].actor_states[0],
        #         environment_result.environment_states[-1].actor_states[0],
        #     )
        #     for environment_result in batch_results.environment_results
        # ]

    @staticmethod
    def _calculate_fitness(begin_state: ActorState, end_state: ActorState) -> float:
        # TODO simulation can continue slightly passed the defined sim time.

        # distance traveled on the xy plane
        return float(
            math.sqrt(
                (begin_state.position[0] - end_state.position[0]) ** 2
                + ((begin_state.position[1] - end_state.position[1]) ** 2)
            )
        )

    def _gather_states(self):
        return []

    def _on_generation_checkpoint(self, session: AsyncSession) -> None:
        session.add(
            DbOptimizerState(
                db_id=self._db_id.fullname,
                generation_index=self.generation_index,
                rng=pickle.dumps(self._rng.getstate()),
                innov_db_body=self._innov_db_body.Serialize(),
                innov_db_brain=self._innov_db_brain.Serialize(),
                simulation_time=self._simulation_time,
                sampling_frequency=self._sampling_frequency,
                control_frequency=self._control_frequency,
                num_generations=self._num_generations,
            )
        )


DbBase = declarative_base()


class DbOptimizerState(DbBase):
    """Optimizer state."""

    __tablename__ = "optimizer"

    db_id = sqlalchemy.Column(
        sqlalchemy.String,
        nullable=False,
        primary_key=True,
    )
    generation_index = sqlalchemy.Column(
        sqlalchemy.Integer, nullable=False, primary_key=True
    )
    rng = sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)
    innov_db_body = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    innov_db_brain = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    simulation_time = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    sampling_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    control_frequency = sqlalchemy.Column(sqlalchemy.Float, nullable=False)
    num_generations = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
