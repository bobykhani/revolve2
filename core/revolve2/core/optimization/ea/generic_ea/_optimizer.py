from __future__ import annotations

import logging
import os
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, Type, TypeVar, Dict

from revolve2.core.database import IncompatibleError, Serializer
from revolve2.core.optimization import DbId, Process
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound
from revolve2.core.modular_robot import MeasureRelative
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v2 import (
    Develop as body_develop)

from revolve2.core.modular_robot.render.render import Render
from ._database import (
    DbBase,
    DbEAOptimizer,
    DbEAOptimizerGeneration,
    DbEAOptimizerIndividual,
    DbEAOptimizerParent,
    DbEAOptimizerState,
)
#from ... import ProcessIdGen

Genotype = TypeVar("Genotype")
#Fitness = TypeVar("Fitness")
Measure = TypeVar("Measure")


class EAOptimizer(Process, Generic[Genotype, Measure]):
    """
    A generic optimizer implementation for evolutionary algorithms.

    Inherit from this class and implement its abstract methods.
    See the `Process` parent class on how to make an instance of your implementation.
    You can run the optimization process using the `run` function.

    Results will be saved every generation in the provided database_karine_params.
    """

    @abstractmethod
    async def _evaluate_generation(
        self,
        genotypes: List[Genotype],
        database: AsyncEngine,
        # process_id: int,
        # process_id_gen,
        db_id: DbId,
    ) -> List[Measure]:
        """
        Evaluate a list of genotypes.

        :param genotypes: The genotypes to evaluate. Must not be altered.
        :param database: Database that can be used to store anything you want to save from the evaluation.
        :param db_id: Unique identifier in the completely program specifically made for this function call.
        :returns: The Measure result.
        """

    @abstractmethod
    def _select_parents(
        self,
        population: List[Genotype],
        measures: List[Measure],
        num_parent_groups: int,
    ) -> List[List[int]]:
        """
        Select groups of parents that will create offspring.

        :param population: The generation to select sets of parents from. Must not be altered.
        :param fitnesses: Fitnesses of the population.
        :param num_parent_groups: Number of groups to create.
        :returns: The selected sets of parents, each integer representing a population index.
        """

    @abstractmethod
    def _select_survivors(
        self,
        old_individuals: List[Genotype],
        old_Measures: List[Measure],
        new_individuals: List[Genotype],
        new_Measures: List[Measure],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        """
        Select survivors from the sets of old and new individuals, which will form the next generation.

        :param old_individuals: Original individuals.
        :param old_fitnesses: Fitnesses of the original individuals.
        :param new_individuals: New individuals.
        :param new_fitnesses: Fitnesses of the new individuals.
        :param num_survivors: How many individuals should be selected.
        :returns: Indices of the old survivors and indices of the new survivors.
        """

    @abstractmethod
    def _crossover(self, parents: List[Genotype]) -> Genotype:
        """
        Combine a set of genotypes into a new genotype.

        :param parents: The set of genotypes to combine. Must not be altered.
        :returns: The new genotype.
        """

    @abstractmethod
    def _mutate(self, genotype: Genotype) -> Genotype:
        """
        Apply mutation to an genotype to create a new genotype.

        :param genotype: The original genotype. Must not be altered.
        :returns: The new genotype.
        """

    @abstractmethod
    def _must_do_next_gen(self) -> bool:
        """
        Decide if the optimizer must do another generation.

        :returns: True if it must.
        """

    @abstractmethod
    def _on_generation_checkpoint(self, session: AsyncSession) -> None:
        """
        Save the results of this generation to the database_karine_params.

        This function is called after a generation is finished and results and state are saved to the database_karine_params.
        Use it to store state and results of the optimizer.
        The session must not be committed, but it may be flushed.

        :param session: The session to use for writing to the database_karine_params. Must not be committed, but can be flushed.
        """

    __database: AsyncEngine

    __db_id: DbId
    __ea_optimizer_id: int

    __genotype_type: Type[Genotype]
    __genotype_serializer: Type[Serializer[Genotype]]
    __measures_type: Type[Measure]
    __measures_serializer: Type[Serializer[Measure]]
    __states_serializer: List[Tuple[float, State]]

    __offspring_size: int
    # __process_id_gen: ProcessIdGen

    __next_individual_id: int

    __latest_population: List[_Individual[Genotype]]
    __latest_measures: Optional[List[Measure]]  # None only for the initial population
    __generation_index: int
    __fitness_measure: str
    __latest_states: List[Tuple[float, State]]

    async def ainit_new(
        self,
        database: AsyncEngine,
        session: AsyncSession,

        db_id: DbId,
        genotype_type: Type[Genotype],
        genotype_serializer: Type[Serializer[Genotype]],
        measures_type: Type[Measure],
        states_serializer: list[Tuple[float, State]],
        measures_serializer: Type[Serializer[Measure]],
        offspring_size: int,
        initial_population: List[Genotype],
        fitness_measure: str,

        experiment_name: str,
        max_modules: int,
        crossover_prob: float,
        mutation_prob: float,
        substrate_radius: str,
        run_simulation: bool,

    ) -> None:
        """
        Initialize this class async.

        Called when creating an instance using `new`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when saving data to the database_karine_params during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param genotype_type: Type of the genotype generic parameter.
        :param genotype_serializer: Serializer for serializing genotypes.
        :param fitness_type: Type of the fitness generic parameter.
        :param fitness_serializer: Serializer for serializing fitnesses.
        :param offspring_size: Number of offspring made by the population each generation.
        :param initial_population: List of genotypes forming generation 0.
        """
        self.__database = database
        self.__db_id = db_id

        self.__genotype_type = genotype_type
        self.__genotype_serializer = genotype_serializer
#        self.__fitness_type = fitness_type
#        self.__fitness_serializer = fitness_serializer
        self.__offspring_size = offspring_size
        self.__next_individual_id = 0
        self.__latest_fitnesses = None
        self.__generation_index = 0
        self.__measures_type = measures_type
        self.__measures_serializer = measures_serializer
        self.__states_serializer = states_serializer
        self.__latest_measures = None
        self.__latest_states = None
        self.__fitness_measure = fitness_measure
        self.__experiment_name = experiment_name
        self.__max_modules = max_modules,
        self.__crossover_prob = crossover_prob,
        self.__mutation_prob = mutation_prob,
        self.__substrate_radius = substrate_radius,


        self.__latest_population = [
            _Individual(self.__gen_next_individual_id(), g, [])
            for g in initial_population
        ]

        await (await session.connection()).run_sync(DbBase.metadata.create_all)
        await self.__genotype_serializer.create_tables(session)
        await self.__measures_serializer.create_tables(session)
        await self.__states_serializer.create_tables(session)

        new_opt = DbEAOptimizer(
            #process_id=process_id,
            db_id=db_id.fullname,
            offspring_size=self.__offspring_size,
            genotype_table=self.__genotype_serializer.identifying_table(),
            measures_table=self.__measures_serializer.identifying_table(),
            states_table=self.__states_serializer.identifying_table(),
            fitness_measure=self.__fitness_measure,
            experiment_name=self.__experiment_name,
            max_modules=self.__max_modules[0],
            crossover_prob=self.__crossover_prob[0],
            mutation_prob=self.__mutation_prob[0],
            substrate_radius=self.__substrate_radius[0],

        )
        session.add(new_opt)
        await session.flush()
        assert new_opt.id is not None  # this is impossible because it's not nullable
        self.__ea_optimizer_id = new_opt.id

        await self.__save_generation_using_session(
            session, None, None, None, self.__latest_population, None, None, None
        )
        print('bobak')

    async def ainit_from_database(
        self,
        database: AsyncEngine,
        session: AsyncSession,
        # process_id: int,
        # process_id_gen: ProcessIdGen,
        db_id: DbId,
        genotype_type: Type[Genotype],
        genotype_serializer: Type[Serializer[Genotype]],
        states_serializer: List[Tuple[float, State]],
        measures_type: Type[Measure],
        measures_serializer: Type[Serializer[Measure]],
        fitness_measure: str,
    ) -> bool:
        """
        Try to initialize this class async from a database_karine_params.

        Called when creating an instance using `from_database`.

        :param database: Database to use for this optimizer.
        :param session: Session to use when loading and saving data to the database_karine_params during initialization.
        :param db_id: Unique identifier in the completely program specifically made for this optimizer.
        :param genotype_type: Type of the genotype generic parameter.
        :param genotype_serializer: Serializer for serializing genotypes.
        :param fitness_type: Type of the fitness generic parameter.
        :param fitness_serializer: Serializer for serializing fitnesses.
        :returns: True if this complete object could be deserialized from the database_karine_params.
        :raises IncompatibleError: In case the database_karine_params is not compatible with this class.
        """
        self.__database = database
        self.__db_id = db_id
        self.__genotype_type = genotype_type
        self.__genotype_serializer = genotype_serializer
        self.__states_serializer = states_serializer
        self.__measures_type = measures_type
        self.__measures_serializer = measures_serializer
        self.__fitness_measure = fitness_measure

        try:
            eo_row = (
                (
                    await session.execute(
                        select(DbEAOptimizer).filter(
                            DbEAOptimizer.db_id == self.__db_id.fullname
                        )
                    )
                )
                .scalars()
                .one()
            )
        except MultipleResultsFound as err:
            raise IncompatibleError() from err
        except (NoResultFound, OperationalError):
            return False

        self.__ea_optimizer_id = eo_row.id
        self.__offspring_size = eo_row.offspring_size

        # TODO: this name 'state' conflicts a bit with the table of states (positions)...
        state_row = (
            (
                await session.execute(
                    select(DbEAOptimizerState)
                    .filter(
                        DbEAOptimizerState.ea_optimizer_id == self.__ea_optimizer_id
                    )
                    .order_by(DbEAOptimizerState.generation_index.desc())
                )
            )
            .scalars()
            .first()
        )

        if state_row is None:
            raise IncompatibleError()  # not possible that there is no saved state but DbEAOptimizer row exists

        self.__generation_index = state_row.generation_index

        gen_rows = (
            (
                await session.execute(
                    select(DbEAOptimizerGeneration)
                    .filter(
                        (
                            DbEAOptimizerGeneration.ea_optimizer_id
                            == self.__ea_optimizer_id
                        )
                        & (
                            DbEAOptimizerGeneration.generation_index
                            == self.__generation_index
                        )
                    )
                    .order_by(DbEAOptimizerGeneration.individual_index)
                )
            )
            .scalars()
            .all()
        )

        individual_ids = [row.individual_id for row in gen_rows]

        individual_rows = (
            (
                await session.execute(
                    select(DbEAOptimizerIndividual).filter(
                        (
                            DbEAOptimizerIndividual.ea_optimizer_id
                            == self.__ea_optimizer_id
                        )
                       & (DbEAOptimizerIndividual.individual_id.in_(individual_ids))
                    )
                )
            )
            .scalars()
            .all()
        )
        individual_map = {i.individual_id: i for i in individual_rows}

        all_individual_rows = (
            (
                await session.execute(
                    select(DbEAOptimizerIndividual).filter(
                        (
                                DbEAOptimizerIndividual.ea_optimizer_id
                                == self.__ea_optimizer_id
                        )
                        & (DbEAOptimizerIndividual.individual_id.in_(individual_ids))
                    )
                )
            )
            .scalars()
            .all()
        )
        all_individual_ids = {i.individual_id for i in all_individual_rows}

        # the highest individual id ever is the highest id overall.
        self.__next_individual_id = max(all_individual_ids) + 1

        if not len(all_individual_ids) == len(individual_rows):
            raise IncompatibleError()

        genotype_ids = [individual_map[id].float_id for id in individual_ids]
        genotypes = await self.__genotype_serializer.from_database(
            session, genotype_ids
        )

        assert len(genotypes) == len(genotype_ids)
        self.__latest_population = [
            _Individual(g_id, g, None) for g_id, g in zip(individual_ids, genotypes)
        ]

        if self.__generation_index == 0:
            self.__latest_measures = {}
            self.__latest_states = {}
        else:
            measures_ids = [individual_map[id].float_id for id in individual_ids]
            measures = await self.__measures_serializer.from_database(
                session, measures_ids
            )
            assert len(measures) == len(measures_ids)
            self.__latest_measures = measures

            states_ids = [individual_map[id].states_id for id in individual_ids]
            states = await self.__states_serializer.from_database(
                session, states_ids
            )
            assert len(states) == len(states_ids)
            self.__latest_states = states

        return True

    def collect_key_value(self, dictionaries, key):
        list = []
        for d in dictionaries:
            list.append(d[key])
        return list


    async def run(self) -> None:
        """Run the optimizer."""
        # evaluate initial population if required
        if self.__latest_measures is None:
            self.__latest_measures, self.__latest_states = await self.__safe_evaluate_generation(
                [i.genotype for i in self.__latest_population],
                self.__database,
                self.__db_id.branch(f"evaluate{self.__generation_index}"),
            )
            initial_population = self.__latest_population
            initial_measures = {}
            initial_states = {}
            initial_relative_measures = {}

            initial_measures = self.__latest_measures
            initial_states = self.__latest_states
            self._pool_and_time_relative_measures(self.__latest_population, self.__latest_measures)

            #self._pool_seasonal_relative_measures(self.__latest_population, self.__latest_measures)
            self._pop_relative_measures()


            relative_measures = []
            for i in range(len(self.__latest_population)):
                relative_measures.append(MeasureRelative(genotype_measures=self.__latest_measures[i])._return_only_relative())

            initial_relative_measures = relative_measures

        else:
            initial_population = None
            initial_measures = None
            initial_states = None
            initial_relative_measures = None


        while self.__safe_must_do_next_gen():

            # let user select parents
            self.__generation_index += 1

            ##any_cond = list(self.__env_conditions.keys())[0]
            # relative measures for pool parents
            ##for cond in self.__env_conditions:
#            self._pool_and_time_relative_measures(self.__latest_population, self.__latest_measures)#[cond])
            #self._pool_seasonal_relative_measures(self.__latest_population, self.__latest_measures)

            # let user select parents
            latest_fitnesses = self.collect_key_value(self.__latest_measures,#[any_cond],
                                                      self.__fitness_measure)
            parent_selections = self.__safe_select_parents(
                [i.genotype for i in self.__latest_population],
                latest_fitnesses,
                self.__offspring_size,
            )

            # let user create offspring
            offspring = [
                self.__safe_mutate(
                    self.__safe_crossover(
                        [self.__latest_population[i].genotype for i in s]
                    )
                )
                for s in parent_selections
            ]

            # let user evaluate offspring
            new_measures, new_states = await self.__safe_evaluate_generation(
                offspring,
                self.__database,
                self.__db_id.branch(f"evaluate{self.__generation_index}"),
            )

            # combine to create list of individuals
            new_individuals = [
                _Individual(
                    -1,  # placeholder until later
                    genotype,
                    [self.__latest_population[i].id for i in parent_indices],
                )
                for parent_indices, genotype in zip(parent_selections, offspring)
            ]

            # let user select survivors between old and new individuals
            new_fitnesses = self.collect_key_value(new_measures, self.__fitness_measure)
            old_survivors, new_survivors = self.__safe_select_survivors(
                [i.genotype for i in self.__latest_population],
                latest_fitnesses,
                [i.genotype for i in new_individuals],
                new_fitnesses,
                len(self.__latest_population),
            )

            survived_new_individuals = [new_individuals[i] for i in new_survivors]
            survived_new_measures = [new_measures[i] for i in new_survivors]
            survived_new_states = [new_states[i] for i in new_survivors]

            # set ids for new individuals
            for individual in new_individuals:
                individual.id = self.__gen_next_individual_id()

            # combine old and new and store as the new generation
            self.__latest_population = [
                self.__latest_population[i] for i in old_survivors
            ] + survived_new_individuals

            self.__latest_measures = [
                                         self.__latest_measures[i] for i in old_survivors
                                     ] + survived_new_measures
            self.__latest_states = [
                                       self.__latest_states[i] for i in old_survivors
                                   ] + survived_new_states

            #self.__generation_index += 1

            # calculates relative measures: it has to be sequential because they may depend on each other in pop level
            # for i in range(len(self.__latest_population)):
            #     self.__latest_measures[i] = MeasureRelative(genotype_measures=self.__latest_measures[i],
            #                                                 neighbours_measures=self.__latest_measures)._diversity()

            # for i in range(len(self.__latest_population)):
            #     self.__latest_measures[i] = MeasureRelative(genotype_measures=self.__latest_measures[i],
            #                                                 neighbours_measures=self.__latest_measures)._dominated_individuals()

            # latest_relative_measures = []
            # for i in range(len(self.__latest_population)):
            #     latest_relative_measures.append(MeasureRelative(
            #         genotype_measures=self.__latest_measures[i])._return_only_relative())

            # save generation and possibly measures of initial population            # and let user save their state


            async with AsyncSession(self.__database) as session:
                async with session.begin():
                    await self.__save_generation_using_session(
                        session,
                        initial_population,
                        initial_measures,
                        initial_states,
                        new_individuals,
                        new_measures,
                        new_states,
                        None,# latest_relative_measures,
                    )
                    self._on_generation_checkpoint(session)
            # in any case they should be none after saving once
            initial_population = None
            initial_measures = None
            initial_states = None

            logging.info(f"Finished generation {self.__generation_index}.")

        assert (
            self.__generation_index > 0
        ), "Must create at least one generation beyond initial population. This behaviour is not supported."  # would break database_karine_params structure

    def _pop_relative_measures(self):
        # interdependent measures must be calculated sequentially (for after for)
        for i in range(len(self.__latest_population)):
            self.__latest_measures[i] = MeasureRelative(genotype_measures=self.__latest_measures[i],
                                                            neighbours_measures=self.__latest_measures)._diversity('pop')

    def _pool_and_time_relative_measures(self, pool_individuals, pool_measures):

        # populational-interdependent measures must be calculated sequentially (for after for)
        # for i in range(len(pool_individuals)):
        #     pool_measures[i] = MeasureRelative(genotype_measures=pool_measures[i],
        #                                        neighbours_measures=pool_measures).\
        #                                                 _age(self.__generation_index)
        #
        #     pool_measures[i] = MeasureRelative(genotype_measures=pool_measures[i],
        #                                        neighbours_measures=pool_measures)._diversity('pool')

        # for i in range(len(pool_individuals)):
        #     pool_measures[i] = MeasureRelative(genotype_measures=pool_measures[i],
        #                                        neighbours_measures=pool_measures)._pool_dominated_individuals()
        pass


    @property
    def generation_index(self) -> Optional[int]:
        """
        Get the current generation.

        The initial generation is numbered 0.

        :returns: The current generation.
        """
        return self.__generation_index

    def __gen_next_individual_id(self) -> int:
        next_id = self.__next_individual_id
        self.__next_individual_id += 1
        return next_id

    async def __safe_evaluate_generation(
        self,
        genotypes: List[Genotype],
        database: AsyncEngine,
        db_id: DbId,
    ) -> List[Measure]:
        measures, states = await self._evaluate_generation(
            genotypes=genotypes,
            database=database,
            db_id=db_id,
        )
        assert type(measures) == list
        assert len(measures) == len(genotypes)
        assert len(states) == len(genotypes)
        # TODO : adapt to new types
        # assert all(type(e) == self.__measures_type for e in measures)
        return measures, states

    def __safe_select_parents(
        self,
        population: List[Genotype],
        fitnesses: List[Measure],
        num_parent_groups: int,
    ) -> List[List[int]]:
        parent_selections = self._select_parents(
            population, fitnesses, num_parent_groups
        )
        assert type(parent_selections) == list
        assert len(parent_selections) == num_parent_groups
        assert all(type(s) == list for s in parent_selections)
        assert all(
            [
                all(type(p) == int and p >= 0 and p < len(population) for p in s)
                for s in parent_selections
            ]
        )
        return parent_selections

    def __safe_crossover(self, parents: List[Genotype]) -> Genotype:
        genotype = self._crossover(parents)
        assert type(genotype) == self.__genotype_type
        return genotype

    def __safe_mutate(self, genotype: Genotype) -> Genotype:
        genotype = self._mutate(genotype)
        assert type(genotype) == self.__genotype_type
        return genotype

    def __safe_select_survivors(
        self,
        old_individuals: List[Genotype],
        old_measures: List[float],
        new_individuals: List[Genotype],
        new_measures: List[float],
        num_survivors: int,
    ) -> Tuple[List[int], List[int]]:
        old_survivors, new_survivors = self._select_survivors(
            old_individuals,
            old_measures,
            new_individuals,
            new_measures,
            num_survivors,
        )
        assert type(old_survivors) == list
        assert type(new_survivors) == list
        assert len(old_survivors) + len(new_survivors) == len(self.__latest_population)
        assert all(type(s) == int for s in old_survivors)
        assert all(type(s) == int for s in new_survivors)
        return (old_survivors, new_survivors)

    def __safe_must_do_next_gen(self) -> bool:
        must_do = self._must_do_next_gen()
        assert type(must_do) == bool
        return must_do

    async def __save_generation_using_session(
        self,
        session: AsyncSession,
        initial_population: Optional[List[_Individual[Genotype]]],
        initial_measures: Optional[List[Measure]],
        initial_states: List[Tuple[float, State]],
        new_individuals: List[_Individual[Genotype]],
        new_measures: Optional[List[Measure]],
        new_states: List[Tuple[float, State]],
        latest_relative_measures: Dict
    ) -> None:
        # TODO this function can probably be simplified as well as optimized.
        # but it works so I'll leave it for now.

        # update measures/states of initial population if provided
        if initial_measures is not None:
            assert initial_population is not None

            measures_ids = await self.__measures_serializer.to_database(
                session, initial_measures
            )
            assert len(measures_ids) == len(initial_measures)

            states_ids = await self.__states_serializer.to_database(
                session, initial_states
            )
            assert len(states_ids) == len(initial_states)

            rows = (
                (
                    await session.execute(
                        select(DbEAOptimizerIndividual)
                        .filter(
                            (
                                    DbEAOptimizerIndividual.ea_optimizer_id
                                    == self.__ea_optimizer_id
                            )
                            & (
                                DbEAOptimizerIndividual.individual_id.in_(
                                    [i.id for i in initial_population]
                                )
                            )
                        )
                        .order_by(DbEAOptimizerIndividual.individual_id)
                    )
                )
                .scalars()
                .all()
            )
            if len(rows) != len(initial_population):
                raise IncompatibleError()

            for i, row in enumerate(rows):
                row.float_id = measures_ids[i]
                row.states_id = states_ids[i]

            # rows = (
            #     (
            #         await session.execute(
            #             select(DbEAOptimizerGeneration)
            #             .filter(
            #                 (
            #                         DbEAOptimizerGeneration.ea_optimizer_id
            #                         == self.__ea_optimizer_id
            #                 )
            #                 & (
            #                     DbEAOptimizerGeneration.individual_id.in_(
            #                         [i.id for i in initial_population]
            #                     )
            #                 )
            #             )
            #             .order_by(DbEAOptimizerGeneration.individual_id)
            #         )
            #     )
            #     .scalars()
            #     .all()
            # )
            # if len(rows) != len(initial_population):
            #     raise IncompatibleError()
            # print(len(rows), len(latest_relative_measures))
            # for i, row in enumerate(rows):
                # row.diversity = latest_relative_measures[i]['diversity']
        #b = body_develop.develop(ind.genotype.body)
        bodies = [body_develop(ind.genotype.body).develop() for ind in self.__latest_population]

        folder = str(self._EAOptimizer__database.engine.url).replace('db.sqlite','').replace('sqlite+aiosqlite:///./','')

        img_directory = f'{folder}/body_images/generation_{self.generation_index}/'
        # Check whether the specified path exists or not
        isExist = os.path.exists(img_directory)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(img_directory)
            print("The new directory is created!")

        # save body images
        for ind, body in zip(self.__latest_population, bodies):
            render = Render()
            id = ind.id
            img_path = f'{folder}/body_images/generation_{self.generation_index}/individual_{id}.png'
            render.render_robot(body[0].core, img_path)



        # save current optimizer state
        session.add(
            DbEAOptimizerState(
                ea_optimizer_id=self.__ea_optimizer_id,
                generation_index=self.__generation_index,
            )
        )

        # save new individuals
        genotype_ids = await self.__genotype_serializer.to_database(
            session, [g.genotype for g in new_individuals]
        )
        assert len(genotype_ids) == len(new_individuals)
        measures_ids2: List[Optional[int]]
        if new_measures is not None:
            measures_ids2 = [
                m
                for m in await self.__measures_serializer.to_database(
                    session, new_measures
                )
            ]  # this extra comprehension is useless but it stops mypy from complaining
            assert len(measures_ids2) == len(new_measures)
        else:
            measures_ids2 = [None for _ in range(len(new_individuals))]

        states_ids2: List[Tuple[float, State]]
        if new_states is not None:
            states_ids2 = [
                s
                for s in await self.__states_serializer.to_database(
                    session, new_states
                )
            ]  # this extra comprehension is useless but it stops mypy from complaining
            assert len(states_ids2) == len(new_states)
        else:
            states_ids2 = [None for _ in range(len(new_individuals))]

        session.add_all(
            [
                DbEAOptimizerIndividual(
                    ea_optimizer_id=self.__ea_optimizer_id,
                    individual_id=i.id,
                    genotype_id=g_id,
                    float_id=m_id,
                    states_id=s_id,
                )
                for i, g_id, m_id, s_id  in zip(new_individuals, genotype_ids, measures_ids2, states_ids2 )
            ]
        )

        # save parents of new individuals
        parents: List[DbEAOptimizerParent] = []
        for individual in new_individuals:
            assert (
                individual.parent_ids is not None
            )  # Cannot be None. They are only None after recovery and then they are already saved.
            for p_id in individual.parent_ids:
                parents.append(
                    DbEAOptimizerParent(
                        ea_optimizer_id=self.__ea_optimizer_id,
                        child_individual_id=individual.id,
                        parent_individual_id=p_id,
                    )
                )
        session.add_all(parents)
        print('heyyyyyyyyyy')
        # save current generation
        for index, individual in enumerate(self.__latest_population):
            # TODO: this could be better, but it has to adapt to
            #  the bizarre fact tht the initial pop gets saved before evaluated
            if latest_relative_measures is None:
                diversity = None
            else:
                diversity = latest_relative_measures[index]['diversity']
            session.add(
                DbEAOptimizerGeneration(
                    ea_optimizer_id=self.__ea_optimizer_id,
                    generation_index=self.__generation_index,
                    individual_index=index,
                    individual_id=individual.id,
                    diversity=diversity
                )
            )


@dataclass
class _Individual(Generic[Genotype]):
    id: int
    genotype: Genotype
    # Empty list of parents means this is from the initial population
    # None means we did not bother loading the parents during recovery because they are not needed.
    parent_ids: Optional[List[int]]
