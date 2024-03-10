"""Genotype for a modular robot body and brain."""
import sys
from dataclasses import dataclass
from random import Random
from typing import List

import multineat
import numpy as np
import sqlalchemy

from robots import *

#from experiments.optimize_modular_v2.body_spider import make_body_spider
from revolve2.core.database import IncompatibleError, Serializer
from revolve2.core.modular_robot import ModularRobot
from revolve2.genotypes.cppnwin import Genotype as CppnwinGenotype
from revolve2.genotypes.cppnwin import GenotypeSerializer as CppnwinGenotypeSerializer
from revolve2.genotypes.cppnwin import crossover_v1, mutate_v1
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v2 import (
    Develop as body_develop
)
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v2 import (
    random_v1 as body_random,
)
from mask_gene.mask_genotype import MaskGenome

# from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_cpg_v1 import (
#     develop_v1 as brain_develop,
# )
# from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_cpg_v1 import (
#     random_v1 as brain_random,
# )

from revolve2.genotypes.cppnwin.modular_robot.brain_genotype_ann_v2 import (random_v1 as brain_random, develop_v1 as brain_develop)

from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select


def _make_multineat_params() -> multineat.Parameters:

    multineat_params = multineat.Parameters()

    multineat_params.OverallMutationRate = 1
    multineat_params.MutateAddLinkProb = 0.5
    multineat_params.MutateRemLinkProb = 0.5
    multineat_params.MutateAddNeuronProb = 0.2
    multineat_params.MutateRemSimpleNeuronProb = 0.2
    multineat_params.RecurrentProb = 0.0
    multineat_params.MutateWeightsProb = 0.8
    multineat_params.WeightMutationMaxPower = 0.5
    multineat_params.WeightReplacementMaxPower = 1.0
    multineat_params.MutateActivationAProb = 0
    multineat_params.ActivationAMutationMaxPower = 0.5
    multineat_params.MinActivationA = 0.05
    multineat_params.MaxActivationA = 6.0
    multineat_params.MaxWeight = 8.0
    multineat_params.MutateNeuronActivationTypeProb = 0
    multineat_params.MutateOutputActivationFunction = False
    multineat_params.MutateNeuronTraitsProb = 0.0
    multineat_params.MutateLinkTraitsProb = 0.0

    multineat_params.AllowLoops = False

    return multineat_params


_MULTINEAT_PARAMS = _make_multineat_params()


@dataclass
class Genotype:
    """Genotype for a modular robot."""

    body: CppnwinGenotype
    brain: CppnwinGenotype
    mask: MaskGenome


class GenotypeSerializer(Serializer[Genotype]):
    """Serializer for storing modular robot genotypes."""

    @classmethod
    async def create_tables(cls, session: AsyncSession) -> None:
        """
        Create all tables required for serialization.

        This function commits. TODO fix this
        :param session: Database session used for creating the tables.
        """
        await (await session.connection()).run_sync(DbBase.metadata.create_all)
        await CppnwinGenotypeSerializer.create_tables(session)

    @classmethod
    def identifying_table(cls) -> str:
        """
        Get the name of the primary table used for storage.

        :returns: The name of the primary table.
        """
        return DbGenotype.__tablename__

    @classmethod
    async def to_database(
        cls, session: AsyncSession, objects: List[Genotype]
    ) -> List[int]:
        """
        Serialize the provided objects to a database_karine_params using the provided session.

        :param session: Session used when serializing to the database_karine_params. This session will not be committed by this function.
        :param objects: The objects to serialize.
        :returns: A list of ids to identify each serialized object.
        """
        body_ids = await CppnwinGenotypeSerializer.to_database(
            session, [o.body for o in objects]
        )
        brain_ids = await CppnwinGenotypeSerializer.to_database(
            session, [o.brain for o in objects]
        )

        dbgenotypes = [
            DbGenotype(body_id=body_id, brain_id=brain_id)
            for body_id, brain_id in zip(body_ids, brain_ids)
        ]

        session.add_all(dbgenotypes)
        await session.flush()
        ids = [
            dbfitness.id for dbfitness in dbgenotypes if dbfitness.id is not None
        ]  # cannot be none because not nullable. check if only there to silence mypy.
        assert len(ids) == len(objects)  # but check just to be sure
        return ids

    @classmethod
    async def from_database(
        cls, session: AsyncSession, ids: List[int]
    ) -> List[Genotype]:
        """
        Deserialize a list of objects from a database_karine_params using the provided session.

        :param session: Session used for deserialization from the database_karine_params. No changes are made to the database_karine_params.
        :param ids: Ids identifying the objects to deserialize.
        :returns: The deserialized objects.
        :raises IncompatibleError: In case the database_karine_params is not compatible with this serializer.
        """
        rows = (
            (await session.execute(select(DbGenotype).filter(DbGenotype.id.in_(ids))))
            .scalars()
            .all()
        )

        if len(rows) != len(ids):
            raise IncompatibleError()

        id_map = {t.id: t for t in rows}
        body_ids = [id_map[id].body_id for id in ids]
        brain_ids = [id_map[id].brain_id for id in ids]

        body_genotypes = await CppnwinGenotypeSerializer.from_database(
            session, body_ids
        )
        brain_genotypes = await CppnwinGenotypeSerializer.from_database(
            session, brain_ids
        )

        genotypes = [
            Genotype(body, brain,MaskGenome(10))
            for body, brain in zip(body_genotypes, brain_genotypes)
        ]

        return genotypes


def random(
    innov_db_body: multineat.InnovationDatabase,
    innov_db_brain: multineat.InnovationDatabase,
    rng: Random,
    num_initial_mutations: int,
    body_fixed = None
) -> Genotype:
    """
    Create a random genotype.

    :param innov_db_body: Multineat innovation database_karine_params for the body. See Multineat library.
    :param innov_db_brain: Multineat innovation database_karine_params for the brain. See Multineat library.
    :param rng: Random number generator.
    :param num_initial_mutations: The number of times to mutate to create a random network. See CPPNWIN genotype.
    :returns: The created genotype.
    """
    multineat_rng = _multineat_rng_from_random(rng)

    body = body_random(
        innov_db_body,
        multineat_rng,
        _MULTINEAT_PARAMS,
        multineat.ActivationFunction.TANH,
        50,
        n_env_conditions = 0,
        plastic_body = 0,
    )


    evolvable_mask = True
    bb = 'evolvable'
    if bb == 'evolvable':
        b = body_develop(body)
        bb = b.develop()
        x = len(bb[0].find_active_hinges())
        if x == 0:
            x = 1
    else:
        if body_fixed == 'spider':
            x = len(spider().find_active_hinges())
        elif body_fixed == 'salamander':
            x = len(salamander().find_active_hinges())
        elif body_fixed == 'snake':
            x = len(snake().find_active_hinges())
        elif body_fixed == 'insect':
            x = len(insect().find_active_hinges())
        elif body_fixed == 'babya':
            x = len(babya().find_active_hinges())
        elif body_fixed == 'babyb':
            x = len(babyb().find_active_hinges())
        elif body_fixed == 'blokky':
            x = len(blokky().find_active_hinges())
        elif body_fixed == 'garrix':
            x = len(garrix().find_active_hinges())
        elif body_fixed == 'gecko':
            x = len(gecko().find_active_hinges())
        elif body_fixed == 'stingray':
            x = len(stingray().find_active_hinges())
        elif body_fixed == 'tinlicker':
            x = len(tinlicker().find_active_hinges())
        elif body_fixed == 'turtle':
            x = len(turtle().find_active_hinges())
        elif body_fixed == 'ww':
            x = len(ww().find_active_hinges())
        elif body_fixed == 'zappa':
            x = len(zappa().find_active_hinges())
        elif body_fixed == 'ant':
            x = len(ant().find_active_hinges())
        elif body_fixed == 'park':
            x = len(park().find_active_hinges())
        elif body_fixed == 'linkin':
            x = len(linkin().find_active_hinges())
        elif body_fixed == 'longleg':
            x = len(longleg().find_active_hinges())
        elif body_fixed == 'penguin':
            x = len(penguin().find_active_hinges())
        elif body_fixed == 'pentapod':
            x = len(pentapod().find_active_hinges())
        elif body_fixed == 'queen':
            x = len(queen().find_active_hinges())
        elif body_fixed == 'squarish':
            x = len(squarish().find_active_hinges())
        elif body_fixed == 'Head':
            x = len(Head().find_active_hinges())
        else:
            # show error message and stop the program
            print('ERROR: no body fixed')
            sys.exit()

    brain = brain_random(
        innov_db_brain,
        multineat_rng,
        _MULTINEAT_PARAMS,
        multineat.ActivationFunction.SIGNED_SINE,
        num_initial_mutations,
        body,
        x
    )

    # mask = MaskGenome(x)
    # if not evolvable_mask:
    #     mask.genome = np.ones(x)

    return Genotype(body, brain, b.mask)


def mutate(
    genotype: Genotype,
    innov_db_body: multineat.InnovationDatabase,
    innov_db_brain: multineat.InnovationDatabase,
    rng: Random,
) -> Genotype:
    """
    Mutate a genotype.

    The genotype will not be changed; a mutated copy will be returned.

    :param genotype: The genotype to mutate. This object is not altered.
    :param innov_db_body: Multineat innovation database_karine_params for the body. See Multineat library.
    :param innov_db_brain: Multineat innovation database_karine_params for the brain. See Multineat library.
    :param rng: Random number generator.
    :returns: A mutated copy of the provided genotype.
    """
    evolvable_mask = True

    multineat_rng = _multineat_rng_from_random(rng)

    # if evolvable_mask:
    #     if (genotype.mask != None):
    #         genotype.mask.mutate(0.02)

    return Genotype(
        mutate_v1(genotype.body, _MULTINEAT_PARAMS, innov_db_body, multineat_rng),
        mutate_v1(genotype.brain, _MULTINEAT_PARAMS, innov_db_brain, multineat_rng),
        genotype.mask
    )


def crossover(
    parent1: Genotype,
    parent2: Genotype,
    rng: Random,
) -> Genotype:
    """
    Perform crossover between two genotypes.

    :param parent1: The first genotype.
    :param parent2: The second genotype.
    :param rng: Random number generator.
    :returns: A newly created genotype.
    """
    multineat_rng = _multineat_rng_from_random(rng)

    evolvable_mask = False

    if evolvable_mask:
        return Genotype(
            crossover_v1(
                parent1.body,
                parent2.body,
                _MULTINEAT_PARAMS,
                multineat_rng,
                False,
                False,
            ),
            crossover_v1(
                parent1.brain,
                parent2.brain,
                _MULTINEAT_PARAMS,
                multineat_rng,
                False,
                False,
            ),
            # parent1.mask.crossover(parent2.mask)
        )
    else:
        return Genotype(
            crossover_v1(
                parent1.body,
                parent2.body,
                _MULTINEAT_PARAMS,
                multineat_rng,
                False,
                False,
            ),
            crossover_v1(
                parent1.brain,
                parent2.brain,
                _MULTINEAT_PARAMS,
                multineat_rng,
                False,
                False,
            ),
            parent1.mask
        )

def develop(genotype: Genotype, robot) -> ModularRobot:
    """
    Develop the genotype into a modular robot.

    :param genotype: The genotype to create the robot from.
    :returns: The created robot.
    """
    bb = 'evolvable'
    if bb != 'evolvable':
        if robot == 'spider':
            body = spider()
        elif robot == 'salamander':
            body = salamander()
        elif robot == 'insect':
            body = insect()
        elif robot == 'babya':
            body = babya()
        elif robot == 'babyb':
            body = babyb()
        elif robot == 'blokky':
            body = blokky()
        elif robot == 'garrix':
            body = garrix()
        elif robot == 'gecko':
            body = gecko()
        elif robot == 'stingray':
            body = stingray()
        elif robot == 'tinlicker':
            body = tinlicker()
        elif robot == 'turtle':
            body = turtle()
        elif robot == 'ww':
            body = ww()
        elif robot == 'zappa':
            body = zappa()
        elif robot == 'ant':
            body = ant()
        elif robot == 'park':
            body = park()
        # Add the missing robots here
        elif robot == 'linkin':
            body = linkin()
        elif robot == 'longleg':
            body = longleg()
        elif robot == 'penguin':
            body = penguin()
        elif robot == 'pentapod':
            body = pentapod()
        elif robot == 'queen':
            body = queen()
        elif robot == 'squarish':
            body = squarish()
        elif robot == 'snake':
            body = snake()
        elif robot == 'Head':
            body = Head()
        else:
            # show error message and stop the program
            print('ERROR: no body fixed')
            sys.exit()

        body = {0: body}
    else:
        b = body_develop(genotype.body)
        body = b.develop()
        x = len(body[0].find_active_hinges())
    brain = brain_develop(genotype.brain, body, b.mask,x)
    return ModularRobot(body, brain)



def _multineat_rng_from_random(rng: Random) -> multineat.RNG:
    multineat_rng = multineat.RNG()
    multineat_rng.Seed(rng.randint(0, 2**31))
    return multineat_rng


DbBase = declarative_base()


class DbGenotype(DbBase):
    """Database model for the genotype."""

    __tablename__ = "genotype"

    id = sqlalchemy.Column(
        sqlalchemy.Integer,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )

    body_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
    brain_id = sqlalchemy.Column(sqlalchemy.Integer, nullable=False)
