"""Visualize and simulate the best robot from the optimization process."""

from genotype import GenotypeSerializer, develop
from revolve2.core.database import open_async_database_sqlite
from revolve2.core.database.serializers import DbFloat
from revolve2.core.optimization.ea.generic_ea import DbEAOptimizerIndividual
from revolve2.runners.mujoco import ModularRobotRerunner
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.future import select
from revolve2.standard_resources import terrains

async def main() -> None:
    """Run the script."""
    db = open_async_database_sqlite("./database")
    async with AsyncSession(db) as session:
        best_individual = (
            await session.execute(
                select(DbEAOptimizerIndividual, DbFloat)#.offset(100)
                .filter(DbEAOptimizerIndividual.float_id == '9')
                #.order_by(DbFloat.speed_y.desc())
            )
        ).first()

        assert best_individual is not None

        print(f"fitness: {best_individual[1].speed_y}")

        genotype = (
            await GenotypeSerializer.from_database(
                session, [best_individual[0].genotype_id]
            )
        )[0]

    rerunner = ModularRobotRerunner()
    await rerunner.rerun(develop(genotype), 60, terrain= terrains.crater([20,20],0.3,1))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
