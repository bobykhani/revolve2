"""SQLAlchemy database_karine_params model for EA."""

from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base

DbBase = declarative_base()


class DbEAOptimizer(DbBase):
    """Model for the optimizer itself, containing static parameters."""

    __tablename__ = "ea_optimizer"

    id = Column(
        Integer,
        nullable=False,
        unique=True,
        autoincrement=True,
        primary_key=True,
    )
    db_id = Column(String, nullable=False, unique=True)
    genotype_table = Column(String, nullable=False)
    measures_table = Column(String, nullable=False)
    states_table = Column(String, nullable=False)
    fitness_measure = Column(String, nullable=True)
    offspring_size = Column(Integer, nullable=True)
    experiment_name = Column(String, nullable=True)
    max_modules = Column(Integer, nullable=True)
    substrate_radius = Column(Integer, nullable=True)
    crossover_prob = Column(Float, nullable=True)
    mutation_prob = Column(Float, nullable=True)

class DbEAOptimizerState(DbBase):
    """State of the optimizer."""

    __tablename__ = "ea_optimizer_state"

    ea_optimizer_id = Column(Integer, nullable=False, primary_key=True)
    generation_index = Column(Integer, nullable=False, primary_key=True)
    individual_index = Column(Integer, nullable=True)
    individual_id = Column(Integer, nullable=True)
    diversity = Column(Float, nullable=True)

class DbEAOptimizerGeneration(DbBase):
    """A single generation."""

    __tablename__ = "ea_optimizer_generation"


    ea_optimizer_id = Column(Integer, nullable=False, primary_key=True)
    generation_index = Column(Integer, nullable=False, primary_key=True)
    individual_index = Column(Integer, nullable=False, primary_key=True)
    individual_id = Column(Integer, nullable=False)
    pop_diversity = Column(Float, nullable=True)
    pool_diversity = Column(Float, nullable=True)
    dominated_quality_youth = Column(Float, nullable=True)
    fullydominated_quality_youth = Column(Float, nullable=True)
    age = Column(Float, nullable=True)
    inverse_age = Column(Float, nullable=True)
    seasonal_dominated = Column(Float, nullable=True)
    seasonal_fullydominated = Column(Float, nullable=True)
    backforth_dominated = Column(Float, nullable=True)
    forthright_dominated = Column(Float, nullable=True)
    diversity = Column(Float, nullable=True)


class DbEAOptimizerIndividual(DbBase):
    """
    An individual with a fitness which may or may not be assigned.

    Can be part of multiple generations.
    """

    __tablename__ = "ea_optimizer_individual"

    ea_optimizer_id = Column(Integer, nullable=False, primary_key=True)
    individual_id = Column(Integer, nullable=False, primary_key=True)
    genotype_id = Column(Integer, nullable=False)
    float_id = Column(Integer, nullable=True)
    states_id = Column(Integer, nullable=True)

class DbEAOptimizerParent(DbBase):
    """Parent-child relationship between two individuals."""

    __tablename__ = "ea_optimizer_parent"

    ea_optimizer_id = Column(Integer, nullable=False, primary_key=True)
    child_individual_id = Column(Integer, nullable=False, primary_key=True)
    parent_individual_id = Column(Integer, nullable=False, primary_key=True)
