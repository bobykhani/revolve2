from typing import Optional

import sqlalchemy
from sqlalchemy import Column, Integer, String, sql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.expression import null

Base = declarative_base()


class DbNode(Base):
    __tablename__ = "nodes"
    id = Column(
        Integer, nullable=False, unique=True, autoincrement=True, primary_key=True
    )
    type = Column(Integer, nullable=False)  # 0: uninitialized, 1: db_data, 2: list
    db_data = Column(String, nullable=True)

    def __init__(
        self, type: int, db_data: Optional[str], id: Optional[int] = None
    ) -> None:
        if id is not None:
            self.id = id
        self.type = type
        self.db_data = db_data


class DbListItem(Base):
    __tablename__ = "list_items"
    id = Column(
        Integer, nullable=False, unique=True, autoincrement=True, primary_key=True
    )
    index = Column(Integer, nullable=False)
    list_node_id = Column(Integer, nullable=False)
    child_node_id = Column(Integer, nullable=False)

    def __init__(self, index: int, list_node_id: int, child_node_id: int) -> None:
        self.index = index
        self.list_node_id = list_node_id
        self.child_node_id = child_node_id
