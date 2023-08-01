from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import JSON

# DATABASE_URL =

# engine = create_engine(DATABASE_URL)

Base = declarative_base()


class UserInteraction(Base):
    __tablename__ = "queries_rating"

    id = Column(Integer, primary_key=True, index=True)
    dt = Column(DateTime(timezone=True), server_default=func.now())
    tg_id = Column(Integer, nullable=False)
    uid = Column(UUID(as_uuid=True), nullable=False)
    liked_book = Column(String(450), nullable=False)
    books = Column(JSON, nullable=False)


# Base.metadata.create_all(bind=engine)
