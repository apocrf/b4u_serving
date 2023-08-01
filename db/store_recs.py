import os
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import UUID
from db.init_db import UserInteraction
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()


# setup
DATABASE_URL = os.environ.get("POSTGRES_DB_FULL_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# create tables
# Base.metadata.create_all(bind=engine)


def db_interaction(db: Session, tg_id: str, liked_book: str, books: list, uid: str):
    interaction = UserInteraction(
        tg_id=tg_id, liked_book=liked_book, books=books, uid=uid
    )
    db.add(interaction)
    db.commit()
    db.refresh(interaction)
