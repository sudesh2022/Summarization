from sqlalchemy import create_engine, inspect

from sqlalchemy import create_engine
engine = create_engine("postgresql+asyncpg://sudesh:sudesh@localhost/summariesdb")
print(engine.url.username)


