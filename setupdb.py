from databases import Database
import sqlalchemy
from sqlalchemy import create_engine

# Use this for SQLAlchemy engine (sync)
DATABASE_URL = "postgresql+asyncpg://sudeshkrishnamoorthy:sudesh@localhost:5432/summariesdb"
# Example:


# Sync engine for table creation
sync_engine = create_engine(DATABASE_URL.replace("+asyncpg", ""))

# Metadata & table
metadata = sqlalchemy.MetaData()

summaries_table = sqlalchemy.Table(
    "summaries",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("filename", sqlalchemy.String),
    sqlalchemy.Column("filehash", sqlalchemy.String, unique=True),
    sqlalchemy.Column("summary", sqlalchemy.Text),
)

# Create table
metadata.create_all(sync_engine)
print("✅ Table created successfully.")
