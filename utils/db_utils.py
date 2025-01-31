from config.config import DATABASE_FILE_PATH, BACKUP_FOLDER, BACKUP_LIMIT
import os
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Boolean, DateTime, ForeignKey, event, ForeignKeyConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from contextlib import contextmanager

Base = declarative_base()

##############################################################################
# TABLE DEFINITIONS (exact same schema, names, and constraints as your code)
##############################################################################

class DefaultAmounts(Base):
    __tablename__ = "default_amounts"
    item_name = Column(String, primary_key=True, nullable=False)
    item_amount = Column(Integer, nullable=False)

class Users(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, unique=True, nullable=False)
    user_name = Column(String)
    registration_date = Column(Integer, nullable=False)
    original_receipts_added = Column(Integer)
    products_added = Column(Integer)
    household_id = Column(String)

class UserPurchases(Base):
    __tablename__ = "user_purchases"
    # Composite primary key: (receipt_id, user_id)
    receipt_id = Column(String, primary_key=True, nullable=False)
    user_id = Column(
        Integer,
        ForeignKey("users.user_id", onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True, 
        nullable=False
    )
    purchase_datetime = Column(DateTime)
    total_sum = Column(Float)
    in_history = Column(Boolean)
    retail_place = Column(String)
    retail_place_address = Column(String)
    company_name = Column(String)
    inn = Column(Integer)

class ReceiptItems(Base):
    __tablename__ = "receipt_items"

    # Use item_id as the primary key (autoincrement):
    item_id = Column(Integer, primary_key=True, autoincrement=True)
    # The same columns we had, but no direct one-column FKs:
    receipt_id = Column(String, nullable=False)
    user_id = Column(Integer, nullable=False)

    quantity = Column(Float)
    percentage = Column(Float)
    amount = Column(String)
    product_name = Column(String)
    portion = Column(String)
    prediction = Column(String)
    user_prediction = Column(String)
    confidence = Column(Float)
    in_history = Column(Boolean)

    # Composite FK referencing the primary key of user_purchases
    __table_args__ = (
        ForeignKeyConstraint(
            ["receipt_id", "user_id"],  # local columns
            ["user_purchases.receipt_id", "user_purchases.user_id"],  # remote columns
            ondelete="CASCADE",
            onupdate="CASCADE"
        ),
    )

class UserSettings(Base):
    __tablename__ = "user_settings"
    user_id = Column(
        Integer,
        ForeignKey("users.user_id", onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
        unique=True,
        nullable=False
    )
    add_to_history = Column(Boolean, nullable=False)
    minimal_prediction_confidence = Column(Float, nullable=False)
    return_excel_document = Column(Boolean, nullable=False)

class Experts(Base):
    __tablename__ = "experts"
    expert_id = Column(Integer, primary_key=True, unique=True, nullable=False)
    full_name = Column(String)
    consulting_price = Column(Float)
    service_name = Column(String)
    additional_services = Column(String)
    phone_consulting = Column(Boolean)
    physical_address = Column(String)
    city = Column(String)
    telegram_id = Column(Integer)
    email = Column(String)
    phone_number = Column(Integer)
    website_link = Column(String)
    blog_link = Column(String)

class Admins(Base):
    __tablename__ = "admins"
    admin_id = Column(Integer, primary_key=True, unique=True, nullable=False)
    login = Column(String, nullable=False)
    password = Column(String, nullable=False)

class UserMetrics(Base):
    __tablename__ = "user_metrics"
    user_id = Column(
        Integer,
        ForeignKey("users.user_id", onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
        unique=True,
        nullable=False
    )
    weight = Column(Float)
    height = Column(Integer)
    age = Column(Float)
    gender = Column(String)
    activity_level = Column(Integer)

class Households(Base):
    __tablename__ = "households"
    household_id = Column(
        String,
        ForeignKey("users.household_id", onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
        unique=True,
        nullable=False
    )
    num_children = Column(Integer)
    num_adults = Column(Integer)
    num_dogs = Column(Integer)
    num_cats = Column(Integer)

##############################################################################
# ENGINE AND SESSION SETUP
##############################################################################

# Create engine pointing to the same database file you had in config
engine = create_engine(f"sqlite:///{DATABASE_FILE_PATH}", echo=False)

# Important: ensure SQLite foreign keys are enforced
@event.listens_for(engine, "connect")
def enable_foreign_keys(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON;")
    cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

##############################################################################
# CONTEXT MANAGER FOR SESSIONS (REPLACES db_connection)
##############################################################################

@contextmanager
def get_connection():
    """
    Yields a SQLAlchemy Session. Commits on success, rolls back on error.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

##############################################################################
# FUNCTION TO CREATE ALL TABLES
##############################################################################

def create_database():
    """
    Creates all necessary tables in the SQLite database 
    (with ON DELETE CASCADE, ON UPDATE CASCADE in the foreign keys).
    """
    Base.metadata.create_all(engine)
    print("Database tables created (or already exist).")

##############################################################################
# FUNCTIONS TO CREATE BACKUPS
##############################################################################

def make_backup():
    """
    Creates a timestamped backup of the database file in the backups folder.
    Keeps the most recent 100 backups and deletes older ones.
    """
    # Ensure backup folder exists
    os.makedirs(BACKUP_FOLDER, exist_ok=True)

    # Generate a timestamped backup file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_file = os.path.join(BACKUP_FOLDER, f"backup_{timestamp}.db")
    
    # Copy the database to the backup folder
    try:
        with open(DATABASE_FILE_PATH, "rb") as source:
            with open(backup_file, "wb") as dest:
                dest.write(source.read())
        print(f"Backup created: {backup_file}")
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return

    # Cleanup older backups
    clean_old_backups()

def clean_old_backups(limit=BACKUP_LIMIT):
    """
    Keeps only the most recent 'limit' backups and deletes older ones.
    """
    try:
        backups = sorted(
            (os.path.join(BACKUP_FOLDER, f) for f in os.listdir(BACKUP_FOLDER) if f.startswith("backup_")),
            key=os.path.getmtime,
            reverse=True
        )
        for old_backup in backups[limit:]:
            os.remove(old_backup)
            print(f"Deleted old backup: {old_backup}")
    except Exception as e:
        print(f"Failed to clean up old backups: {e}")
