from config.config import DATABASE_FILE_PATH, BACKUP_FOLDER, BACKUP_LIMIT
import os
import pandas as pd
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Boolean, DateTime, ForeignKey, event, ForeignKeyConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker
from contextlib import contextmanager
from sqlalchemy.inspection import inspect

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
    price = Column(Float)
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

#small helper function
def to_dict(obj):
    return {c.key: getattr(obj, c.key) for c in inspect(obj).mapper.column_attrs}

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

def export_receipt_items_to_excel(filepath):
    """
    Export all ReceiptItems to an Excel file.
    
    Args:
        filepath (str): Path where the Excel file should be saved
    """
    with get_connection() as session:
        # Query all receipt items
        items = session.query(ReceiptItems).all()
        
        # Convert to list of dictionaries
        items_data = [to_dict(item) for item in items]
        
        # Convert to DataFrame and save to Excel
        if items_data:
            df = pd.DataFrame(items_data)
            df.to_excel(filepath, index=False)
            print(f"Data exported to {filepath}")
        else:
            print("No data to export")

if __name__ == '__main__':
    with get_connection() as session:
        session.query(UserSettings).filter(UserSettings.user_id == 0).update({'minimal_prediction_confidence': 0.05})
        
#     # Create tables if they don't exist
#     create_database()
    
#     from datetime import datetime
#     import random
#     from sqlalchemy import func

#     # -----------------------------------------------------
#     # 1. Insert the fake user and corresponding settings.
#     # -----------------------------------------------------
#     with get_connection() as session:
#         # Check if fake user (user_id=0) exists; if not, create it.
#         if not session.query(Users).filter_by(user_id=0).first():
#             fake_user = Users(
#                 user_id=0,  # 0000 as integer
#                 user_name="Fake User",
#                 registration_date=int(datetime.now().timestamp()),
#                 original_receipts_added=0,
#                 products_added=0,
#                 household_id=None
#             )
#             session.add(fake_user)
#             session.flush()  # Ensure fake_user is inserted so FK constraints pass

#         if not session.query(UserSettings).filter_by(user_id=0).first():
#             fake_settings = UserSettings(
#                 user_id=0,
#                 add_to_history=True,
#                 minimal_prediction_confidence=0.5,
#                 return_excel_document=False
#             )
#             session.add(fake_settings)


#     # -----------------------------------------------------------------
#     # 2. Helper functions to create receipts and copy product data.
#     # -----------------------------------------------------------------
#     def create_receipt(year, month, day_range, r_type):
#         """
#         Generates a purchase datetime and a receipt_id.
#         r_type is either "big" or "small".
#         """
#         day = random.randint(day_range[0], day_range[1])
#         purchase_dt = datetime(
#             year, month, day,
#             random.randint(0, 23),
#             random.randint(0, 59)
#         )
#         # Create a receipt_id that encodes year, month, type, and random values.
#         receipt_id = f"R-{year}-{month:02d}-{r_type}-{random.randint(1,1000)}-{random.randint(1000,9999)}"
#         purchase = UserPurchases(
#             receipt_id=receipt_id,
#             user_id=0,
#             purchase_datetime=purchase_dt,
#             total_sum=round(random.uniform(10.0, 500.0), 2),
#             in_history=True,
#             retail_place=f"Store {random.choice(['A', 'B', 'C', 'D'])}",
#             retail_place_address=f"{random.randint(1, 100)} Fake Street",
#             company_name="Fake Company Inc.",
#             inn=random.randint(100000000, 999999999)
#         )
#         return receipt_id, purchase

#     def get_random_existing_product(session):
#         """
#         Retrieves a random ReceiptItems record from a real user (user_id != 0).
#         """
#         return session.query(ReceiptItems)\
#                       .filter(ReceiptItems.user_id != 0)\
#                       .order_by(func.random())\
#                       .first()

#     def copy_receipt_item(existing_item, new_receipt_id):
#         """
#         Creates a new ReceiptItems instance by copying the fields from an existing item.
#         Updates the receipt_id and user_id for the fake user.
#         """
#         # Process the percentage field: strip "%" if present and convert to float.
#         raw_percentage = existing_item.percentage
#         if isinstance(raw_percentage, str):
#             # Remove the '%' and any surrounding whitespace.
#             raw_percentage = raw_percentage.replace("%", "").strip()
#             try:
#                 raw_percentage = float(raw_percentage)
#             except ValueError:
#                 raw_percentage = 0.0
#         elif raw_percentage is None:
#             raw_percentage = 0.0

#         return ReceiptItems(
#             receipt_id=new_receipt_id,
#             user_id=0,
#             quantity=existing_item.quantity,
#             percentage=raw_percentage,
#             amount=existing_item.amount,
#             price=existing_item.price,
#             product_name=existing_item.product_name,
#             portion=existing_item.portion,
#             prediction=existing_item.prediction,
#             user_prediction=existing_item.user_prediction,
#             confidence=existing_item.confidence,
#             in_history=existing_item.in_history
#         )

#     # -----------------------------------------------------------------
#     # 3. Insert receipts and receipt items for 2024 (all 12 months)
#     # -----------------------------------------------------------------
#     with get_connection() as session:
#         for month in range(1, 13):
#             # BIG RECEIPTS: 4-8 per month; each with 10-45 products.
#             big_receipt_count = random.randint(4, 8)
#             for _ in range(big_receipt_count):
#                 receipt_id, purchase = create_receipt(2024, month, (1, 28), "big")
#                 session.add(purchase)
#                 session.flush()
#                 num_products = random.randint(10, 45)
#                 for _ in range(num_products):
#                     existing_item = get_random_existing_product(session)
#                     if existing_item:
#                         new_item = copy_receipt_item(existing_item, receipt_id)
#                         session.add(new_item)
#             # SMALL RECEIPTS: 3-14 per month; each with 2-10 products.
#             small_receipt_count = random.randint(3, 14)
#             for _ in range(small_receipt_count):
#                 receipt_id, purchase = create_receipt(2024, month, (1, 28), "small")
#                 session.add(purchase)
#                 session.flush()
#                 num_products = random.randint(2, 10)
#                 for _ in range(num_products):
#                     existing_item = get_random_existing_product(session)
#                     if existing_item:
#                         new_item = copy_receipt_item(existing_item, receipt_id)
#                         session.add(new_item)

#     # -----------------------------------------------------------------
#     # 4. Insert receipts for January 2025 (full expected receipts)
#     # -----------------------------------------------------------------
#     with get_connection() as session:
#         # Big receipts for January 2025.
#         big_receipt_count = random.randint(4, 8)
#         for _ in range(big_receipt_count):
#             receipt_id, purchase = create_receipt(2025, 1, (1, 28), "big")
#             session.add(purchase)
#             session.flush()
#             num_products = random.randint(10, 45)
#             for _ in range(num_products):
#                 existing_item = get_random_existing_product(session)
#                 if existing_item:
#                     new_item = copy_receipt_item(existing_item, receipt_id)
#                     session.add(new_item)
#         # Small receipts for January 2025.
#         small_receipt_count = random.randint(3, 14)
#         for _ in range(small_receipt_count):
#             receipt_id, purchase = create_receipt(2025, 1, (1, 28), "small")
#             session.add(purchase)
#             session.flush()
#             num_products = random.randint(2, 10)
#             for _ in range(num_products):
#                 existing_item = get_random_existing_product(session)
#                 if existing_item:
#                     new_item = copy_receipt_item(existing_item, receipt_id)
#                     session.add(new_item)

#     # -----------------------------------------------------------------
#     # 5. Insert receipts for February 2025 (only half, before the 20th)
#     # -----------------------------------------------------------------
#     with get_connection() as session:
#         # For February, force purchase dates before the 20th.
#         expected_big = random.randint(4, 8)
#         actual_big = max(1, expected_big // 2)
#         for _ in range(actual_big):
#             receipt_id, purchase = create_receipt(2025, 2, (1, 19), "big")
#             session.add(purchase)
#             session.flush()
#             num_products = random.randint(10, 45)
#             for _ in range(num_products):
#                 existing_item = get_random_existing_product(session)
#                 if existing_item:
#                     new_item = copy_receipt_item(existing_item, receipt_id)
#                     session.add(new_item)
#         expected_small = random.randint(3, 14)
#         actual_small = max(1, expected_small // 2)
#         for _ in range(actual_small):
#             receipt_id, purchase = create_receipt(2025, 2, (1, 19), "small")
#             session.add(purchase)
#             session.flush()
#             num_products = random.randint(2, 10)
#             for _ in range(num_products):
#                 existing_item = get_random_existing_product(session)
#                 if existing_item:
#                     new_item = copy_receipt_item(existing_item, receipt_id)
#                     session.add(new_item)

#     print("Synthetic data for the fake user's receipts and receipt items inserted successfully!")

# export_receipt_items_to_excel("receipt_items.xlsx")  # Export all ReceiptItems to Excel
