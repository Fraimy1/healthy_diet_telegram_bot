import sqlite3
from contextlib import contextmanager
from config.config import DATABASE_FILE_PATH, BACKUP_FOLDER, BACKUP_LIMIT
import os
from datetime import datetime

@contextmanager
def db_connection(db_path=DATABASE_FILE_PATH):
    """
    A context manager that yields a SQLite connection with row_factory enabled.
    Closes the connection automatically after the with block.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

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

def create_database(db_path=DATABASE_FILE_PATH):
    """
    Creates all necessary tables in the SQLite database using ON DELETE CASCADE for
    appropriate foreign keys to handle automatic deletion of dependent records.
    """
    schema_sql = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS "default_amounts" (
        "item_name" TEXT NOT NULL,
        "item_amount" INTEGER NOT NULL,
        PRIMARY KEY("item_name")
    );

    CREATE TABLE IF NOT EXISTS "users" (
        "user_id" INTEGER NOT NULL UNIQUE,
        "user_name" TEXT,
        "registration_date" INTEGER NOT NULL,
        "original_receipts_added" INTEGER,
        "products_added" INTEGER,
        "household_id" TEXT,
        PRIMARY KEY("user_id")
    );

    CREATE TABLE IF NOT EXISTS "user_purchases" (
        "receipt_id" TEXT NOT NULL,
        "user_id" INTEGER NOT NULL,
        "purchase_datetime" DATETIME,
        "total_sum" REAL,
        "in_history" BOOLEAN,
        "retail_place" TEXT,
        "retail_place_address" TEXT,
        "company_name" TEXT,
        "inn" INTEGER,
        PRIMARY KEY("receipt_id", "user_id"),
        FOREIGN KEY ("user_id") REFERENCES "users"("user_id")
            ON UPDATE CASCADE
            ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS "receipt_items" (
        "item_id" INTEGER NOT NULL UNIQUE,
        "receipt_id" TEXT,
        "user_id" INTEGER NOT NULL,
        "quantity" REAL,
        "percentage" REAL,
        "amount" REAL,
        "product_name" TEXT,
        "portion" REAL,
        "prediction" TEXT,
        "user_prediction" TEXT,
        "confidence" REAL,
        "in_history" BOOLEAN,
        PRIMARY KEY("item_id"),
        FOREIGN KEY ("receipt_id") REFERENCES "user_purchases"("receipt_id")
            ON UPDATE CASCADE
            ON DELETE CASCADE,
        FOREIGN KEY ("user_id") REFERENCES "users"("user_id")
            ON UPDATE CASCADE
            ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS "user_settings" (
        "user_id" INTEGER NOT NULL UNIQUE,
        "add_to_history" BOOLEAN NOT NULL,
        "minimal_prediction_confidence" REAL NOT NULL,
        "return_excel_document" BOOLEAN NOT NULL,
        PRIMARY KEY("user_id"),
        FOREIGN KEY ("user_id") REFERENCES "users"("user_id")
            ON UPDATE CASCADE
            ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS "experts" (
        "expert_id" INTEGER NOT NULL UNIQUE,
        "full_name" TEXT,
        "consulting_price" REAL,
        "service_name" TEXT,
        "additional_services" TEXT,
        "phone_consulting" BOOLEAN,
        "physical_address" TEXT,
        "city" TEXT,
        "telegram_id" INTEGER,
        "email" TEXT,
        "phone_number" INTEGER,
        "website_link" TEXT,
        "blog_link" TEXT,
        PRIMARY KEY("expert_id")
    );

    CREATE TABLE IF NOT EXISTS "admins" (
        "admin_id" INTEGER NOT NULL UNIQUE,
        "login" TEXT NOT NULL,
        "password" TEXT NOT NULL,
        PRIMARY KEY("admin_id")
    );

    CREATE TABLE IF NOT EXISTS "user_metrics" (
        "user_id" INTEGER NOT NULL UNIQUE,
        "weight" REAL,
        "height" INTEGER,
        "age" REAL,
        "gender" TEXT,
        "activity_level" INTEGER,
        PRIMARY KEY("user_id"),
        FOREIGN KEY ("user_id") REFERENCES "users"("user_id")
            ON UPDATE CASCADE
            ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS "households" (
        "household_id" TEXT NOT NULL UNIQUE,
        "num_children" INTEGER,
        "num_adults" INTEGER,
        "num_dogs" INTEGER,
        "num_cats" INTEGER,
        PRIMARY KEY("household_id"),
        FOREIGN KEY ("household_id") REFERENCES "users"("household_id")
            ON UPDATE CASCADE
            ON DELETE CASCADE
    );
    """

    with db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executescript(schema_sql)
        conn.commit()
    print("Database tables created (or already exist).")

def migrate_old_to_new_db(old_db_path, new_db_path):
    """
    Migrate from the old database schema to the new schema.
    """

    with db_connection(old_db_path) as old_conn, db_connection(new_db_path) as new_conn:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        # ---- 1. USERS + USER_SETTINGS ----
        old_cursor.execute("""
            SELECT
                user_id,
                user_name,
                registration_date,
                original_receipts_added,
                products_added,
                add_to_history,
                return_excel_document,
                min_prediction_confidence
            FROM main_table
        """)
        main_rows = old_cursor.fetchall()

        for row in main_rows:
            user_id = row["user_id"]
            user_name = row["user_name"]
            registration_date = row["registration_date"]
            original_receipts_added = row["original_receipts_added"]
            products_added = row["products_added"]

            # for user_settings
            add_to_history = row["add_to_history"]  # keep as 0/1
            return_excel_document = row["return_excel_document"]  # keep as 0/1
            minimal_prediction_confidence = float(row["min_prediction_confidence"])

            # Insert into new 'users'
            new_cursor.execute("""
                INSERT INTO users
                    (user_id, user_name, registration_date,
                     original_receipts_added, products_added, household_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                user_name,
                registration_date,
                original_receipts_added,
                products_added,
                None  # household_id is None
            ))

            # Insert into new 'user_settings'
            new_cursor.execute("""
                INSERT INTO user_settings
                    (user_id, add_to_history, minimal_prediction_confidence, return_excel_document)
                VALUES (?, ?, ?, ?)
            """, (
                user_id,
                add_to_history,
                minimal_prediction_confidence,
                return_excel_document
            ))

        # ---- 2. USER_PURCHASES ----
        old_cursor.execute("""
            SELECT
                receipt_id,
                user_id,
                purchase_datetime,
                total_sum,
                in_history
            FROM user_purchases
        """)
        purchases_rows = old_cursor.fetchall()

        for row in purchases_rows:
            receipt_id = row["receipt_id"]
            user_id = row["user_id"]
            purchase_datetime = row["purchase_datetime"]
            total_sum = row["total_sum"]
            in_history = row["in_history"]

            # Insert into new 'user_purchases', with new columns = NULL
            new_cursor.execute("""
                INSERT INTO user_purchases
                    (receipt_id, user_id, purchase_datetime, total_sum,
                     in_history, retail_place, retail_place_address,
                     company_name, inn)
                VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
            """, (
                receipt_id,
                user_id,
                purchase_datetime,
                total_sum,
                in_history
            ))

        # ---- 3. RECEIPT_ITEMS ----
        old_cursor.execute("""
            SELECT
                item_id,
                receipt_id,
                user_id,
                name,           -- we want this to become product_name
                quantity,
                percentage,
                amount,
                portion,
                prediction,
                user_prediction,
                confidence,
                in_history
            FROM receipt_items
        """)
        items_rows = old_cursor.fetchall()

        for row in items_rows:
            item_id = row["item_id"]
            receipt_id = row["receipt_id"]
            user_id = row["user_id"]
            quantity = row["quantity"]
            percentage = row["percentage"]
            amount = row["amount"]
            portion = row["portion"]
            prediction = row["prediction"]
            user_prediction = row["user_prediction"]
            confidence = row["confidence"]
            in_history = row["in_history"]

            # The new "product_name" = old "name"
            product_name = row["name"]

            new_cursor.execute("""
                INSERT INTO receipt_items
                    (item_id, receipt_id, user_id, quantity, percentage,
                     amount, product_name, portion, prediction,
                     user_prediction, confidence, in_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item_id,
                receipt_id,
                user_id,
                quantity,
                percentage,
                amount,
                product_name,
                portion,
                prediction,
                user_prediction,
                confidence,
                in_history
            ))

        # ---- 4. DEFAULT_AMOUNTS ----
        # Only keep unique (item_name, item_amount). If duplicates exist, just ignore or overwrite.
        old_cursor.execute("""
            SELECT DISTINCT item_name, item_amount
            FROM default_amounts
        """)
        default_rows = old_cursor.fetchall()

        for row in default_rows:
            item_name = row["item_name"]
            item_amount = row["item_amount"]

            # Insert or ignore duplicates
            new_cursor.execute("""
                INSERT OR IGNORE INTO default_amounts
                    (item_name, item_amount)
                VALUES (?, ?)
            """, (item_name, item_amount))

        # Finally commit all changes
        new_conn.commit()

    print("Migration complete!")