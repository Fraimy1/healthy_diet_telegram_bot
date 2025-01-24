import sqlite3
from utils.db_utils import db_connection

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
                minimal_confidence_for_prediction
            FROM main_database
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
            minimal_prediction_confidence = float(row["minimal_confidence_for_prediction"])

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
            SELECT DISTINCT item_name, item_amount_grams
            FROM default_amounts
        """)
        default_rows = old_cursor.fetchall()

        for row in default_rows:
            item_name = row["item_name"]
            item_amount = row["item_amount_grams"]

            # Insert or ignore duplicates
            new_cursor.execute("""
                INSERT OR IGNORE INTO default_amounts
                    (item_name, item_amount)
                VALUES (?, ?)
            """, (item_name, item_amount))

        # Finally commit all changes
        new_conn.commit()

    print("Migration complete!")

migrate_old_to_new_db('data/database.db', 'data/new_database.db')