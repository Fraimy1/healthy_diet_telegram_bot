from migration.migration_utils import *
from config.config import DATABASE_PATH

# for user_folder in os.listdir(DATABASE_PATH):
#     if user_folder.startswith('user_'):
#         user_id = int(user_folder.split('_')[1])
#         for file in os.listdir(os.path.join(DATABASE_PATH, user_folder)):
#             if file.startswith('file'):
#                 add_receipts_to_json(user_id, file)
                
create_database()

for user_folder in [user for user in os.listdir(DATABASE_PATH) if user.startswith('user_')]:
    user_id = int(user_folder.split('_')[1])
    user_profile = get_user_profile(user_id)
    user_profile_to_db(user_profile)
    update_user_profile_db(user_id)