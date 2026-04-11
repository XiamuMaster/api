# database 包入口，统一导出常用对象
from .models import db, User, DetectionRecord
from .models import ROLE_USER, ROLE_ADMIN, ROLE_SUPER_ADMIN, ROLE_LEVELS
from .crud import (
    hash_password,
    get_user_by_username,
    get_user_by_id,
    create_user,
    update_user_last_login,
    update_user_role,
    update_user_status,
    delete_user_by_id,
    list_users_paginated,
    save_detection_record,
    get_history_list,
    get_record_by_id,
    delete_record_by_id,
)

