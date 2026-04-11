
import hashlib
from datetime import datetime

from .models import db, User, DetectionRecord, ROLE_USER, ROLE_ADMIN


def hash_password(password: str) -> str:
    """SHA-256 + 固定盐值哈希密码"""
    salt = 'yolo_system_salt_2026'
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def get_user_by_username(username: str):
    """根据用户名查询用户"""
    return User.query.filter_by(username=username).first()


def get_user_by_id(user_id: int):
    """根据 ID 查询用户"""
    return User.query.get(user_id)


def create_user(username: str, password: str, role: str = ROLE_USER) -> User:
    """创建新用户，密码自动哈希"""
    user = User(
        username=username,
        password=hash_password(password),
        role=role,
        is_active=True
    )
    db.session.add(user)
    db.session.commit()
    return user


def update_user_last_login(user: User):
    """更新用户最后登录时间"""
    user.last_login = datetime.now()
    db.session.commit()


def update_user_role(user: User, new_role: str):
    """修改用户角色"""
    user.role = new_role
    db.session.commit()


def update_user_status(user: User, is_active: bool):
    """启用或禁用用户账号"""
    user.is_active = is_active
    db.session.commit()


def delete_user_by_id(user_id: int) -> bool:
    """删除用户，成功返回 True，不存在返回 False"""
    user = User.query.get(user_id)
    if not user:
        return False
    db.session.delete(user)
    db.session.commit()
    return True


def list_users_paginated(page: int = 1, per_page: int = 20, role_filter: str = None):
    """分页查询用户列表，可按角色过滤"""
    query = User.query
    if role_filter:
        query = query.filter_by(role=role_filter)
    return query.order_by(User.create_time.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )



def save_detection_record(load_filename: str, result_filename: str,
                          model_type: str, detect_file_type: str,
                          user_id: int = None) -> DetectionRecord:
    """保存一条检测记录"""
    record = DetectionRecord(
        user_id=user_id,
        load_filename=load_filename,
        result_filename=result_filename,
        model_type=model_type,
        detect_file_type=detect_file_type
    )
    db.session.add(record)
    db.session.commit()
    return record


def get_history_list(limit: int = 20, user_id: int = None,
                     role: str = None):
    """获取检测历史记录，根据用户角色和user_id过滤"""
    query = DetectionRecord.query
    # 普通用户和管理员只能看自己的数据
    if role in (ROLE_USER, ROLE_ADMIN):
        query = query.filter_by(user_id=user_id)
    # 超级管理员可以看到所有数据
    return query.order_by(
        DetectionRecord.create_time.desc()
    ).limit(limit).all()


def get_record_by_id(record_id: int):
    """根据 ID 查询检测记录"""
    return DetectionRecord.query.filter_by(id=record_id).first()


def delete_record_by_id(record_id: int) -> bool:
    """删除检测记录，成功返回 True，不存在返回 False"""
    record = DetectionRecord.query.filter_by(id=record_id).first()
    if not record:
        return False
    db.session.delete(record)
    db.session.commit()
    return True
