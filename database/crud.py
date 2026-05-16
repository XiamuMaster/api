
import hashlib
import os
from datetime import datetime

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
from sqlalchemy.pool import StaticPool

from .models import User, DetectionRecord, ROLE_USER, ROLE_ADMIN

# 基础类和模型
Base = declarative_base()

# 创建独立的数据库引擎（非 Flask-SQLAlchemy）
db_path = os.path.join(os.path.dirname(__file__), 'db.sqlite3')
engine = create_engine(
    f'sqlite:///{db_path}',
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
SessionFactory = sessionmaker(bind=engine)
Session = scoped_session(SessionFactory)


def get_session():
    """获取当前线程的 session"""
    return Session()


def hash_password(password: str) -> str:
    """SHA-256 + 固定盐值哈希密码"""
    salt = 'yolo_system_salt_2026'
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def get_user_by_username(username: str):
    """根据用户名查询用户"""
    session = get_session()
    try:
        return session.query(User).filter_by(username=username).first()
    finally:
        session.close()


def get_user_by_id(user_id: int):
    """根据 ID 查询用户"""
    session = get_session()
    try:
        return session.query(User).get(user_id)
    finally:
        session.close()


def create_user(username: str, password: str, role: str = ROLE_USER) -> User:
    """创建新用户，密码自动哈希"""
    session = get_session()
    try:
        user = User(
            username=username,
            password=hash_password(password),
            role=role,
            is_active=True
        )
        session.add(user)
        session.commit()
        return user
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def update_user_last_login(user: User):
    """更新用户最后登录时间"""
    session = get_session()
    try:
        u = session.query(User).get(user.id)
        if u:
            u.last_login = datetime.now()
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def update_user_role(user: User, new_role: str):
    """修改用户角色"""
    session = get_session()
    try:
        u = session.query(User).get(user.id)
        if u:
            u.role = new_role
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def update_user_status(user: User, is_active: bool):
    """启用或禁用用户账号"""
    session = get_session()
    try:
        u = session.query(User).get(user.id)
        if u:
            u.is_active = is_active
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def delete_user_by_id(user_id: int) -> bool:
    """删除用户，成功返回 True，不存在返回 False"""
    session = get_session()
    try:
        user = session.query(User).get(user_id)
        if not user:
            return False
        session.delete(user)
        session.commit()
        return True
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def list_users_paginated(page: int = 1, per_page: int = 20, role_filter: str = None):
    """分页查询用户列表，可按角色过滤"""
    session = get_session()
    try:
        query = session.query(User)
        if role_filter:
            query = query.filter_by(role=role_filter)
        return query.order_by(User.create_time.desc()).offset((page - 1) * per_page).limit(per_page).all()
    finally:
        session.close()


# ===== 以下是 Flask-SQLAlchemy 版本（供 app.py 使用）=====

from .models import db


def save_detection_record(load_filename: str, result_filename: str,
                          model_type: str, detect_file_type: str,
                          user_id: int = None) -> DetectionRecord:
    """保存一条检测记录（Flask 上下文）"""
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
    """获取检测历史记录，根据用户角色和user_id过滤（Flask 上下文）"""
    query = DetectionRecord.query
    if role in (ROLE_USER, ROLE_ADMIN):
        query = query.filter_by(user_id=user_id)
    return query.order_by(
        DetectionRecord.create_time.desc()
    ).limit(limit).all()


def get_record_by_id(record_id: int):
    """根据 ID 查询检测记录（Flask 上下文）"""
    return DetectionRecord.query.filter_by(id=record_id).first()


def delete_record_by_id(record_id: int) -> bool:
    """删除检测记录（Flask 上下文）"""
    record = DetectionRecord.query.filter_by(id=record_id).first()
    if not record:
        return False
    db.session.delete(record)
    db.session.commit()
    return True
