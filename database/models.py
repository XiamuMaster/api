from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

#用户等级
ROLE_SUPER_ADMIN = 'super_admin'   # 超级管理员
ROLE_ADMIN       = 'admin'         # 管理员
ROLE_USER        = 'user'          # 普通用户

ROLE_LEVELS = {
    ROLE_USER:        1,
    ROLE_ADMIN:       2,
    ROLE_SUPER_ADMIN: 3,
}

#用户表设置
class User(db.Model):
    __tablename__ = 'users'
    id          = db.Column(db.Integer, primary_key=True)
    username    = db.Column(db.String(64),  unique=True, nullable=False)
    password    = db.Column(db.String(128), nullable=False)           # 哈希后的密码
    role        = db.Column(db.String(32),  default=ROLE_USER)        # user / admin / super_admin
    is_active   = db.Column(db.Boolean,     default=True)             # 是否启用
    create_time = db.Column(db.DateTime,    default=datetime.now)
    last_login  = db.Column(db.DateTime,    nullable=True)

    def to_dict(self):
        return {
            'id':          self.id,
            'username':    self.username,
            'role':        self.role,
            'is_active':   self.is_active,
            'create_time': self.create_time.strftime('%Y-%m-%d %H:%M:%S'),
            'last_login':  self.last_login.strftime('%Y-%m-%d %H:%M:%S') if self.last_login else None,
        }


#检测记录
class DetectionRecord(db.Model):
    __tablename__ = 'YOLOlist'
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # 检测用户ID
    load_filename    = db.Column(db.String(255), nullable=False)  # 原文件名
    result_filename  = db.Column(db.String(255))                  # 结果文件名
    model_type       = db.Column(db.String(255))                  # 模型类型
    detect_file_type = db.Column(db.String(255))                  # 文件类型
    create_time      = db.Column(db.DateTime, default=datetime.now)

    def to_dict(self):
        return {
            'id':               self.id,
            'user_id':          self.user_id,
            'load_filename':    self.load_filename,
            'result_filename':  self.result_filename,
            'model_type':       self.model_type,
            'detect_file_type': self.detect_file_type,
            'create_time':      self.create_time.strftime('%Y-%m-%d %H:%M:%S') if self.create_time else None,
        }
