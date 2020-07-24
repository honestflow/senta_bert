from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash

db = SQLAlchemy()

class Users(db.Model):
    __tablename__ = 'users_info'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(32), nullable=False)
    password = db.Column(db.String(128), nullable=False)
    register_time = db.Column(db.DateTime, nullable=False, default=datetime.now())
    # 我们新增了一个avatar_path字段来存用户头像图片文件的路径
    def __init__(self, *args, **kwargs):
        self.username = kwargs.get('username')
        self.password = kwargs.get('password')


class Questions(db.Model):
    __tablename__ = 'questions_info'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.TEXT, nullable=False)
    author_id = db.Column(db.Integer, db.ForeignKey('users_info.id'))
    create_time = db.Column(db.DateTime, nullable=False, default=datetime.now())

    author = db.relationship('Users', backref=db.backref('questions', order_by=create_time.desc()))

class Comments(db.Model):
    __tablename__ = 'comments_info'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    content = db.Column(db.TEXT, nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('questions_info.id'))
    author_id = db.Column(db.Integer, db.ForeignKey('users_info.id'))
    create_time = db.Column(db.DateTime, nullable=False, default=datetime.now())

    author = db.relationship('Users', backref=db.backref('comments'))
    question = db.relationship('Questions', backref=db.backref('comments', order_by=create_time.desc()))

