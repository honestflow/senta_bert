from werkzeug.security import check_password_hash

from models import Users

def validate(username, password1, password2=None):
    user = Users.query.filter(Users.username == username).first()
    if password2:
        if user:
            return '用户名已经存在'
        else:
            if len(username) < 4:
                return '用户名长度至少4个字符'
            elif password1 != password2:
                return '两次密码不一致'
            elif len(password1) < 6:
                return '密码长度至少6个字符'
            else:
                return '注册成功，请登录'
    else:
        if user:
            if user.password == password1:
                return '登录成功'
            else:
                return '密码错误'
        else:
            return '用户名不存在'



