from sqlalchemy import create_engine  
from sqlalchemy.ext.declarative import declarative_base  
from sqlalchemy import Column, Integer, String  
from sqlalchemy.orm import sessionmaker  

# Создание базы данных SQLite  
engine = create_engine('sqlite:///example.db', echo=True)  

# Определение базового класса для моделей  
Base = declarative_base()  

# Определение модели User  
class User(Base):  
    __tablename__ = 'users'  

    id = Column(Integer, primary_key=True)  
    name = Column(String)  
    age = Column(Integer)  

    def __repr__(self):  
        return f"<User(id={self.id}, name='{self.name}', age={self.age})>"  

# Создание всех таблиц  
Base.metadata.create_all(engine)  

# Создание сессии для работы с базой данных  
Session = sessionmaker(bind=engine)  
session = Session()

# Добавление нового пользователя  
new_user = User(name='Alice', age=30)  
session.add(new_user)  
session.commit()  

# Получение всех пользователей  
all_users = session.query(User).all()  
print(all_users)  

url = 'sqlite:///base.db'


# Закрытие сессии  
session.close()