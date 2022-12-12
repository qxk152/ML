import numpy
import matplotlib
import sklearn
import pandas
class Student :
    name = "qi"
    sum = 0;
    #注意这里的类变量 name 和实例变量self.name不是一个东西
    def __init__(self,name):
        self.name = name
        self.__class__.sum +=1
        self.__score = 0
        print(self.__class__.sum)
    def do_homework(self):
        self.do_english_homework()
    def do_english_homework(self):
        print("English")
    def mark(self,score):
        assert score>= 0,"不能是负"
        self.__score = score
        print(self.name,"score:",self.__score)


s1 = Student("祁星凯")
s1.mark(60)
s1.__score = -1; #python 可以动态创建一个新属性__score 并不是 在self.__score
print(s1.__dict__)
print(s1._Student__score)
