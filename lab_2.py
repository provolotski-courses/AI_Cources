"""Modude second homework"""

from datetime import datetime


class Car:
    """объявление класса"""
    brand = 'No name'  # по умолчанию не знаем что за брэнд
    year = 1885  # год первого авто

    __engine_type = None  # приватный метод. задача 5
    __car_count = 0  # переменная класса. Задача 9

    def __init__(self, brand='unknown model', year=datetime.now().year):
        """конструктор со значениями по умолчанию. Задание 3"""
        self.brand = brand
        self.year = year
        Car.__car_count += 1  # увеличиваем счетчик машин при каждом вызове конструктора. Задание 9

    def __del__(self):
        """Деструктор класса"""
        Car.__car_count -= 1  # уменьшаем счетчик машин при каждом вызове деструктора. Задание 9

    @staticmethod
    def print_count():
        """Статический метод для вывода количества автомобилей класса Car. Задание 9"""
        return Car.__car_count

    def start_engine(self):
        """реализация метода класса. задание 2."""
        print('Двигатель запущен!')

    def __str__(self):
        """реализация вывода атрибутов. задание 4."""
        return f'brand is:  {self.brand}  year is: + {str(self.year)}'

    def set_engine(self, engine_type):
        """реализация публичного метода  для приватного аттрибута класса. задание 5."""
        self.__engine_type = engine_type

    def get_engine(self):
        """реализация публичного метода  для приватного аттрибута класса. задание 5."""
        return self.__engine_type

    @classmethod
    def get_def_car(cls):
        """Метод класса. Задание 8"""
        return cls('unknown car', datetime.now().year - 5)


class ElectricCar(Car):
    """Наследуемый класс. Задание 6"""
    battery_size = 800

    def __init__(self, brand, year, battery_size):
        """реализация конструктора класса с использованием метода super. задание 8."""
        super().__init__(brand, year)
        self.battery_size = battery_size

    def start_engine(self):
        """Переопределенный метод. задание 7."""
        print('Тихий запуск двигателя!')


print(Car.print_count())

# создание экземпляра класса задача 4
my_car = Car('что-то китайское')
print(my_car)
print(Car.print_count())

second_car = ElectricCar('hgf', 122, 876)
# second_car = ElectricCar()
third_car = Car.get_def_car()
print(Car.print_count())
