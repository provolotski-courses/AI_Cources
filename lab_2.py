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

    def __str__(self):
        """реализация вывода атрибутов. задание 4."""
        return f'brand is:  {self.brand}  year is: {str(self.year)}'

    def __repr__(self):
        """реализация вывода атрибутов. задание 4."""
        return f'Car brand is: {self.brand}  year is: {str(self.year)} engine_type: {self.__engine_type}'

    @staticmethod
    def print_count():
        """Статический метод для вывода количества автомобилей класса Car. Задание 9"""
        return Car.__car_count

    def start_engine(self):
        """реализация метода класса. задание 2."""
        print('Двигатель запущен!')

    @property
    def engine_type(self):
        """Декоратор для геттера"""
        return self.__engine_type

    @engine_type.setter
    def engine_type(self, value):
        """Декоратор для сеттера с генерацией исключения"""
        if value not in ['petrol', 'diesel', 'electric']:
            raise ValueError('не корректный тип двигателя')
        self.__engine_type = value

    @classmethod
    def get_def_car(cls):
        """Метод класса. Задание 8"""
        return cls('unknown car', datetime.now().year - 5)


class ElectricCar(Car):
    """Наследуемый класс."""
    battery_size = 800

    def __init__(self, brand, year, battery_size):
        """реализация конструктора класса с использованием метода super. задание 8."""
        super().__init__(brand, year)
        self.battery_size = battery_size

    def start_engine(self):
        """Переопределенный метод. задание 7."""
        print('Тихий запуск двигателя!')


class CarPark:
    def __init__(self, cars):
        self.car_list = cars
        self.max_value = len(cars)
        self.current_value = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_value < self.max_value:
            result = self.car_list[self.current_value]
            self.current_value += 1
            return result
        else:
            raise StopIteration


print(f'Начало выполнения')
# Выводим изначальное количество мащин
print(f'Машин в наличии при старте : {Car.print_count()}')

# создание первого экземпляра класса
my_car = Car('Chevrolet', 1968)
print(my_car)
print(f'Машин в наличии: {Car.print_count()}')

# создание второго экземпляра класса с годом по умолчанию
my_car1 = Car('что-то очень китайское')
print(my_car1)
print(f'Машин в наличии: {Car.print_count()}')

# создание  экземпляра дочернего класса
second_car = ElectricCar('Lada', 170, 876)
second_car.engine_type = 'electric'  # определяем тип двигателя
print(second_car)
print(f'Машин в наличии: {Car.print_count()}')

# создание экземпляра с помощью метода класса
third_car = Car.get_def_car()
print(third_car)
print(f'Машин в наличии: {Car.print_count()}')

# Объединяем машины в лист
car_dict = [my_car, my_car1, second_car, third_car]
print(f'печать листа {car_dict}')

# создаем экземпляр класса-итератора
car_park = CarPark(car_dict)

print('Итерируемся по парку')
for i in car_park:
    print(i)
