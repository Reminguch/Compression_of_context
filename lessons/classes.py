from dataclasses import dataclass


@dataclass
class Address:
    city: str
    street: str
    house: int
    apt: int = 1


@dataclass
class Devices:
    name: str
    price: float
    quantity: int = 1
    description: str = "No description"


@dataclass
class User:
    name: str
    age: int
    email: str = "<Email>"
    address: Address = Address("Moscow", "Lenina", 1, 1)
    devices: list[Devices] = None


user1 = User("John", 25)
print(user1)
user2 = User("Alice", 30, "<Email111>", Address("SPB", "Nevskiy", 10, 5))

address1 = Address("SPB", "Nevskiy", 10, 5)
user3 = User("Alice ", 30, "<Email111>", address1)

print(user2)
if user1 == user2:
    print("Users are equal")
else:
    print("Users are not equal")

user3 = {"name": "Bob", "age": 22, "email": "<Email222>"}


class Car:
    owner = "Default Owner"  # Class variable

    def __init__(
        self, make, model, year, color="Red"
    ):  # Constructor with default color
        self.make = make
        self.model = model
        self.year = year
        self.color = color
        self.date_of_maintenance = "2023-01-01"  # Instance variable with default value

    def display_info(self):
        return f"{self.year} {self.make} {self.model} {self.color} {self.owner} {self.date_of_maintenance}"

    def __repr__(self):
        return f"make='{self.make}', model='{self.model}', year={self.year}, color='{self.color}'"

    def __eq__(self, other):
        if not isinstance(other, Car):
            return NotImplemented
        return (self.year) == (other.year)

    @classmethod
    def from_string(cls, data: str):
        """
        Альтернативный конструктор: создаёт Car из строки формата "make,model,year,color"
        """
        make, model, year, color = data.split(",")
        return cls(make, model, int(year), color)

    @classmethod
    def how_many_cars(cls):
        return f"Всего создано машин: {cls.total_cars}"

    @classmethod
    def change_owner(cls, new_owner):  # Class method to change the owner
        cls.owner = new_owner

    @classmethod
    def schedule_maintenance(
        cls, car_instance, date
    ):  # Class method to schedule maintenance
        car_instance.date_of_maintenance = date
        return f"Maintenance scheduled for {car_instance.model} on {date}."

    @staticmethod
    def honk_horn():
        return "Beep beep!"

    @staticmethod
    def price_estimate(year):
        base_price = 20000
        age = 2024 - year
        depreciation = age * 1000
        return max(base_price - depreciation, 5000)

    def repaint(self, new_color):  # Method to change the color
        self.color = new_color
        return f"The car has been repainted to {self.color}."


car1 = Car("Toyota", "Camry", 2020)
# print(car1.display_info())
print(car1)
car2 = Car.from_string("Honda,Civic,2020,Blue")
# print(Car.price_estimate(2018))

if car1 == car2:
    print("Cars are equal")
else:
    print("Cars are not equal")
