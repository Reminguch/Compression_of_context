from datetime import datetime
from functools import lru_cache as cashe
from functools import wraps

COUNT = 1


def decorator(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        global COUNT
        print(
            "Function is being called for the first time"
            if COUNT == 1
            else f"Function has been called {COUNT} times"
        )
        COUNT += 1
        return func(*args, **kwargs)

    return wrapped


def create_function():
    pass


@decorator
@cashe
def another_function(asset, boolean, *args, cast=None, **kwargs):
    """This is another function."""
    print(args, asset, boolean, cast, kwargs)
    return 1


print(another_function.__doc__)
print(another_function.__name__)


another_function(
    "asset_value",
    True,
    1,
    2,
    3,
    cast="cast_value",
    extra_arg1="extra1",
    extra_arg2="extra2",
)
another_function(
    "asset_value2",
    False,
    4,
    5,
    6,
    cast="cast_value2",
    extra_arg3="extra3",
    extra_arg4="extra4",
)
another_function(
    "asset_value",
    True,
    1,
    2,
    3,
    cast="cast_value",
    extra_arg1="extra1",
    extra_arg2="extra2",
)


def yet_another_function(test: list, number: int = 42) -> str:
    now = datetime.now()
    return f"Test: {test}, Number: {number} now	: {now}"


def multiply(a, b):
    return a * b


# print(yet_another_function(1, number="sdasdsad"))
start_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# for i in start_list:
#     print(i)
#     if i == 5:
#         break
#     if i == 6:
#         continue
#     if i % 2 == 0:
#         m = multiply(i, 2)
#         print(m)
