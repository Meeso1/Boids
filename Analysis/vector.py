from __future__ import annotations
from typing import Any, overload
import numpy as np
from numpy._typing import NDArray


class Vector:

    def __init__(self, x: float, y: float) -> None:
        self.data: NDArray[np.float64] = np.array([x, y])

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"Vector({str(self.data)})"

    def __add__(self, other: Any) -> Vector:
        if isinstance(other, Vector) and len(self) == len(other):
            return Vector(self.x + other.x, self.y + other.y)
        return Vector(self.x + other, self.y + other)

    def __radd__(self, other: Any) -> Vector:
        return self.__add__(other)

    def __neg__(self) -> Vector:
        return Vector(-self.x, -self.y)

    def __sub__(self, other: Any) -> Vector:
        return self + (-other)

    def __rsub__(self, other: Any) -> Vector:
        return self.__sub__(other)

    @overload
    def __mul__(self, other: Vector) -> float:
        ...

    @overload
    def __mul__(self, other: int | float) -> Vector:
        ...

    def __mul__(self, other: float | int | Vector) -> Vector | float:
        if isinstance(other, Vector):
            return self.dot(other)
        return Vector(self.x * other, self.y * other)

    @overload
    def __rmul__(self, other: Vector) -> float:
        ...

    @overload
    def __rmul__(self, other: int | float) -> Vector:   # type: ignore
        ...

    def __rmul__(self, other: int | float | Vector) -> Vector | float:    # type: ignore
        return self.__mul__(other)

    def __truediv__(self, other: int | float) -> Vector:
        if other == 0:
            raise ArithmeticError()
        return self * (1 / other)

    def __abs__(self) -> float:
        return self.length()

    def __eq__(self, other: Vector) -> bool:
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        return False

    def __len__(self) -> int:
        return 2

    def __getitem__(self, i: int) -> float:
        if not isinstance(i, int) or not (0 <= i < 2):
            raise IndexError(f"Incorrect index for Vector: {i}")
        return self.data[i]

    def __setitem__(self, i: int, val: int | float) -> None:
        if not isinstance(i, int) or not (0 <= i < 2):
            raise IndexError(f"Incorrect index for Vector: {i}")
        if not isinstance(val, int) and not isinstance(val, float):
            raise TypeError(f"Cannot assign {type(val)} to Vector coordinate")
        self.data[i] = val

    def __copy__(self):
        return Vector(self.x, self.y)

    def length(self) -> float:
        return np.sqrt(self.dot(self))

    def dot(self, other: Vector) -> float:
        return self.x * other.x + self.y * other.y

    def normalize(self) -> Vector:
        return self / self.length() if not self.length() == 0 else Vector(0, 0)

    @property
    def x(self):
        return self[0]

    @x.getter
    def x(self):
        return self[0]

    @x.setter
    def x(self, value):
        self[0] = value

    # y property
    @property
    def y(self):
        return self[1]

    @y.getter
    def y(self):
        return self[1]

    @y.setter
    def y(self, value):
        self[1] = value
