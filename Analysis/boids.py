from __future__ import annotations
from random import random
from typing import List
from vector import Vector


class RuleConfig:
    def __init__(self, radius: float, strength: float):
        self.radius: float = radius
        self.strength: float = strength


class Environment:
    def __init__(self,
                 speed_range: List[float],
                 scene_size: List[float],
                 separation: RuleConfig,
                 attraction: RuleConfig,
                 alignment: RuleConfig,
                 avoidance: RuleConfig,
                 use_smooth_acc: bool = False
                 ):
        self.max_v: float = speed_range[1]
        self.min_v: float = speed_range[0]
        self.size_x: float = scene_size[0]
        self.size_y: float = scene_size[1]
        self.separation: RuleConfig = separation
        self.attraction: RuleConfig = attraction
        self.alignment: RuleConfig = alignment
        self.avoidance: RuleConfig = avoidance
        self.smooth_acc: bool = use_smooth_acc


class Boids:
    def __init__(self, n: int, environment: Environment):
        self.n: int = n
        self.environment: Environment = environment

        self.t: List[float] = [0]
        self.fish: List[Fish] = [Boids.random_fish(environment, i) for i in range(n)]

    @staticmethod
    def random_fish(env: Environment, index: int) -> Fish:
        def rand_between(a: float, b: float) -> float:
            return random() * (b - a) + a

        def rand_v(min_v: float, max_v: float) -> Vector:
            v: Vector = Vector(rand_between(-1, 1), rand_between(-1, 1))
            l: float = rand_between(min_v, max_v)
            return v.normalize() * l if v != Vector(0, 0) else rand_v(min_v, max_v)

        return Fish(
            rand_between(-env.size_x, env.size_x),
            rand_between(-env.size_y, env.size_y),
            rand_v(env.min_v, env.max_v),
            index
        )

    def update(self, dt: float) -> None:
        self.t.append(self.t[-1] + dt)
        for f in self.fish:
            f.update(dt, self.fish, self.environment)


class Fish:
    def __init__(self, x0: float, y0: float, v0: Vector, index: int):
        self.x: List[float] = [x0]
        self.y: List[float] = [y0]
        self.v: Vector = v0

        self.index: int = index

    def dist(self, fish: Fish):
        return Vector(self.x[-1] - fish.x[-1], self.y[-1] - fish.y[-1]).length()

    def update(self, dt: float, fish: List[Fish], env: Environment) -> None:
        dv: Vector = self.separation(fish, env.separation) \
                     + self.attraction(fish, env.attraction) \
                     + self.alignment(fish, env.alignment) \
                     + self.avoidance(env.size_x, env.size_y, env.alignment)
        if env.smooth_acc:
            dv *= dt

        self.v = self.clip_speed(self.v + dv, env.min_v, env.max_v)

        self.x.append(self.x[-1] + (self.v * dt).x)
        self.y.append(self.y[-1] + (self.v * dt).y)

    def separation(self, fish: List[Fish], params: RuleConfig) -> Vector:
        away: Vector = Vector(0, 0)
        for i, f in enumerate(fish):
            if i == self.index:
                continue
            if self.dist(f) > params.radius:
                continue
            away += Vector(self.x[-1] - f.x[-1], self.y[-1] - f.y[-1])

        return away * params.strength

    def attraction(self, fish: List[Fish], params: RuleConfig) -> Vector:
        center_x: float = 0
        center_y: float = 0
        neighbors: int = 0
        for i, f in enumerate(fish):
            if i == self.index:
                continue
            if self.dist(f) > params.radius:
                continue
            center_x += f.x[-1]
            center_y += f.y[-1]
            neighbors += 1

        if neighbors <= 0:
            return Vector(0, 0)

        to_center: Vector = Vector(center_x/neighbors - self.x[-1], center_y/neighbors - self.y[-1])
        return to_center * params.strength

    def alignment(self, fish: List[Fish], params: RuleConfig) -> Vector:
        v: Vector = Vector(0, 0)
        neighbors: int = 0
        for i, f in enumerate(fish):
            if i == self.index:
                continue
            if self.dist(f) > params.radius:
                continue
            v += f.v
            neighbors += 1

        if neighbors <= 0:
            return Vector(0, 0)

        avg_v: Vector = v / neighbors
        return (avg_v - self.v) * params.strength

    def avoidance(self, size_x: float, size_y: float, params: RuleConfig) -> Vector:
        v: Vector = Vector(0, 0)
        if self.x[-1] < -(size_x - params.radius):
            v += Vector(-(size_x - params.radius) - self.x[-1], 0)
        elif self.x[-1] > size_x - params.radius:
            v += Vector(size_x - params.radius - self.x[-1], 0)

        if self.y[-1] < -(size_y - params.radius):
            v += Vector(0, -(size_y - params.radius) - self.y[-1])
        elif self.y[-1] > size_y - params.radius:
            v += Vector(0, size_y - params.radius - self.y[-1])

        return v * params.strength

    @staticmethod
    def clip_speed(v: Vector, min_v: float, max_v: float) -> Vector:
        if v.length() < min_v:
            return v.normalize() * min_v
        elif v.length() > max_v:
            return v.normalize() * max_v
        return v
