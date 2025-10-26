from abc import ABC, abstractmethod
from base import Material, Matrix, Intersection, Vector, Ray, Point, inverse
# If you need dot function, import it explicitly
from math import fabs,sqrt
from base import dot
from utils import auto_str, EPSILON

class Shape(ABC):
    def __init__(self):
        self._transform: Matrix = Matrix([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        self._inverse_transform = None
        self.material = Material()

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        self._transform = value
        self._inverse_transform = None  # Clear cached inverse when transform changes

    @property
    def inverse_transform(self):
        if self._inverse_transform is None:
            self._inverse_transform = inverse(self._transform)
        return self._inverse_transform

    @abstractmethod
    def local_intersect(self, ray) -> list[Intersection] | None:
        pass

    @abstractmethod
    def local_normal_at(self, point) -> Vector:
        pass


class TestShape(Shape):
    def __init__(self):
        super().__init__()
        self.saved_ray = Ray(Point(0, 0, 0), Vector(0, 0, 0))

    def local_normal_at(self, point):
        return Vector(point.tuple.x, point.tuple.y, point.tuple.z)

    def local_intersect(self, ray):
        self.saved_ray = ray


class Plane(Shape):
    def __init__(self):
        super().__init__()

    def local_normal_at(self, point):
        return Vector(0, 1, 0)

    def local_intersect(self, ray):
        if fabs(ray.direction.tuple.y) < EPSILON:
            return []
        t = -ray.origin.tuple.y / ray.direction.tuple.y
        return [Intersection(t, self)]


@auto_str
class Sphere(Shape):
    def __init__(self):
        super().__init__()

    def local_normal_at(self, point):
        return point - Point(0, 0, 0)

    def local_intersect(self, ray):
        sphere_to_ray = ray.origin - Point(0, 0, 0)  # unit circle for now
        a = dot(ray.direction, ray.direction)
        b = 2 * dot(ray.direction, sphere_to_ray)
        c = dot(sphere_to_ray, sphere_to_ray) - 1

        discriminant = b * b - 4 * a * c
        # print("discriminant: {0}".format(discriminant))

        if discriminant < 0:
            return []

        t1 = (-b - sqrt(discriminant)) / (2 * a)
        t2 = (-b + sqrt(discriminant)) / (2 * a)

        return [Intersection(t1, self), Intersection(t2, self)]

def test_shape():
    return TestShape()
