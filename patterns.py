from abc import ABC, abstractmethod
from rayMath import Matrix, Color, inverse
import math

class AbstractPattern(ABC):
    def __init__(self, color_a, color_b):
        self.a = color_a
        self.b = color_b
        self._transform = Matrix([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
        self._inverse_transform = None

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
    def pattern_at(self, point) -> Color:
        pass


class DefaultPattern(AbstractPattern):
    def __init__(self, color_a, color_b):
        super().__init__(color_a, color_b)

    def pattern_at(self, point):
        if math.floor(point.tuple.x) % 2 == 0:
            return self.a
        else:
            return self.b


class TestPattern(AbstractPattern):
    def __init__(self, color_a, color_b):
        super().__init__(color_a, color_b)

    def pattern_at(self, point):
        return Color(point.tuple.x, point.tuple.y, point.tuple.z)


class GradientPattern(AbstractPattern):
    def __init__(self, color_a, color_b):
        super().__init__(color_a, color_b)

    def pattern_at(self, point):
        distance = self.b - self.a
        fraction = point.tuple.x - math.floor(point.tuple.x)

        return self.a + distance * fraction


class RingPattern(AbstractPattern):
    def __init__(self, color_a, color_b):
        super().__init__(color_a, color_b)

    def pattern_at(self, point):
        if math.floor(math.sqrt(point.tuple.x * point.tuple.x + point.tuple.z * point.tuple.z)) % 2 == 0:
            return self.a
        else:
            return self.b


class CheckerPattern(AbstractPattern):
    def __init__(self, color_a, color_b):
        super().__init__(color_a, color_b)

    def pattern_at(self, point):
        if (math.floor(point.tuple.x)+math.floor(point.tuple.y)+math.floor(point.tuple.z)) % 2 == 0:
            return self.a
        else:
            return self.b

class ZStripePattern(AbstractPattern):
    def __init__(self, color_a, color_b):
        super().__init__(color_a, color_b)

    def pattern_at(self, point):
        if math.floor(point.tuple.z) % 2 == 0:
            return self.a
        else:
            return self.b


class BlendedPattern(AbstractPattern):
    def __init__(self, pattern_a, pattern_b):
        # Dummy colors for parent class
        super().__init__(Color(0,0,0), Color(0,0,0))
        self.pattern_a = pattern_a
        self.pattern_b = pattern_b

    def pattern_at(self, point):
        color_a = self.pattern_a.pattern_at(point)
        color_b = self.pattern_b.pattern_at(point)
        # Simple average blend
        return Color(
            (color_a.tuple.x + color_b.tuple.x) / 2,
            (color_a.tuple.y + color_b.tuple.y) / 2,
            (color_a.tuple.z + color_b.tuple.z) / 2
        )

def gradient_pattern(color_a, color_b):
    return GradientPattern(color_a, color_b)


def ring_pattern(color_a, color_b):
    return RingPattern(color_a, color_b)


def checker_pattern(color_a, color_b):
    return CheckerPattern(color_a, color_b)

def stripe_pattern(color_a, color_b):
    return DefaultPattern(color_a, color_b)

def blended_pattern(pattern_a, pattern_b):
    return BlendedPattern(pattern_a, pattern_b)

def z_stripe_pattern(color_a, color_b):
    return ZStripePattern(color_a, color_b)

def test_pattern():
    black = Color(0, 0, 0)
    white = Color(1, 1, 1)
    return TestPattern(white, black)
