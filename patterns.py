from abc import ABC, abstractmethod
from rayMath import Matrix, Color, inverse
from base import Point
import math
import random

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


class PerlinNoise:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

        # Create permutation table
        self.p = list(range(256))
        random.shuffle(self.p)
        self.p += self.p  # Duplicate to avoid overflow

    def fade(self, t):
        """Fade function as defined by Ken Perlin"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(self, t, a, b):
        """Linear interpolation"""
        return a + t * (b - a)

    def grad(self, hash_val, x, y, z):
        """Convert low 4 bits of hash code into 12 gradient vectors"""
        h = hash_val & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def noise(self, x, y, z):
        """3D Perlin noise"""
        # Find unit cube that contains point
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        Z = int(math.floor(z)) & 255

        # Find relative x,y,z of point in cube
        x -= math.floor(x)
        y -= math.floor(y)
        z -= math.floor(z)

        # Compute fade curves for each of x,y,z
        u = self.fade(x)
        v = self.fade(y)
        w = self.fade(z)

        # Hash coordinates of 8 cube corners
        A = self.p[X] + Y
        AA = self.p[A] + Z
        AB = self.p[A + 1] + Z
        B = self.p[X + 1] + Y
        BA = self.p[B] + Z
        BB = self.p[B + 1] + Z

        # Add blended results from 8 corners of cube
        return self.lerp(w,
            self.lerp(v,
                self.lerp(u, self.grad(self.p[AA], x, y, z),
                             self.grad(self.p[BA], x-1, y, z)),
                self.lerp(u, self.grad(self.p[AB], x, y-1, z),
                             self.grad(self.p[BB], x-1, y-1, z))),
            self.lerp(v,
                self.lerp(u, self.grad(self.p[AA + 1], x, y, z-1),
                             self.grad(self.p[BA + 1], x-1, y, z-1)),
                self.lerp(u, self.grad(self.p[AB + 1], x, y-1, z-1),
                             self.grad(self.p[BB + 1], x-1, y-1, z-1))))


class PerturbedPattern(AbstractPattern):
    def __init__(self, pattern, scale=1.0, amplitude=0.1):
        # Dummy colors for parent class
        super().__init__(Color(0,0,0), Color(0,0,0))
        self.pattern = pattern
        self.scale = scale
        self.amplitude = amplitude
        self.noise = PerlinNoise()

    def pattern_at(self, point):
        # Apply Perlin noise to perturb the point
        noise_x = self.noise.noise(point.tuple.x * self.scale,
                                   point.tuple.y * self.scale,
                                   point.tuple.z * self.scale) * self.amplitude
        noise_y = self.noise.noise((point.tuple.x + 100) * self.scale,
                                   (point.tuple.y + 100) * self.scale,
                                   (point.tuple.z + 100) * self.scale) * self.amplitude
        noise_z = self.noise.noise((point.tuple.x + 200) * self.scale,
                                   (point.tuple.y + 200) * self.scale,
                                   (point.tuple.z + 200) * self.scale) * self.amplitude

        # Create perturbed point
        perturbed_point = Point(
            point.tuple.x + noise_x,
            point.tuple.y + noise_y,
            point.tuple.z + noise_z
        )

        return self.pattern.pattern_at(perturbed_point)

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

def pertrubed_pattern(pattern, scale=1.0, amplitude=0.1):
    return PerturbedPattern(pattern, scale, amplitude)

def test_pattern():
    black = Color(0, 0, 0)
    white = Color(1, 1, 1)
    return TestPattern(white, black)
