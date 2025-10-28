import math
import unittest
import tempfile

# from rayMath import Color, Matrix, Point, Vector, \
#     magnitude, cross, dot, normalize, Canvas, write_pixel, \
#     pixel_at, canvas_to_ppm, translation, scaling, rotation_x, \
#     rotation_y, rotation_z, shearing, TransformationBuilder, Ray, \
#     position_along_ray, Sphere, intersect, Intersection, intersections, hit, \
#     transform, set_transform, normal_at, reflect, PointLight, Material, \
from rayMath import magnitude
from rayMath import normalize
from base import dot
from rayMath import cross
from rayMath import Canvas
from rayMath import write_pixel
from rayMath import pixel_at
from rayMath import canvas_to_ppm
from rayMath import shearing
from rayMath import rotation_x
from rayMath import scaling
from rayMath import translation
from rayMath import rotation_y
from rayMath import rotation_z
from rayMath import TransformationBuilder
from base import Ray
from rayMath import position_along_ray
from shapes import Sphere
from rayMath import intersect
from base import Intersection
from rayMath import intersections
from rayMath import hit
from rayMath import transform
from rayMath import set_transform
from rayMath import normal_at
from rayMath import reflect
from rayMath import PointLight
from base import Material
from rayMath import World
from rayMath import default_world
from rayMath import intersect_world
from rayMath import prepare_computations
from rayMath import shade_hit
from rayMath import color_at
from rayMath import view_transforfmation
from rayMath import Camera
from rayMath import ray_for_pixel
from rayMath import is_shadowed
from utils import EPSILON
from rayMath import stripe_at
from rayMath import set_pattern_transform
from rayMath import pattern_at_shape
#     lighting, World, default_world, intersect_world, prepare_computations, \
#     shade_hit, color_at, view_transforfmation, Camera, ray_for_pixel, render, \
#     is_shadowed, EPSILON, stripe_at, \
#     set_pattern_transform, pattern_at_shape
#
from rayMath import lighting, render
from shapes import Shape, Plane, test_shape
from patterns import test_pattern, stripe_pattern
from patterns import gradient_pattern,ring_pattern, checker_pattern
from base import Tuple, transpose, determinant, submatrix, \
minor, cofactor, inverse, Color, Matrix, Point, Vector
from math import sqrt


# Run: python -m unittest rayMath_test.py

class TestRayMath(unittest.TestCase):
    def assertFloatEqual(self, first, second, msg=None):
        """Assert that two values are equal within floating point tolerance"""
        from math import fabs
        if not (fabs(first - second) < EPSILON):
            raise AssertionError(f'{first} != {second} (difference: {fabs(first - second)}, tolerance: {EPSILON})')

    def test_tuple_is_a_point(self):
        a = Tuple(4.3, -4.2, 3.1, 1.0)
        self.assertEqual(a.x, 4.3)
        self.assertEqual(a.y, -4.2)
        self.assertEqual(a.z, 3.1)
        self.assertEqual(a.w, 1.0)
        self.assertEqual(a.type(), "point")
        self.assertNotEqual(a.type(), "vector")

    def test_tuple_is_a_vector(self):
        a = Tuple(4.3, -4.2, 3.1, 0.0)
        self.assertEqual(a.x, 4.3)
        self.assertEqual(a.y, -4.2)
        self.assertEqual(a.z, 3.1)
        self.assertEqual(a.w, 0.0)
        self.assertEqual(a.type(), "vector")
        self.assertNotEqual(a.type(), "point")

    def test_tuple_equality(self):
        a = Tuple(4.3, -4.2, 3.1, 1.0)
        b = Tuple(4.3, -4.2, 3.1, 1.0)

        self.assertTrue(a == b)

    def test_point_creates_vectors_with_w_equals_1(self):
        p = Point(4.0, -4.0, 3.0)
        t = Tuple(4.0, -4.0, 3.0, 1.0)
        self.assertTrue(p.tuple == t)

    def test_point_creates_vectors_with_w_equals_0(self):
        v = Vector(4.0, -4.0, 3.0)
        t = Tuple(4.0, -4.0, 3.0, 0.0)
        self.assertTrue(v.tuple == t)

    def test_add_two_tuples(self):
        point = Tuple(3, -2, 5, 1)
        vector = Tuple(-2, 3, 1, 0)
        self.assertEqual(point + vector, Tuple(1, 1, 6, 1))

    def test_add_two_vectors(self):
        a1 = Vector(3.0, -2.0, 5.0)
        a2 = Vector(-2.0, 3.0, 1.0)
        self.assertEqual(a1 + a2, Vector(1, 1, 6))

    def test_add_point_and_vector(self):
        point = Point(3, -2, 5)
        vector = Vector(-2, 3, 1)
        self.assertEqual(point + vector, Point(1, 1, 6))

    def test_raise_error_when_two_points_are_added(self):
        point1 = Point(3.0, -2.0, 5.0)
        point2 = Point(-2.0, 3.0, 1.0)
        with self.assertRaises(TypeError):
            point2 + point1

    def test_subtract_two_points_to_produce_vector(self):
        p1 = Point(3, 2, 1)
        p2 = Point(5, 6, 7)
        self.assertEqual(p1 - p2, Vector(-2, -4, -6))

    def test_subtract_vector_from_zero_vector(self):
        zero = Vector(0, 0, 0)
        v = Vector(1, -2, 3)
        self.assertEqual(zero - v, Vector(-1, 2, -3))

    def test_multiply_vector_by_scalar(self):
        self.assertEqual(Vector(11.25, 0, 0), normalize(Vector(4, 0, 0)) * 11.25)

    def test_negate_tuple(self):
        a = Tuple(1, -2, 3, -4)
        self.assertEqual(-a, Tuple(-1, 2, -3, 4))

    def test_multiply_tuple_by_scalar(self):
        a = Tuple(1, -2, 3, -4)
        self.assertEqual(a * 3.5, Tuple(3.5, -7, 10.5, -14))

    def test_multiply_tuple_by_scalar_fraction(self):
        a = Tuple(1, -2, 3, -4)
        self.assertEqual(a * 0.5, Tuple(0.5, -1, 1.5, -2))

    def test_divide_tuple_by_scalar(self):
        a = Tuple(1, -2, 3, -4)
        self.assertEqual(a / 2, Tuple(0.5, -1, 1.5, -2))

    def test_magnitude_of_vector(self):
        v = Vector(1, 0, 0)
        self.assertEqual(magnitude(v), 1)
        v = Vector(0, 0, 1)
        self.assertEqual(magnitude(v), 1)
        v = Vector(1, 2, 3)
        self.assertEqual(magnitude(v), sqrt(14))

    def test_normalize_vector(self):
        v = Vector(4, 0, 0)
        self.assertEqual(normalize(v), Vector(1, 0, 0))

    def test_dot_product(self):
        a = Vector(1, 2, 3)
        b = Vector(2, 3, 4)
        self.assertEqual(dot(a, b), 20)

    def test_cross_product(self):
        a = Vector(1, 2, 3)
        b = Vector(2, 3, 4)
        self.assertEqual(cross(a, b), Vector(-1, 2, -1))
        self.assertEqual(cross(b, a), Vector(1, -2, 1))

    def test_colors_are_red_gree_blue_tuple(self):
        c = Color(-0.5, 0.4, 1.7)
        self.assertEqual(c.red, -0.5)
        self.assertEqual(c.green, 0.4)
        self.assertEqual(c.blue, 1.7)

    def test_adding_colors(self):
        c1 = Color(0.9, 0.6, 0.75)
        c2 = Color(0.7, 0.1, 0.25)

        self.assertEqual(c1 + c2, Color(1.6, 0.7, 1.0))

    def test_subtracting_colors(self):
        c1 = Color(0.9, 0.6, 0.75)
        c2 = Color(0.7, 0.1, 0.25)

        # print(c1 - c2)
        self.assertEqual(c1 - c2, Color(0.2, 0.5, 0.5))

    def test_multiplying_color_by_scalar(self):
        c = Color(0.2, 0.3, 0.4)
        self.assertEqual(c * 2, Color(0.4, 0.6, 0.8))

    def test_multiplying_color_by_color(self):
        c1 = Color(1, 0.2, 0.4)
        c2 = Color(0.9, 1, 0.1)
        self.assertEqual(c1 * c2, Color(0.9, 0.2, 0.04))

    def test_create_canvas(self):
        c = Canvas(10, 20)
        self.assertEqual(c.width, 10)
        self.assertEqual(c.height, 20)
        for row in range(c.height):
            for col in range(c.width):
                self.assertEqual(c.pixels[row][col],
                                 Color(0, 0, 0))

    def test_write_pixel_canvas(self):
        c = Canvas(10, 20)
        red = Color(1, 0, 0)
        write_pixel(c, 2, 3, red)
        self.assertEqual(pixel_at(c, 2, 3), red)

    def test_construct_ppm_header(self):
        c = Canvas(5, 3)
        _, outfile_path = tempfile.mkstemp()
        canvas_to_ppm(c, outfile_path)
        with open(outfile_path, 'r') as f:
            contents = f.readlines()
            self.assertEqual("P3", contents[0].strip())
            self.assertEqual("5 3", contents[1].strip())
            self.assertEqual("255", contents[2].strip())

    def test_construct_ppm_pixel_data(self):
        c = Canvas(5, 3)
        c1 = Color(1.5, 0, 0)
        c2 = Color(0, 0.5, 0)
        c3 = Color(-0.5, 0, 1)

        write_pixel(c, 0, 0, c1)
        write_pixel(c, 2, 1, c2)
        write_pixel(c, 4, 2, c3)

        _, outfile_path = tempfile.mkstemp()

        canvas_to_ppm(c, outfile_path)

        with open(outfile_path, 'r') as f:
            contents = f.readlines()
            self.assertEqual("255 0 0 0 0 0 0 0 0 0 0 0 0 0 0", contents[3].strip())
            self.assertEqual("0 0 0 0 0 0 0 128 0 0 0 0 0 0 0", contents[4].strip())
            self.assertEqual("0 0 0 0 0 0 0 0 0 0 0 0 0 0 255", contents[5].strip())

    def test_split_long_lines_ppm_files(self):
        c = Canvas(10, 2, Color(1, 0.8, 0.6))

        _, outfile_path = tempfile.mkstemp()

        canvas_to_ppm(c, outfile_path)

        with open(outfile_path, 'r') as f:
            contents = f.readlines()
            # print(contents)
            self.assertEqual("255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204", contents[3].strip())
            self.assertEqual("153 255 204 153 255 204 153 255 204 153 255 204 153", contents[4].strip())
            self.assertEqual("255 204 153 255 204 153 255 204 153 255 204 153 255 204 153 255 204", contents[5].strip())
            self.assertEqual("153 255 204 153 255 204 153 255 204 153 255 204 153", contents[6].strip())

    def test_ppm_files_are_terminated_by_newline(self):
        c = Canvas(5, 3)
        _, outfile_path = tempfile.mkstemp()

        canvas_to_ppm(c, outfile_path)

        with open(outfile_path, 'r') as f:
            contents = f.readlines()
            self.assertEqual(contents[-1], "\n")

    def test_construct_and_inspect_4x4_matrix(self):
        M = Matrix([[1, 2, 3, 4],
                    [5.5, 6.5, 7.5, 8.5],
                    [9, 10, 11, 12],
                    [13.5, 14.5, 15.5, 16.5]])

        self.assertEqual(M[0, 0], 1)
        self.assertEqual(M[1, 0], 5.5)
        self.assertEqual(M[1, 0], 5.5)
        self.assertEqual(M[1, 2], 7.5)
        self.assertEqual(M[2, 2], 11)
        self.assertEqual(M[3, 0], 13.5)
        self.assertEqual(M[3, 2], 15.5)

    def test_construct_and_inspect_2x2_matrix(self):
        M = Matrix([[-3, 5],
                    [1, -2]])
        self.assertEqual(M[0, 0], -3)
        self.assertEqual(M[0, 1], 5)
        self.assertEqual(M[1, 0], 1)
        self.assertEqual(M[1, 1], -2)

    def test_construct_and_inspect_3x3_matrix(self):
        M = Matrix([[-3, 5, 0],
                    [1, -2, -7],
                    [0, 1, 1]])
        self.assertEqual(M[0, 0], -3)
        self.assertEqual(M[1, 1], -2)
        self.assertEqual(M[2, 2], 1)

    def test_matrix_equality_with_identical_matrices(self):
        A = Matrix([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 8, 7, 6],
                    [5, 4, 3, 2]])

        B = Matrix([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 8, 7, 6],
                    [5, 4, 3, 2]])

        self.assertEqual(A, B)

    def test_matrix_equality_with_different_matrices(self):
        A = [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 8, 7, 6],
             [5, 4, 3, 2]]

        B = [[2, 3, 4, 8],
             [6, 7, 8, 9],
             [8, 7, 6, 5],
             [4, 3, 2, 1]]

        self.assertNotEqual(A, B)

    def test_multiplying_two_matrices(self):
        A = Matrix([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 8, 7, 6],
                    [5, 4, 3, 2]])

        B = Matrix([[-2, 1, 2, 3],
                    [3, 2, 1, -1],
                    [4, 3, 6, 5],
                    [1, 2, 7, 8]])

        C = Matrix([[20, 22, 50, 48],
                    [44, 54, 114, 108],
                    [40, 58, 110, 102],
                    [16, 26, 46, 42]])
        self.assertEqual(A * B, C)

    def test_multiplying_matrix_by_tuple(self):
        A = Matrix([[1, 2, 3, 4],
                    [2, 4, 4, 2],
                    [8, 6, 4, 1],
                    [0, 0, 0, 1]])

        b = Tuple(1, 2, 3, 1)

        C = Tuple(18, 24, 33, 1)
        self.assertEqual(A * b, C)

    def test_multiplying_matrix_by_identity_matrix(self):
        A = Matrix([[0, 1, 2, 4],
                    [1, 2, 4, 8],
                    [2, 4, 8, 16],
                    [4, 8, 16, 32]])

        identity_matrix = Matrix([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

        self.assertEqual(A * identity_matrix, A)

    def test_multiplying_identity_matrix_by_tuple(self):

        identity_matrix = Matrix([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

        a = Tuple(1, 2, 3, 4)

        self.assertEqual(identity_matrix * a, a)

    def test_transposing_matrix(self):
        A = Matrix([[0, 9, 3, 0],
                    [9, 8, 0, 8],
                    [1, 8, 5, 3],
                    [0, 0, 5, 8]])

        self.assertEqual(transpose(A), Matrix([[0, 9, 1, 0],
                                               [9, 8, 8, 0],
                                               [3, 0, 5, 5],
                                               [0, 8, 3, 8]]))

    def test_transposing_identity_matrix_gives_identity_matrix(self):
        identity_matrix = Matrix([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
        A = transpose(identity_matrix)
        self.assertEqual(A, identity_matrix)

    def test_calculate_determinant_of_2x2_matrix(self):
        A = Matrix([[1, 5], [-3, 2]])

        self.assertFloatEqual(determinant(A), 17)

    def test_submatrix_of_3x3_matrix_is_a_2x2_matrix(self):
        A = Matrix([[1, 5, 0],
                    [-3, 2, 7],
                    [0, 6, 3]])
        self.assertEqual(submatrix(A, 0, 2), Matrix([[-3, 2],
                                                     [0, 6]]))

    def test_submatrix_of_4x4_matrix_is_a_3x3_matrix(self):
        A = Matrix([[-6, 1, 1, 6],
                    [-8, 5, 8, 6],
                    [-1, 0, 8, 2],
                    [-7, 1, -1, 1]])

        self.assertEqual(submatrix(A, 2, 1), Matrix([[-6, 1, 6],
                                                     [-8, 8, 6],
                                                     [-7, -1, 1]]))

    def test_calculate_minor_of_3x3_matrix(self):
        A = Matrix([[3, 5, 0],
                    [2, -1, -7],
                    [6, -1, 5]])

        self.assertFloatEqual(minor(A, 1, 0), 25)

    def test_calculate_cofactor_of_3x3_matrix(self):
        A = Matrix([[3, 5, 0],
                    [2, -1, -7],
                    [6, -1, 5]])

        self.assertFloatEqual(minor(A, 0, 0), -12)
        self.assertFloatEqual(cofactor(A, 0, 0), -12)
        self.assertFloatEqual(minor(A, 1, 0), 25)
        self.assertFloatEqual(cofactor(A, 1, 0), -25)

    def test_calculate_determinant_of_3x3_matrix(self):
        A = Matrix([[1, 2, 6],
                    [-5, 8, -4],
                    [2, 6, 4]])

        self.assertFloatEqual(cofactor(A, 0, 0), 56)
        self.assertFloatEqual(cofactor(A, 0, 1), 12)
        self.assertFloatEqual(cofactor(A, 0, 2), -46)
        self.assertFloatEqual(determinant(A), -196)

    def test_calculate_determinant_of_4x4_matrix(self):
        A = Matrix([[-2, -8, 3, 5],
                    [-3, 1, 7, 3],
                    [1, 2, -9, 6],
                    [-6, 7, 7, -9]])

        self.assertFloatEqual(cofactor(A, 0, 0), 690)
        self.assertFloatEqual(cofactor(A, 0, 1), 447)
        self.assertFloatEqual(cofactor(A, 0, 2), 210)
        self.assertFloatEqual(cofactor(A, 0, 3), 51)
        self.assertFloatEqual(determinant(A), -4071)

    def test_invertible_matrix_for_invertibility(self):
        A = Matrix([[6, 4, 4, 4],
                    [5, 5, 7, 6],
                    [4, -9, 3, -7],
                    [9, 1, 7, -6]])

        self.assertFloatEqual(determinant(A), -2120)
        self.assertEqual(A.invertible(), True)

    def test_noninvertible_matrix_for_invertibility(self):
        A = Matrix([[-4, 2, -2, -3],
                    [9, 6, 2, 6],
                    [0, -5, 1, -5],
                    [0, 0, 0, 0]])

        self.assertFloatEqual(determinant(A), 0)
        self.assertEqual(A.invertible(), False)

    def test_calc_inverse_of_matrix(self):
        A = Matrix([[-5, 2, 6, -8],
                    [1, -5, 1, 8],
                    [7, 7, -6, -7],
                    [1, -3, 7, 4]])

        B = inverse(A)
        self.assertEqual(B, Matrix([[0.21805, 0.45113, 0.24060, -0.04511],
                                    [-0.80827, -1.45677, -0.44361, 0.52068],
                                    [-0.07895, -0.22368, -0.05263, 0.19737],
                                    [-0.52256, -0.81391, -0.30075, 0.30639]]))
        self.assertFloatEqual(determinant(A), 532)
        self.assertFloatEqual(cofactor(A, 2, 3), -160)
        self.assertAlmostEqual(B[3, 2], -160 / 532, 5)
        self.assertFloatEqual(cofactor(A, 3, 2), 105)
        self.assertAlmostEqual(B[2, 3], 105 / 532, 5)

        A = Matrix([[8, -5, 9, 2],
                    [7, 5, 6, 1],
                    [-6, 0, 9, 6],
                    [-3, 0, -9, -4]])

        self.assertEqual(inverse(A), Matrix([[-0.15385, -0.15385, -0.28205, -0.53846],
                                             [-0.07692, 0.12308, 0.02564, 0.03077],
                                             [0.35897, 0.35897, 0.43590, 0.92308],
                                             [-0.69231, -0.69231, -0.76923, -1.92308]]))

        A = Matrix([[9, 3, 0, 9],
                    [-5, -2, -6, -3],
                    [-4, 9, 6, 4],
                    [-7, 6, 6, 2]])

        self.assertEqual(inverse(A), Matrix([[-0.04074, -0.07778, 0.14444, -0.22222],
                                             [-0.07778, 0.03333, 0.36667, -0.33333],
                                             [-0.02901, -0.14630, -0.10926, 0.12963],
                                             [0.17778, 0.06667, -0.26667, 0.33333]]))

    def test_multiply_product_by_inverse(self):
        A = Matrix([[3, -9, 7, 3],
                    [3, -8, 2, -9],
                    [-4, 4, 4, 1],
                    [-6, 5, -1, 1]])

        B = Matrix([[8, 2, 2, 2],
                    [3, -1, 7, 0],
                    [7, 0, 5, 4],
                    [6, -2, 0, 5]])

        C = A * B
        self.assertEqual(C * inverse(B), A)

    def test_multiply_by_translation_matrix(self):
        transform = translation(5, -3, 2)
        p = Point(-3, 4, 5)
        self.assertEqual(transform * p, Point(2, 1, 7))

    def test_multiply_by_inverse_matrix(self):
        transform = translation(5, -3, 2)
        inv = inverse(transform)
        p = Point(-3, 4, 5)
        self.assertEqual(inv * p, Point(-8, 7, 3))

    def test_translation_does_not_effect_others(self):
        transform = translation(5, -3, 2)
        v = Vector(-3, 4, 5)
        self.assertEqual(transform * v, v)

    def test_scaling_matrix_applied_to_point(self):
        transform = scaling(2, 3, 4)
        p = Point(-4, 6, 8)
        self.assertEqual(transform * p, Point(-8, 18, 32))

    def test_scaling_matrix_applied_to_vector(self):
        transform = scaling(2, 3, 4)
        v = Vector(-4, 6, 8)
        self.assertEqual(transform * v, Vector(-8, 18, 32))

    def test_multiply_by_the_inverse_of_scaling_matrix(self):
        transform = scaling(2, 3, 4)
        inv = inverse(transform)
        v = Vector(-4, 6, 8)
        self.assertEqual(inv * v, Vector(-2, 2, 2))

    def test_reflection_is_scaling_by_negative_value(self):
        transform = scaling(-1, 1, 1)
        p = Point(2, 3, 4)
        self.assertEqual(transform * p, Point(-2, 3, 4))

    def test_rotating_around_a_point_around_x(self):
        p = Point(0, 1, 0)
        half_quarter = rotation_x(math.pi / 4)
        full_quarter = rotation_x(math.pi / 2)
        self.assertEqual(half_quarter * p, Point(0, math.sqrt(2) / 2, math.sqrt(2) / 2))
        self.assertEqual(full_quarter * p, Point(0, 0, 1))

    def test_inverse_x_rotation_rotates_opposite_direction(self):
        p = Point(0, 1, 0)
        half_quarter = rotation_x(math.pi / 4)
        inv = inverse(half_quarter)
        self.assertEqual(inv * p, Point(0, math.sqrt(2) / 2, -math.sqrt(2) / 2))

    def test_rotating_around_a_point_around_y(self):
        p = Point(0, 0, 1)
        half_quarter = rotation_y(math.pi / 4)
        full_quarter = rotation_y(math.pi / 2)
        self.assertEqual(half_quarter * p, Point(math.sqrt(2) / 2, 0, math.sqrt(2) / 2))
        self.assertEqual(full_quarter * p, Point(1, 0, 0))

    def test_rotating_around_a_point_around_z(self):
        p = Point(0, 1, 0)
        half_quarter = rotation_z(math.pi / 4)
        full_quarter = rotation_z(math.pi / 2)
        self.assertEqual(half_quarter * p, Point(-math.sqrt(2) / 2, math.sqrt(2) / 2, 0))
        self.assertEqual(full_quarter * p, Point(-1, 0, 0))

    def test_shearing_moves_x_in_proportion_to_y(self):
        transform = shearing(1, 0, 0, 0, 0, 0)
        p = Point(2, 3, 4)
        self.assertEqual(transform * p, Point(5, 3, 4))

    def test_shearing_moves_x_in_proportion_to_z(self):
        transform = shearing(0, 1, 0, 0, 0, 0)
        p = Point(2, 3, 4)
        self.assertEqual(transform * p, Point(6, 3, 4))

    def test_shearing_moves_y_in_proportion_to_x(self):
        transform = shearing(0, 0, 1, 0, 0, 0)
        p = Point(2, 3, 4)
        self.assertEqual(transform * p, Point(2, 5, 4))

    def test_shearing_moves_y_in_proportion_to_z(self):
        transform = shearing(0, 0, 0, 1, 0, 0)
        p = Point(2, 3, 4)
        self.assertEqual(transform * p, Point(2, 7, 4))

    def test_shearing_moves_z_in_proportion_to_x(self):
        transform = shearing(0, 0, 0, 0, 1, 0)
        p = Point(2, 3, 4)
        self.assertEqual(transform * p, Point(2, 3, 6))

    def test_shearing_moves_z_in_proportion_to_y(self):
        transform = shearing(0, 0, 0, 0, 0, 1)
        p = Point(2, 3, 4)
        self.assertEqual(transform * p, Point(2, 3, 7))

    def test_individual_transformations_are_applied_in_sequence(self):
        p = Point(1, 0, 1)
        A = rotation_x(math.pi / 2)
        B = scaling(5, 5, 5)
        C = translation(10, 5, 7)

        # apply rotation first
        p2 = A * p
        self.assertEqual(p2, Point(1, -1, 0))

        # then apply scaling
        p3 = B * p2
        self.assertEqual(p3, Point(5, -5, 0))

        # then apply transformation
        p4 = C * p3
        self.assertEqual(p4, Point(15, 0, 7))

    def test_chained_transformation_must_be_applied_in_reverse_order(self):
        p = Point(1, 0, 1)
        A = rotation_x(math.pi / 2)
        B = scaling(5, 5, 5)
        C = translation(10, 5, 7)

        T = C * B * A
        self.assertEqual(T * p, Point(15, 0, 7))

    def test_chained_transformation_must_be_applied_in_reverse_order_using_builder(self):
        p = Point(1, 0, 1)
        # T = TransformationBuilder().rotate_x(math.pi / 2).scale(5,5,5).translate(10, 5, 7).build()
        T = TransformationBuilder().translate(10, 5, 7).scale(5, 5, 5).rotate_x(math.pi / 2).build()

        self.assertEqual(T * p, Point(15, 0, 7))

    def test_clock_playground(self):
        # twelve o clock (0,0,1)
        # r = rotation_y((math.pi / 6))
        # twelve = Point(0,0,1)
        # three = r * twelve
        # self.assertEqual(three, Point(1,0,0))

        # three o clock (1,0,0)
        r = rotation_y(3 * (math.pi / 6))
        twelve = Point(0, 0, 1)
        three = r * twelve
        self.assertEqual(three, Point(1, 0, 0))

        # six o clock (0,0,-1)
        r = rotation_y(6 * (math.pi / 6))
        twelve = Point(0, 0, 1)
        six = r * twelve
        self.assertEqual(six, Point(0, 0, -1))

        # nine o clock (-1,0,0)
        r = rotation_y(9 * (math.pi / 6))
        twelve = Point(0, 0, 1)
        nine = r * twelve
        self.assertEqual(nine, Point(-1, 0, 0))

    def test_creating_and_querying_a_ray(self):
        origin = Point(1, 2, 3)
        direction = Vector(4, 5, 6)
        r = Ray(origin, direction)
        self.assertEqual(r.origin, origin)
        self.assertEqual(r.direction, direction)

    def test_computing_point_from_distance(self):
        r = Ray(Point(2, 3, 4), Vector(1, 0, 0))
        self.assertEqual(position_along_ray(r, 0), Point(2, 3, 4))
        self.assertEqual(position_along_ray(r, 1), Point(3, 3, 4))
        self.assertEqual(position_along_ray(r, -1), Point(1, 3, 4))
        self.assertEqual(position_along_ray(r, 2.5), Point(4.5, 3, 4))

    def test_ray_intersects_a_sphere_at_two_points(self):
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = Sphere()
        xs = intersect(s, r)
        self.assertEqual(len(xs), 2)
        self.assertEqual(xs[0].t, 4.0)
        self.assertEqual(xs[1].t, 6.0)

    def test_ray_intersects_a_sphere_at_a_tangent(self):
        r = Ray(Point(0, 1, -5), Vector(0, 0, 1))
        s = Sphere()
        xs = intersect(s, r)
        self.assertEqual(len(xs), 2)
        self.assertEqual(xs[0].t, 5.0)
        self.assertEqual(xs[1].t, 5.0)

    def test_ray_misses_a_sphere(self):
        r = Ray(Point(0, 2, -5), Vector(0, 0, 1))
        s = Sphere()
        xs = intersect(s, r)
        self.assertEqual(len(xs), 0)

    def test_ray_originates_inside_a_sphere(self):
        r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
        s = Sphere()
        xs = intersect(s, r)
        self.assertEqual(len(xs), 2)
        self.assertEqual(xs[0].t, -1.0)
        self.assertEqual(xs[1].t, 1.0)

    def test_sphere_is_behind_a_ray(self):
        r = Ray(Point(0, 0, 5), Vector(0, 0, 1))
        s = Sphere()
        xs = intersect(s, r)
        self.assertEqual(len(xs), 2)
        self.assertEqual(xs[0].t, -6.0)
        self.assertEqual(xs[1].t, -4.0)

    def test_an_intersection_encapsulates_t_and_a_object(self):
        s = Sphere()
        i = Intersection(3.5, s)
        self.assertEqual(i.t, 3.5)
        self.assertEqual(i.s_object, s)

    def test_aggregating_intersections(self):
        s = Sphere()
        i1 = Intersection(1, s)
        i2 = Intersection(2, s)

        xs = intersections(i1, i2)
        self.assertEqual(len(xs), 2)
        self.assertEqual(xs[0].t, 1)
        self.assertEqual(xs[1].t, 2)

    def test_intersect_sets_the_object_on_the_intersection(self):
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = Sphere()
        xs = intersect(s, r)
        self.assertEqual(len(xs), 2)
        self.assertEqual(xs[0].s_object, s)
        self.assertEqual(xs[1].s_object, s)

    def test_the_hit_when_all_intersections_have_a_positive_t(self):
        s = Sphere()
        i1 = Intersection(1, s)
        i2 = Intersection(2, s)

        xs = intersections(i2, i1)

        i = hit(xs)
        self.assertEqual(i, i1)

    def test_the_hit_when_some_intersections_have_a_negative_t(self):
        s = Sphere()
        i1 = Intersection(-1, s)
        i2 = Intersection(1, s)

        xs = intersections(i2, i1)

        i = hit(xs)
        self.assertEqual(i, i2)

    def test_the_hit_when_all_intersections_have_a_negative_t(self):
        s = Sphere()
        i1 = Intersection(-2, s)
        i2 = Intersection(-1, s)

        xs = intersections(i2, i1)

        i = hit(xs)
        self.assertEqual(i, None)

    def test_the_hit_is_always_the_lowest_nonnegative_intersection(self):
        s = Sphere()
        i1 = Intersection(5, s)
        i2 = Intersection(7, s)
        i3 = Intersection(-3, s)
        i4 = Intersection(2, s)

        xs = intersections(i1, i2, i3, i4)

        i = hit(xs)
        self.assertEqual(i, i4)

    def test_translating_a_ray(self):
        r = Ray(Point(1, 2, 3), Vector(0, 1, 0))
        m = translation(3, 4, 5)
        r2 = transform(r, m)
        self.assertEqual(r2.origin, Point(4, 6, 8))
        self.assertEqual(r2.direction, Vector(0, 1, 0))

    def test_scaling_a_ray(self):
        r = Ray(Point(1, 2, 3), Vector(0, 1, 0))
        m = scaling(2, 3, 4)
        r2 = transform(r, m)
        self.assertEqual(r2.origin, Point(2, 6, 12))
        self.assertEqual(r2.direction, Vector(0, 3, 0))

    def test_shape_default_transformation(self):
        s = test_shape()
        identity_matrix = Matrix([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
        self.assertEqual(s.transform, identity_matrix)

    def test_change_shape_transformation(self):
        s = test_shape()
        t = translation(2, 3, 4)
        set_transform(s, t)
        self.assertEqual(s.transform, t)

    def test_intersecting_a_scaled_shape_with_a_ray(self):
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = test_shape()
        set_transform(s, scaling(2, 2, 2))
        xs = intersect(s, r)
        self.assertEqual(s.saved_ray.origin, Point(0, 0, -2.5))
        self.assertEqual(s.saved_ray.direction, Vector(0, 0, 0.5))

    def test_intersecting_a_translated_shape_with_a_ray(self):
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = test_shape()
        set_transform(s, translation(5, 0, 0))
        xs = intersect(s, r)
        self.assertEqual(s.saved_ray.origin, Point(-5, 0, -5))
        self.assertEqual(s.saved_ray.direction, Vector(0, 0, 1))

    def test_normal_on_a_sphere_at_a_point_on_the_x_axis(self):
        s = Sphere()
        n = normal_at(s, Point(1, 0, 0))
        self.assertEqual(n, Vector(1, 0, 0))

    def test_normal_on_a_sphere_at_a_point_on_the_y_axis(self):
        s = Sphere()
        n = normal_at(s, Point(0, 1, 0))
        self.assertEqual(n, Vector(0, 1, 0))

    def test_normal_on_a_sphere_at_a_point_on_the_z_axis(self):
        s = Sphere()
        n = normal_at(s, Point(0, 0, 1))
        self.assertEqual(n, Vector(0, 0, 1))

    def test_normal_on_a_sphere_at_a_nonaxial_point(self):
        s = Sphere()
        n = normal_at(s, Point(sqrt(3) / 3, sqrt(3) / 3, sqrt(3) / 3))
        self.assertEqual(n, Vector(sqrt(3) / 3, sqrt(3) / 3, sqrt(3) / 3))

    def test_normal_is_a_normalized_vector(self):
        s = Sphere()
        n = normal_at(s, Point(sqrt(3) / 3, sqrt(3) / 3, sqrt(3) / 3))
        self.assertEqual(n, normalize(n))

    def test_compute_normal_on_translated_shape(self):
        s = test_shape()
        set_transform(s, translation(0, 1, 0))
        n = normal_at(s, Point(0, 1.70711, -0.70711))
        self.assertEqual(n, Vector(0, 0.70711, -0.70711))

    def test_compute_normal_on_transformed_shape(self):
        s = test_shape()
        m = scaling(1, 0.5, 1) * rotation_z(math.pi / 5)
        set_transform(s, m)
        n = normal_at(s, Point(0, math.sqrt(2) / 2, -math.sqrt(2) / 2))
        self.assertEqual(n, Vector(0, 0.97014, -0.24254))

    def test_reflecting_vector_approaching_45_degrees(self):
        v = Vector(1, -1, 0)
        n = Vector(0, 1, 0)
        r = reflect(v, n)
        self.assertEqual(r, Vector(1, 1, 0))

    def test_reflecting_vector_off_slanted_surface(self):
        v = Vector(0, -1, 0)
        n = Vector(math.sqrt(2) / 2, math.sqrt(2) / 2, 0)
        r = reflect(v, n)
        self.assertEqual(r, Vector(1, 0, 0))

    def test_point_light_has_position_and_intensity(self):
        intensity = Color(1, 1, 1)
        position = Point(0, 0, 0)
        light = PointLight(position, intensity)
        self.assertEqual(light.position, position)
        self.assertEqual(light.intensity, intensity)

    def test_default_material(self):
        m = Material()
        self.assertEqual(m.color, Color(1, 1, 1))
        self.assertEqual(m.ambient, 0.1)
        self.assertEqual(m.diffuse, 0.9)
        self.assertEqual(m.specular, 0.9)
        self.assertEqual(m.shininess, 200.0)

    def test_shape_has_default_material(self):
        s = test_shape()
        m = s.material
        self.assertEqual(m, Material())

    def test_shape_may_be_assigned_a_material(self):
        s = test_shape()
        m = Material()
        m.ambient = 1
        s.material = m
        self.assertEqual(s.material, m)

    def test_lighting_with_eye_between_ligth_and_surface(self):
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, -10), Color(1, 1, 1))
        obj = Sphere()
        result = lighting(m, obj, light, position, eyev, normalv)
        self.assertEqual(result, Color(1.9, 1.9, 1.9))

    def test_lighting_with_eye_between_light_and_surface_eye_offset_45_degrees(self):
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, math.sqrt(2) / 2, -math.sqrt(2) / 2)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, -10), Color(1, 1, 1))
        obj = Sphere()
        result = lighting(m, obj, light, position, eyev, normalv)
        self.assertEqual(result, Color(1.0, 1.0, 1.0))

    def test_lighting_with_eye_opposite_surface_light_offset_45_degrees(self):
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 10, -10), Color(1, 1, 1))
        obj = Sphere()
        result = lighting(m, obj, light, position, eyev, normalv)
        self.assertEqual(result, Color(0.7364, 0.7364, 0.7364))

    def test_lighting_with_eye_in_path_of_the_reflection_vector(self):
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, -math.sqrt(2) / 2, -math.sqrt(2) / 2)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 10, -10), Color(1, 1, 1))
        obj = Sphere()
        result = lighting(m, obj, light, position, eyev, normalv)
        self.assertEqual(result, Color(1.6364, 1.6364, 1.6364))

    def test_lighting_with_light_behind_the_surface(self):
        m = Material()
        position = Point(0, 0, 0)
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, 10), Color(1, 1, 1))
        obj = Sphere()
        result = lighting(m, obj, light, position, eyev, normalv)
        self.assertEqual(result, Color(0.1, 0.1, 0.1))

    def test_create_a_world(self):
        w = World()
        self.assertEqual(w.objects, [])
        self.assertEqual(w.light, None)

    def test_default_world(self):
        w = default_world()

        self.assertEqual(w.objects[0].material.color, Color(0.8, 1.0, 0.6))
        self.assertEqual(w.objects[1].material.color, Color(1, 1, 1))

    def test_intersect_a_world_with_a_ray(self):
        w = default_world()

        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))

        xs = intersect_world(w, r)

        self.assertEqual(len(xs), 4)
        self.assertEqual(xs[0].t, 4)
        self.assertEqual(xs[1].t, 4.5)
        self.assertEqual(xs[2].t, 5.5)
        self.assertEqual(xs[3].t, 6)

    def test_precomute_state_of_an_intersection(self):
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = Sphere()
        i = Intersection(4, s)

        comps = prepare_computations(i, r)
        self.assertEqual(comps.t, i.t)
        self.assertEqual(comps.object, i.s_object)
        self.assertEqual(comps.point, Point(0, 0, -1))
        self.assertEqual(comps.eyev, Vector(0, 0, -1))
        self.assertEqual(comps.normalv, Vector(0, 0, -1))

    def test_when_an_intersection_occurs_outside(self):
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s = Sphere()
        i = Intersection(4, s)

        comps = prepare_computations(i, r)
        self.assertEqual(comps.inside, False)

    def test_when_an_intersection_occurs_inside(self):
        r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
        s = Sphere()
        i = Intersection(1, s)

        comps = prepare_computations(i, r)
        self.assertEqual(comps.point, Point(0, 0, 1))
        self.assertEqual(comps.eyev, Vector(0, 0, -1))
        self.assertEqual(comps.inside, True)
        self.assertEqual(comps.normalv, Vector(0, 0, -1))

    def test_shade_an_intersection(self):
        w = default_world()

        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))

        shape = w.objects[0]

        i = Intersection(4, shape)
        comps = prepare_computations(i, r)
        c = shade_hit(w, comps)
        self.assertEqual(c, Color(0.38066, 0.47583, 0.2855))

    def test_shade_an_intersection_from_the_inside(self):
        w = default_world()
        w.light = PointLight(Point(0, 0.25, 0), Color(1, 1, 1))
        r = Ray(Point(0, 0, 0), Vector(0, 0, 1))

        shape = w.objects[1]

        i = Intersection(0.5, shape)
        comps = prepare_computations(i, r)
        c = shade_hit(w, comps)
        self.assertEqual(c, Color(0.90498, 0.90498, 0.90498))

    def test_color_when_ray_misses(self):
        w = default_world()
        r = Ray(Point(0, 0, -5), Vector(0, 1, 0))
        c = color_at(w, r)
        self.assertEqual(c, Color(0, 0, 0))

    def test_color_when_ray_hit(self):
        w = default_world()
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        c = color_at(w, r)
        self.assertEqual(c, Color(0.38066, 0.47583, 0.2855))

    def test_color_with_intersection_behind_ray(self):
        w = default_world()
        outer = w.objects[0]
        outer.material.ambient = 1
        inner = w.objects[1]
        inner.material.ambient = 1
        r = Ray(Point(0, 0, 0.75), Vector(0, 0, -1))
        c = color_at(w, r)
        self.assertEqual(c, inner.material.color)

    def test_transformtion_matrix_for_default_orientation(self):
        from_param = Point(0, 0, 0)
        to_param = Point(0, 0, -1)
        up_param = Vector(0, 1, 0)

        t = view_transforfmation(from_param, to_param, up_param)
        self.assertEqual(t, Matrix([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]]))

    def test_transformtion_matrix_looking_in_positive_z_direction(self):
        from_param = Point(0, 0, 0)
        to_param = Point(0, 0, 1)
        up_param = Vector(0, 1, 0)

        t = view_transforfmation(from_param, to_param, up_param)
        self.assertEqual(t, scaling(-1, 1, -1))

    def test_transformtion_moves_the_world(self):
        from_param = Point(0, 0, 8)
        to_param = Point(0, 0, 0)
        up_param = Vector(0, 1, 0)

        t = view_transforfmation(from_param, to_param, up_param)
        self.assertEqual(t, translation(0, 0, -8))

    def test_arbitrary_view_transformation(self):
        from_param = Point(1, 3, 2)
        to_param = Point(4, -2, 8)
        up_param = Vector(1, 1, 0)

        t = view_transforfmation(from_param, to_param, up_param)
        self.assertEqual(t, Matrix([[-0.50709, 0.50709, 0.67612, -2.36643],
                                    [0.76772, 0.60609, 0.12122, -2.82843],
                                    [-0.35857, 0.59761, -0.71714, 0.00000],
                                    [0.00000, 0.00000, 0.00000, 1.00000]]))

    def test_constructing_a_camera(self):
        hsize = 160
        vsize = 120
        field_of_view = math.pi / 2

        c = Camera(hsize, vsize, field_of_view)

        self.assertEqual(c.hsize, 160)
        self.assertEqual(c.vsize, 120)
        self.assertEqual(c.field_of_view, math.pi / 2)
        self.assertEqual(c.transform, Matrix([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]]))

    def test_pixel_size_horizontal_canvas(self):
        c = Camera(200, 125, math.pi / 2)
        self.assertEqual(round(c.pixel_size, 2), 0.01)

    def test_pixel_size_vertical_canvas(self):
        c = Camera(125, 200, math.pi / 2)
        self.assertEqual(round(c.pixel_size, 2), 0.01)

    def test_construct_ray_through_center_of_canvas(self):
        c = Camera(201, 101, math.pi / 2)
        r = ray_for_pixel(c, 100, 50)
        self.assertEqual(r.origin, Point(0, 0, 0))
        self.assertEqual(r.direction, Vector(0, 0, -1))

    def test_construct_ray_through_corner_of_canvas(self):
        c = Camera(201, 101, math.pi / 2)
        r = ray_for_pixel(c, 0, 0)
        self.assertEqual(r.origin, Point(0, 0, 0))
        self.assertEqual(r.direction, Vector(0.66519, 0.33259, -0.66851))

    def test_construct_ray_when_the_camera_is_transformed(self):
        c = Camera(201, 101, math.pi / 2)
        c.transform = TransformationBuilder().rotate_y(math.pi / 4).translate(0, -2, 5).build()
        r = ray_for_pixel(c, 100, 50)
        self.assertEqual(r.origin, Point(0, 2, -5))
        self.assertEqual(r.direction, Vector(math.sqrt(2) / 2, 0, -math.sqrt(2) / 2))

    def test_rendering_a_world_with_a_camera(self):
        w = default_world()
        c = Camera(11, 11, math.pi / 2)
        from_vector = Point(0, 0, -5)
        to_vector = Point(0, 0, 0)
        up_vector = Vector(0, 1, 0)

        c.transform = view_transforfmation(from_vector, to_vector, up_vector)

        image = render(c, w)
        self.assertEqual(pixel_at(image, 5, 5), Color(0.38066, 0.47583, 0.2855))

    def test_lighting_with_the_surface_in_shadow(self):
        eyev = Vector(0, 0 - 1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, -10), Color(1, 1, 1))
        in_shadow = True
        m = Material()
        p = Point(0, 0, 0)
        obj = Sphere()
        result = lighting(m, obj, light, p, eyev, normalv, in_shadow)
        self.assertEqual(result, Color(0.1, 0.1, 0.1))

    def test_no_shadow_when_nothing_is_collinear_with_point_and_light(self):
        w = default_world()
        p = Point(0, 10, 0)
        self.assertEqual(is_shadowed(w, p), False)

    def test_shadow_when_an_object_between_the_point_and_the_light(self):
        w = default_world()
        p = Point(10, -10, 10)
        self.assertEqual(is_shadowed(w, p), True)

    def test_no_shadow_when_object_is_behing_the_light(self):
        w = default_world()
        p = Point(-20, 20, -20)
        self.assertEqual(is_shadowed(w, p), False)

    def test_no_shadow__when_object_is_behind_the_point(self):
        w = default_world()
        p = Point(-2, 2, -2)
        self.assertEqual(is_shadowed(w, p), False)

    def test_shade_hit_is_given_an_intersection_in_shadow(self):
        w = World()
        w.light = PointLight(Point(0, 0, -10), Color(1, 1, 1))

        s1 = Sphere()

        w.objects.append(s1)

        s2 = Sphere()
        s2.transform = translation(0, 0, 10)

        w.objects.append(s2)

        r = Ray(Point(0, 0, 5), Vector(0, 0, 1))
        i = Intersection(4, s2)

        comps = prepare_computations(i, r)

        c = shade_hit(w, comps)

        self.assertEqual(c, Color(0.1, 0.1, 0.1))

    def test_hit_should_offset_point(self):
        r = Ray(Point(0, 0, -5), Vector(0, 0, 1))
        s1 = Sphere()
        s1.transform = translation(0, 0, 1)

        i = Intersection(5, s1)

        comps = prepare_computations(i, r)

        self.assertEqual(comps.over_point.tuple.z < -EPSILON / 2, True)
        self.assertEqual(comps.point.tuple.z > comps.over_point.tuple.z, True)

    def test_is_sphere_a_shape(self):
        self.assertTrue(issubclass(Sphere, Shape))

    def test_normal_plane_is_constant_everywhere(self):
        p = Plane()
        n1 = p.local_normal_at(Point(0, 0, 0))
        n2 = p.local_normal_at(Point(10, 0, -10))
        n3 = p.local_normal_at(Point(-5, 0, 150))

        self.assertEqual(n1, Vector(0, 1, 0))
        self.assertEqual(n2, Vector(0, 1, 0))
        self.assertEqual(n3, Vector(0, 1, 0))

    def test_intersect_with_ray_parallel_to_plane(self):
        p = Plane()
        r = Ray(Point(0, 10, 0), Vector(0, 0, 1))
        xs = p.local_intersect(r)
        self.assertEqual(xs, [])

    def test_intersect_with_coplanar_ray(self):
        p = Plane()
        r = Ray(Point(0, 0, 0), Vector(0, 0, 1))
        xs = p.local_intersect(r)
        self.assertEqual(xs, [])

    def test_ray_intersecting_a_plane_from_above(self):
        p = Plane()
        r = Ray(Point(0, 1, 0), Vector(0, -1, 0))
        xs = p.local_intersect(r)
        self.assertEqual(len(xs), 1)
        self.assertEqual(xs[0].t, 1)
        self.assertEqual(xs[0].s_object, p)

    def test_ray_intersecting_a_plane_from_below(self):
        p = Plane()
        r = Ray(Point(0, -1, 0), Vector(0, 1, 0))
        xs = p.local_intersect(r)
        self.assertEqual(len(xs), 1)
        self.assertEqual(xs[0].t, 1)
        self.assertEqual(xs[0].s_object, p)

    def test_create_a_stripe_pattern(self):
        black = Color(0, 0, 0)
        white = Color(1, 1, 1)
        pattern = stripe_pattern(white, black)

        self.assertEqual(pattern.a, white)
        self.assertEqual(pattern.b, black)

    def test_stripe_pattern_constant_in_y(self):
        black = Color(0, 0, 0)
        white = Color(1, 1, 1)
        pattern = stripe_pattern(white, black)
        self.assertEqual(stripe_at(pattern, Point(0, 0, 0)), white)
        self.assertEqual(stripe_at(pattern, Point(0, 1, 0)), white)
        self.assertEqual(stripe_at(pattern, Point(0, 2, 0)), white)

    def test_stripe_pattern_constant_in_z(self):
        black = Color(0, 0, 0)
        white = Color(1, 1, 1)
        pattern = stripe_pattern(white, black)
        self.assertEqual(stripe_at(pattern, Point(0, 0, 0)), white)
        self.assertEqual(stripe_at(pattern, Point(0, 1, 0)), white)
        self.assertEqual(stripe_at(pattern, Point(0, 2, 0)), white)

    def test_stripe_pattern_alternates_in_x(self):
        black = Color(0, 0, 0)
        white = Color(1, 1, 1)
        pattern = stripe_pattern(white, black)
        self.assertEqual(stripe_at(pattern, Point(0, 0, 0)), white)
        self.assertEqual(stripe_at(pattern, Point(0.9, 1, 0)), white)
        self.assertEqual(stripe_at(pattern, Point(1, 0, 0)), black)
        self.assertEqual(stripe_at(pattern, Point(-0.1, 0, 0)), black)
        self.assertEqual(stripe_at(pattern, Point(-1, 0, 0)), black)
        self.assertEqual(stripe_at(pattern, Point(-1.1, 0, 0)), white)

    def test_lighting_with_pattern_applied(self):
        m = Material()
        black = Color(0, 0, 0)
        white = Color(1, 1, 1)
        m.pattern = stripe_pattern(white, black)
        m.ambient = 1
        m.diffuse = 0
        m.specular = 0
        eyev = Vector(0, 0, -1)
        normalv = Vector(0, 0, -1)
        light = PointLight(Point(0, 0, -10), Color(1, 1, 1))
        obj = Sphere()
        c1 = lighting(m, obj, light, Point(0.9, 0, 0), eyev, normalv, False)
        c2 = lighting(m, obj, light, Point(1.1, 0, 0), eyev, normalv, False)
        self.assertEqual(c1, white)
        self.assertEqual(c2, black)

    def test_stripes_with_an_object_transformation(self):
        obj = Sphere()
        set_transform(obj, scaling(2, 2, 2))
        pattern = test_pattern()
        c = pattern_at_shape(pattern, obj, Point(2, 3, 4))
        self.assertEqual(c, Color(1, 1.5, 2))

    def test_stripes_with_an_pattern_transformation(self):
        obj = Sphere()
        pattern = test_pattern()
        set_pattern_transform(pattern, scaling(2, 2, 2))
        c = pattern_at_shape(pattern, obj, Point(2, 3, 4))
        self.assertEqual(c, Color(1, 1.5, 2))

    def test_stripes_with_both_object_and_pattern_transformation(self):
        obj = Sphere()
        set_transform(obj, scaling(2, 2, 2))
        pattern = test_pattern()
        set_pattern_transform(pattern, translation(0.5, 1, 1.5))
        c = pattern_at_shape(pattern, obj, Point(2.5, 3, 3.5))
        self.assertEqual(c, Color(0.75, 0.5, 0.25))

    def test_default_pattern_transform(self):
        pattern = test_pattern()
        pattern.transform = Matrix([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

    def test_assigning_a_transformation(self):
        pattern = test_pattern()
        set_pattern_transform(pattern, translation(1, 2, 3))
        self.assertEqual(pattern.transform, translation(1, 2, 3))

    def test_gradient_linearly_interpolates_between_colors(self):
        black = Color(0, 0, 0)
        white = Color(1, 1, 1)
        pattern = gradient_pattern(white, black)
        self.assertEqual(pattern.pattern_at(Point(0, 0, 0)), white)
        self.assertEqual(pattern.pattern_at(Point(0.25, 0, 0)), Color(0.75, 0.75, 0.75))
        self.assertEqual(pattern.pattern_at(Point(0.5, 0, 0)), Color(0.5, 0.5, 0.5))
        self.assertEqual(pattern.pattern_at(Point(0.75, 0, 0)), Color(0.25, 0.25, 0.25))

    def test_ring_should_extend_in_both_x_and_z(self):
        black = Color(0, 0, 0)
        white = Color(1, 1, 1)
        pattern = ring_pattern(white, black)
        self.assertEqual(pattern.pattern_at(Point(0, 0, 0)), white)
        self.assertEqual(pattern.pattern_at(Point(1, 0, 0)), black)
        self.assertEqual(pattern.pattern_at(Point(0, 0, 1)), black)
        self.assertEqual(pattern.pattern_at(Point(0.708, 0, 0.708)), black)

    def test_checkers_should_repeat_in_x(self):
        black = Color(0, 0, 0)
        white = Color(1, 1, 1)
        pattern = checker_pattern(white, black)
        self.assertEqual(pattern.pattern_at(Point(0, 0, 0)), white)
        self.assertEqual(pattern.pattern_at(Point(0.99, 0, 0)), white)
        self.assertEqual(pattern.pattern_at(Point(1.01, 0, 0)), black)

    def test_checkers_should_repeat_in_y(self):
        black = Color(0, 0, 0)
        white = Color(1, 1, 1)
        pattern = checker_pattern(white, black)
        self.assertEqual(pattern.pattern_at(Point(0, 0, 0)), white)
        self.assertEqual(pattern.pattern_at(Point(0, 0.99, 0)), white)
        self.assertEqual(pattern.pattern_at(Point(0, 1.01, 0)), black)

    def test_checkers_should_repeat_in_z(self):
        black = Color(0, 0, 0)
        white = Color(1, 1, 1)
        pattern = checker_pattern(white, black)
        self.assertEqual(pattern.pattern_at(Point(0, 0, 0)), white)
        self.assertEqual(pattern.pattern_at(Point(0, 0, 0.99)), white)
        self.assertEqual(pattern.pattern_at(Point(0, 0, 1.01)), black)

    def test_reflectivity_for_default(self):
        m = Material()
        self.assertEqual(m.reflective, 0.0)

    def test_precompute_reflection_vector(self):
        shape = Plane()
        r = Ray(Point(0,1,-1), Vector(0, -math.sqrt(2) / 2, math.sqrt(2) / 2 )   )
        i = Intersection(math.sqrt(2), shape)
        comps = prepare_computations(i, r)
        self.assertEqual(comps.reflectv,Vector(0, math.sqrt(2) / 2, math.sqrt(2) / 2))




if __name__ == '__main__':
    unittest.main()
