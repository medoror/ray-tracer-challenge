import math
import unittest
import tempfile

from rayMath import Color, Matrix, Tuple, Point, Vector, \
    magnitude, cross, dot, normalize, Canvas, write_pixel, \
    pixel_at, canvas_to_ppm, transpose, determinant, submatrix, \
    minor, cofactor, inverse, translation, scaling, rotation_x, \
    rotation_y, rotation_z, shearing
from math import sqrt


class TestRayMath(unittest.TestCase):
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

        print(c1 - c2)
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
            print(contents)
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

        self.assertEqual(determinant(A), 17)

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

        self.assertEqual(minor(A, 1, 0), 25)

    def test_calculate_cofactor_of_3x3_matrix(self):
        A = Matrix([[3, 5, 0],
                    [2, -1, -7],
                    [6, -1, 5]])

        self.assertEqual(minor(A, 0, 0), -12)
        self.assertEqual(cofactor(A, 0, 0), -12)
        self.assertEqual(minor(A, 1, 0), 25)
        self.assertEqual(cofactor(A, 1, 0), -25)

    def test_calculate_determinant_of_3x3_matrix(self):
        A = Matrix([[1, 2, 6],
                    [-5, 8, -4],
                    [2, 6, 4]])

        self.assertEqual(cofactor(A, 0, 0), 56)
        self.assertEqual(cofactor(A, 0, 1), 12)
        self.assertEqual(cofactor(A, 0, 2), -46)
        self.assertEqual(determinant(A), -196)

    def test_calculate_determinant_of_4x4_matrix(self):
        A = Matrix([[-2, -8, 3, 5],
                    [-3, 1, 7, 3],
                    [1, 2, -9, 6],
                    [-6, 7, 7, -9]])

        self.assertEqual(cofactor(A, 0, 0), 690)
        self.assertEqual(cofactor(A, 0, 1), 447)
        self.assertEqual(cofactor(A, 0, 2), 210)
        self.assertEqual(cofactor(A, 0, 3), 51)
        self.assertEqual(determinant(A), -4071)

    def test_invertible_matrix_for_invertibility(self):
        A = Matrix([[6, 4, 4, 4],
                    [5, 5, 7, 6],
                    [4, -9, 3, -7],
                    [9, 1, 7, -6]])

        self.assertEqual(determinant(A), -2120)
        self.assertEqual(A.invertible(), True)

    def test_noninvertible_matrix_for_invertibility(self):
        A = Matrix([[-4, 2, -2, -3],
                    [9, 6, 2, 6],
                    [0, -5, 1, -5],
                    [0, 0, 0, 0]])

        self.assertEqual(determinant(A), 0)
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
        self.assertEqual(determinant(A), 532)
        self.assertEqual(cofactor(A, 2, 3), -160)
        self.assertEqual(B[3, 2], -160 / 532, 5)
        self.assertEqual(cofactor(A, 3, 2), 105)
        self.assertEqual(B[2, 3], 105 / 532, 5)

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


if __name__ == '__main__':
    unittest.main()
