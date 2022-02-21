import unittest
import tempfile

from rayMath import Color, Matrix, Tuple, Point, Vector, \
    magnitude, cross, dot, normalize, Canvas, write_pixel, \
    pixel_at, canvas_to_ppm, transpose, determinant, submatrix
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
        self.assertEqual(submatrix(A, 0, 2), Matrix([[-3, 2], [0, 6]]))

    def test_submatrix_of_4x4_matrix_is_a_3x3_matrix(self):
        A = Matrix([[-6, 1, 1, 6],
                    [-8, 5, 8, 6],
                    [-1, 0, 8, 2],
                    [-7, 1, -1, 1]])

        self.assertEqual(submatrix(A, 2, 1), Matrix([[-6, 1, 6],
                                                     [-8, 8, 6],
                                                     [-7, -1, 1]]))


if __name__ == '__main__':
    unittest.main()
