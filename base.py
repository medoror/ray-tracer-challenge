from utils import auto_str, EPSILON
from math import fabs
from utils import MATRIX_EPSILON
import numpy as np

@auto_str
class Tuple:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def type(self):
        return "point" if self.w == 1.0 else "vector"

    def __neg__(self):
        return Tuple(-self.x, -self.y, -self.z, -self.w)

    def __eq__(self, other):
        if isinstance(other, Tuple):
            # return self.x == other.x and self.y == other.y and self.z == other.z and self.w == other.w
            return Tuple.equals(self.x, other.x) and Tuple.equals(self.y, other.y) and Tuple.equals(self.z, other.z) \
                and Tuple.equals(self.w, other.w)
        return False

    # can I create my own floating point and override equal?
    @staticmethod
    def equals(a, b):
        return fabs(a - b) < EPSILON

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        w = self.w + other.w
        return Tuple(x, y, z, w)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        w = self.w - other.w
        return Tuple(x, y, z, w)

    def __mul__(self, scalar):
        x = self.x * scalar
        y = self.y * scalar
        z = self.z * scalar
        w = self.w * scalar
        return Tuple(x, y, z, w)

    def __truediv__(self, scalar):
        x = self.x / scalar
        y = self.y / scalar
        z = self.z / scalar
        w = self.w / scalar
        return Tuple(x, y, z, w)


@auto_str
class Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.tuple = Tuple()
        self.tuple.x = x
        self.tuple.y = y
        self.tuple.z = z
        self.tuple.w = 1.0

    def __sub__(self, other):
        t1 = self.tuple - other.tuple
        return Vector(t1.x, t1.y, t1.z)

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.tuple == other.tuple
        return False

    def __add__(self, other):
        if isinstance(other, Vector):
            t1 = self.tuple + other.tuple
            return Point(t1.x, t1.y, t1.z)
        else:
            raise TypeError("Points cannot be added to points")


# TOOD: Think about extending tuple to create a point and vector instead of composition
@auto_str
class Vector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.tuple = Tuple()
        self.tuple.x = x
        self.tuple.y = y
        self.tuple.z = z
        self.tuple.w = 0.0

    def __add__(self, other):
        t1 = self.tuple + other.tuple
        return Vector(t1.x, t1.y, t1.z)

    def __sub__(self, other):
        t1 = self.tuple - other.tuple
        return Vector(t1.x, t1.y, t1.z)

    def __mul__(self, other):
        t1 = self.tuple * other
        return Vector(t1.x, t1.y, t1.z)

    def __neg__(self):
        t = Tuple(self.tuple.x, self.tuple.y, self.tuple.z, self.tuple.w)
        return Vector(-t.x, -t.y, -t.z)

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.tuple == other.tuple
        return False


@auto_str
class Color:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.tuple = Tuple()
        self.tuple.x = x
        self.tuple.y = y
        self.tuple.z = z
        self.tuple.w = 0.0

        self.red = self.tuple.x
        self.green = self.tuple.y
        self.blue = self.tuple.z

    def __eq__(self, other):
        if isinstance(other, Color):
            return self.tuple == other.tuple
        return False

    def __add__(self, other):
        t1 = self.tuple + other.tuple
        return Color(t1.x, t1.y, t1.z)

    def __sub__(self, other):
        t1 = self.tuple - other.tuple
        return Color(t1.x, t1.y, t1.z)

    def __mul__(self, other):
        if isinstance(other, Color):
            return self.hadamard_product_color(self.tuple, other.tuple)
        else:
            t1 = self.tuple * other
            return Color(t1.x, t1.y, t1.z)

    @staticmethod
    def hadamard_product_color(c1, c2):
        red = c1.x * c2.x
        green = c1.y * c2.y
        blue = c1.z * c2.z
        return Color(red, green, blue)

@auto_str
class Material:
    def __init__(self):
        self.color = Color(1, 1, 1)
        self.ambient = 0.1
        self.diffuse = 0.9
        self.specular = 0.9
        self.shininess = 200.0
        self.pattern = None
        self.reflective = 0.0

    def __eq__(self, other):
        if isinstance(other, Material):
            return self.color == other.color and \
                self.ambient == other.ambient and \
                self.diffuse == other.diffuse and \
                self.specular == other.specular and \
                self.shininess == other.shininess

        return False

@auto_str
class Intersection:
    def __init__(self, distance_t, s_object):
        self.t = distance_t
        self.s_object = s_object  # s_object -> x_object??


@auto_str
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction


@auto_str
class Matrix:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=np.float64)

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, indices):
        if type(indices) is tuple:
            return self.matrix[indices[0], indices[1]]
        else:
            return self.matrix[indices]

    def __eq__(self, other):
        if isinstance(other, Matrix):
            # assumes matricies are the same size!
            rows = len(self.matrix)
            columns = len(self.matrix[0])

            for row in range(rows):
                for col in range(columns):
                    if not Matrix.equals(self.matrix[row][col], other.matrix[row][col]):
                        return False
            return True
        return False

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return self.matrix_multiply(self.matrix, other.matrix)

        if isinstance(other, Tuple):
            convert_to_matrix = Matrix([[other.x],
                                        [other.y],
                                        [other.z],
                                        [other.w]])
            new_matrix = self.matrix_multiply(self.matrix, convert_to_matrix.matrix)
            return Tuple(new_matrix[0, 0],
                         new_matrix[1, 0],
                         new_matrix[2, 0],
                         new_matrix[3, 0])

        if isinstance(other, Point):
            convert_to_matrix = Matrix([[other.tuple.x],
                                        [other.tuple.y],
                                        [other.tuple.z],
                                        [other.tuple.w]])
            new_matrix = self.matrix_multiply(self.matrix, convert_to_matrix.matrix)
            return Point(new_matrix[0, 0],
                         new_matrix[1, 0],
                         new_matrix[2, 0])

        if isinstance(other, Vector):
            convert_to_matrix = Matrix([[other.tuple.x],
                                        [other.tuple.y],
                                        [other.tuple.z],
                                        [other.tuple.w]])
            new_matrix = self.matrix_multiply(self.matrix, convert_to_matrix.matrix)
            return Vector(new_matrix[0, 0],
                          new_matrix[1, 0],
                          new_matrix[2, 0])

    def invertible(self):
        return determinant(self) != 0

    @staticmethod
    def equals(a, b):
        return fabs(a - b) < MATRIX_EPSILON

    @staticmethod
    def matrix_multiply(m1, m2):
        # m1 and m2 are already numpy arrays from Matrix.matrix
        result = np.dot(m1, m2)
        return Matrix(result)

def transpose(matrix):
    rows = len(matrix)
    columns = len(matrix[0])

    return_matrix = []
    for j in range(columns):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        return_matrix.append(row)
    return Matrix(return_matrix)


def determinant(matrix):
    return np.linalg.det(matrix.matrix)


def submatrix(matrix, row, col):
    # Use NumPy boolean indexing for fast submatrix creation
    np_matrix = matrix.matrix
    row_mask = np.ones(np_matrix.shape[0], dtype=bool)
    col_mask = np.ones(np_matrix.shape[1], dtype=bool)
    row_mask[row] = False
    col_mask[col] = False
    result = np_matrix[np.ix_(row_mask, col_mask)]
    return Matrix(result)

def generate_zero_matrix(square_size):
    if square_size == 2:
        return Matrix([[0, 0],
                       [0, 0],
                       [0, 0],
                       [0, 0]])
    elif square_size == 3:
        return Matrix([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    elif square_size == 4:
        return Matrix([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]])
    else:
        raise ValueError("Unsupported matrix size")

def minor(matrix, row, column):
    A = submatrix(matrix, row, column)
    return determinant(A)


def cofactor(matrix, row, column):
    m = minor(matrix, row, column)
    return m * -1 if (row + column) % 2 != 0 else m

def inverse(matrix):
    if not matrix.invertible():
        raise ValueError("Matrix is not invertible")

    inv_matrix = np.linalg.inv(matrix.matrix)
    return Matrix(inv_matrix)

# think about throwing error if vector is not passed in
def dot(a, b):
    return a.tuple.x * b.tuple.x + a.tuple.y * b.tuple.y + a.tuple.z * b.tuple.z
