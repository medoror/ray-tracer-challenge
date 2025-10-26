from utils import auto_str, EPSILON
from math import fabs
from utils import MATRIX_EPSILON
import copy

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
        self.matrix = matrix

    def __len__(self):
        if type(self.matrix) is list:
            return len(self.matrix)

    def __getitem__(self, indices):
        if type(indices) is tuple:
            return self.matrix[indices[0]][indices[1]]
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
        subRow = []
        returnMatrix = []
        for i in range(len(m1)):
            for j in range(len(m2[0])):
                sums = 0
                for k in range(len(m2)):
                    sums = sums + (m1[i][k] * m2[k][j])
                subRow.append(sums)
            returnMatrix.append(subRow)
            subRow = []
        return Matrix(returnMatrix)

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
    det = 0
    if len(matrix) == 2:
        det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    else:
        for column in range(len(matrix)):
            det = det + matrix[0, column] * cofactor(matrix, 0, column)
    return det


# Can this be done better?
# more pythonic?
def submatrix(matrix, row, col):
    # Build submatrix manually without expensive deepcopy
    size = len(matrix)
    result = []

    for r in range(size):
        if r == row:
            continue  # Skip the excluded row
        new_row = []
        for c in range(size):
            if c == col:
                continue  # Skip the excluded column
            new_row.append(matrix[r][c])
        result.append(new_row)

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

    rows = len(matrix)
    columns = len(matrix[0])
    determinant_value = determinant(matrix)
    return_matrix = generate_zero_matrix(rows)

    for row in range(rows):
        for col in range(columns):
            c = cofactor(matrix, row, col)
            # the book used floating point nums to 5 decimals.  I dont think we care here
            # so the rounding should be dropped. Work towards have a better equals method
            return_matrix[col][row] = c / determinant_value

    return return_matrix

# think about throwing error if vector is not passed in
def dot(a, b):
    return a.tuple.x * b.tuple.x + a.tuple.y * b.tuple.y + a.tuple.z * b.tuple.z
