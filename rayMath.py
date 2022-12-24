import math
from math import sqrt, fabs, ceil

import sys

import copy

TUPLE_EPSILON = 0.0001
MATRIX_EPSILON = 0.01
MAX_CHARACTER_LENGTH = 70

class Intersection:
    def __init__(self, distance_t, s_object):
        self.t = distance_t
        self.s_object = s_object

class Sphere:
    def __init__(self):
        self.transform = Matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

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

    # Will i really need to convert float matrix to an int matrix?
    # @staticmethod
    # def float_matrix_to_int(matrix):
    #     return Matrix([[float(y) for y in x] for x in matrix.matrix])


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
        return fabs(a - b) < TUPLE_EPSILON

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

    # Untested!!
    def __str__(self):
        return "Tuple: ({0},{1},{2})".format(self.x, self.y, self.z, self.w)


class Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.tuple = Tuple()
        self.tuple.x = x
        self.tuple.y = y
        self.tuple.z = z
        self.tuple.w = 1.0

    def __str__(self):
        return "Point: ({0},{1}, {2})".format(self.tuple.x, self.tuple.y, self.tuple.z)

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

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.tuple == other.tuple
        return False

    # Untested!!
    def __str__(self):
        return "Vector: ({0},{1},{2})".format(self.tuple.x, self.tuple.y, self.tuple.z, self.tuple.w)


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

    def __str__(self):
        return "Color: ({0},{1},{2})".format(self.tuple.x, self.tuple.y, self.tuple.z, self.tuple.w)


class Canvas:
    def __init__(self, width, height, fill=Color(0, 0, 0), max_color_value=255):
        self.width = width
        self.height = height
        self.pixels = [[fill for x in range(width)] for y in range(height)]
        self.max_color_value = max_color_value

# TODO: write a variant of this that takes a point
def write_pixel(canvas, x, y, color):
    # print(x, y)
    canvas.pixels[y][x] = color  # python y address major


def pixel_at(canvas, x, y):
    return canvas.pixels[y][x]


def canvas_to_ppm(canvas, file_path="canvas.ppm"):
    with open(file_path, "w") as file:
        file.write("P3\n")
        file.write("{0} {1}\n".format(canvas.width, canvas.height))
        file.write("{0}\n".format(canvas.max_color_value))

        charsLength = 0
        for row in range(canvas.height):
            for col in range(canvas.width):
                color = canvas.pixels[row][col]
                ppm_red = "{0} ".format(calc_ppm_value(color.tuple.x))
                ppm_green = "{0} ".format(calc_ppm_value(color.tuple.y))
                ppm_blue = "{0} ".format(calc_ppm_value(color.tuple.z))

                vals = ppm_red, ppm_green, ppm_blue
                for val in vals:
                    if charsLength + len(val) <= MAX_CHARACTER_LENGTH:
                        file.write(val)
                        charsLength += len(val)
                    else:
                        file.write("\n")
                        charsLength = 0
                        file.write(val)
                        charsLength += len(val)
            file.write("\n")
            charsLength = 0
        file.write("\n")


def calc_ppm_value(color_value, max_value=255):
    if color_value >= 1:
        return max_value
    elif color_value <= 0:
        return 0
    else:
        return int(ceil(max_value * color_value))


def magnitude(vector):
    return sqrt(vector.tuple.x * vector.tuple.x + \
                vector.tuple.y * vector.tuple.y + \
                vector.tuple.z * vector.tuple.z + \
                vector.tuple.w * vector.tuple.w)


# Will not compute w component as it is not needed
# Will also just create a vector here for the time being
# think about throwing error if vector is not passed in
def normalize(vector):
    return Vector(vector.tuple.x / magnitude(vector),
                  vector.tuple.y / magnitude(vector),
                  vector.tuple.z / magnitude(vector))


# think about throwing error if vector is not passed in
def dot(a, b):
    return a.tuple.x * b.tuple.x + a.tuple.y * b.tuple.y + a.tuple.z * b.tuple.z


def cross(a, b):
    return Vector(a.tuple.y * b.tuple.z - a.tuple.z * b.tuple.y,
                  a.tuple.z * b.tuple.x - a.tuple.x * b.tuple.z,
                  a.tuple.x * b.tuple.y - a.tuple.y * b.tuple.x)


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
    # make a deep copy of the matrix which will be returned
    return_matrix = copy.deepcopy(matrix.matrix)
    # remove the row first
    return_matrix.remove(matrix[row])
    # delete the columns cells in place
    columns = len(matrix[0]) - 1
    for i in range(columns):
        del (return_matrix[i][col])

    return Matrix(return_matrix)


def minor(matrix, row, column):
    A = submatrix(matrix, row, column)
    return determinant(A)


def cofactor(matrix, row, column):
    m = minor(matrix, row, column)
    return m * -1 if (row + column) % 2 != 0 else m


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


def inverse(matrix):
    if not matrix.invertible():
        raise ValueError("Matrix is not invertible")

    rows = len(matrix)
    columns = len(matrix[0])
    determinantValue = determinant(matrix)
    return_matrix = generate_zero_matrix(rows)

    for row in range(rows):
        for col in range(columns):
            c = cofactor(matrix, row, col)
            # the book used floating point nums to 5 decimals.  I dont think we care here
            # so the rounding should be dropped. Work towards have a better equals method
            return_matrix[col][row] = c / determinantValue

    return return_matrix

class TransformationBuilder:
    def __init__(self):
        self.operations = []
 
    # is this correct?
    def identity(self):
        self.operations.append(Matrix([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]]) )
        return self

    def rotate_x(self, radius):
        self.operations.append(rotation_x(radius))
        return self

    def rotate_y(self, radius):
        self.operations.append(rotation_y(radius))
        return self
     
    def rotate_z(self, radius):
        self.operations.append(rotation_z(radius))
        return self

    def scale(self,x,y,z):
        self.operations.append(scaling(x,y,z))
        return self

    def shear(self, xy, xz, yx, yz, zx, zy):
        self.operations.append(shearing(xy, xz, yx, yz, zx, zy))
        return self

    def translate(self,x,y,z):
        self.operations.append(translation(x,y,z))
        return self

    def build(self):
        if not self.operations:
            raise Exception("Gotta add things before you build!!")
        else:
            current = self.operations.pop()
            while self.operations:
                next = self.operations.pop()
                current = current * next
            return current

def translation(x, y, z):
    return Matrix([[1, 0, 0, x],
                   [0, 1, 0, y],
                   [0, 0, 1, z],
                   [0, 0, 0, 1]])


def scaling(x, y, z):
    return Matrix([[x, 0, 0, 0],
                   [0, y, 0, 0],
                   [0, 0, z, 0],
                   [0, 0, 0, 1]])


def rotation_x(radians):
    return Matrix([[1, 0, 0, 0],
                   [0, math.cos(radians), -math.sin(radians), 0],
                   [0, math.sin(radians), math.cos(radians), 0],
                   [0, 0, 0, 1]])


def rotation_y(radians):
    return Matrix([[math.cos(radians), 0, math.sin(radians), 0],
                   [0, 1, 0, 0],
                   [-math.sin(radians), 0, math.cos(radians), 0],
                   [0, 0, 0, 1]])


def rotation_z(radians):
    return Matrix([[math.cos(radians), -math.sin(radians), 0, 0],
                   [math.sin(radians), math.cos(radians), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])


def shearing(xy, xz, yx, yz, zx, zy):
    return Matrix([[1, xy, xz, 0],
                   [yx, 1, yz, 0],
                   [zx, zy, 1, 0],
                   [0, 0, 0, 1]])

def position(ray, t):
    return ray.origin + ray.direction * t

def intersect(sphere, ray):
    transformed_ray = transform(ray, inverse(sphere.transform))
    sphere_to_ray = transformed_ray.origin - Point(0,0,0) # unit circle for now
    a = dot(transformed_ray.direction, transformed_ray.direction)
    b = 2 * dot(transformed_ray.direction, sphere_to_ray)
    c = dot(sphere_to_ray, sphere_to_ray) - 1

    discriminant = b*b - 4 * a * c
    # print("discriminant: {0}".format(discriminant))

    if discriminant < 0:
        return []

    t1 = (-b - math.sqrt(discriminant)) / (2 * a)
    t2 = (-b + math.sqrt(discriminant)) / (2 * a)

    return [Intersection(t1, sphere), Intersection(t2, sphere)]


def intersections(*intersections):
    return intersections

def hit(intersections):
    current_smallest = Intersection(sys.maxsize, None)
    for intersection in intersections:
        if intersection.t < 0:
            continue
        if current_smallest.t >= intersection.t:
            current_smallest = intersection
    return None if current_smallest.t == sys.maxsize else current_smallest

def transform(ray, matrix):
    return Ray(matrix * ray.origin,  matrix * ray.direction)

def set_transform(sphere, translation):
    sphere.transform = translation