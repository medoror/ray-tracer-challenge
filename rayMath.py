from math import sqrt, fabs, ceil

EPSILON = 0.0001


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

    # def red(self):
    #     return self.tuple.x
    #
    # def green(self):
    #     return self.tuple.y
    #
    # def blue(self):
    #     return self.tuple.z

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


def write_pixel(canvas, x, y, color):
    print(x, y)
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

                if charsLength + len(ppm_red) <= 70:
                    file.write(ppm_red)
                    charsLength += len(ppm_red)
                else:
                    file.write("\n")
                    charsLength = 0
                    file.write(ppm_red)
                    charsLength += len(ppm_red)

                if charsLength + len(ppm_green) <= 70:
                    file.write(ppm_green)
                    charsLength += len(ppm_green)
                else:
                    file.write("\n")
                    charsLength = 0
                    file.write(ppm_green)
                    charsLength += len(ppm_green)
                if charsLength + len(ppm_blue) <= 70:
                    file.write(ppm_blue)
                    charsLength += len(ppm_blue)
                else:
                    file.write("\n")
                    charsLength = 0
                    file.write(ppm_blue)
                    charsLength += len(ppm_blue)

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
