import math
from math import sqrt, ceil
import sys
from base import Material, Matrix, Intersection, Vector, Ray, Point, Color, dot
from utils import auto_str, EPSILON, MAX_CHARACTER_LENGTH
from base import inverse
from base import transpose
from shapes import Sphere


@auto_str
class Camera:
    def __init__(self, hsize, vsize, field_of_view):
        self.hsize = hsize
        self.vsize = vsize
        self.field_of_view = field_of_view
        self.transform: Matrix = Matrix([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])

        half_view = math.tan(field_of_view / 2)
        aspect = hsize / vsize

        if aspect >= 1:
            self.half_width = half_view
            self.half_height = half_view / aspect
        else:
            self.half_width = half_view * aspect
            self.half_height = half_view
        self.pixel_size = (self.half_width * 2) / hsize


@auto_str
class PrepareComputations:
    def __init__(self, intersection, ray):
        self.t = intersection.t
        self.object = intersection.s_object
        self.point = position_along_ray(ray, self.t)
        self.eyev = -ray.direction
        self.normalv = normal_at(self.object, self.point)
        self.over_point = Point(0, 0, 0)
        self.inside = False
        self.reflectv = None


@auto_str
class World:
    def __init__(self):
        self.objects = []
        self.light = None

@auto_str
class PointLight:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity


@auto_str
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



def cross(a, b):
    return Vector(a.tuple.y * b.tuple.z - a.tuple.z * b.tuple.y,
                  a.tuple.z * b.tuple.x - a.tuple.x * b.tuple.z,
                  a.tuple.x * b.tuple.y - a.tuple.y * b.tuple.x)


@auto_str
class TransformationBuilder:
    def __init__(self):
        self.operations = []

    # is this correct?
    def identity(self):
        self.operations.append(Matrix([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]]))
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

    def scale(self, x, y, z):
        self.operations.append(scaling(x, y, z))
        return self

    def shear(self, xy, xz, yx, yz, zx, zy):
        self.operations.append(shearing(xy, xz, yx, yz, zx, zy))
        return self

    def translate(self, x, y, z):
        self.operations.append(translation(x, y, z))
        return self

    def build(self):
        if not self.operations:
            raise Exception("Gotta add things before you build!!")
        else:
            current = self.operations.pop()
            while self.operations:
                next = self.operations.pop()
                # current = current * next
                current = next * current
            return current


def translation(x, y, z) -> Matrix:
    return Matrix([[1, 0, 0, x],
                   [0, 1, 0, y],
                   [0, 0, 1, z],
                   [0, 0, 0, 1]])


def scaling(x, y, z) -> Matrix:
    return Matrix([[x, 0, 0, 0],
                   [0, y, 0, 0],
                   [0, 0, z, 0],
                   [0, 0, 0, 1]])


def rotation_x(radians) -> Matrix:
    return Matrix([[1, 0, 0, 0],
                   [0, math.cos(radians), -math.sin(radians), 0],
                   [0, math.sin(radians), math.cos(radians), 0],
                   [0, 0, 0, 1]])


def rotation_y(radians) -> Matrix:
    return Matrix([[math.cos(radians), 0, math.sin(radians), 0],
                   [0, 1, 0, 0],
                   [-math.sin(radians), 0, math.cos(radians), 0],
                   [0, 0, 0, 1]])


def rotation_z(radians) -> Matrix:
    return Matrix([[math.cos(radians), -math.sin(radians), 0, 0],
                   [math.sin(radians), math.cos(radians), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])


def shearing(xy, xz, yx, yz, zx, zy) -> Matrix:
    return Matrix([[1, xy, xz, 0],
                   [yx, 1, yz, 0],
                   [zx, zy, 1, 0],
                   [0, 0, 0, 1]])


def position_along_ray(ray, t):
    return ray.origin + ray.direction * t


def intersect(shape, ray):
    local_ray = transform(ray, inverse(shape.transform))
    return shape.local_intersect(local_ray)


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
    return Ray(matrix * ray.origin, matrix * ray.direction)


def set_transform(sphere, translation):
    sphere.transform = translation


def normal_at(shape, world_point):
    local_point = inverse(shape.transform) * world_point
    local_normal = shape.local_normal_at(local_point)
    world_normal = transpose(inverse(shape.transform)) * local_normal
    world_normal.tuple.w = 0
    return normalize(world_normal)


def reflect(in_vector, normal):
    return in_vector - normal * 2 * dot(in_vector, normal)


def lighting(material, obj, light, point, eyev, normalv, in_shadow=False):
    color = Color(0, 0, 0)
    if material.pattern is None:
        color = material.color
    else:
        color = pattern_at_shape(material.pattern, obj, point)

    # combine the surface color with the lights color/intensity
    effective_color = color * light.intensity
    # find the direction to the light source
    lightv = normalize(light.position - point)
    # compute the ambient contribution
    ambient = effective_color * material.ambient

    if in_shadow:
        return ambient

    # light_dot_normal represents the cosine of the angle between the light_vector
    # and the normal vector.  A negative number means the light is on the other side
    # of the surface
    light_dot_normal = dot(lightv, normalv)
    if light_dot_normal < 0:
        # can i make this a const
        diffuse = Color(0, 0, 0)
        specular = Color(0, 0, 0)
    else:
        # compute the diffuse contribution
        diffuse = effective_color * material.diffuse * light_dot_normal

        # reflect_dot_eye represents the cosine of the angle between the reflection vector
        # and the eye vector.  A negative number means the light reflects away from the eye
        reflectv = reflect(-lightv, normalv)
        reflect_dot_eye = dot(reflectv, eyev)

        if reflect_dot_eye <= 0:
            specular = Color(0, 0, 0)  # can i make this a const
        else:
            # compute the specular contribution
            factor = math.pow(reflect_dot_eye, material.shininess)
            specular = light.intensity * material.specular * factor

    return ambient + diffuse + specular


def default_world() -> World:
    w = World()

    s1 = Sphere()
    m = Material()
    m.color = Color(0.8, 1.0, 0.6)
    m.diffuse = 0.7
    m.specular = 0.2
    s1.material = m

    s2 = Sphere()
    s2.transform = scaling(0.5, 0.5, 0.5)

    w.objects = [s1, s2]
    w.light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))
    return w


def intersect_world(world, ray):
    # there has got to be cleaner way to do this
    returned_intersections = []
    for shape in world.objects:
        computed_intersections = intersect(shape, ray)
        if len(computed_intersections) == 0:
            continue
        returned_intersections.extend(computed_intersections)
    returned_intersections.sort(key=lambda i: i.t)
    return returned_intersections


def prepare_computations(intersection, ray):
    comps = PrepareComputations(intersection, ray)

    if dot(comps.normalv, comps.eyev) < 0:
        comps.inside = True
        comps.normalv = -comps.normalv
        comps.reflectv = reflect(ray.direction, comps.normalv)
        comps.over_point = comps.point + comps.normalv * EPSILON
    else:
        comps.over_point = comps.point + comps.normalv * EPSILON
        comps.inside = False
    return comps


def shade_hit(world, comps):
    shadow = is_shadowed(world, comps.over_point)
    return lighting(comps.object.material, comps.object, world.light, comps.over_point, comps.eyev, comps.normalv,
                    shadow)


def color_at(world, ray):
    xs = intersect_world(world, ray)

    possible_intersection = hit(xs)
    if possible_intersection is None:
        return Color(0, 0, 0)
    else:
        comps = prepare_computations(possible_intersection, ray)
        return shade_hit(world, comps)


def view_transforfmation(from_vector, to_vector, up_vector) -> Matrix:
    forward = normalize(to_vector - from_vector)
    upn = normalize(up_vector)
    left = cross(forward, upn)
    true_up = cross(left, forward)

    orientation = Matrix([[left.tuple.x, left.tuple.y, left.tuple.z, 0],
                          [true_up.tuple.x, true_up.tuple.y, true_up.tuple.z, 0],
                          [-forward.tuple.x, -forward.tuple.y, -forward.tuple.z, 0],
                          [0, 0, 0, 1]])

    return orientation * translation(-from_vector.tuple.x, -from_vector.tuple.y, -from_vector.tuple.z)


def ray_for_pixel(camera, px, py):
    # the offset from the edge of the canvas to the pixels center
    xoffset = (px + 0.5) * camera.pixel_size
    yoffset = (py + 0.5) * camera.pixel_size

    # the untransformed coordinates of the pixel in the world space.
    # (remember that the camera looks toward -z, so +x is to the left)

    world_x = camera.half_width - xoffset
    world_y = camera.half_height - yoffset

    # using the camera matrix, transform the canvas point and the origin, and then
    # compute the ray's direction vector
    # (remember that the canvas is at z =-1)

    camera_inverse = inverse(camera.transform)
    pixel = camera_inverse * Point(world_x, world_y, -1)
    origin = camera_inverse * Point(0, 0, 0)
    direction = normalize(pixel - origin)

    return Ray(origin, direction)


def render(camera, world):
    image = Canvas(camera.hsize, camera.vsize)

    for y in range(camera.vsize - 1):
        for x in range(camera.hsize - 1):
            ray = ray_for_pixel(camera, x, y)
            color = color_at(world, ray)
            write_pixel(image, x, y, color)

    return image


def is_shadowed(world, point):
    v = world.light.position - point
    distance = magnitude(v)
    direction = normalize(v)

    r = Ray(point, direction)

    intersections = intersect_world(world, r)

    h = hit(intersections)
    if h is not None and h.t < distance:
        return True
    else:
        return False


def stripe_at(pattern, point):
    if math.floor(point.tuple.x) % 2 == 0:
        return pattern.a
    else:
        return pattern.b


def set_pattern_transform(pattern, transform):
    pattern.transform = transform

def pattern_at_shape(pattern, shape, world_point):
    object_point = shape.inverse_transform * world_point
    pattern_point = pattern.inverse_transform * object_point

    return pattern.pattern_at(pattern_point)
