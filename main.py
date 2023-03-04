import math
from rayMath import Point, Vector, normalize, Canvas, canvas_to_ppm, \
    Color, write_pixel, translation, rotation_x, rotation_y, rotation_z, TransformationBuilder, \
    Sphere, Ray, intersect, hit, scaling, rotation_z, shearing, Material, lighting, \
    PointLight, position_along_ray, normal_at, view_transforfmation, World, Camera, render, Plane


# Run: python3 main.py
class Projectile:
    def __init__(self, pos=Point(), vel=Vector()):
        self.position = pos
        self.velocity = vel

    def __str__(self):
        return self.position.__str__() + " " + self.velocity.__str__()


class Environment:
    def __init__(self, grav=Vector(), wind=Vector()):
        self.gravity = grav
        self.wind = wind


def tick(env, proj):
    position = proj.position + proj.velocity
    velocity = proj.velocity + env.gravity + env.wind
    return Projectile(position, velocity)


def basic_projectile():
    p = Projectile(Point(0, 1, 0), normalize(Vector(1, 1, 0)))
    e = Environment(Vector(0, -0.1, 0), Vector(-0.01, 0, 0))

    while p.position.tuple.y > 0:
        print(p)
        p = tick(e, p)


def convert_to_pixel_space(raw_coord):
    return int(raw_coord)


def outside_of_canvas(canvas, pos_x, pos_y):
    if pos_x < 0 or pos_x > canvas.width:
        return True
    # elif pos_y < 0 or pos_y > canvas.height:
    #     return True
    else:
        return False


def ppm_projectile():
    c = Canvas(900, 550)
    start = Point(0, 1, 0)
    start_velocity = normalize(Vector(1, 1, 0)) * 11.25
    p = Projectile(start, start_velocity)
    gravity = Vector(0, -0.1, 0)
    wind = Vector(-0.01, 0, 0)
    e = Environment(gravity, wind)

    while p.position.tuple.y > 0:
        # print(p)
        canvas_pos_x = convert_to_pixel_space(p.position.tuple.x)
        canvas_pos_y = c.height - convert_to_pixel_space(p.position.tuple.y)
        print(canvas_pos_x, canvas_pos_y)
        if not outside_of_canvas(c, canvas_pos_x, canvas_pos_y):
            write_pixel(c, canvas_pos_x, canvas_pos_y, Color(1, 1, 0))
        p = tick(e, p)

    canvas_to_ppm(c)


def clock():
    # orientation: from book (looking down the -y axis)
    # 12 hour mark- Point(0,0,1)
    # 3 hour mark - Point(1,0,0)
    # 6 hour mark - Point(0, 0, -1)
    # 9 hour mark - Point(-1, 0, 0)

    square_dimension = 900
    c = Canvas(square_dimension, square_dimension)
    point_color = Color(0, 1, 0)  # green
    # Move each point from the middle. We scale here to spread the points a part from one another
    middle_of_canvas_matrix = TransformationBuilder().scale(80, 0, 80).translate(square_dimension / 2, 0,
                                                                                 square_dimension / 2).build()

    twelve = Point(0, 0, 1)

    for interval in range(0, 12):
        r = rotation_y(interval * (math.pi / 6))
        hour_mark = r * twelve
        translated_point = middle_of_canvas_matrix * hour_mark
        print(translated_point)
        write_pixel(c, convert_to_pixel_space(translated_point.tuple.x),
                    convert_to_pixel_space(translated_point.tuple.z), point_color)

    canvas_to_ppm(c)


def ray_cast_sphere_lighting():
    canvas_pixels = 500
    canvas = Canvas(canvas_pixels, canvas_pixels)
    sphere = Sphere()
    sphere.material = Material()
    sphere.material.color = Color(1, 0.2, 1)
    sphere.transform = shearing(1, 0, 0, 0, 0, 0) * scaling(0.5, 1, 1)

    # light source
    light_position = Point(-10, 10, -10)
    light_color = Color(1, 1, 1)
    light = PointLight(light_position, light_color)

    ray_origin = Point(0, 0, -5)
    wall_z = 10
    wall_size = 7.0

    pixel_size = wall_size / canvas_pixels

    half = wall_size / 2

    # for each row of pixels in the canvas
    for y in range(canvas_pixels - 1):
        world_y = half - pixel_size * y

        # for each pixel in the row
        for x in range(canvas_pixels - 1):
            # compute the world x coordinate
            world_x = -half + pixel_size * x
            # describe the point on the wall that the ray will target
            point_on_wall_position = Point(world_x, world_y, wall_z)
            r = Ray(ray_origin, normalize(point_on_wall_position - ray_origin))
            xs = intersect(sphere, r)

            if hit(xs):
                point = position_along_ray(r, xs[0].t)  # todo: this intersection should be the closest
                normal = normal_at(xs[0].s_object, point)
                eye = -r.direction
                color = lighting(xs[0].s_object.material, light, point, eye, normal)
                write_pixel(canvas, x, y, color)

    canvas_to_ppm(canvas)


def ray_cast_sphere():
    canvas_pixels = 100
    canvas = Canvas(canvas_pixels, canvas_pixels)
    color = Color(1, 0, 0)
    shape = Sphere()
    # shrink it along the y axis
    # shape.transform = scaling(1, 0.5, 1)

    # shrink it along the x axis
    # shape.transform = scaling(0.5, 1, 1)

    # shrink it, and rotate it!
    # shape.transform = rotation_z(math.pi / 4) * scaling(0.5,1,1)

    # shrink it, and skew it!
    shape.transform = shearing(1, 0, 0, 0, 0, 0) * scaling(0.5, 1, 1)

    ray_origin = Point(0, 0, -5)
    wall_z = 10
    wall_size = 7.0

    pixel_size = wall_size / canvas_pixels

    half = wall_size / 2

    # for each row of pixels in the canvas
    for y in range(canvas_pixels - 1):
        world_y = half - pixel_size * y

        # for each pixel in the row
        for x in range(canvas_pixels - 1):
            # compute the world x coordinate
            world_x = -half + pixel_size * x
            # describe the point on the wall that the ray will target
            position = Point(world_x, world_y, wall_z)
            r = Ray(ray_origin, normalize(position - ray_origin))
            xs = intersect(shape, r)

            if hit(xs):
                write_pixel(canvas, x, y, color)

    canvas_to_ppm(canvas)


def create_scene():
    world = World()
    world.light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))

    floor = Sphere()
    floor.transform = scaling(10, 0.01, 10)
    floor.material = Material()
    floor.material.color = Color(1, 0.9, 0.9)
    floor.material.specular = 0
    world.objects.append(floor)

    left_wall = Sphere()
    left_wall.transform = TransformationBuilder().translate(0, 0, 5).rotate_y(-math.pi / 4).rotate_x(math.pi / 2).scale(
        10, 0.01, 10).build()
    # left_wall.transform = TransformationBuilder().scale(10,0.01,10).rotate_x(math.pi/2).rotate_y(-math.pi/4).translate(0,0,5).build()
    left_wall.material = floor.material
    world.objects.append(left_wall)

    right_wall = Sphere()
    right_wall.transform = TransformationBuilder().translate(0, 0, 5).rotate_y(math.pi / 4).rotate_x(math.pi / 2).scale(
        10, 0.01, 10).build()
    right_wall.material = floor.material
    world.objects.append(right_wall)

    middle = Sphere()
    middle.transform = translation(-0.5, 1, 0.5)
    middle.material = Material()
    middle.material.color = Color(0.1, 1, 0.5)
    middle.material.diffuse = 0.7
    middle.material.specular = 0.3
    world.objects.append(middle)

    right = Sphere()
    right.transform = translation(1.5, 0.5, -0.5) * scaling(0.5, 0.5, 0.5)
    right.material = Material()
    right.material.color = Color(0.5, 1, 0.1)
    right.material.diffuse = 0.7
    right.material.specular = 0.3
    world.objects.append(right)

    left = Sphere()
    left.transform = translation(-1.5, 0.33, -0.75) * scaling(0.33, 0.33, 0.33)
    left.material = Material()
    left.material.color = Color(1, 0.8, 0.1)
    left.material.diffuse = 0.7
    left.material.specular = 0.3
    world.objects.append(left)

    camera = Camera(100, 50, math.pi / 3)

    camera.transform = view_transforfmation(Point(-5, 1.5, -5), Point(0, 1, 0), Vector(0, 1, 0))

    canvas = render(camera, world)
    canvas_to_ppm(canvas)


def create_shaded_scene():
    world = World()
    world.light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))

    floor = Sphere()
    floor.transform = scaling(10, 0.01, 10)
    floor.material = Material()
    floor.material.color = Color(1, 0.9, 0.9)
    floor.material.specular = 0
    world.objects.append(floor)

    left_wall = Sphere()
    left_wall.transform = TransformationBuilder().translate(0, 0, 5).rotate_y(-math.pi / 4).rotate_x(math.pi / 2).scale(
        10, 0.01, 10).build()
    left_wall.material = floor.material
    world.objects.append(left_wall)

    right_wall = Sphere()
    right_wall.transform = TransformationBuilder().translate(0, 0, 5).rotate_y(math.pi / 4).rotate_x(math.pi / 2).scale(
        10, 0.01, 10).build()
    right_wall.material = floor.material
    world.objects.append(right_wall)

    middle = Sphere()
    middle.transform = TransformationBuilder().translate(-0.5, 1, 0.5).scale(2, 0.01, 2).build()
    middle.material = Material()
    middle.material.color = Color(0.1, 1, 0.5)
    middle.material.diffuse = 0.7
    middle.material.specular = 0.3
    world.objects.append(middle)

    right = Sphere()
    right.transform = translation(1.5, 0.5, -0.5) * scaling(0.5, 0.5, 0.5)
    right.material = Material()
    right.material.color = Color(0.5, 1, 0.1)
    right.material.diffuse = 0.7
    right.material.specular = 0.3
    world.objects.append(right)

    left = Sphere()
    left.transform = translation(-1.5, 0.33, -0.75) * scaling(0.33, 0.33, 0.33)
    left.material = Material()
    left.material.color = Color(1, 0.8, 0.1)
    left.material.diffuse = 0.7
    left.material.specular = 0.3
    world.objects.append(left)

    camera = Camera(100, 50, math.pi / 3)

    camera.transform = view_transforfmation(Point(0, 1.5, -5), Point(0, 1, 0), Vector(0, 1, 0))

    canvas = render(camera, world)
    canvas_to_ppm(canvas)


def create_plane_scene():
    world = World()
    world.light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))

    floor = Plane()
    floor.material = Material()
    floor.material.color = Color(1, 0.9, 0.9)
    floor.material.specular = 0
    world.objects.append(floor)

    backdrop = Plane()
    backdrop.transform = TransformationBuilder().rotate_x(math.pi / 2).build()
    backdrop.material = Material()
    backdrop.material.color = Color(1, 0.9, 0.9)
    backdrop.material.specular = 0
    world.objects.append(backdrop)

    left_wall = Plane()
    left_wall.transform = TransformationBuilder().translate(10, 0, 0).rotate_z(math.pi / 2).build()
    left_wall.material = Material()
    left_wall.material.color = Color(1, 0.9, 0.9)
    left_wall.material.specular = 0
    world.objects.append(left_wall)

    middle = Sphere()
    middle.transform = TransformationBuilder().translate(-0.5, 1, 0.5).build()
    middle.material = Material()
    middle.material.color = Color(0.1, 1, 0.5)
    middle.material.diffuse = 0.7
    middle.material.specular = 0.3
    world.objects.append(middle)

    right = Sphere()
    right.transform = translation(1.5, 0.5, -0.5) * scaling(0.5, 0.5, 0.5)
    right.material = Material()
    right.material.color = Color(0.5, 1, 0.1)
    right.material.diffuse = 0.7
    right.material.specular = 0.3
    world.objects.append(right)

    left = Sphere()
    left.transform = translation(-1.5, 0.33, -0.75) * scaling(0.33, 0.33, 0.33)
    left.material = Material()
    left.material.color = Color(1, 0.8, 0.1)
    left.material.diffuse = 0.7
    left.material.specular = 0.3
    world.objects.append(left)

    camera = Camera(100, 50, math.pi / 3)

    camera.transform = view_transforfmation(Point(0, 1.5, -5), Point(0, 1, 0), Vector(0, 1, 0))

    canvas = render(camera, world)
    canvas_to_ppm(canvas)


if __name__ == '__main__':
    # basic_projectile()
    # ppm_projectile()
    # clock()
    # ray_cast_sphere()
    # ray_cast_sphere_lighting()
    # create_scene()
    # create_shaded_scene()
    create_plane_scene()
