import math
from rayMath import Point, Vector, normalize, Canvas, canvas_to_ppm, \
    Color, write_pixel, translation, rotation_x, rotation_y, rotation_z, TransformationBuilder, \
    Sphere, Ray, intersect, hit, scaling, rotation_z, shearing, Material, lighting, \
    PointLight, position_along_ray, normal_at, view_transforfmation, World, Camera, render

from shapes import Plane
from patterns import checker_pattern, ring_pattern, gradient_pattern, stripe_pattern, blended_pattern, z_stripe_pattern, pertrubed_pattern


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
    elif pos_y < 0 or pos_y > canvas.height:
        return True
    else:
        return False


def draw_circle(canvas, center_x, center_y, radius, color):
    """Draw a filled circle on the canvas"""
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                px, py = center_x + dx, center_y + dy
                if not outside_of_canvas(canvas, px, py):
                    write_pixel(canvas, px, py, color)


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

    twelve = Point(0, 0, 1)
    scale_factor = 350
    translate_offset = square_dimension / 2

    for interval in range(0, 12):
        r = rotation_y(interval * (math.pi / 6))
        hour_mark = r * twelve

        # Manual transformation: scale first, then translate
        scaled_point = Point(hour_mark.tuple.x * scale_factor, hour_mark.tuple.y, hour_mark.tuple.z * scale_factor)
        translated_point = Point(scaled_point.tuple.x + translate_offset, scaled_point.tuple.y, scaled_point.tuple.z + translate_offset)

        x = convert_to_pixel_space(translated_point.tuple.x)
        y = convert_to_pixel_space(translated_point.tuple.z)

        # Draw a larger, more visible hour mark
        draw_circle(c, x, y, 8, point_color)

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
                color = lighting(xs[0].s_object.material, xs[0].s_object, light, point, eye, normal)
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

def create_pattern_scene():
    world = World()
    world.light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))

    floor = Plane()
    floor.material = Material()
    floor.material.color = Color(1, 0.9, 0.9)
    floor.material.specular = 0
    world.objects.append(floor)

    backdrop = Plane()
    black = Color(0, 0, 0)
    white = Color(1, 1, 1)
    pattern = stripe_pattern(white, black)
    pattern.transform = scaling(2, 2, 2)
    backdrop.transform = TransformationBuilder().rotate_x(math.pi / 2).build()
    backdrop.material = Material()
    backdrop.material.color = Color(1, 0.9, 0.9)
    backdrop.material.specular = 0
    backdrop.material.pattern = pattern
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
    black = Color(0, 0, 0)
    white = Color(1, 1, 1)
    pattern = stripe_pattern(white, black)
    pattern.transform = scaling(2, 2, 2)  # Make stripes much larger
    left.transform = translation(-1.5, 0.33, -0.75) * scaling(0.33, 0.33, 0.33)
    left.material = Material()
    left.material.color = Color(1, 0.8, 0.1)
    left.material.diffuse = 0.7
    left.material.specular = 0.3
    left.material.pattern = pattern
    world.objects.append(left)

    camera = Camera(100, 50, math.pi / 3)

    camera.transform = view_transforfmation(Point(0, 1.5, -5), Point(0, 1, 0), Vector(0, 1, 0))

    canvas = render(camera, world)
    canvas_to_ppm(canvas)

def create_blended_pattern_scene():
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

    # Create blended pattern - crossing stripes
    left = Sphere()
    green = Color(0, 1, 0)
    white = Color(1, 1, 1)

    # Horizontal stripes
    horizontal_stripes = stripe_pattern(green, white)
    horizontal_stripes.transform = scaling(0.2, 0.2, 0.2)

    # Vertical stripes (rotated 90 degrees)
    vertical_stripes = stripe_pattern(green, white)
    vertical_stripes.transform = scaling(0.2, 0.2, 0.2) * rotation_z(math.pi / 2)

    # Blend the two patterns
    blended = blended_pattern(horizontal_stripes, vertical_stripes)

    left.transform = translation(-1.5, 0.33, -0.75) * scaling(0.33, 0.33, 0.33)
    left.material = Material()
    left.material.color = Color(1, 0.8, 0.1)  # This will be overridden by pattern
    left.material.diffuse = 0.7
    left.material.specular = 0.3
    left.material.pattern = blended
    world.objects.append(left)

    camera = Camera(100, 50, math.pi / 3)

    camera.transform = view_transforfmation(Point(0, 1.5, -5), Point(0, 1, 0), Vector(0, 1, 0))

    canvas = render(camera, world)
    canvas_to_ppm(canvas)

def create_plane_pattern_scene():
    world = World()
    world.light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))

    # Just the floor plane with pattern
    floor = Plane()

    # Your crossing stripes
    green = Color(0, 1, 0)
    white = Color(1, 1, 1)

    # Horizontal stripes (along X-axis)
    horizontal_stripes = stripe_pattern(green, white)
    horizontal_stripes.transform = scaling(0.2, 0.2, 0.2)

    # Vertical stripes (along Z-axis) - use Z-stripe pattern for true crossing
    vertical_stripes = z_stripe_pattern(green, white)
    vertical_stripes.transform = scaling(0.2, 0.2, 0.2)

    blended = blended_pattern(horizontal_stripes, vertical_stripes)

    floor.material = Material()
    floor.material.pattern = blended
    floor.material.specular = 0
    world.objects.append(floor)

    # Camera looking down at angle
    camera = Camera(400, 400, math.pi/3)
    camera.transform = view_transforfmation(Point(0, 2, -5), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = render(camera, world)
    canvas_to_ppm(canvas)

def create_perturbed_scene():
    world = World()
    world.light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))

    # Just the floor plane with pattern
    floor = Plane()

    # Your crossing stripes
    red = Color(1, 0, 0)
    white = Color(1, 1, 1)

    # Horizontal stripes (along X-axis)
    horizontal_stripes = stripe_pattern(red, white)
    horizontal_stripes.transform = scaling(0.2, 0.2, 0.2)

    # Vertical stripes (along Z-axis) - use Z-stripe pattern for true crossing
    vertical_stripes = z_stripe_pattern(red, white)
    vertical_stripes.transform = scaling(0.2, 0.2, 0.2)

    blended = blended_pattern(horizontal_stripes, vertical_stripes)

    floor.material = Material()
    floor.material.pattern = blended
    floor.material.specular = 0
    world.objects.append(floor)

    middle = Sphere()
    middle.transform = TransformationBuilder().translate(-0.5, 1, 0.5).build()
    middle.material = Material()
    middle.material.color = Color(0.1, 1, 0.5)
    middle.material.diffuse = 0.7
    middle.material.specular = 0.3
    base_pattern = ring_pattern(Color(1, 0, 0), Color(0, 1, 0))
    perturbed = pertrubed_pattern(base_pattern, scale=2.0, amplitude=0.3)
    middle.material.pattern = perturbed
    world.objects.append(middle)

    right = Sphere()
    right.transform = translation(1.5, 0.5, -0.5) * scaling(0.5, 0.5, 0.5)
    right.material = Material()
    right.material.color = Color(0.5, 1, 0.1)
    right.material.diffuse = 0.7
    right.material.specular = 0.3
    world.objects.append(right)

    left = Sphere()
    black = Color(0, 0, 0)
    white = Color(1, 1, 1)
    pattern = stripe_pattern(white, black)
    pattern.transform = scaling(2, 2, 2)  # Make stripes much larger
    left.transform = translation(-1.5, 0.33, -0.75) * scaling(0.33, 0.33, 0.33)
    left.material = Material()
    left.material.color = Color(1, 0.8, 0.1)
    left.material.diffuse = 0.7
    left.material.specular = 0.3
    left.material.pattern = pattern
    world.objects.append(left)

    # Camera looking down at angle
    camera = Camera(400, 400, math.pi/3)
    camera.transform = view_transforfmation(Point(0, 2, -5), Point(0, 0, 0), Vector(0, 1, 0))

    canvas = render(camera, world)
    canvas_to_ppm(canvas)




if __name__ == '__main__':
    print("Use the justfile to run each project")
