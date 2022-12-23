import math
from rayMath import Point, Vector, normalize, Canvas, canvas_to_ppm, \
Color, write_pixel, translation, rotation_x, rotation_y, rotation_z, TransformationBuilder

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
    point_color = Color(0,1,0) # green
    # Move each point from the middle. We scale here to spread the points a part from one another
    middle_of_canvas_matrix = TransformationBuilder().scale(80, 0, 80).translate(square_dimension / 2, 0, square_dimension / 2).build()

    twelve = Point(0,0,1)

    for interval in range(0,12):
        r = rotation_y(interval * (math.pi / 6))
        hour_mark = r * twelve
        translated_point = middle_of_canvas_matrix * hour_mark
        print(translated_point)
        write_pixel(c, convert_to_pixel_space(translated_point.tuple.x), convert_to_pixel_space(translated_point.tuple.z), point_color)

    canvas_to_ppm(c)


if __name__ == '__main__':
    # basic_projectile()
    # ppm_projectile()
    clock()