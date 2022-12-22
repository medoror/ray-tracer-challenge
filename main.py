import math
from rayMath import Point, Vector, normalize, Canvas, canvas_to_ppm, Color, write_pixel, translation, rotation_x, rotation_y, rotation_z

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
    # how can i orient my points to start in the middle like Point(0, 1, 0) ?
    c = Canvas(900, 550)
    middle_of_canvas_point = Point(int(c.width / 2), int(c.height / 2), 0)
    transform = translation(0, -200, 0)
    twelve_point = transform * middle_of_canvas_point
    write_pixel(c, convert_to_pixel_space(twelve_point.tuple.x), convert_to_pixel_space(twelve_point.tuple.y), Color(0,1,0))

    r = rotation_z(3 * math.pi/6)

    three_point = r * twelve_point

    write_pixel(c, convert_to_pixel_space(three_point.tuple.x), convert_to_pixel_space(three_point.tuple.y), Color(0,1,0))
    
    canvas_to_ppm(c)


if __name__ == '__main__':
    # basic_projectile()
    # ppm_projectile()
    clock()