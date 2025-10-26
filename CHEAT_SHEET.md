# Ray Tracer Cheat Sheet

A quick reference guide for understanding the ray tracer implementation.

## Coordinate System

### Right-Handed Coordinate System
```
              +Y (up)
               ^
               |
               |
               |
  -X <---------+---------> +X (right)
              /|
             / |
            /  |
           /   |
          v    |
        -Z     |
      (into    |
       screen) |
               |
               v
              -Y (down)

         +Z (out of screen)
```

### Key Conventions
- **Camera looks toward -Z** (into screen)
- **Canvas positioned at z = -1**
- **+X = Right, +Y = Up, +Z = Out of screen**

### Clock Reference (looking down -Y axis)
- 12 o'clock: `Point(0,0,1)` (+Z direction)
- 3 o'clock: `Point(1,0,0)` (+X direction)  
- 6 o'clock: `Point(0,0,-1)` (-Z direction)
- 9 o'clock: `Point(-1,0,0)` (-X direction)

## Vector vs Ray

### Vector
- Mathematical object with **magnitude and direction**
- **No fixed position** in space - represents displacement
- Has `w = 0` in homogeneous coordinates
- Examples: velocity, force, normal directions

```python
Vector(3, 4, 0)  # "Move 3 right, 4 up"
# Can be placed anywhere in space
```

### Ray  
- Geometric object with **origin point + direction vector**
- Extends infinitely from origin in one direction
- Ray equation: `point = origin + direction * t`

```python
Ray(Point(1,1,-5), Vector(0,0,1))
# Starts at (1,1,-5), shoots toward +Z
# At t=2: Point(1,1,-3)
# At t=5: Point(1,1,0)
```

### Usage in Ray Tracing
```python
def ray_for_pixel(camera, px, py):
    # ... calculations ...
    return Ray(origin, direction)  # Camera position + pixel direction

def position_along_ray(ray, t):
    return ray.origin + ray.direction * t  # Find point at distance t
```

## Transformations - ORDER MATTERS!

### TransformationBuilder Applies in REVERSE Order
```python
# This chain:
TransformationBuilder()
    .scale(2, 2, 2)      # Applied FIRST
    .translate(1, 0, 0)  # Applied SECOND
    .build()

# Actual execution:
# 1. Scale: (1,0,0) → (2,0,0)  
# 2. Translate: (2,0,0) → (3,0,0)
```

### Common Patterns

#### Wrong Order - Translation Gets Scaled!
```python
# DON'T DO THIS - translation gets affected by scale
.translate(5, 0, 0)  # This 5 becomes 10 after scaling!
.scale(2, 1, 1)
# Result: object ends up at (10, 0, 0)
```

#### Correct Order - Local then World
```python
# DO THIS - local operations first, world positioning last
.translate(5, 0, 0)     # LAST: Position in world
.rotate_y(math.pi/4)    # MIDDLE: Orient object
.scale(2, 1, 1)         # FIRST: Resize object
```

### Real Example from Code
```python
# Wall creation - notice the order!
left_wall.transform = TransformationBuilder()
    .translate(0, 0, 5)       # LAST: Move to world position
    .rotate_y(-math.pi / 4)   # 3rd: Angle the wall  
    .rotate_x(math.pi / 2)    # 2nd: Rotate to vertical
    .scale(10, 0.01, 10)      # FIRST: Flatten to thin plane
    .build()
```

### General Rule
Think: **"What do I do to object locally, then how do I place it in the world?"**

## Camera System

### 1. Camera Creation
```python
camera = Camera(width, height, field_of_view)
# width/height: image dimensions in pixels
# field_of_view: angle in radians (math.pi/3 = 60°)
```

### 2. Camera Positioning
```python
camera.transform = view_transforfmation(from_point, to_point, up_vector)

# Example: Camera behind and above, looking at origin
view_transforfmation(
    Point(0, 2, -5),   # Camera position
    Point(0, 0, 0),    # Look at point  
    Vector(0, 1, 0)    # Up direction
)
```

### 3. How It Works

#### Field of View Calculation
```python
half_view = math.tan(field_of_view / 2)
aspect = width / height

# Landscape (width > height)
if aspect >= 1:
    half_width = half_view
    half_height = half_view / aspect
# Portrait (height > width)  
else:
    half_width = half_view * aspect
    half_height = half_view
    
pixel_size = (half_width * 2) / width
```

#### Ray Generation per Pixel
```python
def ray_for_pixel(camera, px, py):
    # 1. Pixel offsets (center of pixel)
    xoffset = (px + 0.5) * camera.pixel_size
    yoffset = (py + 0.5) * camera.pixel_size
    
    # 2. World coordinates (canvas at z = -1)
    # Note: +X is LEFT because camera looks down -Z
    world_x = camera.half_width - xoffset   # Flip X
    world_y = camera.half_height - yoffset  
    
    # 3. Transform through camera matrix
    camera_inverse = inverse(camera.transform)
    pixel = camera_inverse * Point(world_x, world_y, -1)
    origin = camera_inverse * Point(0, 0, 0)
    
    # 4. Create ray
    direction = normalize(pixel - origin)
    return Ray(origin, direction)
```

### 4. Rendering Process
```python
def render(camera, world):
    image = Canvas(camera.hsize, camera.vsize)
    
    for y in range(camera.vsize):
        for x in range(camera.hsize):
            ray = ray_for_pixel(camera, x, y)    # Generate ray
            color = color_at(world, ray)         # Trace through world
            write_pixel(image, x, y, color)      # Store result
            
    return image
```

## Key Classes

### Base Types
```python
Point(x, y, z)    # Position in space (w = 1)
Vector(x, y, z)   # Direction/displacement (w = 0)  
Ray(origin, direction)  # Point + Vector
Color(r, g, b)    # RGB color values
```

### Transformations
```python
# Individual transforms
translation(x, y, z)
scaling(x, y, z)  
rotation_x/y/z(radians)
shearing(xy, xz, yx, yz, zx, zy)

# Chained transforms (remember: reverse order!)
TransformationBuilder()
    .translate(...)
    .rotate_y(...)  
    .scale(...)
    .build()
```

### Scene Objects
```python
Camera(width, height, fov)
World()  # Contains objects and lights
Sphere() # Basic shape with transform and material
Material() # Surface properties (color, shininess, etc.)
PointLight(position, intensity)
```

## Common Patterns

### Typical Scene Setup
```python
# Create world with objects
world = World()
world.light = PointLight(Point(-10, 10, -10), Color(1, 1, 1))

# Add objects with transforms
sphere = Sphere()
sphere.transform = TransformationBuilder()
    .translate(0, 1, 0)     # Position
    .scale(0.5, 0.5, 0.5)   # Size
    .build()
sphere.material.color = Color(1, 0, 0)
world.objects.append(sphere)

# Setup camera
camera = Camera(400, 400, math.pi/3)
camera.transform = view_transforfmation(
    Point(0, 1.5, -5),  # Camera back and up
    Point(0, 1, 0),     # Looking at scene center
    Vector(0, 1, 0)     # Y is up
)

# Render
image = render(camera, world)
canvas_to_ppm(image, "output.ppm")
```

### Ray-Object Intersection
```python
# Transform ray to object space
local_ray = transform(ray, inverse(object.transform))

# Test intersection in object's local coordinate system
intersections = object.local_intersect(local_ray)

# Transform results back to world space
```

### Lighting Calculation
```python
def lighting(material, object, light, point, eyev, normalv, in_shadow):
    # Combine material color with light
    effective_color = material.color * light.intensity
    
    # Calculate ambient, diffuse, and specular components
    ambient = effective_color * material.ambient
    
    if not in_shadow:
        diffuse = calculate_diffuse(...)
        specular = calculate_specular(...)
    
    return ambient + diffuse + specular
```

## Quick Tips

1. **Debugging Transformations**: Create simple test cases with known points
2. **Camera Positioning**: Start with camera at (0,0,-5) looking at origin  
3. **Field of View**: math.pi/3 (60°) is a good default
4. **Coordinate Conversion**: Remember canvas coordinates are flipped
5. **Matrix Operations**: Always check order - matrices don't commute!
6. **Ray Direction**: Should be normalized for consistent lighting calculations

## Common Gotchas

- ⚠️ **Transformation order matters** - TransformationBuilder applies in reverse
- ⚠️ **Camera looks toward -Z** - positive Z is toward viewer
- ⚠️ **Canvas X is flipped** - +X on canvas is -X in world when camera at origin
- ⚠️ **Matrix multiplication order** - `A * B != B * A`
- ⚠️ **Floating point precision** - Use EPSILON for comparisons