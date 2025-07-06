# List all available commands
default:
    @just --list

# Run the basic projectile simulation
projectile:
    @python -c "from main import basic_projectile; basic_projectile()"

# Create a projectile animation as PPM
ppm:
    @python -c "from main import ppm_projectile; ppm_projectile()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Draw a clock
clock:
    @python -c "from main import clock; clock()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Ray cast a basic sphere
sphere:
    @python -c "from main import ray_cast_sphere; ray_cast_sphere()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Ray cast a sphere with lighting
lighting:
    @python -c "from main import ray_cast_sphere_lighting; ray_cast_sphere_lighting()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Create a 3D scene
scene:
    @python -c "from main import create_scene; create_scene()"

# Create a shaded scene
shaded:
    @python -c "from main import create_shaded_scene; create_shaded_scene()"

# Create a scene with planes
plane:
    @python -c "from main import create_plane_scene; create_plane_scene()"

# Create a scene with patterns
pattern:
    @python -c "from main import create_pattern_scene; create_pattern_scene()"

# Run a method with profiling
profile METHOD:
    @python -c "import cProfile, pstats; from main import {{METHOD}}; \
                profile = cProfile.Profile(); \
                profile.runcall({{METHOD}}); \
                stats = pstats.Stats(profile); \
                stats.sort_stats(pstats.SortKey.TIME).print_stats()"
