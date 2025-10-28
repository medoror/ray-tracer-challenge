# Detect the correct Python command (prefer python3, fallback to python)
python_cmd := `if command -v python3 &> /dev/null; then echo python3; elif command -v python &> /dev/null; then echo python; else echo "echo 'No Python found!' && exit 1"; fi`

# List all available commands
default:
    @just --list

# Run the basic projectile simulation
projectile:
    @{{python_cmd}} -c "from main import basic_projectile; basic_projectile()"

# Create a projectile animation as PPM
ppm:
    @{{python_cmd}} -c "from main import ppm_projectile; ppm_projectile()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Draw a clock
clock:
    @{{python_cmd}} -c "from main import clock; clock()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Ray cast a basic sphere
sphere:
    @{{python_cmd}} -c "from main import ray_cast_sphere; ray_cast_sphere()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Ray cast a sphere with lighting
lighting:
    @{{python_cmd}} -c "from main import ray_cast_sphere_lighting; ray_cast_sphere_lighting()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Create a 3D scene
scene:
    @{{python_cmd}} -c "from main import create_scene; create_scene()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Create a shaded scene
shaded:
    @{{python_cmd}} -c "from main import create_shaded_scene; create_shaded_scene()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Create a scene with planes
plane:
    @{{python_cmd}} -c "from main import create_plane_scene; create_plane_scene()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Create a scene with patterns
pattern:
    @{{python_cmd}} -c "from main import create_pattern_scene; create_pattern_scene()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

# Create a scene with blended patterns
blended:
    @{{python_cmd}} -c "from main import create_blended_pattern_scene; create_blended_pattern_scene()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

plane_pattern:
    @{{python_cmd}} -c "from main import create_plane_pattern_scene; create_plane_pattern_scene()"
    @open canvas.ppm || echo "Could not open canvas.ppm"

create_perturbed_scene:
    @{{python_cmd}} -c "from main import create_perturbed_scene; create_perturbed_scene()"
    @open canvas.ppm || echo "Could not open canvas.ppm"


test:
    @{{python_cmd}} -m unittest rayMath_test.py

# Run a method with profiling
profile METHOD:
    @echo "Using Python: $(which {{python_cmd}})"
    @{{python_cmd}} -c "import cProfile, pstats; from main import {{METHOD}}; \
                       profile = cProfile.Profile(); \
                       profile.runcall({{METHOD}}); \
                       stats = pstats.Stats(profile); \
                       stats.sort_stats(pstats.SortKey.TIME).print_stats()"
