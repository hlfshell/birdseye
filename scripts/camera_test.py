import time

import carla

client = carla.Client('localhost', 2000)
world = client.get_world()

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

vehicle_bp = bp_lib.find('vehicle.mini.cooper_s')
vehicle_bp.set_attribute('color', "152, 64, 42")
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[3])

time.sleep(3)

spectator = world.get_spectator()
transform = carla.Transform(vehicle.get_transform().transform(
    carla.Location(x=-4, z=2.5)), vehicle.get_transform().rotation)
spectator.set_transform(transform)

time.sleep(3)

camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", "800")
camera_bp.set_attribute("image_size_y", "800")
camera_bp.set_attribute("fov", "90")
# camera_init_transform = carla.Transform(carla.Location(z=2)) #Change this to move camera
camera_init_transform = carla.Transform(
    carla.Location(z=1.5), carla.Rotation(pitch=90))
camera = world.spawn_actor(camera_bp, camera_init_transform, attach_to=vehicle)
time.sleep(1)
spectator.set_transform(camera.get_transform())

taken = False


def take_picture(image):
    global taken

    if taken == False:
        taken = True
        image.save_to_disk("output.png")


time.sleep(3)
camera.listen(take_picture)
time.sleep(3)

vehicle.destroy()
camera.destroy()

# FRONT
# camera_init_transform = carla.Transform(carla.Location(x=1, z=1.2))

# REAR
# camera_init_transform = carla.Transform(carla.Location(x=-1.5, z=1.2), carla.Rotation(yaw=180))

# Passenger side
# camera_init_transform = carla.Transform(carla.Location(y=0.75, z=1.2), carla.Rotation(yaw=145))

# Driver Side
# camera_init_transform = carla.Transform(carla.Location(y=-0.75, z=1.2), carla.Rotation(yaw=-145))

# TOPDOWN
# camera_init_transform = carla.Transform(carla.Location(z=50), carla.Rotation(pitch=-90))

# Roof -> UP
# camera_init_transform = carla.Transform(carla.Location(z=1.5), carla.Rotation(pitch=90))
