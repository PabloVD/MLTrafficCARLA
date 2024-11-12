import carla
import os
import time

client = carla.Client()
world = client.get_world()
# world_settings = world.get_settings()
# world_settings.fixed_delta_seconds = 0.01
# world.apply_settings(world_settings)

logfile = os.getcwd()+"/logs/record.log"

print(client.show_recorder_file_info(logfile, True))

# client.set_replayer_time_factor(0.1)
# 24, 31
ego_id = 25
print(client.replay_file(logfile, 0, 0, ego_id, False))

# # Spawn a camera in a vehicle
# camdir = "cameraout"
# if not os.path.exists(camdir+"/"):
#     os.system("mkdir "+camdir+"/")
# hero_v = world.get_actor(ego_id)
# camera_init_trans = carla.Transform(carla.Location(x=-6, z=6),carla.Rotation(pitch=-20))
# camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
# camera_bp.set_attribute('image_size_x', '1920')
# camera_bp.set_attribute('image_size_y', '1080')
# camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=hero_v)
# camera.listen(lambda image: image.save_to_disk(camdir+'/%06d.png' % image.frame))

# while True:
#     #frame = world.wait_for_tick()
#     frame = world.tick()
#     #print(frame)
