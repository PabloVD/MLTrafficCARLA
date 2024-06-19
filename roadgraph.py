import numpy as np

class RoadGraph():
     
    def __init__(self, world, precision = 0.5):
        
        self.precision = precision
        self.world = world
        self.map = world.get_map()
        self.topology = self.map.get_topology()
        self.each_road_waypoints = self.generate_road_waypoints()
        self.waypoints = self.map.generate_waypoints(5)
        self.center = np.array([[waypoint.transform.location.x, waypoint.transform.location.y] for waypoint in self.waypoints]).mean(0)
          
    def generate_road_waypoints(self):
        """Return all, precisely located waypoints from the map.

        Topology contains simplified representation (a start and an end
        waypoint for each road segment). By expanding each until another
        road segment is found, we explore all possible waypoints on the map.

        Returns a list of waypoints for each road segment.
        """
        road_segments_starts: carla.Waypoint = [
            road_start for road_start, road_end in self.topology
        ]

        each_road_waypoints = []
        for road_start_waypoint in road_segments_starts:
            road_waypoints = [road_start_waypoint]

            # Generate as long as it's the same road
            next_waypoints = road_start_waypoint.next(self.precision)

            if len(next_waypoints) > 0:
                # Always take first (may be at intersection)
                next_waypoint = next_waypoints[0]
                while next_waypoint.road_id == road_start_waypoint.road_id:
                    road_waypoints.append(next_waypoint)
                    next_waypoint = next_waypoint.next(self.precision)

                    if len(next_waypoint) > 0:
                        next_waypoint = next_waypoint[0]
                    else:
                        # Reached the end of road segment
                        break
            each_road_waypoints.append(road_waypoints)
        return each_road_waypoints
    

if __name__=="__main__":

    import carla
    import cv2
    import matplotlib.pyplot as plt

    raster_size = 224
    displacement = np.array([[raster_size // 2, raster_size // 2]])

    client = carla.Client()
    world = client.get_world()

    roadmap = np.ones((raster_size, raster_size, 3), dtype=np.uint8)*256

    roadnet = RoadGraph(world)
    waypointslist = roadnet.each_road_waypoints
    center = roadnet.center
    
    for waypoints in waypointslist:

        road = np.array([[waypoint.transform.location.x, waypoint.transform.location.y] for waypoint in waypoints])

        road = road - center + displacement

        roadmap = cv2.polylines(roadmap,[road.astype(int)],False,(0,0,0))

    plt.imshow(roadmap)

    plt.savefig("road.png")
        
    