import numpy as np

class RoadGraph():
     
    def __init__(self, world, precision = 0.5):
        
        self.precision = precision
        self.world = world
        self.map = world.get_map()
        self.topology = self.map.get_topology()
        self.each_road_waypoints = self.generate_road_waypoints()
        self.center = self.get_center()
        self.tl_lanes = None


    def get_center(self):
        waypoints = self.map.generate_waypoints(5) # Get waypoints just to compute the center
        center = np.array([[waypoint.transform.location.x, waypoint.transform.location.y] for waypoint in waypoints]).mean(0)
        return center


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
    

    def get_tl_lanes(self, traffic_lights):

        tl_lanes = []

        for tl in traffic_lights:

            affected_lanes_wps = tl.get_affected_lane_waypoints()

            affected_lanes = []

            for wp in affected_lanes_wps:
                for id, road_wps in enumerate(self.each_road_waypoints):
                    for waypoint in road_wps:
                        if waypoint.is_junction:
                            dist = np.sqrt((waypoint.transform.location.x - wp.transform.location.x)**2. + (waypoint.transform.location.y - wp.transform.location.y)**2.)
                            if dist<self.precision:
                                affected_lanes.append(id)

            tl_lanes.append(np.unique(affected_lanes))

        self.tl_lanes = tl_lanes

        return self.tl_lanes
    

if __name__=="__main__":

    import carla
    import cv2
    import matplotlib.pyplot as plt

    raster_size = 224
    displacement = np.array([[raster_size // 2, raster_size // 2]])
    color_to_rgb = { "Red":(255,0,0), "Yellow":(255,255,0),"Green":(0,255,0) }

    client = carla.Client()
    world = client.get_world()

    roadmap = np.ones((raster_size, raster_size, 3), dtype=np.uint8)*255

    roadnet = RoadGraph(world)
    waypointslist = roadnet.each_road_waypoints
    center = roadnet.center

    traffic_lights = world.get_actors().filter('traffic.traffic_light*')

    tl_lanes = roadnet.get_tl_lanes(traffic_lights)
    tl_cols = [ tl.get_state() for tl in traffic_lights ]
    
    for id, road_wps in enumerate(waypointslist):

        road = np.array([[waypoint.transform.location.x, waypoint.transform.location.y] for waypoint in road_wps])

        road = road - center + displacement

        road_color = (0,0,0)

        for j, tl in enumerate(tl_lanes):
            if id in tl:
                road_color = color_to_rgb[str(tl_cols[j])]

        roadmap = cv2.polylines(roadmap,[road.astype(int)],False,road_color)

    plt.imshow(roadmap)

    plt.savefig("road.png")
        
    