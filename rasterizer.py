import numpy as np
import cv2


MAX_PIXEL_VALUE = 255

color_to_rgb = { "Red":(255,0,0), "Yellow":(255,255,0),"Green":(0,255,0) }

raster_size = 224
zoom_fact = 1.3
n_channels = 11

displacement = np.array([[raster_size // 4, raster_size // 2]])

def rasterize_input(agents_arr, bb_npcs, roadnet, tl_states):

    XY = agents_arr[:,:,:2]
    YAWS = agents_arr[:,:,2]*np.pi/180.
    lengths = bb_npcs[:,0]
    widths = bb_npcs[:,1]

    raster_list = []

    agents_ids = np.array(list(range(len(XY))),dtype=int)
    unique_agent_ids = agents_ids

    # Generate rasterized input for each agent
    for i, (xy,yawvec) in enumerate(zip(XY,YAWS)):

        road_map = np.ones((raster_size, raster_size, 3), dtype=np.uint8) * MAX_PIXEL_VALUE
        ego_map = [np.zeros((raster_size, raster_size, 1), dtype=np.uint8) for _ in range(n_channels)]
        other_map = [np.zeros((raster_size, raster_size, 1), dtype=np.uint8) for _ in range(n_channels)]

        unscaled_center_xy = xy[-1].reshape(1, -1)
        center_xy = unscaled_center_xy*zoom_fact
        yawt = yawvec[-1]
        rot_matrix = np.array(
            [
                [np.cos(yawt), -np.sin(yawt)],
                [np.sin(yawt), np.cos(yawt)],
            ]
        )

        centered_others = (XY.reshape(-1, 2)*zoom_fact - center_xy) @ rot_matrix + displacement
        centered_others = centered_others.reshape(len(unique_agent_ids), n_channels, 2)

        
        # Roads

        roads = roadnet.each_road_waypoints
        tl_lanes = roadnet.tl_lanes

        for id, road in enumerate(roads):

            road = np.array([[waypoint.transform.location.x, waypoint.transform.location.y] for waypoint in road])

            road = (road*zoom_fact - center_xy) @ rot_matrix + displacement

            road_color = (0,0,0)

            for j, tl in enumerate(tl_lanes):
                if id in tl:
                    road_color = color_to_rgb[str(tl_states[j])]

            road_map = cv2.polylines(road_map,[road.astype(int)],False,road_color)

        # Agents

        agent_id = i

        is_ego = False
        for other_agent_id in unique_agent_ids:
            other_agent_id = int(other_agent_id)
            
            if other_agent_id == agent_id:
                is_ego = True
            else:
                is_ego = False

            agent_lane = centered_others[agents_ids == other_agent_id][0]
            # agent_valid = agents_valid[agents_ids == other_agent_id]
            agent_yaw = YAWS[agents_ids == other_agent_id]

            agent_l = lengths[agents_ids == other_agent_id]
            agent_w = widths[agents_ids == other_agent_id]

            for timestamp, (coord, past_yaw) in enumerate(
                zip(
                    agent_lane,
                    agent_yaw.flatten(),
                )
            ):

                box_points = (
                    np.array(
                        [
                            -agent_l,
                            -agent_w,
                            agent_l,
                            -agent_w,
                            agent_l,
                            agent_w,
                            -agent_l,
                            agent_w,
                        ]
                    )
                    .reshape(4, 2)
                    .astype(np.float32)
                    *zoom_fact
                    / 2
                )

                _coord = np.array([coord])

                rot_matrix = np.array([
                            [np.cos(yawt - past_yaw), -np.sin(yawt - past_yaw)],
                            [np.sin(yawt - past_yaw), np.cos(yawt - past_yaw)],
                        ])
                
                box_points = box_points @ rot_matrix + _coord
                box_points = box_points.reshape(1, -1, 2).astype(np.int32)

                if is_ego:
                    cv2.fillPoly(
                        ego_map[timestamp],
                        box_points,
                        color=MAX_PIXEL_VALUE
                    )
                else:
                    cv2.fillPoly(
                        other_map[timestamp],
                        box_points,
                        color=MAX_PIXEL_VALUE
                    )

        raster = np.concatenate([road_map] + ego_map + other_map, axis=2)
        raster_list.append(raster)

    raster = np.array(raster_list)
    
    raster = raster.transpose(0, 3, 2, 1) / 255

    return raster

    

