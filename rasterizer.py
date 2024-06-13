import numpy as np
import cv2


MAX_PIXEL_VALUE = 255
N_ROADS = 21
road_colors = [int(x) for x in np.linspace(1, MAX_PIXEL_VALUE, N_ROADS).astype("uint8")]

# Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
# Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
state_to_color = {1:"red",4:"red",7:"red",2:"yellow",5:"yellow",8:"yellow",3:"green",6:"green"}
color_to_rgb = { "red":(255,0,0), "yellow":(255,255,0),"green":(0,255,0) }

raster_size = 224
zoom_fact = 3#1.3
n_channels = 11

displacement = np.array([[raster_size // 4, raster_size // 2]])


def draw_roads(roadmap, centered_roadlines, roadlines_ids, roadlines_types, tl_dict):

    roadmap = cv2.polylines(roadmap,[centered_roadlines.astype(int)],False,road_colors[0])

    # unique_road_ids = np.unique(roadlines_ids)
    # for road_id in unique_road_ids:
    #     if road_id >= 0:
    #         roadline = centered_roadlines[roadlines_ids == road_id]
    #         road_type = roadlines_types[roadlines_ids == road_id].flatten()[0]

    #         road_color = road_colors[road_type]
    #         for col in color_to_rgb.keys():
    #             if road_id in tl_dict[col]:
    #                 road_color = color_to_rgb[col]

    #         print(roadline.shape)

    #         roadmap = cv2.polylines(
    #             roadmap,
    #             [roadline.astype(int).reshape(1,1,2)],
    #             False,
    #             road_color
    #         )

    return roadmap

def rasterize_input(agents_arr, bb_npcs, roads):

    XY = agents_arr[:,:,:2]
    YAWS = agents_arr[:,:,2]
    lengths = bb_npcs[:,0]
    widths = bb_npcs[:,1]

    raster_list = []

    agents_ids = np.array(list(range(len(XY))),dtype=int)
    unique_agent_ids = agents_ids
    #agents_valid = np.ones(len(XY),dtype=int)

    # roadlines_ids = np.array(list(range(len(roads))),dtype=int)
    # roadlines_types = np.ones(len(roads),dtype=int)

    # For each agent
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

        #centered_roadlines = (roads*zoom_fact - center_xy) @ rot_matrix + displacement
        centered_others = (XY.reshape(-1, 2)*zoom_fact - center_xy) @ rot_matrix + displacement
        centered_others = centered_others.reshape(len(unique_agent_ids), n_channels, 2)

        # tl_dict = {"green": set(), "yellow": set(), "red": set()}
        # tl_dict = get_tl_dict(tl_states_hist[:,-1], tl_ids, tl_valid_hist[:,-1])

        #road_map = draw_roads(road_map, centered_roadlines, roadlines_ids, roadlines_types, tl_dict)

        for road in roads:
            road = np.array([[waypoint.transform.location.x, waypoint.transform.location.y] for waypoint in road])
            road = (road*zoom_fact - center_xy) @ rot_matrix + displacement
            road_map = cv2.polylines(road_map,[road.astype(int)],False,road_colors[0])

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
                
                #yawt = yawvec[-1]

                box_points = (
                    box_points
                    @ np.array(
                        (
                            (np.cos(yawt - past_yaw), -np.sin(yawt - past_yaw)),
                            (np.sin(yawt - past_yaw), np.cos(yawt - past_yaw)),
                        )
                    ).reshape(2, 2)
                )

                box_points = box_points + _coord
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

    

