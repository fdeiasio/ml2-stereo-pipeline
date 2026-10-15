import socket

from .parser import parse_wcam_packet
from .decoder import decode_raw_frame
from .rectifier import load_rectification_maps, rectify_frame
from .disparity import compute_disparity, preview_from_disparity
from .point_cloud import load_q_matrix, create_point_cloud_from_disparity
from .pose_estimation import load_model, estimate_model_pose
from .previewer import show_preview_window, close_preview_window
from .visualize_3d import view_3d_preview, close_3d_preview
from .model_preview import show_model_preview, close_model_preview
from .utils import camera_pair_to_ids, parse_mode
from .config import (
    PREVIEW_MODE, STREAM_TO_REMOTE, CAMERA_PAIR, 
    DEFAULT_POSITION, DEFAULT_ORIENTATION, 
    STREAMING_PORT, STREAMING_ADDR
)


def handle_client_connection(client_socket: socket.socket, client_address: tuple[str, int]) -> None:
    """
    This function contains the main logic of the application.
    It handles the whole data pipeline, from receiving raw frames to streaming
    the estimated pose to an outside application.
    """
    
    allow_streaming = STREAM_TO_REMOTE

    # Parse and validate the preview mode from the configuration
    preview_mode: str = parse_mode(PREVIEW_MODE)
    if preview_mode is None:
        print(f"[ERR] Invalid preview mode specified: {PREVIEW_MODE}.")
        return


    # Parse and validate the camera pair from the configuration if needed for the preview mode
    if preview_mode in {"rect", "disp", "view3d", "model"} or STREAM_TO_REMOTE:
        camera_ids = camera_pair_to_ids(CAMERA_PAIR)
        if camera_ids is None:
            print(f"[WARN] Unsupported camera pair specified: {CAMERA_PAIR}. Falling back to raw preview.")
            preview_mode = "raw"
            allow_streaming = False


    # Load rectification maps if needed for the preview mode
    if preview_mode in {"rect", "disp", "view3d", "model"} or STREAM_TO_REMOTE:
        maps_by_camera_id = load_rectification_maps(CAMERA_PAIR)
        if maps_by_camera_id is None:
            print("[WARN] Failed to load rectification maps. Falling back to raw preview.")
            preview_mode = "raw"
            allow_streaming = False


    # Load Q matrix if needed for the point cloud computation
    if preview_mode in {"view3d", 'model'} or STREAM_TO_REMOTE:
        Q_matrix = load_q_matrix(CAMERA_PAIR)
        if Q_matrix is None:
            print("[WARN] Failed to load Q matrix for 3D view. Falling back to disparity preview.")
            preview_mode = "disp"
            allow_streaming = False


    # Load model .obj and default pose if needed for pose estimation
    if preview_mode in {"model"} or STREAM_TO_REMOTE:
        prev_position = DEFAULT_POSITION
        prev_orientation = DEFAULT_ORIENTATION
        model = load_model()
        if model is None:
            print("[WARN] Failed to load 3D model. Falling back to 3D view preview.")
            preview_mode = "view3d"
            allow_streaming = False

    
    # Set up streaming socket if needed 
    if STREAM_TO_REMOTE:
        try:
            streaming_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if STREAMING_ADDR == 'default':
                streaming_socket.connect((client_address[0], STREAMING_PORT))
            else:
                streaming_socket.connect((STREAMING_ADDR, STREAMING_PORT))
        except Exception as exc:
            print(f"[WARN] Failed to set up streaming socket to {client_address[0]}:{STREAMING_PORT}: {exc}")
            allow_streaming = False


    with client_socket:
        print(f"Client connected: {client_address}")
        packet_count = 0

        # Main loop
        should_continue = True
        while should_continue:
            # raw packet -> raw frames
            try:
                packet_frames = parse_wcam_packet(client_socket)
            except ConnectionError:
                break
            except ValueError as exc:
                print(f"[WARN] Packet parse error from {client_address}: {exc}")
                continue
            packet_count += 1
            print(f"[INFO] Received packet #{packet_count} from {client_address} | frameCount={len(packet_frames)}")

            if preview_mode == "none" and not allow_streaming:
                continue


            # raw frames -> opencv frames
            raw_frames = {}
            for metadata, image_bytes in packet_frames:
                camera_id = str(metadata["cameraId"]).strip().lower()
                raw_frame = decode_raw_frame(metadata, image_bytes)
                if raw_frame is None:
                    print(f"[WARN] Failed to decode raw frame for camera '{camera_id}' in packet #{packet_count} from {client_address}")
                    continue
                raw_frames[camera_id] = raw_frame

            if "raw" == preview_mode:
                should_continue = show_preview_window(raw_frames)
                if not allow_streaming:
                    continue

            # opencv frames -> rectified frames
            rectified_frames = {}
            for camera_id in camera_ids:
                rectified_frame = rectify_frame(
                    frame=raw_frames[camera_id],
                    map=maps_by_camera_id[camera_id]
                )
                if rectified_frame is None:
                    print(f"[WARN] Failed to rectify frame for camera '{camera_id}' in packet #{packet_count} from {client_address}")
                    continue
                rectified_frames[camera_id] = rectified_frame

            if "rect" == preview_mode:
                should_continue = show_preview_window(rectified_frames)
                if not allow_streaming:
                    continue


            # rectified frames -> disparity
            left_id, right_id = camera_ids
            disparity = compute_disparity(  
                left_rectified=rectified_frames[left_id],
                right_rectified=rectified_frames[right_id],
            )
            if disparity is None:
                print(f"[WARN] Failed to compute disparity for packet #{packet_count} from {client_address}")
                continue
            
            if "disp" == preview_mode:
                should_continue = show_preview_window({
                    "disp": preview_from_disparity(disparity)
                })
                if not allow_streaming:
                    continue
            

            # disparity -> point cloud
            point_cloud = create_point_cloud_from_disparity(disparity, Q_matrix)
            if point_cloud is None:
                print(f"[WARN] Failed to create point cloud for packet #{packet_count} from {client_address}")
                continue

            if "view3d" == preview_mode:
                should_continue = view_3d_preview(point_cloud)
                if not allow_streaming:
                    continue
            

            # point cloud -> model pose
            position, orientation = estimate_model_pose(point_cloud, model, prev_position, prev_orientation)
            if position is not None and orientation is not None:
                prev_position, prev_orientation = position, orientation
            else:
                print(f"[WARN] Failed to estimate model pose for packet #{packet_count} from {client_address}")
            
            if "model" == preview_mode:
                should_continue = show_model_preview(prev_position, prev_orientation, point_cloud)
                if not allow_streaming:
                    continue
        
            
            # model pose -> stream
            try:
                format_string = f"{prev_position[0]:.8f},{prev_position[1]:.8f},{prev_position[2]:.8f}," \
                                f"{prev_orientation[0]:.8f},{prev_orientation[1]:.8f},{prev_orientation[2]:.8f},{prev_orientation[3]:.8f}"
                data_bytes = format_string.encode("utf-8")
                streaming_socket.send(data_bytes)
            except Exception as exc:
                print(f"[WARN] Failed to stream model pose data to {client_address}: {exc}")
            

        if preview_mode in {"none", "raw", "rect", "disp"}:
            close_preview_window()
        
        if preview_mode in {"view3d"}:
            close_3d_preview()

        if preview_mode in {"model"}:
            close_model_preview()


        print(f"Client disconnected: {client_address}")
