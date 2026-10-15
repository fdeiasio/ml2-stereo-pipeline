# Client
"""
This parameter controls which cameras are used for the preview and point cloud computation, by specifying which rectification maps to apply.
The "left-center" pair use the left and center cameras, which are closer together 
The "center-right" pair use the center and right cameras, which are farther apart 
"""
CAMERA_PAIR = "left-center" 

"""
This parameter controls which type of preview is shown
Options:
- "none": no preview.
- "raw": show the raw frames without any processing, in separate windows for each camera.
- "rect": show the rectified frames, in separate windows for each camera.
- "disp": show the computed disparity map as a grayscale image.
- "view3d": show a 3D point cloud visualization using Open3D.
- "model": show the pose of a 3D model estimated from the point cloud
"""
PREVIEW_MODE =  "model" 

"""
This parameter controls if the application streams the estimated model pose or not.

If False, the application will show the preview window for the selected modeand NOT COMPUTE OR STREAM any further. 
Ex. if PREVIEW_MODE is "disp", it will compute the disparity map and show it, but not compute the point cloud or model pose, nor stream anything).

If True, the application will show the preview selected but it WILL COMPUTE AND STREAM all stages of the pipeline.
Ex. if PREVIEW_MODE is "disp", it will compute the disparity map and show it, but it will also compute the point cloud and model pose, 
Ex. if PREVIEW_MODE is "none", it will not show any preview, but will compute everything and stream the model pose.
"""
STREAM_TO_REMOTE = True








# Server
"""
Network config:
- HOST: the IP address the server listens on (0.0.0.0 means all interfaces)
- PORT: the TCP port the server listens on 
- SOCKET_TIMEOUT: the timeout for socket operations in seconds, to allow periodic checks for graceful shutdown
"""
HOST = "0.0.0.0"
PORT = 9000
SOCKET_TIMEOUT = 0.5









# Parser
"""
Parser config:
- MAGIC_HEADER: the expected header bytes at the start of each frame, used for basic validation and synchronization.
- EXPECTED_FRAME_COUNT: the expected number of frames in each packet, used for validation.
"""
MAGIC_HEADER = b"WCAM"
EXPECTED_FRAME_COUNT = 3









# Rectifier
LEFT_CENTER_MAPS_PATH = "data/stereo_rectification_maps_left_center_pair.npz"
CENTER_RIGHT_MAPS_PATH = "data/stereo_rectification_maps_center_right_pair.npz"








# Disparity
"""
This parameter empirically sets the distance for which the algorithm works best. 
16 * 8 seems to be about right for arm's lenght distance, which is the main use case for this application.
Higher means more computational cost, lower means more noise and less precision, especially for farther objects.
"""
DEFAULT_NUM_DISPARITIES = 16 * 8

"""
This parameter sets the size of the block used for stereo matching.
Practically, it controls the amount of smoothing applied to the disparity map.
Higher means more smoothing and less noise, but also less precision and more loss of detail, especially for smaller objects.
Lower means more precision and detail, but also more noise and less stability, especially for textureless surfaces.
"""
DEFAULT_BLOCK_SIZE = 5

"""
This parameter sets the window size for speckle filtering of the disparity map.
Higher means more aggressive filtering and less noise, but also more loss of detail.
Lower means less filtering and more detail, but also more noise.
"""
DISPARITY_SPECKLE_WINDOW = 100
DISPARITY_SPECKLE_MAX_DIFF = 16








# Point Cloud
LEFT_CENTER_CALIBRATION_PATH = "data/stereo_calibration_left_center_pair.json"
CENTER_RIGHT_CALIBRATION_PATH = "data/stereo_calibration_center_right_pair.json"


"""
Point cloud parameters:
- MAX_POINT_COUNT: the maximum number of points to keep in the point cloud, for performance reasons.
- MIN_VALID_DISPARITY: the minimum disparity value to consider a point valid, to filter out noise and outliers.

The rest are filters
"""
MAX_POINT_COUNT = 3_000
MIN_VALID_DISPARITY = 1300.0
DISPARITY_LOW_PERCENTILE = 1.0
DISPARITY_HIGH_PERCENTILE = 99.0
DEPTH_LOW_PERCENTILE = 1.0
DEPTH_HIGH_PERCENTILE = 99.0








# ICP
MODEL_PATH = "data/model.obj"

"""
For some reason the 3d printed model doesn't match well with the scale of the point cloud.
I don't know what the issue could be, but this scale factor makes the model fit more precisely
"""
MODEL_SCALE_FACTOR = 2.0 / 3.0

"""
These are sensible values that i empirically found to work well for the model initialization. 
It corresponds more or less to the toracoscope being held at arms lenght with your left hand (for left-center pair)
The arm should be roughly parallel to the ground, angles 30 degrees away from your body center line
All of this with your head straight and looking forward.

Change this if you want to use center-right setup or if the behaviour expected should be different.
"""
DEFAULT_POSITION = [-8.0, 2.0, 29.0]
DEFAULT_ORIENTATION = [-0.4, -0.4, -0.3, 0.8]


"""
These parameters control the ICP algorithm for model pose estimation:
- MODEL_SAMPLE_POINTS: the number of points to sample from the model for ICP, for performance reasons.
- MODEL_VOXEL_SIZE: the voxel size for downsampling the model point cloud, for performance reasons.
- SCENE_VOXEL_SIZE: the voxel size for downsampling the scene point cloud, for performance reasons.
- ICP_MAX_ITERATIONS: the maximum number of iterations for the ICP algorithm, to prevent infinite loops.
- ICP_MIN_FITNESS: the minimum fitness score for the ICP result to be considered valid
"""
MODEL_SAMPLE_POINTS = MAX_POINT_COUNT
MODEL_VOXEL_SIZE = 0.008
SCENE_VOXEL_SIZE = 0.01
ICP_MAX_ITERATIONS = 30
ICP_MIN_FITNESS = 0.70








# Streaming
"""
Streaming network config:
- STREAMING_ADDR: the IP address to stream the estimated model pose to
- STREAMING_PORT: the UDP port used for streaming the estimated model pose to the client.

If streaming addr is set to ""default"" it will stream to the same address as the client that sent the frames
If you use the setup with adb, that will just be localhost

WARN: if you are using adb, it does not support UDP port forwarding, so you won't be able to receive the streamed 
pose on the device. To implement this you'd need to change the protocol to TCP and implement a handshake
between the device and the host. 
"""
STREAMING_ADDR = "default"
STREAMING_PORT = 12345






# Previewer
WINDOW_NAME = "Magic Leap Stream"
SAVED_IMAGES_DIR = "data/images"
