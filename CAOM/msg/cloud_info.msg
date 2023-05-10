Header header 

int32[] startRingIndex
int32[] endRingIndex

float32 startOrientation
float32 endOrientation
float32 orientationDiff

bool[]    segmentedCloudGroundFlag # true - ground point, false - other points
uint32[]  segmentedCloudColInd # point column index in range image
float32[] segmentedCloudRange # point range 

# Point cloud messages
sensor_msgs/PointCloud2 segmentedCloud  # filtered cloud 
sensor_msgs/PointCloud2 cloud_corner    # extracted corner features
sensor_msgs/PointCloud2 cloud_surface   # extracted surface features by curvature
sensor_msgs/PointCloud2 cloud_outlier   # 
sensor_msgs/PointCloud2 cloud_lines   # structural features

