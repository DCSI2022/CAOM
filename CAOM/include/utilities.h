//
// Created by joe on 2020/10/3.
//

#ifndef STRUCTURAL_MAPPING_UTILITIES_H
#define STRUCTURAL_MAPPING_UTILITIES_H

#define PCL_NO_PRECOMPILE

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>

#include <octomap/octomap.h>
#include <octomap_ros/conversions.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

#include <ros/ros.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Quaternion.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/console/time.h>
#include <pcl/registration/ndt.h>
//#include <pcl/registration/icp.h>

#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <assert.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <omp.h>

#include "structural_mapping/cloud_info.h"

#define PI 3.14159265

// the following are UBUNTU/LINUX ONLY terminal color codes.
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */

using namespace std;

//region # POINT TYPE
// point struct to save 6dof pose
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                           (float, z, z) (float, intensity, intensity)
                                           (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                           (double, time, time)
)

// point struct with ring info
struct PointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRT,
                                   (float, x, x) (float, y, y)
                                           (float, z, z) (float, intensity, intensity)
                                           (float, ring, ring) (float, time, time)
)

struct RsPointXYZIRT
{
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(RsPointXYZIRT,
                                  (float, x, x)(float, y, y)(float, z, z)
                                          (float, intensity, intensity)(float, time, time)
                                          (uint16_t, ring, ring))

struct OusterPointXYZIRT
{
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    // uint16_t noise;
    uint16_t ambient;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
                                  (float, x, x)(float, y, y)(float, z, z)
                                          (float, intensity, intensity)(uint32_t, t, t)(uint16_t, reflectivity, reflectivity)(uint8_t, ring, ring)
                                          //   (uint16_t,noise, noise)
                                          (uint16_t, ambient, ambient)
                                          (uint32_t, range, range))


typedef PointXYZIRPYT  PointTypePose;
typedef PointXYZIRT  PointTypeRing;

typedef pcl::PointXYZI  PointType;
typedef pcl::PointXYZINormal  PointInfoType;

typedef pcl::PointCloud<pcl::PointXYZI>::Ptr  pcXYZIptr;
typedef pcl::PointCloud<PointTypePose>::Ptr  pcPosePtr;
typedef pcl::PointCloud<PointInfoType>::Ptr  pcInfoPtr;
typedef pcl::PointCloud<pcl::PointXYZI>  pcXYZI;
typedef vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;

class PointFeature
{
public:
    PointFeature() : idx_(0), laser_idx_(0), type_('n') {}
    size_t idx_;
    size_t laser_idx_;
    Eigen::Vector3d point_;
    Eigen::VectorXd coeffs_;
    Eigen::MatrixXd jaco_;
    char type_;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
class FeatureWithScore
{
public:
    FeatureWithScore(const int &idx, const double &score, const Eigen::MatrixXd &jaco)
            : idx_(idx), score_(score), jaco_(jaco) {}

    bool operator<(const FeatureWithScore &fws) const{
        return this->score_ < fws.score_;
    }

    size_t idx_;
    double score_;
    Eigen::MatrixXd jaco_;
};
// endregion

/////////////////////////////////////////// PARAMETERS /////////////////////////////////////
extern const int numofCore ;
extern const bool useOMP ;
// data preprocess params
extern bool labelGround ;
extern bool segmentCloud ;
extern bool filterEdge ;
extern bool saveImg ;
extern std::string projPath;
extern std::string cloudTopicName ; // /velodyne_points

extern const int weightModeltype ;  /// 1:t / 2:Cauchy / 3:MoEP

extern bool useRingInfo;

/// region VLP-16
extern const int N_SCAN;
extern const int Horizon_SCAN;  // 360\0.2
extern const float ang_res_x ;
extern const float ang_res_y ;
extern std::vector<float> ang_y_Vec;
extern const float ang_bottom ;
extern const int groundScanInd;
extern const float dataDriftZ ;
extern const Eigen::Matrix3d COV_MEASUREMENT ;
// endregion

///region VLP-32C
//extern const int N_SCAN = 32;
//extern const int Horizon_SCAN = 1800;
//extern const float ang_res_x = 0.2;
//extern const float ang_res_y = 0.5;
//extern std::vector<float> ang_y_Vec{-25, -1, -1.667, -15.639, -11.31, 0, -0.667, -8.843,
//                                    -7.254, 0.333, -0.333, -6.148, -5.333, 1.333, 0.667,
//                                    -4, -4.667, 1.667, 1, -3.667, -3.333, 3.333, 2.333,
//                                    -2.667, -3, 7, 4.667, -2.333, -2, 15, 10.333, -1.333};
//extern const float ang_bottom = 25;
//extern const int groundScanInd = 16;
//extern const float dataDriftZ = 0;
//extern const Eigen::Matrix3d COV_MEASUREMENT = Eigen::Vector3d(0.0025, 0.0025, 0.0025).asDiagonal();
// endregion

/// region HDL-32E
//extern const int N_SCAN = 32;
//extern const int Horizon_SCAN = 2250;
//extern const float ang_res_x = 0.16;
//extern const float ang_res_y = 1.33;
//extern std::vector<float> ang_y_Vec{ -30.67, -9.33,  -29.33, -8.0, -28.0, -6.66, -26.66, -5.33,
//                                    -25.33, -4.0, -24.0, -2.67, -22.67, -1.33, -21.33, 0, -20,
//                                    1.33, -18.67, 2.67, -17.33, 4, -16, 5.33, -14.67, 6.67,
//                                    -13.33, 8, -12, 9.33, -10.67, 10.67 };
//
//extern const float ang_bottom = 30.67;
//extern const int groundScanInd = 20;
//extern const float dataDriftZ = 0;
//extern const Eigen::Matrix3d COV_MEASUREMENT = Eigen::Vector3d(0.0025, 0.0025, 0.0025).asDiagonal();
// endregion

/// region pandar-40
//extern const float ang_res_x = 0.2;  // 0.08~0.35
//extern const int Horizon_SCAN = 1800;  // 2083
//extern const int N_SCAN = 40;
//extern const float ang_res_y_top = 0.33;  // [-6~2], 24
//extern const float ang_res_y = 1;  // 26.9/63
//extern const float ang_res_y_bottom = 1;  // [-16~-6], 10
//extern std::vector<float> ang_y_Vec{-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, 0,
//                                    -0.33, -0.67, -1, -1.33, -1.67, -2, -2.33, -2.67, -3,
//                                    -3.33, -3.67, -4, -4.33, -4.67, -5, -5.33, -5.67, 0.33,
//                                    0.67, 1, 1.33, 1.67, 2, 3, 4, 5, 6, 7};
//extern const float ang_bottom = 16;
//extern const int groundScanInd = 30;
//extern const float dataDriftZ = 0;
//extern const Eigen::Matrix3d COV_MEASUREMENT = Eigen::Vector3d(0.0004, 0.0004, 0.0004).asDiagonal(); // 2cm
// endregion

/// region HDL-64e
//extern const float ang_res_x = 0.1728;  // 0.08~0.35
//extern const int Horizon_SCAN = 2083;  // 2083
//extern const int N_SCAN = 64;
//extern const float ang_res_y_top = 0.339;  // [-8.5~2], 10.5/31
//extern const float ang_res_y = 0.4;  // 26.9/63
//extern const float ang_res_y_bottom = 0.516;  // [-24.87~-8.87], 16/31
//extern std::vector<float> ang_y_Vec{2, 1.661, 1.322, 0.983, 0.644, 0.305, -0.034, -0.373, 0.712, -1.051, -1.39,
//                                    -1.729, -2.068, -2.407, -2.746, -3.085, -3.424, -3.763, -4.102, -4.441,
//                                    -4.78, -5.119, -5.458, -5.797, -6.136, -6.475, -6.814, -7.153, -7.492,
//                                    -7.831, -8.17, -8.509, -8.848,
//                                    -8.87, -9.386, -9.902, -10.418, -10.934, -11.45, -11.966, -12.482,
//                                    -12.998, -13.514, -14.03, -14.546, -15.062, -15.578, -16.094, -16.61,
//                                    -17.126, -17.642, -18.158, -18.674, -19.19, -19.706, -20.222, -20.738,
//                                    -21.254, -21.77, -22.286, -22.802, -23.318, -23.834, -24.35, -24.87};
//extern const float ang_bottom = 24.87;
//extern const int groundScanInd = 55;
//extern const float dataDriftZ = 0;
//extern const Eigen::Matrix3d COV_MEASUREMENT = Eigen::Vector3d(0.0004, 0.0004, 0.0004).asDiagonal(); // 2cm
// endregion

/// region param for OS1-64
//extern const int N_SCAN = 64;
//extern const int groundScanInd = 32;
//extern const int Horizon_SCAN = 1024;
//extern const float ang_res_x = 360.0/float(Horizon_SCAN);
//extern const float ang_res_y = 33.2/float(64-1);
//extern const float ang_bottom = 16.6+0.1;
//extern const float dataDriftZ = 0;
// endregion

/// region OS1-16
//extern const int N_SCAN = 16;
//extern const int Horizon_SCAN = 1024;  // 360\0.2
//extern const float ang_res_x = 0.3515625;
//extern const float ang_res_y = 2.2133;
//extern std::vector<float> ang_y_Vec{-16.6, -14.387, -12.173, -9.96, -7.747, -5.533, -3.32, -1.107,
//                                    1.107, 3.32, 5.533, 7.747, 9.96, 12.1734, 14.387, 16.6};
//extern const float ang_bottom = 16.6;
//extern const int groundScanInd = 7;
//extern const float dataDriftZ = 0;
//extern const Eigen::Matrix3d COV_MEASUREMENT = Eigen::Vector3d(0.0025, 0.0025, 0.0025).asDiagonal();
// endregion

// region Ouster OS0-128
//extern const bool useCloudRing = 0; // if true, ang_res_y and ang_bottom are not used
//extern const int N_SCAN = 128;
//extern const int groundScanInd = 50;  //
//extern const int Horizon_SCAN = 1024;
//extern const float ang_res_x = 360.0/float(Horizon_SCAN);
//extern const float ang_res_y = 0.7;  // 91.97
//extern const float ang_bottom = 45.0+0.1;  //
//extern const float dataDriftZ = 15;  //
//extern const Eigen::Matrix3d COV_MEASUREMENT = Eigen::Vector3d(0.0025, 0.0025, 0.0025).asDiagonal();
//extern const std::vector<float> ang_y_Vec = {
//        65, 44, 24, 4, 64, 44, 24, 5, 63, 43,24, 5, 62, 43, 24, 6,
//        61, 43, 25, 7, 60, 42, 25, 7, 60, 42, 25, 8, 59, 42, 25, 8,
//        59, 42, 25, 8, 58, 42, 25, 9, 58, 41, 25, 9, 57, 41, 25, 9, 57, 41, 25, 9,
//        57, 41, 25, 9, 57, 41, 25, 9, 57, 41, 25, 9, 57, 41, 25, 9, 57, 41, 25, 9,
//        57, 41, 25, 9, 57, 41, 25, 8, 57, 41, 24, 8, 57, 41, 24, 8, 57, 41, 24, 7,
//        57, 41, 24, 7, 58, 41, 24, 7, 58, 41, 23, 6, 58, 41, 23, 5, 59, 41, 23, 4,
//        59, 41, 22, 4, 60, 41, 22, 3, 61, 41, 22, 1, 62, 42, 21, 0 };
// endregion


extern const float scanPeriod ;
extern const int systemDelay ;
extern const int imuQueLength ;
extern const string imuTopic ;

extern const float sensorMountAngle ;
extern const float segmentTheta ;  // 60.0/180.0*M_PI
extern const int segmentValidPointNum ;
extern const int segmentValidLineNum ;
extern const float segmentAlphaX ;
extern const float segmentAlphaY;

// feature extraction params
extern const int edgeFeatureNum ;
extern const int surfFeatureNum ;
extern const int sectiohnsTotal ;
extern float edgeThreshold ;
extern float surfThreshold ;
extern const float nearestFeatureSearchSqDist ;

// odom estimation params
extern bool  odom_useOctomap ;
extern bool  odom_octoFilterMap ;
extern bool  odom_save_pose;
extern bool  odom_localize_mode ;
extern float odom_ds_leafsize  ;
extern float odom_map_ds_leafsize ;
extern float odom_octomap_res ;
extern float odom_octomap_odd_thre;
extern int   odom_localmapRadi  ;

// mapping params
extern int splineT ;  // 0:B-spline  1：catmull-rom spline
extern float ds_leaf_mapping ;
extern const float surroundingKeyframeSearchRadius ;
extern const int   surroundingKeyframeSearchNum ;

extern const float historyKeyframeSearchRadius;
extern const int   historyKeyframeSearchNum ;
extern const float historyKeyframeFitnessScore;

extern const float globalMapVisualizationSearchRadius ;

extern const bool loopClosureEnableFlag ;
extern const double mappingProcessInterval ;

// line fitting params
extern string poseTopic ;
extern float minlineLength ;
extern float maxDPdist ;
extern int minlinePtNum ;
extern float ds_map_leaf ;

#endif //STRUCTURALMAPPING_ROS_UTILITIES_H
