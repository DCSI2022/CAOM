//
// Created by cyz on 2023/4/13.
//

#include "tools.h"

/// region GLOBAL VARIABLES
const int numofCore = 4;
const bool useOMP = true;
// data preprocess params
bool labelGround = false;
bool segmentCloud = true;
bool filterEdge = false;
bool saveImg = false;
std::string projPath = "nan";
std::string cloudTopicName = "/velodyne_points"; // /velodyne_points

const int weightModeltype = 3;  /// 1:t / 2:Cauchy / 3:MoEP

bool useRingInfo = 0;
/// region VLP-16
const int N_SCAN = 16;
const int Horizon_SCAN = 1800;  // 360\0.2
const float ang_res_x = 0.2;
const float ang_res_y = 2.0;
std::vector<float> ang_y_Vec{-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15};
const float ang_bottom = 15.0 + 0.1;
const int groundScanInd = 7;
const float dataDriftZ = 0;
const Eigen::Matrix3d COV_MEASUREMENT = Eigen::Vector3d(0.0025, 0.0025, 0.0025).asDiagonal();
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


const float scanPeriod = 0.1;
const int systemDelay = 0;
const int imuQueLength = 200;
const string imuTopic = "/imu/data";

const float sensorMountAngle = 0.0;
const float segmentTheta = 30.0/180.0*M_PI;  // 60.0/180.0*M_PI
const int segmentValidPointNum = 5;
const int segmentValidLineNum = 3;
const float segmentAlphaX = ang_res_x / 180.0 * M_PI;
const float segmentAlphaY = ang_res_y / 180.0 * M_PI;

// feature extraction params
const int edgeFeatureNum = 2;
const int surfFeatureNum = 4;
const int sectiohnsTotal = 6;
float edgeThreshold = 0.1;
float surfThreshold = 0.1;
const float nearestFeatureSearchSqDist = 25;

// odom estimation params
bool  odom_useOctomap = false;
bool  odom_octoFilterMap = false;
bool  odom_save_pose = true;
bool  odom_localize_mode = false;
float odom_ds_leafsize = 0.3f;
float odom_map_ds_leafsize = 0.2f;
float odom_octomap_res = 0.3f;
float odom_octomap_odd_thre = 0.4f;
int   odom_localmapRadi = 80;

// mapping params
int splineT = 1;  // 0:B-spline  1：catmull-rom spline
float ds_leaf_mapping = 0.2f;
const float surroundingKeyframeSearchRadius = 50.0;
const int   surroundingKeyframeSearchNum = 50;

const float historyKeyframeSearchRadius = 15.0;
const int   historyKeyframeSearchNum = 25;
const float historyKeyframeFitnessScore = 0.3;

const float globalMapVisualizationSearchRadius = 50.0;

const bool loopClosureEnableFlag = true;
const double mappingProcessInterval = 0.3;

// line fitting params
string poseTopic = "nan";
float minlineLength = 0.6;
float maxDPdist = 0.15f;
int minlinePtNum = 8;
float ds_map_leaf = 0.05f;
// endregion

//////////////////// Evaluating the overlapping area between two polygons /////////////////////////////////
// 计算线段ab和cd的交点坐标
PolygonOverlapHelper::Point PolygonOverlapHelper::intersection(Point a, Point b, Point c, Point d){

    Point p = a;
    double t =((a.x-c.x)*(c.y-d.y)-(a.y-c.y)*(c.x-d.x))/((a.x-b.x)*(c.y-d.y)-(a.y-b.y)*(c.x-d.x));
    p.x += (b.x-a.x)*t;
    p.y += (b.y-a.y)*t;
    cout << "intersection p.x=" << p.x << ", p.y=" << p.y << endl;
    return p;
}

// 计算多边形面积，将多边形拆解成连续三个顶点组合成的多个三角形进行计算，这个循环计算一次其实是计算两次多边形的面积。
double PolygonOverlapHelper::PolygonArea(Point p[], int n){

    if(n < 3) return 0.0;
    double s = p[0].y * (p[n - 1].     x - p[1].x);
    for(int i = 1; i < n - 1; ++ i) {
        s += p[i].y * (p[i - 1].x - p[i + 1].x);
        // cout << "p[i-1].x =" << p[i-1].x << ", p[i-1].y=" << p[i-1].y << endl;
        // cout << "p[i].x =" << p[i].x << ", p[i].y=" << p[i].y << endl;
        // cout << "p[i+1].x =" << p[i+1].x << ", p[i+1].y=" << p[i+1].y << endl;
    }
    s += p[n - 1].y * (p[n - 2].x - p[0].x);
    cout << "s =" << s << endl;
    return fabs(s * 0.5);
}

// ConvexPolygonIntersectArea
double PolygonOverlapHelper::CPIA(Point a[], Point b[], int na, int nb) {

    Point p[20], tmp[20];
    int tn, sflag, eflag;
    memcpy(p,b,sizeof(Point)*(nb));

    for(int i = 0; i < na && nb > 2; i++){

        if (i == na - 1)   // last
            sflag = dcmp(cross(a[0], p[0],a[i]));
        else
            sflag = dcmp(cross(a[i + 1], p[0],a[i]));

        for(int j = tn = 0; j < nb; j++, sflag = eflag){

            if(sflag >= 0)
                tmp[tn++] = p[j];

            if (i == na - 1) {
                if (j == nb -1) {
                    eflag = dcmp(cross(a[0], p[0], a[i]));
                } else {
                    eflag = dcmp(cross(a[0], p[j + 1], a[i])); // 计算下一个连续点在矢量线段的位置
                }
            } else {
                if (j == nb -1) {
                    eflag = dcmp(cross(a[i + 1], p[0], a[i]));
                } else {
                    eflag = dcmp(cross(a[i + 1], p[j + 1], a[i]));
                }
            }
            if((sflag ^ eflag) == -2) {  // 1和-1的异或为-2，也就是两个点分别在矢量线段的两侧
                if (i == na - 1) {
                    if (j == nb -1) {
                        tmp[tn++] = intersection(a[i], a[0], p[j], p[0]); //求交点
                    } else {
                        tmp[tn++] = intersection(a[i], a[0], p[j], p[j + 1]);
                    }
                } else {
                    if (j == nb -1) {
                        tmp[tn++] = intersection(a[i], a[i + 1], p[j], p[0]);
                    } else {
                        tmp[tn++] = intersection(a[i], a[i + 1], p[j], p[j + 1]);
                    }
                }
            }
        }
        memcpy(p, tmp, sizeof(Point) * tn);
        nb = tn, p[nb] = p[0];
    }
    if(nb < 3) return 0.0;
    return PolygonArea(p, nb);
}

// SimplePolygonIntersectArea 调用此函数
double PolygonOverlapHelper::SPIA(Point a[], Point b[], int na, int nb) {

    int i, j;
    Point t1[na], t2[nb];
    double res = 0, num1, num2;
    t1[0] = a[0], t2[0] = b[0];
    for(i = 2; i < na; i++)
    {
        t1[1] = a[i-1], t1[2] = a[i];
        num1 = dcmp(cross(t1[1], t1[2], t1[0]));  // 根据差积公式来计算t1[2]在矢量线段（t1[0], t1[1]）的左侧还是右侧，
        // 值为负数在矢量线段左侧，值为正数在矢量线段右侧
        if(num1 < 0) swap(t1[1], t1[2]);  // 按逆时针进行排序
        for(j = 2; j < nb; j++)
        {
            t2[1] = b[j - 1], t2[2] = b[j];
            num2 = dcmp(cross(t2[1], t2[2], t2[0]));
            if(num2 < 0) swap(t2[1], t2[2]);
            res += CPIA(t1, t2, 3, 3) * num1 * num2;
        }
    }
    cout << "Sum::res=" <<res << endl;
    return res;
}

//////////////////////////////////// Octomap for map filter //////////////////////////////////////////////////
// copy from octomap_ros/conversion.h
void OctomapManager::pointCloud2ToOctomap(const sensor_msgs::PointCloud2& cloud,
                                          octomap::Pointcloud& octomapCloud){

    octomapCloud.reserve(cloud.data.size() / cloud.point_step);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z){
        // Check if the point is invalid
        if (!std::isnan (*iter_x) && !std::isnan (*iter_y) && !std::isnan (*iter_z))
            octomapCloud.push_back(*iter_x, *iter_y, *iter_z);
    }
}

void OctomapManager::insertCloud2OctoMap(const pcl::PointCloud<PointType>::Ptr cloudin, double max_range){

    octomap::Pointcloud octoCloud;
    sensor_msgs::PointCloud2 cloud2;
    pcl::toROSMsg(*cloudin, cloud2);
//        octomap::pointCloud2ToOctomap(cloud2, octoCloud);  // undefined
    pointCloud2ToOctomap(cloud2, octoCloud);

    octoMap->insertPointCloud(octoCloud, octomap::point3d(0,0,0), max_range);
}

void OctomapManager::insertCloud2OctoMap(const pcl::PointCloud<PointType>::Ptr cloudin, const PointType& origin,
                                         double max_range){

    octomap::Pointcloud octoCloud;
    sensor_msgs::PointCloud2 cloud2;
    pcl::toROSMsg(*cloudin, cloud2);
//        octomap::pointCloud2ToOctomap(cloud2, octoCloud);  // undefined
    pointCloud2ToOctomap(cloud2, octoCloud);

    octoMap->insertPointCloud(octoCloud, octomap::point3d(origin.x,origin.y,origin.z), max_range);
}

void OctomapManager::filterLocalOccupancyMap(pcl::PointCloud<PointType>::Ptr cloud){

    octoMap->updateInnerOccupancy();

    int n = cloud->points.size();
#pragma omp parallel for
    for (int j = 0; j < n; ++j) {

        double odds = (octoMap->search(cloud->points[j].x,
                                       cloud->points[j].y,
                                       cloud->points[j].z))->getOccupancy();
        if(odds < odom_octomap_odd_thre)
            cloud->points[j] = nanPoint;
    }

//        std::vector<int> tmpVec;
//        pcl::removeNaNFromPointCloud(*localmap_surf, *localmap_surf, tmpVec);
//        pcl::removeNaNFromPointCloud(*localmap_corner, *localmap_corner, tmpVec);
}

void OctomapManager::saveOctoMap(const string filename){

    pcl::PointCloud<PointType>::Ptr occupied_nodes(new pcl::PointCloud<PointType>());

    for(octomap::OcTree::leaf_iterator it = octoMap->begin_leafs(), end = octoMap->end_leafs();it != end; ++it){

        PointType cube_center;
        cube_center.x = it.getX();
        cube_center.y = it.getY();
        cube_center.z = it.getZ();
        cube_center.intensity = it.getDepth();

        if(octoMap->isNodeOccupied(*it))
            occupied_nodes->points.push_back(cube_center);
    }
    if(!occupied_nodes->points.empty())
        pcl::io::savePCDFileBinary(filename, *occupied_nodes);
}

bool OctomapManager::OctoMaptoRosMsg(octomap_msgs::Octomap &octoMsg){

    return  octomap_msgs::fullMapToMsg(*octoMap, octoMsg);
}

bool OctomapManager::rayCasting(const PointType& origin_, const PointType& direction_, PointType& endPt_){

    octomap::point3d origin_Octo(origin_.x, origin_.y, origin_.z), end_Octo;
    octomap::point3d direction_Octo(direction_.x, direction_.y, direction_.z);
    if(octoMap->castRay(origin_Octo, direction_Octo, end_Octo)){

        endPt_.x = end_Octo.x();
        endPt_.y = end_Octo.y();
        endPt_.z = end_Octo.z();
        if (pointRange(endPt_) > 200) return false;

        return true;
    }else
        return false;
}


//////////////////////////////////// common functions /////////////////////////////////////////////////////
pcl::PointCloud<PointType>::Ptr transformPointCloud(const pcl::PointCloud<PointType>::Ptr cloudIn,
                                                    const PointTypePose* transformIn){

    if(cloudIn->empty())
        cout << BOLDMAGENTA << "[ Warning ] NO CLOUD DATA" << RESET << endl;

    Eigen::Quaternionf quat(transformIn->intensity, transformIn->roll, transformIn->pitch, transformIn->yaw);
//    cout << "[ Debug ] before nomalization " << quat.coeffs() << endl;
    quat.normalize();  /// the coeff wont change if it is a rotation
//    cout << "[ Debug ] after nomalization " << quat.coeffs() << endl;
    Eigen::Vector3f trans(transformIn->x, transformIn->y, transformIn->z);

    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    int cloudSize = cloudIn->points.size();
    cloudOut->points.resize(cloudSize);

#pragma omp parallel for
    for (int i = 0; i < cloudSize; ++i){

        Eigen::Vector3f ptVec = quat*(cloudIn->points[i].getVector3fMap()) + trans;
        PointType pointTo;
        pointTo.x = ptVec(0);
        pointTo.y = ptVec(1);
        pointTo.z = ptVec(2);
        pointTo.intensity = cloudIn->points[i].intensity;

        cloudOut->points[i] = pointTo;
    }
    return cloudOut;
}

Eigen::Isometry3d getTransformMatrix(const PointTypePose& pose){

    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    Eigen::Quaterniond estimated_rot(pose.intensity, pose.roll, pose.pitch,pose.yaw);
    estimated_rot.normalize();
    transform.rotate(estimated_rot);
    transform.pretranslate(Eigen::Vector3d(pose.x, pose.y, pose.z));
    return transform;
}

Eigen::Matrix4d getTransformMatrix4d(const PointTypePose& pose){

    Eigen::Quaterniond estimated_rot(pose.intensity, pose.roll, pose.pitch,pose.yaw);
    estimated_rot.normalize();
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.block<3,3>(0,0) = estimated_rot.toRotationMatrix();
    transform(0,3) = pose.x;
    transform(1,3) = pose.y;
    transform(2,3) = pose.z;
    return transform;
}

PointTypePose matrix4fToPose(const Eigen::Matrix4f &T, double time ){

    PointTypePose po;
    Eigen::Matrix3f rot = T.block(0,0, 3,3);
    Eigen::Quaternionf quat(rot);
    quat.normalize();
    po.roll  = quat.x();
    po.pitch = quat.y();
    po.yaw   = quat.z();
    po.intensity = quat.w();
    po.x = T(0,3);
    po.y = T(1,3);
    po.z = T(2,3);
    po.time = time;
    return po;
}

PointTypePose getRelativePose(const PointTypePose &p1, const PointTypePose &p2){

    Eigen::Matrix4d p1T = getTransformMatrix4d(p1);
    Eigen::Matrix4d p2T = getTransformMatrix4d(p2);
    Eigen::Matrix4d rT = p1T.inverse() * p2T;

    return matrix4fToPose(rT.cast<float>());
}

float pointRange(const PointType& pt){
    return sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
}

float pointDistBet(const PointType& pt1, const PointType& pt2){
    return sqrt((pt2.x-pt1.x)*(pt2.x-pt1.x) + (pt2.y-pt1.y)*(pt2.y-pt1.y) + (pt2.z-pt1.z)*(pt2.z-pt1.z));
}

void saveQuanPoseToFile(const std::string &file, const pcPosePtr pcQuanpose){

    FILE *fp = fopen(file.data(),"w");
    for(auto pose : pcQuanpose->points){

        fprintf(fp, "%lf %f %f %f %f %f %f %f\n", pose.time,
                pose.x, pose.y, pose.z,
                pose.roll, pose.pitch, pose.yaw,
                pose.intensity);
    }
    fclose(fp);
}

int readQuanPosefromfile(const std::string &file, pcPosePtr pcQuanpose, bool switchxyz) {

    PointTypePose ptRPY;
    char line[256];
    ifstream infile(file.c_str());
    if (infile.is_open()) {
        while (!infile.eof()) {
            infile.getline(line, 256);
            if(switchxyz)
                sscanf(line, "%lf %f %f %f %f %f %f %f\n", &ptRPY.time, &ptRPY.y, &ptRPY.z, &ptRPY.x,
                       &ptRPY.pitch, &ptRPY.yaw, &ptRPY.roll, &ptRPY.intensity); // roll=q.x  pitch=q.y  yaw=q.z  intensity=q.w
            else
                sscanf(line, "%lf %f %f %f %f %f %f %f\n", &ptRPY.time, &ptRPY.x, &ptRPY.y, &ptRPY.z,
                       &ptRPY.roll, &ptRPY.pitch, &ptRPY.yaw, &ptRPY.intensity); // roll=q.x  pitch=q.y  yaw=q.z  intensity=q.w

            pcQuanpose->push_back(ptRPY);
        }
    }
    infile.close();
    int keyposeSize = pcQuanpose->points.size() - 1;  //最后一个位姿读了两遍

    return keyposeSize;
}

pcXYZIptr transformPointCloudbyQuanPose(const pcXYZIptr cloudIn, PointTypePose transformIn, double maxR) {

    Eigen::Quaternionf quat(transformIn.intensity, transformIn.roll, transformIn.pitch, transformIn.yaw);
    Eigen::Matrix3f R = quat.toRotationMatrix();
    Eigen::Vector3f t(transformIn.x, transformIn.y, transformIn.z);
    Eigen::Vector3f transPt;

    pcXYZIptr cloudOut(new pcXYZI());

    pcl::PointXYZI *pointFrom;
    pcl::PointXYZI pointTo;

    int cloudSize = cloudIn->points.size();
    cloudOut->reserve(cloudSize);

    for (int i = 0; i < cloudSize; ++i) {

        pointFrom = &cloudIn->points[i];
        if (pointFrom->getVector3fMap().norm() > maxR)
            continue;
        transPt = R*pointFrom->getVector3fMap() + t;  // Rp+t
        pointTo.x = transPt(0);
        pointTo.y = transPt(1);
        pointTo.z = transPt(2);
        pointTo.intensity = pointFrom->intensity;

        cloudOut->points.emplace_back(pointTo);
    }
    return cloudOut;
}

bool getAndsaveglobalmapQuanTraj(const string &scanspath,
                                 const pcl::PointCloud<PointTypePose>::Ptr &pcQuatpose){

    pcXYZIptr globalmap(new pcXYZI());
    pcXYZIptr scan(new pcXYZI());

    int poseSize = pcQuatpose->points.size();
    std::string filename;

    for (int k = 0; k < poseSize; ++k) {

        filename = scanspath + to_string(pcQuatpose->points[k].time) + ".pcd";
        if(pcl::io::loadPCDFile<pcl::PointXYZI>(filename, *scan) != -1){

            *globalmap += *transformPointCloudbyQuanPose(scan, pcQuatpose->points[k], 70.0);
//            ccviewer.showCloud(globalmap);
//            ccviewer.wasStwopped(100000);
            cout << " —— scan " << k << " registered." << endl;
        } else {
            cout << "#no correspondent scan !" << endl;
            continue;
        }
    }

    if(!globalmap->empty()){
        pcl::io::savePCDFile(scanspath + "registeredCloud.pcd", *globalmap);
        cout << BOLDGREEN << "[ MAP ] Global map registred. " << RESET << endl;
    }

    return true;
}
// folder with '/' in the end
void buildmapfromQuatPosefile(const std::string &posefile,const std::string &scanfolder){

    pcPosePtr pcPoses(new pcl::PointCloud<PointTypePose>());
    readQuanPosefromfile(posefile, pcPoses);
    getAndsaveglobalmapQuanTraj(scanfolder, pcPoses);
}