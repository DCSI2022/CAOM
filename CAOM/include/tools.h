//
// Created by cyz on 2023/4/13.
//

#ifndef STRUCTURAL_MAPPING_TOOLS_H
#define STRUCTURAL_MAPPING_TOOLS_H

using namespace std;
#include "utilities.h"

//////////////////////////////////// common functions /////////////////////////////////////////////////////
// find nearest val in vector, return position
template<typename T>
size_t findClosest(const vector<T>& vec, T val){

    size_t found = 0;
    if (vec.front() < vec.back())
        found = upper_bound(vec.begin(), vec.end(), val) - vec.begin();
    else
        found = vec.rend() - upper_bound(vec.rbegin(), vec.rend(), val);

    if (found == 0)
        return found;

    if (found == vec.size())
        return found - 1;

    auto diff_next = abs(vec[found] - val);
    auto diff_prev = abs(val - vec[found - 1]);
    return diff_next < diff_prev ? found : found - 1;
}

struct smoothness_t{
    float value;
    size_t ind;
};

struct by_value{
    bool operator()(smoothness_t const &left, smoothness_t const &right) {
        return left.value < right.value;
    }
};

inline Eigen::Matrix3d skew(const Eigen::Vector3d& mat_in){

    Eigen::Matrix<double,3,3> skew_mat;
    skew_mat.setZero();
    skew_mat(0,1) = -mat_in(2);
    skew_mat(0,2) =  mat_in(1);
    skew_mat(1,2) = -mat_in(0);
    skew_mat(1,0) =  mat_in(2);
    skew_mat(2,0) = -mat_in(1);
    skew_mat(2,1) =  mat_in(0);
    return skew_mat;
}

template <typename MatrixType>
typename MatrixType::Scalar logDet(const MatrixType &M, bool use_cholesky = false){

    using namespace Eigen;
    typedef typename MatrixType::Scalar Scalar;

    Scalar ld = 0;
    if (use_cholesky)
    {
        LLT<Matrix<Scalar, Dynamic, Dynamic>> chol(M);
        auto &U = chol.matrixL();
        for (unsigned i = 0; i < M.rows(); ++i)
            ld += std::log(U(i, i)); // or ld+= std::log(prod(U.diagonal()))
        ld *= 2;
    }
    else
    {
        PartialPivLU<Matrix<Scalar, Dynamic, Dynamic>> lu(M);
        auto &LU = lu.matrixLU();
        Scalar c = lu.permutationP().determinant(); // -1 or 1
        for (unsigned i = 0; i < LU.rows(); ++i)
        {
            const auto &lii = LU(i, i);
            if (lii < Scalar(0))
                c *= -1;
            ld += std::log(abs(lii));
        }
        ld += std::log(c);
    }
    return ld;
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(const pcl::PointCloud<PointType>::Ptr cloudIn,
                                                    const PointTypePose* transformIn);


Eigen::Isometry3d getTransformMatrix(const PointTypePose& pose);

Eigen::Matrix4d getTransformMatrix4d(const PointTypePose& pose);

PointTypePose matrix4fToPose(const Eigen::Matrix4f &T, double time = 0);

PointTypePose getRelativePose(const PointTypePose &p1, const PointTypePose &p2);

float pointRange(const PointType& pt);

float pointDistBet(const PointType& pt1, const PointType& pt2);

void saveQuanPoseToFile(const std::string &file, const pcPosePtr pcQuanpose);

int readQuanPosefromfile(const std::string &file, pcPosePtr pcQuanpose, bool switchxyz = false) ;

pcXYZIptr transformPointCloudbyQuanPose(const pcXYZIptr cloudIn, PointTypePose transformIn,
                                        double maxR = 99) ;

bool getAndsaveglobalmapQuanTraj(const string &scanspath,
                                 const pcl::PointCloud<PointTypePose>::Ptr &pcQuatpose);
// folder with '/' in the end
void buildmapfromQuatPosefile(const std::string &posefile,const std::string &scanfolder);

////// Evaluating the overlapping area between two polygons /////////////////////////////////
class PolygonOverlapHelper{

    const double eps = 1e-6;

public:

    // 位置标识
    int dcmp(double x){
        if(x > eps) return 1;
        return x < -eps ? -1 : 0;
    }

    struct Point{
        double x, y;
    };

    double cross(Point a, Point b, Point c){
        return (a.x-c.x)*(b.y-c.y)-(b.x-c.x)*(a.y-c.y);  // 叉积公式
    }

    // 计算线段ab和cd的交点坐标
    Point intersection(Point a, Point b, Point c, Point d);

    // 计算多边形面积，将多边形拆解成连续三个顶点组合成的多个三角形进行计算，这个循环计算一次其实是计算两次多边形的面积。
    double PolygonArea(Point p[], int n);

    // ConvexPolygonIntersectArea
    double CPIA(Point a[], Point b[], int na, int nb) ;

    // SimplePolygonIntersectArea 调用此函数
    double SPIA(Point a[], Point b[], int na, int nb) ;
};

////////////////// Octomap for map filter //////////////////////////////////////////////////
class OctomapManager{

    shared_ptr<octomap::OcTree> octoMap;
    PointType nanPoint;

public:

    OctomapManager(float octoRes_ = odom_octomap_res) { resetPara(octoRes_); }
    OctomapManager(const pcl::PointCloud<PointType>::Ptr cloud, float octoRes_ = odom_octomap_res){

        resetPara(octoRes_);
        insertCloud2OctoMap(cloud);
    }

    void resetPara(float octoRes){

        octoMap.reset(new octomap::OcTree(octoRes));

//        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
//        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
//        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.x = 0;
        nanPoint.y = 0;
        nanPoint.z = 0;
        nanPoint.intensity = -1;
    }

    // copy from octomap_ros/conversion.h
    void pointCloud2ToOctomap(const sensor_msgs::PointCloud2& cloud,
                              octomap::Pointcloud& octomapCloud);

    void insertCloud2OctoMap(const pcl::PointCloud<PointType>::Ptr cloudin, double max_range = 300);

    void insertCloud2OctoMap(const pcl::PointCloud<PointType>::Ptr cloudin, const PointType& origin,
                             double max_range = 300);

    void filterLocalOccupancyMap(pcl::PointCloud<PointType>::Ptr cloud);

    void saveOctoMap(const string filename);

    bool OctoMaptoRosMsg(octomap_msgs::Octomap &octoMsg);

    bool rayCasting(const PointType& origin_, const PointType& direction_, PointType& endPt_);
};

/// structure for incremental kdtree from Nanoflann ///////////////////////////////////////
template <typename T>
struct PointCloudforKDTREE
{
    struct Point
    {
        T  x,y,z;
    };

    std::vector<Point>  pts;

    // Must return the number of data points
    size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};

#endif //STRUCTURAL_MAPPING_TOOLS_H
