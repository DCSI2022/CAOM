//
// Created by joe on 2020/10/10.
//

#ifndef STRUCTURAL_MAPPING_LINESTRUCTURE_H
#define STRUCTURAL_MAPPING_LINESTRUCTURE_H

#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>

template <class PointType>
class LineStructureClass{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double time;
    std::vector<int> indices_;  // indices of points on line in this scan
    std::vector<int> inliersOfscan_; // inliers to this line
    int scanlineid;  // 扫描线ID  VLP16: 0 ~ 15
    int numofpts; // 内点数
    Eigen::VectorXf coeffs;  // [ptx, pty, ptz, dirc1, d2, d3]
    double length;
    double linearity;

    bool onPlane; // 是否存在某面上
    int planeID; // 包含面的ID

    LineStructureClass(){

        time = -1;

        scanlineid=-1;
        numofpts=-1;
        length=-1;
        linearity=-1;
        onPlane = false;
        planeID = -1;
    }

    ~LineStructureClass()= default;

    // 获得线的方向向量
    Eigen::Vector3f getDirectionVec(){
        direction_(0,0) = coeffs(3,0);
        direction_(1,0) = coeffs(4,0);
        direction_(2,0) = coeffs(5,0);
        return direction_;
    }

    // 获得线上的点坐标
    Eigen::Vector3f getPointOnline(){
        ptOnline_(0,0) = coeffs(0,0);
        ptOnline_(1,0) = coeffs(1,0);
        ptOnline_(2,0) = coeffs(2,0);
        return ptOnline_;
    }

    // 利用新的坐标点更新线元参数
    bool updateCoeffs(const typename pcl::PointCloud<PointType>::Ptr& inCloud){

        std::vector<int> inliers;

        typename pcl::SampleConsensusModelLine<PointType>::Ptr model
                (new pcl::SampleConsensusModelLine<PointType>(inCloud, indices_));

        pcl::RandomSampleConsensus<PointType> ransacer(model);
        ransacer.setDistanceThreshold(0.1);
        ransacer.computeModel();
        ransacer.getModelCoefficients(coeffs);
        ransacer.getInliers(inliers);

//        if(inliers.size() < 0.7*indices_.size())
//        {
//            cout<<"× × × New line is not good enough. "<<endl;
//            return false;
//        }
        indices_.swap(inliers);

        numofpts = indices_.size();
        std::vector<int>().swap(inliers);

        return true;
    }

    // 计算线元各项参数
    void calculateLineParas(const typename pcl::PointCloud<PointType>::Ptr &inCloud, int id, const std::vector<int> &indices){

        indices_.assign(indices.begin(), indices.end());

        std::vector<int> inliers;
        Eigen::Vector3f cento;
        //计算直线模型参数
        typename pcl::SampleConsensusModelLine<PointType>::Ptr lineModel
                (new pcl::SampleConsensusModelLine<PointType>(inCloud, indices_));
//        pcl::RandomSampleConsensus<PointType> ransacer(lineModel); // TODO SIG!!
//        ransacer.setDistanceThreshold(0.03);  // 3cm
//        ransacer.computeModel();
//        ransacer.getModelCoefficients(coeffs);
//        ransacer.getInliers(inliers);
        typename pcl::RandomSampleConsensus<PointType>::Ptr ransacer(new pcl::RandomSampleConsensus<PointType>(lineModel));
        ransacer->setDistanceThreshold(0.03);  // 3cm
        ransacer->computeModel();
        ransacer->getModelCoefficients(coeffs);
        ransacer->getInliers(inliers);

        lineModel->getDistancesToModel(coeffs, squaredists_);
        if(inliers.size() < indices_.size()*0.5)
            cout<<"|LINE "<<id<<" is not good.|"<<endl;
//        cout<<"After RANSAC, inliers is "<<inliers.size()<<endl;

        //利用所有点到拟合直线的距离作为衡量线性程度的标准
        linearity = 0;
        for (int j = 0; j < squaredists_.size(); ++j)
            linearity += sqrt(squaredists_[j]);
        linearity /= squaredists_.size();

//                coeffpcl.values.resize(6);
//                for (int j = 0; j < 6; ++j)
//                    coeffpcl.values[j] = coeffsVec(j,0);

        scanlineid = id;
        numofpts = indices_.size();
        cento = (inCloud->points[indices_[0]].getVector3fMap() +
                 inCloud->points[indices_[numofpts-1]].getVector3fMap()) / 2;

        // find the cloest point on line to cento
        // and refer it as the point of coeff para
        double mindist = 1000, dist=0;
        PointType pt1;
        for (int k = 0; k < numofpts; ++k) {
            dist = (cento(0,0)-inCloud->points[indices_[k]].x)*
                   (cento(0,0)-inCloud->points[indices_[k]].x) +
                   (cento(1,0)-inCloud->points[indices_[k]].y)*
                   (cento(1,0)-inCloud->points[indices_[k]].y) +
                   (cento(2,0)-inCloud->points[indices_[k]].z)*
                   (cento(2,0)-inCloud->points[indices_[k]].z);
            if(dist < mindist){
                mindist = dist;
                pt1 = inCloud->points[indices_[k]];
            }
        }
        coeffs(0,0) = pt1.x;
        coeffs(1,0) = pt1.y;
        coeffs(2,0) = pt1.z;

        length = (inCloud->points[indices_[0]].getVector3fMap() -
                  inCloud->points[indices_[numofpts-1]].getVector3fMap()).norm();

        squaredists_.clear();
    }

    // 提取与线元距离一定的内点：inliersOfscan_
    void extractInliersByCoeffs(const typename pcl::PointCloud<PointType>::Ptr& inCloud, double distThre){

        inliersOfscan_.clear();

        Eigen::Vector3f linedirection, vec, centro;
        double d = 0;
        int cloudsize = inCloud->points.size();

        centro = getPointOnline();
        linedirection = getDirectionVec();
        linedirection = linedirection / linedirection.norm();

        for (int i = 0; i < cloudsize; ++i) {

            vec = inCloud->points[i].getVector3fMap() - centro;
            d = (vec.cross(linedirection)).norm() ; // distance from a point to line
            if(d < distThre)
                inliersOfscan_.push_back(i);
        }

    }


private:
    std::vector<double> squaredists_; // for temp use
    Eigen::Vector3f direction_; // direction of line
    Eigen::Vector3f ptOnline_;  // point on line

};


#endif //STRUCTURAL_MAPPING_LINESTRUCTURE_H
