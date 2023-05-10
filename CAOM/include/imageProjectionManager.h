//
// Created by cyz on 2022/2/17.
//

#ifndef STRUCTURAL_MAPPING_IMAGEPROJECTIONMANAGER_H
#define STRUCTURAL_MAPPING_IMAGEPROJECTIONMANAGER_H

#define MAX_RANGE 200.0
#define MIN_RANGE 1.0

#include "tools.h"

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

// the probability is saved in labelMat
#define Prob_New 0.55
#define Prob_Change 0.6
#define Prob_Static 0.4

class ImageProjectionManager{

    size_t rowN, colN;
    double verticalFov, vertical_res, horizon_res;
    OctomapManager octomapManager;

public:
    cv::Mat rangeMat, labelMat, indMat;

    ImageProjectionManager(double horizon_res_ = 0.02,
                           double vertical_res_ = 0.05,
                           double verticalFov_ = 40.0): horizon_res(horizon_res_),
                                                        vertical_res(vertical_res_),
                                                        verticalFov(verticalFov_){

        colN = 360.0 / horizon_res;
        rowN = verticalFov / vertical_res + 1;
        rangeMat = cv::Mat(rowN, colN, CV_32FC1, cv::Scalar::all(-1));
        labelMat = cv::Mat(rowN, colN, CV_32FC1, cv::Scalar::all(0.5));
        indMat = cv::Mat(rowN, colN, CV_32FC1, cv::Scalar::all(0.0));

        octomapManager.resetPara(2*horizon_res);
    }
    ~ImageProjectionManager(){}


    // from point cloud to range image
    int fromPointCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn,
                       Eigen::Vector3f origin = Eigen::Vector3f(0,0,0)){

        rangeMat = cv::Mat(rowN, colN, CV_32FC1, cv::Scalar::all(-1));
        indMat   = cv::Mat(rowN, colN, CV_32FC1, cv::Scalar::all(0));

        int invalidPtsN = 0, N = cloudIn->points.size();
        size_t u, v;
//#pragma omp parallel for num_threads(8)
        for(size_t i=0; i<N; i++){

            PointType pt = cloudIn->points[i];
            pt.x -= origin[0];
            pt.y -= origin[1];
            pt.z -= origin[2];
            float rangeCur = pointRange(pt);
            if (rangeCur < MAX_RANGE && projectPoint(pt, u, v)){

                float rangeBef = rangeMat.at<float>(u,v);
                if (rangeBef < 0 || rangeBef > rangeCur){
                    indMat.at<float>(u,v) = i;  // should be float as the ind may exceed the UINT range!
                    rangeMat.at<float>(u,v) = rangeCur;
//                    rangeMat.at<float>(u,v) = pt.intensity;  // for image visualize
//                    if (pt.intensity > 0) cout << "distance: " << pt.intensity << endl;
                }
            }else invalidPtsN++;
        }
        return invalidPtsN;
    }

    // from range image to point cloud
    int toPointCloud(pcl::PointCloud<PointType>::Ptr& cloud_selected){

        cloud_selected->clear();
        cloud_selected->reserve(rowN * colN);
        size_t emptyPixN = 0;
        PointType endPt;
        for (int i = 0; i < rowN; ++i) {
            for (int j = 0; j < colN; ++j) {
                if(projectPointInv(i, j, rangeMat.at<float>(i,j), endPt)){
                    endPt.intensity = labelMat.at<float>(i,j);
                    cloud_selected->points.emplace_back(endPt);
                }
                else emptyPixN++;
            }
        }
        return emptyPixN;
    }

    // from indmat to point cloud
    int toPointCloud(const pcl::PointCloud<PointType>::Ptr& cloud_Ori,
                     pcl::PointCloud<PointType>::Ptr& cloud_selected){

        cloud_selected->clear();
        cloud_selected->reserve(rowN * colN);
        size_t emptyPixN = 0;
        PointType endPt;
        for (int i = 0; i < rowN; ++i) {
            for (int j = 0; j < colN; ++j) {

                if(indMat.at<float>(i,j) > 0){
                    endPt = cloud_Ori->points[int(indMat.at<float>(i,j))];
                    endPt.intensity = labelMat.at<float>(i,j);
                    cloud_selected->points.emplace_back(endPt);
                }
                else emptyPixN++;
            }
        }
        return emptyPixN;
    }

    // Fixme: slow and center drift
    size_t generateImageByRayCasting(const pcl::PointCloud<PointType>::Ptr &cloudIn,
                                     const PointType& origin,
                                     pcl::PointCloud<PointType>::Ptr& cloud_selected){

//        pcl::PointCloud<PointType>::Ptr cloud_selected(new pcl::PointCloud<PointType>());
        cloud_selected->reserve(rowN * colN);

        size_t emptyPixN = 0;
        octomapManager.insertCloud2OctoMap(cloudIn, origin, MAX_RANGE);
        cout << GREEN << " [ octoMap ] Cloud inserted to OCTREE structure. " << RESET << endl;

        PointType directionPt, endPt;
        for (int i = 0; i < rowN; ++i) {
            for (int j = 0; j < colN; ++j) {
                projectPointInv(i, j, 1, directionPt);
                if(octomapManager.rayCasting(origin, directionPt, endPt)){

                    cloud_selected->points.emplace_back(endPt);
                    rangeMat.at<float>(i,j) = pointRange(endPt);
//                    labelMat.at<float>(i,j) = 0;
                }else emptyPixN++;
            }
        }
        return emptyPixN;
    }

    // from point to pixel
    bool projectPoint(const PointType thisPoint, size_t& rowIdn, size_t& columnIdn){

        // atan2~(-PI, PI]
        double verticalAngle = atan2(thisPoint.z,
                                     sqrt(thisPoint.x*thisPoint.x + thisPoint.y*thisPoint.y)) *180.0/M_PI;
        double ind = (verticalAngle + ang_bottom) / vertical_res;
        rowIdn = std::round(ind);
        if ( rowIdn < 0 || rowIdn >= rowN) return false;

        double horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180.0/M_PI;
        ind = (horizonAngle + 180.0)/horizon_res ;
        columnIdn = round(ind);
        if (columnIdn >= colN) columnIdn -= colN;
        if ( columnIdn < 0 || columnIdn >= colN) return false;
        return true;
    }

    // from pixel to point
    bool projectPointInv(const size_t& rowIdn, const size_t& columnIdn, const float& range, PointType& ptOut){

        if (range < 0 || range > MAX_RANGE) return false;
        float verticalAngle = (rowIdn * vertical_res - ang_bottom) * M_PI/180.0;
        float horizonAngle = (columnIdn * horizon_res - 90.0) * M_PI/180.0;
        ptOut.x = -range * cos(verticalAngle) * cos(horizonAngle);
        ptOut.y = range * cos(verticalAngle) * sin(horizonAngle);
        ptOut.z = range * sin(verticalAngle);
        return true;
    }

    bool writeRangeImage(const string& file){

        cv::Mat rangeMat_normalized, output;
//        rangeMat_normalized = cv::Mat(rowN, colN, CV_16UC1, cv::Scalar::all(0));
//        for (int j = 0; j < rowN; ++j) {
//            for (int k = 0; k < colN; ++k) {
//                if (rangeMat.at<float>(j,k) < MAX_RANGE)
//                    rangeMat_normalized.at<float>(j,k) = int(1.7 * rangeMat.at<float>(j,k));
//            }
//        }
        cv::normalize(rangeMat, rangeMat_normalized, 0, 255, CV_MINMAX, CV_8U);
        cv::applyColorMap(rangeMat_normalized, rangeMat_normalized, cv::COLORMAP_JET);
//        rangeMat_normalized.convertTo(rangeMat_normalized, CV_8UC1);

//        cv::resize(rangeMat_normalized, output,
//                    cv::Size(colN/5, rowN*5), 0, 0, cv::INTER_AREA);
//        cv::pyrDown(rangeMat_normalized, output,
//                    cv::Size(rangeMat_normalized.cols/5, rangeMat_normalized.rows/2));
//        cv::imshow( "pyrdown", output);
        if(!cv::imwrite(file , rangeMat_normalized))  return false;
        return true;
    }

    /// SRI: Spherical Range Image
    static void rangeImageDiff(const cv::Mat& SRI_MLS,const cv::Mat& SRI_SLAM,
                               cv::Mat& L_MLS, cv::Mat& L_SLAM){

        const float ratio = 0.06, maxDist = MAX_RANGE/2.0;
#pragma omp parallel for
        for (int row = 0; row < SRI_MLS.rows; ++row)
            for (int col = 0; col < SRI_MLS.cols; ++col) {

                float range1 = SRI_MLS.at<float>(row, col);
                float range2 = SRI_SLAM.at<float>(row, col);
                if (range1 > maxDist || range2 > maxDist) continue;
                float diff = range1 - range2;

                if (range2 < 0)  // SLAM is empty
                    continue;
                else if (range1 < 0)  // MLS is empty
                    L_SLAM.at<float>(row, col) = Prob_New;  // new
                else if (diff > range1*ratio)  // MLS is further
                    L_SLAM.at<float>(row, col) = Prob_Change;  // appeared
                else if (diff < -range1*ratio)  // SLAM is further
                    L_MLS.at<float>(row, col) = Prob_Change;  // disappeared
                else{
                    L_MLS.at<float>(row, col) = Prob_Static;  // no change
                    L_SLAM.at<float>(row, col) = Prob_Static;
                }
            }
    }

    void updateCloudStatus(pcl::PointCloud<PointType>::Ptr& cloudIn,
                           const vector<int>& ind_ORI){
#pragma omp parallel for
        for (int row = 0; row < rowN; ++row)
            for (int col = 0; col < colN; ++col) {

                int id = int(indMat.at<float>(row, col));
                if (id > 0) {
                    float prob = labelMat.at<float>(row, col);
                    float original = cloudIn->points[ind_ORI[id]].intensity;

                    if (original > 0 && !isnan(original))
                        original += log(prob / (1-prob));
                    else
                        original = 0.5;

                    cloudIn->points[ind_ORI[id]].intensity = min<float>(1.0, max<float>(0.0, original));
                }
            }
    }

};



#endif //STRUCTURAL_MAPPING_IMAGEPROJECTIONMANAGER_H
