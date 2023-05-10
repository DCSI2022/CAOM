/////////////////////////////////////////////////////////////////////
// Created by joe on 2020/10/26.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef STRUCTURAL_MAPPING_POSEESTIMATIONLIB_HPP
#define STRUCTURAL_MAPPING_POSEESTIMATIONLIB_HPP

#include "tools.h"
#include "ceresAnalyticCost.h"
#include "constraintSphere.hpp"
//#include "point_with_cov.h"

#include <bits/stdc++.h>
#include <Eigen/Sparse>
#include <random>

#define GNC_Factor 11.8
#define selectedptThre 150  // number of selected feature points according to the list

#define MAX_FEATURE_SELECT_TIME 20
#define MAX_RANDOM_QUEUE_TIME 20
#define NUM_MATCH_POINTS 5

template <typename T>
struct RandomGeneratorInt{

    std::random_device random_device;
    std::mt19937 m_random_engine;
    std::uniform_int_distribution<T> m_dist;
    RandomGeneratorInt() : m_random_engine(std::random_device{}()){};
    ~RandomGeneratorInt(){};

    T geneRandUniform(T low = 0, T hight = 100){
        m_dist = std::uniform_int_distribution<T>(low, hight);
        return m_dist(m_random_engine);
    }

    T *geneRandUniformArray(T low = 0, T hight = 100, size_t numbers = 100){
        T *res = new T[numbers];
        m_dist = std::uniform_int_distribution<T>(low, hight);
        for (size_t i = 0; i < numbers; i++)
            res[i] = m_dist(m_random_engine);

        return res;
    }

    T *geneRandArrayNoRepeat(T low, T high, T k){
        T n = high - low;
        T *res_array = new T[k];
        std::vector<T> foo;
        foo.resize(n);
        for (T i = 1; i <= n; ++i)
            foo[i] = i + low;
        std::shuffle(foo.begin(), foo.end(), m_random_engine);
        for (T i = 0; i < k; ++i){
            res_array[i] = foo[i];
            // std::cout << foo[ i ] << " ";
        }
        return res_array;
    }
};

class PoseEstimationManager{

    // translation first
//    double transformation[7] = {0, 0, 0, 0, 0, 0, 1};
//    Eigen::Map<Eigen::Quaterniond> q_w_curr = Eigen::Map<Eigen::Quaterniond>(transformation+3);
//    Eigen::Map<Eigen::Vector3d> t_w_curr = Eigen::Map<Eigen::Vector3d>(transformation);

    double transformation[7] = {0, 0, 0, 1, 0, 0, 0};
    Eigen::Map<Eigen::Quaterniond> q_w_curr = Eigen::Map<Eigen::Quaterniond>(transformation);
    Eigen::Map<Eigen::Vector3d> t_w_curr = Eigen::Map<Eigen::Vector3d>(transformation + 4);

    deque<pair<Eigen::Isometry3d, double> > deltaPoses;

    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> poseCov;
    Eigen::Matrix4d poseT;
    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> poseCov_pre;
    Eigen::Matrix4d poseT_pre;

    Eigen::Isometry3d odom, odom_last, odom_delta;

    int optiCnt, skipframeNum, minOptiCnt = 3;

    pcl::PointCloud<PointType>::Ptr localmap_corner_;
    pcl::PointCloud<PointType>::Ptr localmap_surf_;
    pcl::PointCloud<PointType>::Ptr globalmap_localize;
    pcl::PointCloud<PointType>::Ptr localmap_all;

    pcl::PointCloud<PointType>::Ptr curCloud_corner;
    pcl::PointCloud<PointType>::Ptr curCloud_surf;
    pcl::PointCloud<PointType>::Ptr surfptsSelected;
    pcl::PointCloud<PointInfoType>::Ptr ptsInfoUsed;

    pcl::KdTreeFLANN<PointType>::Ptr kdtree_map_corner;
    pcl::KdTreeFLANN<PointType>::Ptr kdtree_map_surf;

    pcl::CropBox<PointType>::Ptr cropBoxFilter;
    pcl::VoxelGrid<PointType> downSampler_map;
    float map_ds_leaf_size = 0;

    bool useOctomap = false, localizeMode = false, useObservationSelection = 0;
    bool constrainted = false, poseEstimated = false;

    int curCornerNum, curSurfNum;
    int corner_numUsed, surf_numUsed;
    int mapCornerNum, mapSurfNum;
    int frameNum = 0;
    double costTotal, costTotal_pre;
    Eigen::Matrix<double, 1, Eigen::Dynamic> edgeWeights;  // cost^2
    Eigen::Matrix<double, 1, Eigen::Dynamic> edgeResi;
    Eigen::Matrix<double, 1, Eigen::Dynamic> surfWeights;
    Eigen::Matrix<double, 1, Eigen::Dynamic> surfResi;
    std::mutex factor_mutex;

    std::unique_ptr<ceres::Problem> problemPtr;
    std::unique_ptr<ConstraintSphere> constraintSpherePtr;
    std::vector<ceres::ResidualBlockId > resiBlockIds;


    bool addEdgeCostFactor(ceres::LossFunction *loss_func, int& iter){

        curCornerNum = curCloud_corner->points.size();
        corner_numUsed = 0;
//        edgeResi = Eigen::Matrix<double, 1, Eigen::Dynamic>(curCornerNum);
//        edgeResi.setZero();
        std::unique_lock<std::mutex> unique_lock(factor_mutex, std::defer_lock);

//#pragma omp parallel for
        for (int i = 0; i < curCornerNum; i++){
            PointType point_trans;
            pointAssociateToMap(&(curCloud_corner->points[i]), &point_trans);

            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtree_map_corner->nearestKSearch(point_trans, 5, pointSearchInd, pointSearchSqDis);
            if (pointSearchSqDis[4] < 1.0){

                std::vector<Eigen::Vector3d> nearCorners;
                Eigen::Vector3d center(0, 0, 0);
                for (int j = 0; j < 5; j++){
                    Eigen::Vector3d tmp(localmap_corner_->points[pointSearchInd[j]].x,
                                        localmap_corner_->points[pointSearchInd[j]].y,
                                        localmap_corner_->points[pointSearchInd[j]].z);
                    center = center + tmp;
                    nearCorners.push_back(tmp);
                }
                center = center / 5.0;

                Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                for (int j = 0; j < 5; j++){
                    Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                    covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                }

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                Eigen::Vector3d curr_point(curCloud_corner->points[i].x, curCloud_corner->points[i].y, curCloud_corner->points[i].z);
                if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]){

                    Eigen::Vector3d point_on_line = center;
                    Eigen::Vector3d point_a, point_b;
                    point_a = 0.1 * unit_direction + point_on_line;
                    point_b = -0.1 * unit_direction + point_on_line;

                    PointInfoType ptInfo;
                    ptInfo.x = point_trans.x;
                    ptInfo.y = point_trans.y;
                    ptInfo.z = point_trans.z;
                    ptInfo.intensity = point_trans.intensity;
                    unit_direction.normalize();
                    // 计算点到直线的垂线方向向量( https://blog.csdn.net/tanmengwen/article/details/8472849#commentBox )
                    Eigen::Vector3f v1 = point_trans.getVector3fMap() - point_a.cast<float>();
                    float t = (v1.dot(unit_direction.cast<float>())) / (unit_direction.squaredNorm());
                    v1 = point_a.cast<float>() + t*unit_direction.cast<float>();  // 垂点（位于直线上）
                    Eigen::Vector3f v2 = v1 - point_trans.getVector3fMap();  // vector of two points
                    ptInfo.curvature = v2.norm();  // dist
                    v2.normalize();

                    if(iter > 1 && abs(ptInfo.curvature) > 0.2)  // todo : dynamic points?
                        continue;

                    ceres::CostFunction *cost_function = new EdgeAnalyticCostFunction(curr_point, point_a,
                                                                                      point_b, edgeWeights(i));
                    resiBlockIds.emplace_back(problemPtr->AddResidualBlock(cost_function,
                                                                           loss_func,
                                                                           transformation));

                    ptInfo.normal_x = v2(0);
                    ptInfo.normal_y = v2(1);
                    ptInfo.normal_z = v2(2);
                    {
                        unique_lock.lock();
                        ptsInfoUsed->points.emplace_back(ptInfo);
                        costTotal += ptInfo.curvature;
                        unique_lock.unlock();
                    }

                    edgeResi(i) = std::pow(ptInfo.curvature, 2);
                    corner_numUsed++;
                }
            }
        }
        if(corner_numUsed < 20)
            cout << BOLDYELLOW <<"[ PEM ] WARN : not enough Cost edge points ! " << endl;
        cout << GREEN << "[ PEM ] Cost edge points : " << corner_numUsed << RESET << endl;

        return true;
    }

    bool addSurfCostFactor(ceres::LossFunction *loss_func,
                           std::vector<ceres::CostFunction*> &surfcostFuncs_all,
                           int& iter){

        curSurfNum = curCloud_surf->points.size();
        surf_numUsed = 0;
//        surfResi = Eigen::Matrix<double, 1, Eigen::Dynamic>(1, curSurfNum);
//        surfResi.setZero();
        std::unique_lock<std::mutex> unique_lock(factor_mutex, std::defer_lock);

//#pragma omp parallel for
        for (int i = 0; i < curSurfNum; i++){

            PointType point_trans;
            pointAssociateToMap(&(curCloud_surf->points[i]), &point_trans);
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            kdtree_map_surf->nearestKSearch(point_trans, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
            if (pointSearchSqDis[4] < 1.0){

                for (int j = 0; j < 5; j++)
                {
                    matA0(j, 0) = localmap_surf_->points[pointSearchInd[j]].x;
                    matA0(j, 1) = localmap_surf_->points[pointSearchInd[j]].y;
                    matA0(j, 2) = localmap_surf_->points[pointSearchInd[j]].z;
                }
                // find the norm of plane
                Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                double negative_OA_dot_norm = 1 / norm.norm();
                norm.normalize();

                bool planeValid = true;
                for (int j = 0; j < 5; j++){
                    // if OX * n > 0.2, then plane is not fit well
                    if (fabs(norm(0) * localmap_surf_->points[pointSearchInd[j]].x +
                             norm(1) * localmap_surf_->points[pointSearchInd[j]].y +
                             norm(2) * localmap_surf_->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                    {
                        planeValid = false;
                        break;
                    }
                }
                Eigen::Vector3d curr_point(curCloud_surf->points[i].x, curCloud_surf->points[i].y, curCloud_surf->points[i].z);
                if (planeValid){

                    PointInfoType ptInfo;
                    ptInfo.x = curr_point(0);
                    ptInfo.y = curr_point(1);
                    ptInfo.z = curr_point(2);
//                    ptInfo.x = point_trans.x;
//                    ptInfo.y = point_trans.y;
//                    ptInfo.z = point_trans.z;
                    ptInfo.intensity = point_trans.intensity;
                    Eigen::Vector3f direcVec = point_trans.getVector3fMap() -
                                               localmap_surf_->points[pointSearchInd[0]].getVector3fMap();

                    ptInfo.curvature = (norm.cast<float>()).dot(point_trans.getVector3fMap())
                                       + negative_OA_dot_norm;  // point to plane dist as weight

                    if(iter > 1 && abs(ptInfo.curvature) > 0.2)  // todo : dynamic points?
                        continue;

                    ceres::CostFunction *cost_function = new SurfNormAnalyticCostFunction(curr_point, norm,
                                                                                          negative_OA_dot_norm,
                                                                                          surfWeights(i));
                    surfcostFuncs_all.emplace_back(cost_function);
//                    resiBlockIds.emplace_back(problemPtr->AddResidualBlock(cost_function,
//                                                                           loss_func,
//                                                                           transformation));
                    // distance should be positive
                    if(ptInfo.curvature < 0)
                        ptInfo.curvature = -1 * ptInfo.curvature;
                    else if((norm.cast<float>()).dot(direcVec) > 0)
                        norm = -1 * norm;  // point lies in the opposite direction of plane normal

                    ptInfo.normal_x = norm(0);
                    ptInfo.normal_y = norm(1);
                    ptInfo.normal_z = norm(2);
                    {
                        unique_lock.lock();
                        ptsInfoUsed->points.emplace_back(ptInfo);
                        costTotal += ptInfo.curvature;
                        unique_lock.unlock();
                    }
//                    surfResi(i) = std::pow(ptInfo.curvature, 2);
                    surf_numUsed++;
                }
            }

        }
        if(surf_numUsed < 20){
            cout << BOLDYELLOW <<"[ PEM ] WARN : not enough Cost SURF points ! " << endl;
            return false;
        }
        cout << GREEN << "[ PEM ] Cost SURF points : " << surf_numUsed << RESET << endl;
        return true;
    }

    // update the weights for Truncated Least Square
    bool updateWeightsTLS(Eigen::Matrix<double, 1, Eigen::Dynamic> &weights,
                          Eigen::Matrix<double, 1, Eigen::Dynamic> &residuals,
                          double size, double noiseBound_sq, double thre1, double thre2, double mu){

        for (int i = 0; i < size; ++i) {
            if(residuals(i) == 0)
                continue;
            if(residuals(i) >= thre1)
                weights(i) = 0;
            else if(residuals(i) <= thre2)
                weights(i) = 1.0;
            else{
                weights(i) = std::sqrt(noiseBound_sq *mu* (mu+1) / residuals(i)) - mu;
                assert(weights(i)>=0 && weights(i)<=1.0);
            }
        }
        return true;
    }

    inline void pointAssociateToMap(const PointType* pi, PointType* po){

        Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
        Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
        po->x = point_w.x();
        po->y = point_w.y();
        po->z = point_w.z();
        po->intensity = pi->intensity;
    }

    // filter out points with low occupancy
    void filterLocalOccupancyMap(){

        if(!useOctomap)
            return;
        cout << MAGENTA << "[ PEM ] Updating Local map Occupancy..." << RESET << endl;

        octomapManager.filterLocalOccupancyMap(localmap_corner_);
        octomapManager.filterLocalOccupancyMap(localmap_surf_);
    }

    bool setMapKdtree(){

        if(localizeMode){
            if(localmap_all->points.size() < 100)
                return false;

            kdtree_map_surf->setInputCloud(localmap_all);
            kdtree_map_corner->setInputCloud(localmap_all);
//            pcl::io::savePCDFileBinaryCompressed(projPath+"localmap_all.pcd", *localmap_all);
            return true;
        }

        mapSurfNum = localmap_surf_->points.size();
        mapCornerNum = localmap_corner_->points.size();

        if(mapSurfNum < 50 || mapCornerNum < 10){
            cout << YELLOW << "[ PEM ] Local map is small with " << mapSurfNum << " planar points. " << RESET << endl;
            return false;
        }

        kdtree_map_corner->setInputCloud(localmap_corner_);
        kdtree_map_surf->setInputCloud(localmap_surf_);
        return true;
    }


public:
    OctomapManager octomapManager;

    PoseEstimationManager(float ds_leafsize = 0.3f,
                          int skipframeNum_ = 1,
                          bool useConstraintSphere = 1,
                          int optiCnt_ = 12){

        optiCnt = optiCnt_;
        skipframeNum = skipframeNum_;
        map_ds_leaf_size = ds_leafsize;
        useObservationSelection = useConstraintSphere;

        cout << "[ PEM ] optiCnt :" << optiCnt << endl;
        cout << "[ PEM ] skipframeNum :" << skipframeNum << endl;
        cout << "[ PEM ] map_ds_leaf_size :"  << map_ds_leaf_size << endl;
        cout << "[ PEM ] useObservationSelection :"  << useObservationSelection << endl;

        localmap_corner_.reset(new pcl::PointCloud<PointType>());
        localmap_surf_  .reset(new pcl::PointCloud<PointType>());

        ptsInfoUsed  .reset(new pcl::PointCloud<PointInfoType>());

        curCloud_surf  .reset(new pcl::PointCloud<PointType>());
        curCloud_corner  .reset(new pcl::PointCloud<PointType>());
        surfptsSelected  .reset(new pcl::PointCloud<PointType>());

        localmap_all  .reset(new pcl::PointCloud<PointType>());
        globalmap_localize.reset(new pcl::PointCloud<PointType>());

        kdtree_map_surf  .reset(new pcl::KdTreeFLANN<PointType>());
        kdtree_map_corner.reset(new pcl::KdTreeFLANN<PointType>());

        cropBoxFilter.reset(new pcl::CropBox<PointType>());

        downSampler_map.setLeafSize(ds_leafsize, ds_leafsize, ds_leafsize);

        odom = Eigen::Isometry3d::Identity();
        odom_last = Eigen::Isometry3d::Identity();

        poseCov = Eigen::Matrix<double , 6, 6, Eigen::RowMajor>::Zero();
        poseT = Eigen::Matrix4d::Zero();
    }

    void init(const pcl::PointCloud<PointType>::Ptr& cloudcornerIn,
              const pcl::PointCloud<PointType>::Ptr& cloudsurfIn,
              bool useOctomap_ = false, string mapfile = "",
              bool localizeMode_ = false){

        *localmap_corner_ += *cloudcornerIn;
        *localmap_surf_ += *cloudsurfIn;

        useOctomap = useOctomap_;
        localizeMode = localizeMode_;

        if(useOctomap){
            octomapManager.insertCloud2OctoMap(localmap_corner_);
            octomapManager.insertCloud2OctoMap(localmap_surf_);
        }

        if(localizeMode){

            if(pcl::io::loadPCDFile(mapfile, *globalmap_localize) != -1)
                cout << GREEN << "[ PEM ] Load local map success. " << RESET << endl;
            else {
                useOctomap = false;
                return;
            }

            double x_min =  - odom_localmapRadi;
            double y_min =  - odom_localmapRadi;
            double z_min =  - odom_localmapRadi;
            double x_max =  + odom_localmapRadi;
            double y_max =  + odom_localmapRadi;
            double z_max =  + odom_localmapRadi;

            cropBoxFilter->setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
            cropBoxFilter->setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
            cropBoxFilter->setNegative(false);

            cropBoxFilter->setInputCloud(globalmap_localize);
            cropBoxFilter->filter(*localmap_all);
            downSampler_map.setInputCloud(localmap_all);
            downSampler_map.filter(*localmap_all);
        }

    }

    void resetPara(){

        optiCnt = 8;
        frameNum = 0;

        localmap_corner_->clear();
        localmap_surf_->clear();
        localmap_all->clear();

        curCloud_surf->clear();
        curCloud_corner->clear();

        odom = Eigen::Isometry3d::Identity();
        odom_last = Eigen::Isometry3d::Identity();
        odom_delta = Eigen::Isometry3d::Identity();

        deltaPoses.clear();
    }


    inline void deriveMap(pcl::PointCloud<PointType>::Ptr& cloudcornerIn,
                          pcl::PointCloud<PointType>::Ptr& cloudsurfIn){

        pcl::copyPointCloud(*localmap_corner_, *cloudcornerIn);
        cloudcornerIn->width = cloudcornerIn->points.size();
        cloudcornerIn->height = 1;
        pcl::copyPointCloud(*localmap_surf_, *cloudsurfIn);
        cloudsurfIn->width = cloudsurfIn->points.size();
        cloudsurfIn->height = 1;
        if(localizeMode){
            pcl::copyPointCloud(*localmap_all, *cloudsurfIn);
            cloudsurfIn->width = cloudsurfIn->points.size();
            cloudsurfIn->height = 1;
        }
    }

    inline void deriveSelectedSurf(pcl::PointCloud<PointType>::Ptr& cloudsurfIn){

        if(surfptsSelected->empty())
            pcl::copyPointCloud(*curCloud_surf, *cloudsurfIn);
        else
            pcl::copyPointCloud(*surfptsSelected, *cloudsurfIn);

        cloudsurfIn->width = cloudsurfIn->points.size();
        cloudsurfIn->height = 1;
    }

    void evalHessian(const ceres::CRSMatrix &jaco, Eigen::Matrix<double, 6, 6> &mat_H){

        if (jaco.num_rows == 0) return;
        Eigen::SparseMatrix<double, Eigen::RowMajor> mat_J; // Jacobian is a diagonal matrix
        mat_J.resize(jaco.num_rows, jaco.num_cols);
//        mat_J.setZero();
        for (auto row = 0; row < jaco.num_rows; row++)
        {
            int start = jaco.rows[row];
            int end = jaco.rows[row + 1] - 1;
            for (auto i = start; i <= end; i++)
            {
                int col = jaco.cols[i];
                mat_J.coeffRef(row, col) = jaco.values[i];
            }
        }
        Eigen::SparseMatrix<double, Eigen::RowMajor> mat_Jt = mat_J.transpose();
        Eigen::MatrixXd mat_JtJ = mat_Jt * mat_J;
        mat_H = mat_JtJ.block(0, 0, 6, 6);  // normalized the hessian matrix for pair uncertainty evaluation
    }

    /* !!! should be called after optimization */
    inline double getCurPose(double* pose, bool getCov = false,
                             double* poseCovMatrix = new double[36]){

        if(!poseEstimated) return -1;
        double uncertainty = -1;
        for (int i=0; i<7; i++)
            pose[i] = transformation[i];

        bool ifcompound = false;
        if (poseCov.sum() > 0 ){
            poseCov_pre = poseCov;
            poseT_pre = odom_last.matrix();
            ifcompound = true;
        }
        std::ofstream txtor(projPath+"pose_evas.txt",std::ios::app);

        /// option1: use ceres to estimate covariance
//        ceres::Covariance::Options covOption;
//        covOption.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
//        covOption.num_threads = 4;
//        covOption.apply_loss_function = false;
//        ceres::Covariance covariance(covOption);
//
//        std::vector<std::pair<const double*, const double*> > covBlocks;
//        covBlocks.push_back(std::make_pair(transformation, transformation));
//        if(covariance.Compute(covBlocks, problemPtr.get())){
//            covariance.GetCovarianceBlockInTangentSpace(transformation, transformation,  // in tangent space
//                                                        poseCov.data());
//            uncertainty = poseCov.trace();
//            cout << BOLDWHITE << "[ PEM ] Pose Uncertainty(trace) : " << uncertainty << RESET << endl;
//
//            txtor << setprecision(5) << uncertainty << " ";
//        }

        /// option2: self-defined covariance estimation based on Hessian J^TJ
        std::vector<double *> parablocks;
        parablocks.push_back(transformation);
        ceres::Problem::EvaluateOptions e_options;
        e_options.parameter_blocks = parablocks;
        e_options.residual_blocks = resiBlockIds;
        ceres::CRSMatrix jaco;
        Eigen::Matrix<double, 6, 6> mat_H; // mat_H / 134 = normlized_mat_H
        double *costs = new double[resiBlockIds.size()];
        std::vector<double> residuals_Vec;
        problemPtr->Evaluate(e_options, costs, &residuals_Vec, nullptr, &jaco);
        evalHessian(jaco, mat_H);
        poseCov = mat_H.inverse();
        if (ifcompound){
            Eigen::Matrix<double, 6, 6> poseCov_comp;
            compoundPoseCovariance(poseCov_pre, poseCov, poseT_pre, poseCov_comp);
            poseCov = poseCov_comp;
        }
        uncertainty = poseCov.trace();
        cout << BOLDWHITE << "[ PEM ] Pose Uncertainty(JTJ) : " << uncertainty << RESET << endl;
        txtor << setprecision(5) << uncertainty << " ";

        const int n_costs = resiBlockIds.size();
        Eigen::Map<Eigen::VectorXd > costVec(costs, n_costs);
        float sumCosts = costVec.sum();
        float sumResi = accumulate(residuals_Vec.begin(), residuals_Vec.end(), 0);
        if(!isnormal(sumCosts) || sumCosts < 0) sumCosts = 0;
        cout << BOLDWHITE << "[ PEM ] SUM of COSTS : " << sumCosts << RESET << endl;
        cout << BOLDWHITE << "[ PEM ] SUM of residuals : " << sumResi << RESET << endl;  // Fixme : why 0?
        txtor << setprecision(5) << sumCosts << " ";
        txtor << setprecision(5) << costTotal/(corner_numUsed+surf_numUsed) << endl;
        txtor.close();

        // for weighted initial pose estimation
        odom_delta = (odom_last.inverse()) * odom;
        deltaPoses.emplace_back(std::make_pair(odom_delta, uncertainty));
        if(deltaPoses.size() > 6)
            deltaPoses.pop_front();

        if(getCov)
            std::memcpy(poseCovMatrix, poseCov.data(), 36* sizeof(double));
        return uncertainty;
    }

    // fixme : weighted averaging of delta poses in SE3 ?
    void predictDelta(Eigen::Isometry3d &deltaT){

        double n = static_cast<double>(deltaPoses.size());
        if(n==0.0) return;

        double w = 2.0 / (n*(n+1)), sum = 0.0, weight, no;
        Eigen::Quaterniond rotDelta = Eigen::Quaterniond::Identity(), rotQua;
        const Eigen::Quaterniond identity = Eigen::Quaterniond::Identity();
        Eigen::Vector3d tDelta;

        for (int i = 0; i < (int)n; ++i)
            sum += deltaPoses[i].second;
        sum = 1.0/sum;  // >1
        w *= sum;

        for (int i = 0; i < (int)n; ++i) {

            no = static_cast<double>(i+1);
            weight = w * no * deltaPoses[i].second;
            if(weight > 1) cout << "[ Warning ] slerp ratio > 1" << endl;

            tDelta += rotDelta * (weight * deltaPoses[i].first.translation());
            rotQua = Eigen::Quaterniond(deltaPoses[i].first.rotation());
            rotDelta = rotDelta * (identity.slerp(weight, rotQua));
        }
        deltaT.linear() = rotDelta.toRotationMatrix();
        deltaT.translation() = tDelta;

//        cout << "[ PEM ] Latest delta : \n " << deltaPoses.back().first.matrix() << endl;
        //cout << "[ PEM ] Predicted delta : \n " << deltaT.matrix() << endl;
    }

    // region # greedy based feature selection 20230403 #
    void evaluateFeatJacobianMatching(const ceres::CostFunction* costFactor,
                                      PointFeature &feature){

        int SIZE_POSE = 7;

        double **param = new double *[1];
        param[0] = new double[SIZE_POSE];
        param[0][0] = t_w_curr.x();
        param[0][1] = t_w_curr.y();
        param[0][2] = t_w_curr.z();
        param[0][3] = q_w_curr.x();
        param[0][4] = q_w_curr.y();
        param[0][5] = q_w_curr.z();
        param[0][6] = q_w_curr.w();

        double *res = new double[1];
        double **jaco = new double *[1];
        jaco[0] = new double[1 * 7];
        // if (feature.type_ == 's')
        // {

        costFactor->Evaluate(param, res, jaco);
        // std::cout << "after lidarmapplanenormfactor" << std::endl;
        // }
        // else if (feature.type_ == 'c')
        // {
        //     LidarMapEdgeFactor f(feature.point_, feature.coeffs_, cov_matrix);
        //     f.Evaluate(param, res, jaco);
        // }

        Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> mat_jacobian(jaco[0]);
        feature.jaco_ = mat_jacobian.topLeftCorner<1, 6>();

        /* double *rho = new double[3];
         double sqr_error = res[0] * res[0] + res[1] * res[1] + res[0] * res[0];
         ceres::LossFunction *loss_function_;
         loss_function_->Evaluate(sqr_error, rho);
         std::cout << "after loss_funciton" << std::endl;
         std::cout << "jaco: " << jaco[0] << std::endl;
         // feature.jaco_ *= sqrt(std::max(0.0, rho[1])); // TODO
         std::cout << "error: " << sqrt(sqr_error) << ", rho_der: " << rho[1]
                   << ", logd: " << common::logDet(feature.jaco_.transpose() * feature.jaco_, true);*/

        delete[] res;
        // delete[] rho;
        delete[] jaco[0];
        delete[] jaco;
        delete[] param[0];
        delete[] param;
    }

    void goodFeatureSelect(std::vector<int> &sel_feature_idx,
                           const std::vector<ceres::CostFunction*>& surfCostFuncVec,
                           double gf_ratio_cur, const std::string gf_method){

        cout << "Using Greedy based feature selection..." << endl;
        RandomGeneratorInt<size_t> rgi_;
        std::vector<PointFeature> all_features;
        size_t num_all_features = surf_numUsed;  // !!!
        all_features.resize(num_all_features);
        Eigen::Matrix<double, 6, 6> sub_mat_H = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;;

        std::vector<size_t> all_feature_idx(num_all_features);
        std::vector<int> feature_visited(num_all_features, -1);
        std::iota(all_feature_idx.begin(), all_feature_idx.end(), 0);

        if (gf_method == "full"){
            sel_feature_idx.resize(num_all_features);
            std::iota(sel_feature_idx.begin(), sel_feature_idx.end(), 0);
            return;
        }

        size_t num_use_features;
        num_use_features = static_cast<size_t>(num_all_features * gf_ratio_cur);
        sel_feature_idx.resize(num_use_features);
        size_t num_sel_features = 0;
        pcl::console::TicToc t_sel_feature;

        bool b_match;
        double cur_det;
        size_t num_rnd_que;
        size_t n_neigh = 5;

        while (true){
            if ((num_sel_features >= num_use_features) ||
                (all_feature_idx.size() == 0))
//                || (t_sel_feature.toc() > MAX_FEATURE_SELECT_TIME))
                break;

            size_t size_rnd_subset = static_cast<size_t>(1.0 * num_all_features / num_use_features);
            std::priority_queue<FeatureWithScore,
                    std::vector<FeatureWithScore>,
                    std::less<FeatureWithScore> > heap_subset;
            while (true){

                if (all_feature_idx.size() == 0) break;
                num_rnd_que = 0;
                size_t j;
                while (num_rnd_que < MAX_RANDOM_QUEUE_TIME){

                    j = rgi_.geneRandUniform(0, all_feature_idx.size() - 1);
                    if (feature_visited[j] < int(num_sel_features)){

                        feature_visited[j] = int(num_sel_features);
                        break;
                    }
                    num_rnd_que++;
                }
                if (num_rnd_que >= MAX_RANDOM_QUEUE_TIME )
//                || t_sel_feature.toc() > MAX_FEATURE_SELECT_TIME)
                    break;

                size_t que_idx = all_feature_idx[j];

                //  FIXME: w/o ua,cov is zero
//                Eigen::Matrix3d cov_matrix;
//                    Eigen::Matrix3d cov_point = Eigen::Matrix3d::Zero();
//                    common::PointIWithCov point_cov(feats_ori->points[que_idx], cov_point.cast<float>());
//                    common::extractCov(point_cov, cov_matrix);

                evaluateFeatJacobianMatching(surfCostFuncVec[que_idx], all_features[que_idx]);


                const Eigen::MatrixXd &jaco = all_features[que_idx].jaco_;
                cur_det = logDet(sub_mat_H + jaco.transpose() * jaco, true);
                heap_subset.push(FeatureWithScore(que_idx, cur_det, jaco));

                if (heap_subset.size() >= size_rnd_subset){

                    const FeatureWithScore &fws = heap_subset.top();
                    std::vector<size_t>::iterator iter = std::find(all_feature_idx.begin(), all_feature_idx.end(), fws.idx_);
                    if (iter == all_feature_idx.end()){
                        std::cerr << "[ goodFeatureMatching ]: not exist feature idx !" << std::endl;
                        break;
                    }
                    sub_mat_H += fws.jaco_.transpose() * fws.jaco_;

                    size_t position = iter - all_feature_idx.begin();
                    all_feature_idx.erase(all_feature_idx.begin() + position);
                    feature_visited.erase(feature_visited.begin() + position);
                    sel_feature_idx[num_sel_features] = fws.idx_;
                    num_sel_features++;
                    // printf("position: %lu, num: %lu\n", position, num_rnd_que);
                    break;
                }
            }
            if (num_rnd_que >= MAX_RANDOM_QUEUE_TIME )
//            || t_sel_feature.toc() > MAX_FEATURE_SELECT_TIME)
                break;
        }
        if (num_rnd_que >= MAX_RANDOM_QUEUE_TIME )
//        || t_sel_feature.toc() > MAX_FEATURE_SELECT_TIME)
        {
            std::cerr << "mapping [goodFeatureMatching]: early termination!" << std::endl;
            std::cout << "early termination: " << num_rnd_que << ", " << t_sel_feature.toc() << std::endl;
        }

        sel_feature_idx.resize(num_sel_features);
        printf("num of all features: %lu, sel features: %lu\n", num_all_features, num_sel_features);
    }
    //endregion

    bool ceresSolver(const pcl::PointCloud<PointType>::Ptr& cloudcornerIn,
                     const pcl::PointCloud<PointType>::Ptr& cloudsurfIn,
                     bool setInitVal = false,
                     Eigen::Isometry3d initPose = Eigen::Isometry3d::Identity()){

        if(!setMapKdtree())
            return false;

        pcl::copyPointCloud(*cloudcornerIn, *curCloud_corner);
        pcl::copyPointCloud(*cloudsurfIn, *curCloud_surf);

        if(optiCnt > minOptiCnt)
            optiCnt --;
        cout << CYAN << "[ PEM ] Optimize Count " << optiCnt << RESET << endl;
        Eigen::Isometry3d odom_predict, odomDelta(Eigen::Isometry3d::Identity()), odom_deltaT;  // todo :
        double resdidualThre = 1;

        if(!setInitVal){
//            predictDelta(odomDelta);
            odomDelta = (odom_last.inverse()) * odom;
//            odomDelta.translation().z() = 0;  // fixme
//            double norm = odomDelta.rotation().eulerAngles(2,1,0).determinant();
//            if(norm > M_PI/2)
            odom_predict = odom * odomDelta;
        }else
            odom_predict = initPose;

        odom_last = odom;
        odom = odom_predict;

        q_w_curr = Eigen::Quaterniond(odom.rotation());
        t_w_curr = odom.translation();

        std::vector<ceres::CostFunction*> surfcostFuncAll;
        bool degenerated, updateWeights = true;
        double noiseBound = 0.15;  // fixme: upper bound of noise ?
        double costThre = 0.0000000005;  //
        double mu = 1.0, noiseBoundsquare = std::pow(noiseBound, 2);
        curCornerNum = curCloud_corner->points.size();
        curSurfNum = curCloud_surf->points.size();
        edgeResi = Eigen::Matrix<double, 1, Eigen::Dynamic>::Zero(curCornerNum);
        surfResi = Eigen::Matrix<double, 1, Eigen::Dynamic>::Zero(curSurfNum);
        edgeWeights = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(curCornerNum);
        surfWeights = Eigen::Matrix<double, 1, Eigen::Dynamic>::Ones(curSurfNum);

        costTotal_pre = 0;

        for (int iter = 0; iter < optiCnt; iter++) {

//            ceres::LossFunction *loss_func = new ceres::HuberLoss(0.1);
            ceres::LossFunction *loss_func = new ceres::CauchyLoss(0.1);
            ceres::Problem::Options problem_options;

            problemPtr.reset(new ceres::Problem(problem_options));
            problemPtr->AddParameterBlock(transformation, 7, new PoseSE3Parameterization());

            degenerated = false;
            poseEstimated = false;
            resiBlockIds.resize(0);
            surfcostFuncAll.resize(0);
            ptsInfoUsed->clear();
            surfptsSelected->clear();
            costTotal = 0;

            // multi-thread
            std::vector<std::future<bool>> factor_build_thread(2);
            std::vector<bool> results(2);
            factor_build_thread[0] = std::async(std::launch::async|std::launch::deferred,
                                                &PoseEstimationManager::addEdgeCostFactor, this,
                                                std::ref(loss_func), std::ref(iter));
            factor_build_thread[1] = std::async(std::launch::async|std::launch::deferred,
                                                &PoseEstimationManager::addSurfCostFactor,this,
                                                std::ref(loss_func), std::ref(surfcostFuncAll), std::ref(iter));
            for (int fac_th = 0; fac_th < 2; ++fac_th)
                results[fac_th] = factor_build_thread[fac_th].get();
            if(!results[1]){
                degenerated = true;
                cout << BOLDYELLOW << "[ PEM ] Warining: degenerated! " << RESET << endl;
                continue;
            }

            assert(ptsInfoUsed->size() == (corner_numUsed + surf_numUsed));
            assert(surfcostFuncAll.size() == surf_numUsed);

//            addEdgeCostFactor(loss_func, iter);
//            if (!addSurfCostFactor(loss_func, surfcostFuncAll, iter)){
//                degenerated = true;
//                continue;  // dont update pose
//            }

            if(useObservationSelection) {  // both is effective especially in indoor environments
                const clock_t begT = clock();
                std::vector<int> indsUsed;
//                buildConstraintSphere(ptsInfoUsed, indsUsed);      // opt1: select feature points by direction sphere
//                observationSelection(ptsInfoUsed,indsUsed);        // opt2: select top feature points
                goodFeatureSelect(indsUsed, surfcostFuncAll, 0.4, "gf");  // opt3: greedy search feature points

//            cout << "[ Debug ] Iteration " << iter << ", Indices num in surf to be added : "<< indsUsed.size() << endl;

                float seconds = float(clock( ) - begT) / 1000; //最小精度到ms
                ofstream timeTxt(projPath + "observTime.txt", ios::app);
                timeTxt << seconds << endl;
                timeTxt.close();

                for (auto id : indsUsed) {
                    resiBlockIds.push_back(problemPtr->AddResidualBlock(surfcostFuncAll[id],  // index of surf costs
                                                                        loss_func,
                                                                        transformation));
//                    surfptsSelected->points.emplace_back(curCloud_surf->points[id]);
//                problemPtr->RemoveResidualBlock(resiBlockIds[id]);
                    surfResi(id) = std::pow(ptsInfoUsed->points[id + corner_numUsed].curvature, 2);
                }
            }else{
                for(int id = 0 ; id < surf_numUsed; id++){
                    resiBlockIds.push_back(problemPtr->AddResidualBlock(surfcostFuncAll[id], loss_func, transformation));
//                problemPtr->RemoveResidualBlock(resiBlockIds[id]);
                    surfResi(id) = std::pow(ptsInfoUsed->points[id + corner_numUsed].curvature, 2);
//                    surfptsSelected->points.emplace_back(curCloud_surf->points[id]);
                }
            }

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.trust_region_strategy_type = ceres::DOGLEG;
            options.dogleg_type = ceres::SUBSPACE_DOGLEG;
            options.max_num_iterations = 4;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;

//            options.minimizer_type = ceres::MinimizerType::LINE_SEARCH;
//            options.line_search_direction_type = ceres::LineSearchDirectionType::BFGS;
            ceres::Solver::Summary summary;
            ceres::Solve(options, problemPtr.get(), &summary);
//            cout << BLACK << summary.BriefReport() << RESET << endl;


            if (iter == 0){  // calculate the initial mu
                double maxEdgeResi = edgeResi.maxCoeff(), maxSurfResi = surfResi.maxCoeff();
                double maxResi = std::max(maxEdgeResi, maxSurfResi);
                mu = 1 / (2*maxResi/noiseBoundsquare - 1.0);
                if (mu <= 0) mu = 1e-10;
            }
            // get thresholds for GNC
            double thre_1 = (mu+1) / mu * noiseBoundsquare;
            double thre_2 = mu / (mu+1) * noiseBoundsquare;
            if(updateWeights){
                updateWeightsTLS(edgeWeights, edgeResi, curCornerNum, noiseBoundsquare, thre_1, thre_2, mu);
                updateWeightsTLS(surfWeights, surfResi, curSurfNum, noiseBoundsquare, thre_1, thre_2, mu);
            }
            // incremental update mu
            mu = mu * std::exp(double(iter+1) * GNC_Factor);

            cout << MAGENTA << "[ PEM ] iter " << iter << ", current COST : " << costTotal << RESET << endl;
            if(abs(costTotal - costTotal_pre) < costThre)
                break;
            costTotal_pre = costTotal;
        }

        if(!degenerated){

            odom = Eigen::Isometry3d::Identity();
            odom.linear() = q_w_curr.toRotationMatrix();
            odom.translation() = t_w_curr;

            Eigen::Isometry3d odom_delta2;
            odom_delta2 = odom_predict.inverse() * odom;
            odom_deltaT = odom_delta2.inverse() * odomDelta;  // deviation from constant velocity model
            resdidualThre = 2 * 80 * sin(0.5 * std::acos(0.5 * (odom_deltaT.rotation().trace()-1)))  // rot err
                            + odom_deltaT.translation().norm() ;  // trans err
            cout << GREEN << "[ DEBUG ] resdidualThre: " << resdidualThre << RESET << endl;
        }
        poseT  = odom.matrix();
        // cout << GREEN << "[ PEM ] Transformation Estimated. \n" << poseT << RESET << endl;
        poseEstimated = true;

        return true;
    }

    void updatelocalMap(const pcl::PointCloud<PointType>::Ptr curCloud_corner_,
                        const pcl::PointCloud<PointType>::Ptr curCloud_surf_){

        pcl::PointCloud<PointType>::Ptr tmpCloud(new pcl::PointCloud<PointType>());
        curCornerNum = curCloud_corner_->points.size();
        mapCornerNum = localmap_corner_->points.size();
        //cout << CYAN << "[ PEM ] Local corner map size " << mapCornerNum << RESET << endl;
        localmap_corner_->points.resize(mapCornerNum + curCornerNum);
        tmpCloud->points.resize(curCornerNum);
#pragma omp parallel for
        for (int i = 0; i < curCornerNum; i++){

            PointType point_temp;
            pointAssociateToMap(&curCloud_corner_->points[i], &point_temp);
//            Eigen::Matrix3d ptCov;
//            evalPointUncertainty(curCloud_surf_->points[i], poseT, poseCov, ptCov);
//            point_temp.intensity = ptCov.trace();
//            localmap_corner_->push_back(point_temp);
            tmpCloud->points[i] = point_temp;
            localmap_corner_->points[mapCornerNum+i] = point_temp;
//            octoMap->insertRay(octomap::point3d(0, 0 ,0),
//                               octomap::point3d(point_temp.x, point_temp.y, point_temp.z));
        }
        if(useOctomap)
            octomapManager.insertCloud2OctoMap(tmpCloud);

        curSurfNum = curCloud_surf_->points.size();
        mapSurfNum = localmap_surf_->points.size();
        //cout << CYAN << "[ PEM ] Local surf map size " << mapSurfNum << RESET << endl;
        localmap_surf_->points.resize(mapSurfNum + curSurfNum);
        tmpCloud->points.resize(curSurfNum);
#pragma omp parallel for
        for (int i = 0; i < curSurfNum; i++){

            PointType point_temp;
            pointAssociateToMap(&curCloud_surf_->points[i], &point_temp);
            Eigen::Matrix3d ptCov;
            evalPointUncertainty(curCloud_surf_->points[i], poseT, poseCov, ptCov);
            point_temp.intensity = ptCov.trace();
//            if(ptCov.trace() > ptUncertainThre)
//                continue;
//            localmap_surf_->push_back(point_temp);
//            tmpCloud->points.emplace_back(point_temp);
            tmpCloud->points[i] = point_temp;
            localmap_surf_->points[mapSurfNum+i] = point_temp;
//            octoMap->insertRay(octomap::point3d(0, 0 ,0),
//                               octomap::point3d(point_temp.x, point_temp.y, point_temp.z));
        }
        if(useOctomap)
            octomapManager.insertCloud2OctoMap(tmpCloud);

        frameNum++;
        if(frameNum % skipframeNum)
            return;

        // debug
        if(!localmap_corner_->empty())
            pcl::io::savePCDFileBinaryCompressed("/home/cyz/workspace/testData/mapUncertainty/"+
                                                 to_string(rand()%500)+".pcd",
                                                 (*localmap_surf_+*localmap_corner_));

        //cout << MAGENTA << "[ odom ] Cropping Local map..." << RESET << endl;
        double x_min = odom.translation().x() - odom_localmapRadi;
        double y_min = odom.translation().y() - odom_localmapRadi;
        double z_min = odom.translation().z() - odom_localmapRadi;
        double x_max = odom.translation().x() + odom_localmapRadi;
        double y_max = odom.translation().y() + odom_localmapRadi;
        double z_max = odom.translation().z() + odom_localmapRadi;
        cropBoxFilter->setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
        cropBoxFilter->setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
        cropBoxFilter->setNegative(false);

        if(localizeMode){

            localmap_all->clear();
            cropBoxFilter->setInputCloud(globalmap_localize);
            cropBoxFilter->filter(*localmap_all);
            downSampler_map.setInputCloud(localmap_all);
            downSampler_map.filter(*localmap_all);

            return;
        }

        tmpCloud->clear();
        cropBoxFilter->setInputCloud(localmap_surf_);
        cropBoxFilter->filter(*tmpCloud);
        downSampler_map.setLeafSize(map_ds_leaf_size, map_ds_leaf_size, map_ds_leaf_size);  // todo
//        downSampler_map.setLeafSize(2*map_ds_leaf_size, 2*map_ds_leaf_size, 2*map_ds_leaf_size);
        downSampler_map.setInputCloud(tmpCloud);
        downSampler_map.filter(*localmap_surf_);

        tmpCloud->clear();
        cropBoxFilter->setInputCloud(localmap_corner_);
        cropBoxFilter->filter(*tmpCloud);
        downSampler_map.setLeafSize(map_ds_leaf_size, map_ds_leaf_size, map_ds_leaf_size);
        downSampler_map.setInputCloud(tmpCloud);
        downSampler_map.filter(*localmap_corner_);

        if(!(frameNum % 2*skipframeNum) && odom_octoFilterMap)  // TODO odom_octoFilterMap
            filterLocalOccupancyMap();
    }


    // TODO: memory-efficient voxel structure for hashing 20230130
    void updatelocalMapVoxelHash(const pcl::PointCloud<PointType>::Ptr curCloud_corner_,
                                 const pcl::PointCloud<PointType>::Ptr curCloud_surf_){

        pcl::PointCloud<PointType>::Ptr tmpCloud(new pcl::PointCloud<PointType>());
        curCornerNum = curCloud_corner_->points.size();
        mapCornerNum = localmap_corner_->points.size();
        //cout << CYAN << "[ PEM ] Local corner map size " << mapCornerNum << RESET << endl;
        localmap_corner_->points.resize(mapCornerNum + curCornerNum);
        tmpCloud->points.resize(curCornerNum);
#pragma omp parallel for
        for (int i = 0; i < curCornerNum; i++){

            PointType point_temp;
            pointAssociateToMap(&curCloud_corner_->points[i], &point_temp);
//            Eigen::Matrix3d ptCov;
//            evalPointUncertainty(curCloud_surf_->points[i], poseT, poseCov, ptCov);
//            point_temp.intensity = ptCov.trace();
//            localmap_corner_->push_back(point_temp);
            tmpCloud->points[i] = point_temp;
            localmap_corner_->points[mapCornerNum+i] = point_temp;
//            octoMap->insertRay(octomap::point3d(0, 0 ,0),
//                               octomap::point3d(point_temp.x, point_temp.y, point_temp.z));
        }
        if(useOctomap)
            octomapManager.insertCloud2OctoMap(tmpCloud);

        curSurfNum = curCloud_surf_->points.size();
        mapSurfNum = localmap_surf_->points.size();
        //cout << CYAN << "[ PEM ] Local surf map size " << mapSurfNum << RESET << endl;
        localmap_surf_->points.resize(mapSurfNum + curSurfNum);
        tmpCloud->points.resize(curSurfNum);
#pragma omp parallel for
        for (int i = 0; i < curSurfNum; i++){

            PointType point_temp;
            pointAssociateToMap(&curCloud_surf_->points[i], &point_temp);
            Eigen::Matrix3d ptCov;
            evalPointUncertainty(curCloud_surf_->points[i], poseT, poseCov, ptCov);
            point_temp.intensity = ptCov.trace();
//            if(ptCov.trace() > ptUncertainThre)
//                continue;
//            localmap_surf_->push_back(point_temp);
//            tmpCloud->points.emplace_back(point_temp);
            tmpCloud->points[i] = point_temp;
            localmap_surf_->points[mapSurfNum+i] = point_temp;
//            octoMap->insertRay(octomap::point3d(0, 0 ,0),
//                               octomap::point3d(point_temp.x, point_temp.y, point_temp.z));
        }
        if(useOctomap)
            octomapManager.insertCloud2OctoMap(tmpCloud);

        frameNum++;
        if(frameNum % skipframeNum)
            return;

        // debug
        if(!localmap_corner_->empty())
            pcl::io::savePCDFileBinaryCompressed("/home/cyz/workspace/testData/mapUncertainty/"+
                                                 to_string(rand()%500)+".pcd",
                                                 (*localmap_surf_+*localmap_corner_));

        //cout << MAGENTA << "[ odom ] Cropping Local map..." << RESET << endl;
        double x_min = odom.translation().x() - odom_localmapRadi;
        double y_min = odom.translation().y() - odom_localmapRadi;
        double z_min = odom.translation().z() - odom_localmapRadi;
        double x_max = odom.translation().x() + odom_localmapRadi;
        double y_max = odom.translation().y() + odom_localmapRadi;
        double z_max = odom.translation().z() + odom_localmapRadi;
        cropBoxFilter->setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
        cropBoxFilter->setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
        cropBoxFilter->setNegative(false);

        if(localizeMode){

            localmap_all->clear();
            cropBoxFilter->setInputCloud(globalmap_localize);
            cropBoxFilter->filter(*localmap_all);
            downSampler_map.setInputCloud(localmap_all);
            downSampler_map.filter(*localmap_all);

            return;
        }

        tmpCloud->clear();
        cropBoxFilter->setInputCloud(localmap_surf_);
        cropBoxFilter->filter(*tmpCloud);
        downSampler_map.setLeafSize(map_ds_leaf_size, map_ds_leaf_size, map_ds_leaf_size);  // todo
//        downSampler_map.setLeafSize(2*map_ds_leaf_size, 2*map_ds_leaf_size, 2*map_ds_leaf_size);
        downSampler_map.setInputCloud(tmpCloud);
        downSampler_map.filter(*localmap_surf_);

        tmpCloud->clear();
        cropBoxFilter->setInputCloud(localmap_corner_);
        cropBoxFilter->filter(*tmpCloud);
        downSampler_map.setLeafSize(map_ds_leaf_size, map_ds_leaf_size, map_ds_leaf_size);
        downSampler_map.setInputCloud(tmpCloud);
        downSampler_map.filter(*localmap_corner_);

        if(!(frameNum % 2*skipframeNum) && odom_octoFilterMap)  // TODO odom_octoFilterMap
            filterLocalOccupancyMap();
    }

    void setlocalMap(const pcl::PointCloud<PointType>::Ptr& cloudcornerIn,
                     const pcl::PointCloud<PointType>::Ptr& cloudsurfIn){

        pcl::copyPointCloud(*cloudcornerIn, *localmap_corner_);
        pcl::copyPointCloud(*cloudsurfIn, *localmap_surf_);
        cout << GREEN << "[ PEM ] Set Corner Map " << localmap_corner_->size() <<
             ", Surf Map " << localmap_surf_->size() << RESET << endl;
    }

    // https://blog.csdn.net/dsoftware/article/details/107184116 TODO
    void farthestPointSampling(const pcl::PointCloud<PointType>::Ptr& cloudIn,
                               pcl::PointCloud<PointType>::Ptr& cloudOut, int num){

        int n = cloudIn->size();
        Eigen::Vector4f centro;
        pcl::compute3DCentroid(*cloudIn, centro);
        Eigen::Vector3f centro3f(centro(0), centro(1), centro(2));
        auto* dist = new float[n];
        float maxDist = 0;
        int oriInd = -1;
        for (int i = 0; i < n; ++i) {  // choose the farthest point from centroid as start point
            dist[i] = (cloudIn->points[i].getVector3fMap() - centro3f).norm();
            if(dist[i] > maxDist){
                oriInd = i;
                maxDist = dist[i];
            }
        }
        std::max_element(dist, dist+n);
    }

    /// Build sphere using normal direction to visualize the constraints equality
    /// \param[in] cloudInfo : all built costs with point direction info
    /// \param[out] indices : selected surf cost function to be added to problem
    void buildConstraintSphere(pcInfoPtr& cloudInfo,
                               std::vector<int>& indices){
        cout << "Using Constraint Sphere based feature selection..." << endl;
        //cout << BOLDRED << "[ PEM ] Final total cost(sum of distances) " << costTotal << RESET << endl;
        indices.resize(0);
        constraintSpherePtr.reset(new ConstraintSphere(std::floor(sqrt(costTotal/2))));

        int n = cloudInfo->size();
        Eigen::Matrix<double, 3, 1> radsquare;

        // add all corner cost
        for (int j = 0; j < corner_numUsed; ++j) {

            auto const & normal = ptsInfoUsed->points[j].getNormalVector3fMap();
//            double const & weight = 1.0/ (1.0+exp(ptsInfoUsed->points[j].curvature)); // todo : weight function
            double const & weight = edgeWeights(j);

            constraintSpherePtr->addConstraint(normal, weight);
        }
        radsquare = constraintSpherePtr->radii().cwiseProduct(constraintSpherePtr->radii());

        ofstream ofs(projPath + "constraintSphereCorner.txt", ios::app);
        ofs << radsquare.transpose() << std::endl;
        ofs.close();
        cout << "[ Debug ] Corner points radii : " << radsquare.transpose() << endl;

        constraintSpherePtr->setThreshold( 1.5*sqrt(radsquare.maxCoeff()));  // TODO threshold

        // Index, score, normal, weight
        std::vector<std::tuple<size_t, double, Eigen::Vector3f, double> > scores;

        for( size_t i = 0; i < surf_numUsed; ++i ){

//            if( used[i] ) continue;

            auto const & point = cloudInfo->at(i + corner_numUsed);
            if( point.getVector3fMap().norm() < 0.1 ) continue; // points at origin

            Eigen::Vector3f normal(point.normal_x, point.normal_y, point.normal_z);
//            double const score = constraintSpherePtr->getScore(normal) * point.curvature;
            double const score = constraintSpherePtr->getScore(normal);
//            double weight = 1.0 / (1.0+exp(point.curvature));  // todo : weight function
            double weight = surfWeights(i);

            if( !std::isnormal(score) ) continue;
            scores.emplace_back(i, score, normal, weight);
        }

        // Sort the scores, so the highest score is in the back
        std::sort(scores.begin(), scores.end(),
                  [](decltype(scores)::value_type const & a, decltype(scores)::value_type const & b)
                  { return std::get<1>(a) < std::get<1>(b); });

        constrainted = false;
        std::random_device rd;
        std::mt19937 gen(rd());
        int cnt = 0;
//        for(auto s : scores){
        while (!scores.empty()){

            cnt ++;
            auto const & s = scores.back();
            auto const & normal = std::get<2>(s);
            double const & weight = std::get<3>(s);
            double const & score = constraintSpherePtr->getScore(normal) * weight;  // it's updating
            if(weight > 1.0) continue;  // cost error

//            if(score > 0)
//                std::cout << "score: " << score << " " << weight << std::endl;
//            std::bernoulli_distribution dis(std::max(score, 0.05));
//            std::bernoulli_distribution dis(score);
//            if(!dis(gen)) continue;

            size_t const index = std::get<0>(s);

//            // Compute the covariance
//            nrt::Indices const neighborhood = useKNN ? getNeighborhoodKNN(cloud, index, search) :
//                                              is32 ? getNeighborhoodVelodyne<32>(cloud, index) :
//                                              getNeighborhoodVelodyne<64>(cloud, index);
//            Eigen::Matrix3d covariance = getCovariance( cloud, neighborhood );
//            auto & c = cloud.get<Covariance>(index).get<Covariance>();
//            c.value = covariance;
//            c.valid = true;

            constraintSpherePtr->addConstraint(normal, weight);
            indices.emplace_back(index);

            auto const & radii = constraintSpherePtr->radii();
            if(radii[2] > constraintSpherePtr->getThreshold() &&
               radii[0] / radii[2] < 1.3f){
                constrainted = true;
                ROS_INFO("Point Constraint Filled");
                break;
            }
            scores.pop_back();

            if(cnt % 15) continue;  // TODO re-order per ? points

            for(auto& item : scores)  // update scores
                std::get<1>(item) = constraintSpherePtr->getScore(std::get<2>(item));
            std::sort(scores.begin(), scores.end(),
                      [](decltype(scores)::value_type const & a, decltype(scores)::value_type const & b)
                      { return std::get<1>(a) < std::get<1>(b); });
        }

        radsquare = constraintSpherePtr->radii().cwiseProduct(constraintSpherePtr->radii());

        if(constrainted){
            ofstream ofs(projPath + "constraintSphere.txt", ios::app);
            ofs << radsquare.transpose() << std::endl;
            ofs.close();
        }
//        for(auto item1 : scores)  // left
//            indices.emplace_back(std::get<0>(item1));

        cout << "  -> Added " << indices.size() << " / " << n-corner_numUsed << " points --- Final radii: "
             << radsquare.transpose() << std::endl;
    }

    static bool sort_list_function(std::pair<int, float> a, std::pair<int, float> b) {
        return (a.second > b.second);
    }
    // Select better observations to build cost function
    void observationSelection(pcInfoPtr& cloudIn,
                              std::vector<int>& indices){
        cout << "Using List based feature selection..." << endl;

        indices.clear();
        int allsize = cloudIn->size();
        assert(corner_numUsed + surf_numUsed == allsize);

        PointType pointOri, pointSel;
        std::vector<float> planarScalar_vector;
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > normal_vector;
        // opt1: using the direction of cost vector
        for (int k = corner_numUsed; k < allsize; ++k) {  // begin from surf points
            planarScalar_vector.emplace_back(cloudIn->points[k].curvature);
//            planarScalar_vector.emplace_back(cloudIn->points[k].curvature*surfWeights(k-corner_numUsed));
            Eigen::Vector3f norm(cloudIn->points[k].normal_x,
                                 cloudIn->points[k].normal_y,
                                 cloudIn->points[k].normal_z);
            normal_vector.emplace_back(norm);
        }
        // region opt2: recalculate planarity using neighorhood PCA
//        for (int k = corner_numUsed; k < allsize; ++k) {  // begin from surf points
//
//            pointOri.x = cloudIn->points[k].x;
//            pointOri.y = cloudIn->points[k].y;
//            pointOri.z = cloudIn->points[k].z;
//            pointAssociateToMap(&pointOri, &pointSel);
//            std::vector<int> pointSearchInd;
//            std::vector<float> pointSearchSqDis;
//            kdtree_map_surf->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
//
//            Eigen::Vector3f currentPoint(pointSel.x, pointSel.y, pointSel.z);
//            Eigen::Vector3f average(currentPoint);
//            int pointsNum = pointSearchInd.size();
//            for (int i = 0; i < pointsNum; ++i)
//                average += localmap_surf_->points[pointSearchInd[i]].getVector3fMap();
//
//            average /= float(pointsNum + 1);
//            Eigen::Matrix<float, 3, Eigen::Dynamic> centeredPoints_mat = Eigen::Matrix<float, 3, Eigen::Dynamic>(
//                    3, pointsNum + 1);
//            for (int j = 0; j < pointsNum; ++j)
//                centeredPoints_mat.block<3, 1>(0, j) = localmap_surf_->points[pointSearchInd[j]].getVector3fMap()
//                                                       - average;
//
//            centeredPoints_mat.block<3, 1>(0, pointsNum) = currentPoint - average;
//            Eigen::Matrix3f covariance_mat = centeredPoints_mat * centeredPoints_mat.transpose();
//            covariance_mat /= float(pointsNum + 1);
//
//            const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 3, 3> > solver(covariance_mat);
//            Eigen::Vector3f eigenValues = solver.eigenvalues();
//
//            // note Eigen library sort eigenvalues in increasing order
////        cout << "sort eigenvalues in increasing order: " << eigenValues(0) << " " << eigenValues(1) << " " << eigenValues(2) << std::endl;
//            float planarScalar = (sqrt(eigenValues(1)) - sqrt(eigenValues(0)))
//                    / sqrt(eigenValues(2));
//            planarScalar_vector.push_back(planarScalar);
//
//            const Eigen::Matrix3f eigenVectors = solver.eigenvectors();
//            Eigen::Vector3f normal(eigenVectors.col(0));
//            normal.normalize();
//            normal_vector.push_back(normal);
//        }
// endregion

        assert(planarScalar_vector.size() == surf_numUsed && normal_vector.size() == surf_numUsed);

        // create nine List which can be the measurement of observability contribution and sort
        Eigen::Vector3f x_axis(1, 0, 0);
        Eigen::Vector3f y_axis(0, 1, 0);
        Eigen::Vector3f z_axis(0, 0, 1);

        std::vector<std::list<std::pair<int, float> > > pointScoreList;
        pointScoreList.resize(9);
        for (int l = 0; l < surf_numUsed; ++l) {  // only select surf feature points

            pointOri.x = cloudIn->points[l + corner_numUsed].x;
            pointOri.y = cloudIn->points[l + corner_numUsed].y;
            pointOri.z = cloudIn->points[l + corner_numUsed].z;
            pointAssociateToMap(&pointOri, &pointSel);

            Eigen::Vector3f currentPoint(pointSel.x, pointSel.y, pointSel.z);
            pointScoreList[0].push_back(std::make_pair(l, planarScalar_vector[l] * planarScalar_vector[l] *
                                                          ((currentPoint.cross(normal_vector[l])).dot(x_axis))));
            pointScoreList[1].push_back(std::make_pair(l, -planarScalar_vector[l] * planarScalar_vector[l] *
                                                          ((currentPoint.cross(normal_vector[l])).dot(x_axis))));
            pointScoreList[2].push_back(std::make_pair(l, planarScalar_vector[l] * planarScalar_vector[l] *
                                                          ((currentPoint.cross(normal_vector[l])).dot(y_axis))));
            pointScoreList[3].push_back(std::make_pair(l, -planarScalar_vector[l] * planarScalar_vector[l] *
                                                          ((currentPoint.cross(normal_vector[l])).dot(y_axis))));
            pointScoreList[4].push_back(std::make_pair(l, planarScalar_vector[l] * planarScalar_vector[l] *
                                                          ((currentPoint.cross(normal_vector[l])).dot(z_axis))));
            pointScoreList[5].push_back(std::make_pair(l, -planarScalar_vector[l] * planarScalar_vector[l] *
                                                          ((currentPoint.cross(normal_vector[l])).dot(z_axis))));
            pointScoreList[6].push_back(std::make_pair(l, planarScalar_vector[l] * planarScalar_vector[l] *
                                                          (fabs(normal_vector[l].dot(x_axis)))));
            pointScoreList[7].push_back(std::make_pair(l, planarScalar_vector[l] * planarScalar_vector[l] *
                                                          (fabs(normal_vector[l].dot(y_axis)))));
            pointScoreList[8].push_back(std::make_pair(l, planarScalar_vector[l] * planarScalar_vector[l] *
                                                          (fabs(normal_vector[l].dot(z_axis)))));
        }
        // sort list in an decreasing order
        for (int m = 0; m < 9; ++m)
            pointScoreList[m].sort(sort_list_function);

        for (int j = 0; j < 9; ++j) {
            int selectednum = 0;
            while (selectednum < selectedptThre && !pointScoreList[j].empty()){
                auto id = pointScoreList[j].front().first;
                pointScoreList[j].pop_front();

                if(std::count(indices.begin(), indices.end(), id))  // found
                    continue;
                indices.push_back(id);
                selectednum++;
//                cout << "[ DEBUG ] List size " << pointScoreList[j].size() << endl;
            }
            cout << "[ DEBUG ] List '" << j << "' points num : " << selectednum << endl;
        }
    }

    inline void extractIndices(const std::vector<int>& inds,
                               pcl::PointCloud<PointInfoType>::Ptr cloudout){

//        pcl::ExtractIndices<PointInfoType> extractIndices;
//        extractIndices.setInputCloud(ptsInfoUsed);
//        extractIndices.setIndices(boost::make_shared<vector<int> >(inds));
//        extractIndices.setNegative(true);
//        extractIndices.getRemovedIndices();
        pcl::copyPointCloud<PointInfoType>(*ptsInfoUsed, inds, *cloudout);
    }

    ////////////////// covariance utilities [ Copied from M-LOAM ] ////////////////////////////////
    inline Eigen::Matrix<double, 6, 6> adjointMatrix(const Eigen::Matrix4d &T){

        Eigen::Matrix<double, 6, 6> AdT = Eigen::Matrix<double, 6, 6>::Zero();
        AdT.topLeftCorner<3, 3>() = T.topLeftCorner<3, 3>();
        AdT.topRightCorner<3, 3>() = skewSymmetric(T.topRightCorner<3, 1>()) * T.topLeftCorner<3, 3>();
        AdT.bottomRightCorner<3, 3>() = T.topLeftCorner<3, 3>();
        return AdT;
    }

    template <typename Derived>
    static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
    {
        Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
        ans << typename Derived::Scalar(0), -q(2), q(1),
                q(2), typename Derived::Scalar(0), -q(0),
                -q(1), q(0), typename Derived::Scalar(0);
        return ans;
    }

    /// Compound the covariance from two pose to one
    inline Eigen::Matrix3d covop1(const Eigen::Matrix3d &B){
        Eigen::Matrix3d A = -B.trace() * Eigen::Matrix3d::Identity() + B;
        return A;
    }
    inline Eigen::Matrix3d covop2(const Eigen::Matrix3d &B, const Eigen::Matrix3d &C){
        Eigen::Matrix3d A = covop1(B) * covop1(C) * covop1(C * B);
        return A;
    }
    inline void compoundPoseCovariance(const Eigen::Matrix<double, 6, 6> &cov_1,
                                       const Eigen::Matrix<double, 6, 6> &cov_2,
                                       const Eigen::Matrix4d &T1,
                                       Eigen::Matrix<double, 6, 6> &cov_cp){

        Eigen::Matrix<double, 6, 6> AdT1 = adjointMatrix(T1); // the adjoint matrix of T1
        Eigen::Matrix<double, 6, 6> cov_2_prime = AdT1 * cov_2 * AdT1.transpose();
        Eigen::Matrix3d cov_1_rr = cov_1.topLeftCorner<3, 3>();
        Eigen::Matrix3d cov_1_rp = cov_1.topRightCorner<3, 3>();
        Eigen::Matrix3d cov_1_pp = cov_1.bottomRightCorner<3, 3>();

        Eigen::Matrix3d cov_2_rr = cov_2_prime.topLeftCorner<3, 3>();
        Eigen::Matrix3d cov_2_rp = cov_2_prime.topRightCorner<3, 3>();
        Eigen::Matrix3d cov_2_pp = cov_2_prime.bottomRightCorner<3, 3>();

        Eigen::Matrix<double, 6, 6> A1 = Eigen::Matrix<double, 6, 6>::Zero();
        A1.topLeftCorner<3, 3>() = covop1(cov_1_pp);
        A1.topRightCorner<3, 3>() = covop1(cov_1_rp + cov_1_rp.transpose());
        A1.bottomRightCorner<3, 3>() = covop1(cov_1_pp);

        Eigen::Matrix<double, 6, 6> A2 = Eigen::Matrix<double, 6, 6>::Zero();
        A2.topLeftCorner<3, 3>() = covop1(cov_2_pp);
        A2.topRightCorner<3, 3>() = covop1(cov_2_rp + cov_2_rp.transpose());
        A2.bottomRightCorner<3, 3>() = covop1(cov_2_pp);

        Eigen::Matrix3d Brr = covop2(cov_1_pp, cov_2_rr) +
                              covop2(cov_1_rp.transpose(), cov_2_rp) +
                              covop2(cov_1_rp, cov_2_rp.transpose()) +
                              covop2(cov_1_rr, cov_2_pp);
        Eigen::Matrix3d Brp = covop2(cov_1_pp, cov_2_rp.transpose()) +
                              covop2(cov_1_rp.transpose(), cov_2_pp);
        Eigen::Matrix3d Bpp = covop2(cov_1_pp, cov_2_pp);
        Eigen::Matrix<double, 6, 6> B = Eigen::Matrix<double, 6, 6>::Zero();
        B.topLeftCorner<3, 3>() = Brr;
        B.topRightCorner<3, 3>() = Brp;
        B.bottomLeftCorner<3, 3>() = Brp.transpose();
        B.bottomRightCorner<3, 3>() = Bpp;

        cov_cp = cov_1 + cov_2_prime + (A1 * cov_2_prime +
                                        cov_2_prime * A1.transpose() +
                                        A2 * cov_1 + cov_1 * A2.transpose()) / 12 + B / 4;
    }

    // pointToFS turns a 4x1 homogeneous point into a special 4x6 matrix
    inline Eigen::Matrix<double, 4, 6> pointToFS(const Eigen::Vector4d &point){

        Eigen::Matrix<double, 4, 6> G = Eigen::Matrix<double, 4, 6>::Zero();
        G.block<3, 3>(0, 0) = point(3) * Eigen::Matrix3d::Identity();
        G.block<3, 3>(0, 3) = -skewSymmetric(point.block<3, 1>(0, 0));
        return G;
    }
    /// Propagate the covariance(uncertainty) from pose to 3D point
    /// \param pi : the original point for evaluating uncertainty
    /// \param cov_point : associated covariance of the point
    /// \param T : 4X4 transformation of pose
    /// \param cov_pose : associated covariance of the pose
    inline void evalPointUncertainty(const PointType &pi,
                                     const Eigen::Matrix4d &T,
                                     const Eigen::Matrix<double, 6, 6> &cov_pose,
                                     Eigen::Matrix3d &cov_point){

        // THETA: diag(P, Phi, Z) includes the translation, rotation, measurement uncertainty
        Eigen::Matrix<double, 9, 9> cov_input = Eigen::Matrix<double, 9, 9>::Zero();
        cov_input.topLeftCorner<6, 6>() = cov_pose;
        cov_input.bottomRightCorner<3, 3>() = COV_MEASUREMENT;

        Eigen::Vector4d point_curr(pi.x, pi.y, pi.z, 1);
        Eigen::Matrix<double, 4, 3> D;
        D << 1, 0, 0,
                0, 1, 0,
                0, 0, 1,
                0, 0, 0;
        Eigen::Matrix<double, 4, 9> G = Eigen::Matrix<double, 4, 9>::Zero();
        G.block<4, 6>(0, 0) = pointToFS(T * point_curr);
        G.block<4, 3>(0, 6) = T * D;
        cov_point = Eigen::Matrix4d(G * cov_input * G.transpose()).topLeftCorner<3, 3>(); // 3x3
        // std::cout << cov_input << std::endl;
        // std::cout << G << std::endl;
        // std::cout << "evalUncertainty:" << std::endl
        //           << point_curr.transpose() << std::endl
        //           << cov_point << std::endl;
        // exit(EXIT_FAILURE);
    }

};

#endif //STRUCTURAL_MAPPING_POSEESTIMATIONLIB_HPP
