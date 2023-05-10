//
// Created by joe on 2020/11/1.
//

#ifndef STRUCTURAL_MAPPING_GTSAMHELPER_HPP
#define STRUCTURAL_MAPPING_GTSAMHELPER_HPP

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

class GtsamHelper{

    NonlinearFactorGraph gtSamGraph;
    Values initialEstimates, latestEstimates;
    ISAM2* isam;

    bool useISAM;

    gtsam::Vector6 vector6;
    gtsam::Vector3 vector3;

    noiseModel::Diagonal::shared_ptr priorNoise;
    noiseModel::Diagonal::shared_ptr odometryNoise;
    noiseModel::Diagonal::shared_ptr constraintNoise;

    std::vector<gtsam::PriorFactor<Pose3> > factorContainer;

public:
    GtsamHelper(){

        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
//        parameters.setOptimizationParams();
        parameters.print();
        isam = new ISAM2(parameters);

        vector6 << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-6;  // basic
        vector3 << 1e-6, 1e-6, 1e-6;  // basic
        priorNoise = noiseModel::Diagonal::Variances(vector6);
        odometryNoise = noiseModel::Diagonal::Variances(vector6);
        constraintNoise = noiseModel::Diagonal::Variances(vector6);
    }

    std::vector<gtsam::PriorFactor<Pose3> > getPriorFactorsPose3(){

        return factorContainer;
    }

    void addpriorFactor(const Eigen::Isometry3d &pose, int id = 0){

        Pose3 origin = Pose3(pose.matrix());
        gtSamGraph.add(PriorFactor<Pose3>(id, origin, priorNoise));
        initialEstimates.insert(id , origin);
    }
    void addpriorFactor(const Eigen::Matrix4d &pose, int id, double noise = 1.0, bool saveFactor = false){

        priorNoise = noiseModel::Diagonal::Variances(noise*vector6);
        Pose3 origin = Pose3(pose.matrix());
        gtsam::PriorFactor<Pose3> pF(id, origin, priorNoise);
        gtSamGraph.add(pF);
//        gtSamGraph.add(PriorFactor<Pose3>(id, origin, priorNoise));

        if (saveFactor) factorContainer.emplace_back(pF);
    }
    void addpriorFactor(const gtsam::PriorFactor<Pose3>& fac){

        gtSamGraph.add(fac);
    }

    void addGPSFactor(const Eigen::Vector3d &pos_gps, int id, double noise = 1){

        priorNoise = noiseModel::Diagonal::Variances(noise * vector3);
        gtSamGraph.add(gtsam::GPSFactor(id, gtsam::Point3(pos_gps[0], pos_gps[1], pos_gps[2]),
                                        priorNoise));

    }


    void addBetweenFactor(const Eigen::Isometry3d &pose1, int id1,
                          const Eigen::Isometry3d &pose2, int id2,
                          bool addval = true, double noise = 1.0){

        addBetweenFactor(pose1.matrix(), id1, pose2.matrix(), id2, addval, noise);
    }

    void addBetweenFactor(const Eigen::Matrix4d &pose1, int id1,
                          const Eigen::Matrix4d &pose2, int id2,
                          bool addval = true, double noise = 1.0){

        Pose3 poseBegin, poseEnd;
        poseBegin = Pose3(pose1);
        poseEnd = Pose3(pose2);

//        cout << BOLDCYAN << " [ DEBUG ] " ;
//        cout << "Matrix Between "<< id1 << ": " << id2 << endl ;
//        cout << (pose1.inverse() * pose2).matrix() << RESET << endl;
//        cout << "GTSAM: "  << poseBegin << RESET << endl;

        odometryNoise = noiseModel::Diagonal::Variances((id2-id1) * noise * vector6);
        gtSamGraph.add(BetweenFactor<Pose3>(id1, id2,
                                            poseBegin.between(poseEnd), odometryNoise));

        if(!addval) return;

        if((!initialEstimates.exists(id1)) && (!latestEstimates.exists(id1)))
            initialEstimates.insert(id1, poseBegin);
        if(!latestEstimates.exists(id2))
            initialEstimates.insert(id2, poseEnd);
    }

    void addloopFactor(int id1, int id2, double noise, Eigen::Matrix4f relaT){

        constraintNoise = noiseModel::Diagonal::Variances(noise * vector6);
        gtSamGraph.add(BetweenFactor<Pose3>(id1, id2, Pose3(relaT.cast<double>()), constraintNoise));
        isam->update(gtSamGraph);
        isam->update();
        gtSamGraph.resize(0);
    }

    void addloopFactor(int id1, int id2, double noise,const Eigen::Matrix4d &T1,const Eigen::Matrix4d &T2){

        constraintNoise = noiseModel::Diagonal::Variances(noise * vector6);
        gtSamGraph.add(BetweenFactor<Pose3>(id1, id2, Pose3(T1.inverse()*T2), constraintNoise));
        isam->update(gtSamGraph);
        isam->update();
        gtSamGraph.resize(0);
    }

    void optimizeGraph(bool useISAM_ = true){

        useISAM = useISAM_;
//        gtSamGraph.print();
//        gtSamGraph.printErrors();

        cout << YELLOW << "[ GTSAM ] Updating Graph. " << endl;

        if(useISAM){
            // FixMe: stuck here. [ can't bind with Ceres ];
            isam->update(gtSamGraph, initialEstimates);
            isam->update();
//            isam->print();

            latestEstimates = isam->calculateEstimate();
        }else{
            // local graph optimization
            LevenbergMarquardtParams optiPara;
            optiPara.setMaxIterations(15);
            optiPara.setLinearSolverType("MULTIFRONTAL_QR");
            LevenbergMarquardtOptimizer optimizer(gtSamGraph, initialEstimates, optiPara);
            latestEstimates = optimizer.optimize();
        }

        cout << GREEN << "[ GTSAM ] Graph Solved. " << RESET << endl;

        gtSamGraph.resize(0);
        initialEstimates.clear();
    }

    void correctPoseCloud(pcl::PointCloud<PointTypePose>::Ptr &poseCloud,
                          int startInd_, int endInd_ ){

//        gtsam::Marginals marginals(gtSamGraph, latestEstimates);
//        Eigen::MatrixXd infoMatrix = marginals.marginalInformation(startInd_);

        cout << GREEN << "[ GTSAM ] Updating poses from "<< startInd_
             << " to " << endInd_-1 << " withIN " << poseCloud->points.size() << endl;

#pragma omp parallel for
        for (int i = startInd_; i < endInd_; ++i) {

            if (!latestEstimates.exists(i))
                continue;
            Pose3 poseNew = latestEstimates.at<Pose3>(i);

            poseCloud->points[i].x = poseNew.translation().x();
            poseCloud->points[i].y = poseNew.translation().y();
            poseCloud->points[i].z = poseNew.translation().z();

            gtsam::Quaternion quaternion = poseNew.rotation().toQuaternion();
//            poseNew.matrix();

            poseCloud->points[i].roll  = quaternion.x();
            poseCloud->points[i].pitch = quaternion.y();
            poseCloud->points[i].yaw   = quaternion.z();
            poseCloud->points[i].intensity = quaternion.w();
        }
        cout << "[ GTSAM ] Global poses updated. " << RESET << endl;

        if(!useISAM) latestEstimates.clear();
    }

    void reset(){

        isam->clear();
        initialEstimates.clear();
        latestEstimates.clear();
        gtSamGraph.resize(0);
        factorContainer.clear();
    }
};

#endif //STRUCTURAL_MAPPING_GTSAMHELPER_HPP
