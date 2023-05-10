
//
// Created by joe on 2020/10/27.
//

//#include "poseEstimationLib.hpp"

#include "tools.h"
#include "structural_mapping/poseInSlidingWin.h"
#include "structural_mapping/localmapWithPoseMsg.h"
#include "gtsamHelper.hpp"

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <pcl/surface/convex_hull.h>

#include "pcl/registration/super4pcs.h"

#include <ctime>

using namespace gtsam;

class GraphOptimization{

    ros::NodeHandle nh;
    ros::Subscriber subCloudInfo,
            subLaserOdometry,
            subposeWin,
            subOdomPose,
            sublocalmapWithPose;

    ros::Publisher pubposesGraph, pubgraphEdges;

    pcl::VoxelGrid<PointType> downSampler;
    fast_gicp::FastGICP<PointType , PointType> fastGicp;

    structural_mapping::cloud_info featureCloudinfo;
    structural_mapping::poseInSlidingWin curPoseWinMsg;
    structural_mapping::localmapWithPoseMsg curlocalmapMsg;

    deque<structural_mapping::cloud_infoConstPtr> featureCloudinfoQue;
    deque<structural_mapping::poseInSlidingWinConstPtr> poseWinMsgQue;
    deque<structural_mapping::localmapWithPoseMsgConstPtr> localmapMsgQue;

    pcl::PointCloud<PointTypePose>::Ptr mapPoseOri;  // save the original poses of map units
    pcl::PointCloud<PointTypePose>::Ptr mapPoseOpted;  // save the optimized poses of map units
    pcl::PointCloud<PointType>::Ptr mapPoseOpted_xy;  // save the optimized poses for loop detection
    pcl::KdTreeFLANN<PointType>::Ptr kdtreePoses_xy;  // to find loop

    vector<pair<int, PointTypePose> > poseInSlidingWindow;  // poses to be optimized
    pcl::PointCloud<PointTypePose>::Ptr globalposesOdom;  // save the original odom poses
    pcl::PointCloud<PointTypePose>::Ptr globalposesOpted;  // save the optimized poses

    // save for reconstruction
    vector<pcl::PointCloud<PointType>::Ptr> surfcloudsVec;
    vector<pcl::PointCloud<PointType>::Ptr> cornercloudsVec;
    vector<pcl::PointCloud<PointType>::Ptr> localmapVec;  // map units in local frame

    pcl::PointCloud<PointType>::Ptr cornerCloudmapInWin;
    pcl::PointCloud<PointType>::Ptr surfCloudmapInWin;

    mutex mtx;

    vector<int> badregisteredFrameInds;
    double timeLaserOdometry ,timeCloudinfo, localmapDura;
    int startInd, endInd;
    int winSize;
    bool useISAM, globalReg = false, loopIsClosed = false;
    float maxCorrDist_loc, maxCorrDist_glo, maxCorrDist_glo_refi;

    GtsamHelper gtsamHelper;
    PointTypePose anchorPose;
    int mapUnitNum = -1;

public:
    GraphOptimization():nh("~"){

//        std::mktime();

        nh.getParam("/projpath", projPath);  // global param add '/' before name
        nh.param<bool>  ("/graphOptimization/useISAM", useISAM, false);
        nh.param<int>   ("/mapOptiBacktracing/winSize", winSize, 30);
        nh.param<float>   ("/graphOptimization/maxCorrDist_loc", maxCorrDist_loc, 1.5);
        nh.param<float>   ("/graphOptimization/maxCorrDist_glo", maxCorrDist_glo, 2.5);
        nh.param<float>   ("/graphOptimization/maxCorrDist_glo_refi", maxCorrDist_glo_refi, 0.5);
        nh.param<double>("/mapOptiSpline/localmapDura", localmapDura, 20.f);

        cout << "[ GraphOpt ] projPath :" << projPath <<endl;
        cout << "[ GraphOpt ] useISAM  :" << useISAM  <<endl;
        cout << "[ GraphOpt ] winSize  :" << winSize  <<endl;
        cout << "[ GraphOpt ] maxCorrDist_loc :"     << maxCorrDist_loc      <<endl;
        cout << "[ GraphOpt ] maxCorrDist_glo :"     << maxCorrDist_glo      <<endl;
        cout << "[ GraphOpt ] maxCorrDist_glo_refi:" << maxCorrDist_glo_refi <<endl;
        cout << "[ GraphOpt ] localmapDura:" << localmapDura <<endl;


//        subCloudInfo = nh.subscribe<structural_mapping::cloud_info>("/feature_cloud_info", 1,
//                                                            &GraphOptimization::laserCloudInfoHandler, this);
//        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/odom_pose", 1,
//                                                            &GraphOptimization::laserOdometryHandler, this);
        subposeWin = nh.subscribe<structural_mapping::poseInSlidingWin>("/mapBT_poses_Win", 1,
                                                                        &GraphOptimization::poseWinMsgHandler, this);
        subOdomPose = nh.subscribe<sensor_msgs::PointCloud2>("/poseCloud_odom", 1,
                                                             &GraphOptimization::odomPoseCloudHandler, this);

        sublocalmapWithPose = nh.subscribe<structural_mapping::localmapWithPoseMsg>("/localmapWithPose", 1,
                                                                                    &GraphOptimization::localmapWithPoseHandler, this); //原始map轨迹（无闭环优化），被transformFusion订阅


        pubposesGraph = nh.advertise<sensor_msgs::PointCloud2>("/poses_graph", 1);
        pubgraphEdges = nh.advertise<visualization_msgs::Marker>("/poses_graph_edge", 1);

        allocateMemo();
    }

    void allocateMemo(){

        poseInSlidingWindow.resize(winSize);

        cornerCloudmapInWin.reset(new pcl::PointCloud<PointType>());
        surfCloudmapInWin.reset(new pcl::PointCloud<PointType>());

        globalposesOpted.reset(new pcl::PointCloud<PointTypePose>());
        globalposesOdom.reset(new pcl::PointCloud<PointTypePose>());

        mapPoseOri.reset(new pcl::PointCloud<PointTypePose>());
        mapPoseOpted.reset(new pcl::PointCloud<PointTypePose>());

        mapPoseOpted_xy.reset(new pcl::PointCloud<PointType>());
        kdtreePoses_xy.reset(new pcl::KdTreeFLANN<PointType>());
    }

    void poseWinMsgHandler(const structural_mapping::poseInSlidingWinConstPtr& msgIn) {

        std::lock_guard<mutex> lockGuard(mtx);
        poseWinMsgQue.emplace_back(msgIn);
    }
    void laserCloudInfoHandler(const structural_mapping::cloud_infoConstPtr& msgIn) {

        std::lock_guard<mutex> lockGuard(mtx);
        featureCloudinfoQue.emplace_back(msgIn);
    }
    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry){

        timeLaserOdometry = laserOdometry->header.stamp.toSec();

        PointTypePose tmp;
        tmp.roll  = laserOdometry->pose.pose.orientation.x;
        tmp.pitch = laserOdometry->pose.pose.orientation.y;
        tmp.yaw   = laserOdometry->pose.pose.orientation.z;
        tmp.intensity = laserOdometry->pose.pose.orientation.w;
        tmp.x = laserOdometry->pose.pose.position.x;
        tmp.y = laserOdometry->pose.pose.position.y;
        tmp.z = laserOdometry->pose.pose.position.z;
        tmp.time = timeLaserOdometry;

        std::lock_guard<mutex> lockGuard(mtx);
        globalposesOdom->points.emplace_back(tmp);
        globalposesOpted->points.emplace_back(tmp);
        cout << "RECEIVED ODOM  " << to_string(timeLaserOdometry) << endl;
    }
    void odomPoseCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){

        std::lock_guard<mutex> lockGuard(mtx);
        pcl::fromROSMsg<PointTypePose>(*msgIn, *globalposesOdom);
        globalposesOpted->points.resize(globalposesOdom->points.size());
    }
    void localmapWithPoseHandler(const structural_mapping::localmapWithPoseMsgConstPtr& msgIn){

        std::lock_guard<mutex> lockGuard(mtx);
        localmapMsgQue.emplace_back(msgIn);
        cout << BOLDWHITE << "[ Graph ] Received local map " << to_string(msgIn->header.stamp.toSec()) << RESET << endl;
    }


    bool parseData(){

        std::lock_guard<mutex> lockGuard(mtx);

        if(poseWinMsgQue.empty())
            return false;

        curPoseWinMsg = *poseWinMsgQue.front();
        poseWinMsgQue.pop_front();

        return true;
    }

    /// add relative edges within the pose windows backward and forward
    void updatePoseGraph(){

        if(startInd == 0){  // fix the first pose

            gtsamHelper.addpriorFactor(getTransformMatrix(globalposesOdom->points.front()));
            cout << GREEN << "[ mapBT ] Graph Origin set. " << RESET << endl;
        }

        Eigen::Isometry3d poseBegin, poseEnd;
        poseBegin = getTransformMatrix(globalposesOdom->points[startInd]);
        // forwarding relative pose constraint factors
        for (int j = 1; j < winSize; ++j) {

            poseEnd = getTransformMatrix(globalposesOdom->
                    points[poseInSlidingWindow[j].first]);

            gtsamHelper.addBetweenFactor(poseBegin, startInd,
                                         poseEnd, poseInSlidingWindow[j].first);
        }

        poseEnd = getTransformMatrix(poseInSlidingWindow.back().second);
        // reversing relative pose constraint factors
        for (int k = winSize-2; k >= 0 ; --k) {

            poseBegin = getTransformMatrix(poseInSlidingWindow[k].second);

            gtsamHelper.addBetweenFactor(poseBegin, poseInSlidingWindow[k].first,
                                         poseEnd, endInd, false);
        }

        gtsamHelper.optimizeGraph(useISAM);

        gtsamHelper.correctPoseCloud(globalposesOpted, startInd, endInd);
    }

    void publishCloud(){

        sensor_msgs::PointCloud2 cloudmsg;
        pcl::toROSMsg(*globalposesOpted, cloudmsg);
        cloudmsg.header.stamp = curPoseWinMsg.header.stamp;
        cloudmsg.header.frame_id = "/base_link";
        pubposesGraph.publish(cloudmsg);

        if(!globalposesOpted->empty())
            pcl::io::savePCDFileBinary(projPath + "globalposesOpted.pcd", *globalposesOpted);
        cout << BOLDGREEN << "[ Graph ] Final Poses saved to " << projPath << RESET << endl;

    }

    void updatePoseInWin(){

        for (int j = 0; j < winSize; ++j) {

            poseInSlidingWindow[j].first = curPoseWinMsg.inds[j];
            poseInSlidingWindow[j].second.roll  = curPoseWinMsg.poseCloud_bt.poses[j].orientation.x;
            poseInSlidingWindow[j].second.pitch = curPoseWinMsg.poseCloud_bt.poses[j].orientation.y;
            poseInSlidingWindow[j].second.yaw   = curPoseWinMsg.poseCloud_bt.poses[j].orientation.z;
            poseInSlidingWindow[j].second.intensity = curPoseWinMsg.poseCloud_bt.poses[j].orientation.w;
            poseInSlidingWindow[j].second.x = curPoseWinMsg.poseCloud_bt.poses[j].position.x;
            poseInSlidingWindow[j].second.y = curPoseWinMsg.poseCloud_bt.poses[j].position.y;
            poseInSlidingWindow[j].second.z = curPoseWinMsg.poseCloud_bt.poses[j].position.z;
        }

        cout << GREEN << "[ Graph ] Poses in sliding window updated. " << RESET << endl;
    }

    /// map unit graph optimization part
    bool parseMap(){

        if (localmapMsgQue.empty())
            return false;

        {
            std::lock_guard<mutex> lockGuard(mtx);
            curlocalmapMsg = *localmapMsgQue.front();
            localmapMsgQue.pop_front();
        }

//        cout << MAGENTA << "[ DEBUG ] derive local map MSG." << RESET << endl;

        pcl::PointCloud<PointType>::Ptr tempCloud(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(curlocalmapMsg.localmap, *tempCloud);
        localmapVec.emplace_back(tempCloud);

        PointTypePose mapPose;
        mapPose.x = curlocalmapMsg.pose.position.x;
        mapPose.y = curlocalmapMsg.pose.position.y;
        mapPose.z = curlocalmapMsg.pose.position.z;
        mapPose.roll  = curlocalmapMsg.pose.orientation.x;
        mapPose.pitch = curlocalmapMsg.pose.orientation.y;
        mapPose.yaw   = curlocalmapMsg.pose.orientation.z;
        mapPose.intensity = curlocalmapMsg.pose.orientation.w;
        mapPose.time = curlocalmapMsg.header.stamp.toSec();
        mapPoseOri->points.emplace_back(mapPose);
        mapPoseOpted->points.emplace_back(mapPose);

        if (curlocalmapMsg.badregistered)
            badregisteredFrameInds.emplace_back(localmapVec.size()-1);

        return true;
    }

    void updatePoseGraph_Map(){

        mapUnitNum = localmapVec.size();
        if( mapUnitNum < 2 ){  // fix the first pose

            gtsamHelper.addpriorFactor(getTransformMatrix(mapPoseOpted->points.front()));
            cout << GREEN << "[ mapUnit ] Graph Origin set. " << RESET << endl;
        }else{

            Eigen::Isometry3d poseBegin, poseEnd;
            poseBegin = getTransformMatrix(mapPoseOpted->points[mapUnitNum-2]);
            poseEnd = getTransformMatrix(mapPoseOpted->points[mapUnitNum-1]);

            gtsamHelper.addBetweenFactor(poseBegin, mapUnitNum-2,
                                         poseEnd, mapUnitNum-1);
        }

        gtsamHelper.optimizeGraph(useISAM);
//        if (loopIsClosed){  // update all poses
        gtsamHelper.correctPoseCloud(mapPoseOpted, 0, mapPoseOpted->size());
//            loopIsClosed = false;
//        }else
//            gtsamHelper.correctPoseCloud(mapPoseOpted, mapPoseOpted->size()-1, mapPoseOpted->size());
    }

    Eigen::Matrix4f getTransMatrixT(const Eigen::Quaternionf &q, const Eigen::Vector3f t){

        Eigen::Matrix4f curT;
        curT.block(0,0,3,3) = q.matrix();
        curT.block(0,3,3,1) = t;
        curT.block(3,0,1,4) = Eigen::Vector4f(0, 0, 0, 1).transpose();
        return curT;
    }

    void detectLoopByPosition(){

        ros::Rate looprate(0.2);
        int curFrameID, lastID = -1;
        while (ros::ok()) {

            looprate.sleep();
            std::lock_guard<mutex> locker(mtx);
            assert_equal(localmapVec.size(), mapPoseOpted->size());
            curFrameID = mapPoseOpted->size() - 1;
            if(curFrameID == lastID) continue;
            if (curFrameID < 1) continue;
            cout << BOLDBLUE << "[Graph] Detecting loop by positions ... " << RESET << endl;
            loopIsClosed = false;

            mapPoseOpted_xy->resize(curFrameID);
            for (int j = 0; j < curFrameID; ++j) {
                mapPoseOpted_xy->points[j].x = mapPoseOpted->points[j].x;
                mapPoseOpted_xy->points[j].y = mapPoseOpted->points[j].y;
                mapPoseOpted_xy->points[j].z = 0;
            }
            kdtreePoses_xy->setInputCloud(mapPoseOpted_xy);

            PointType curPosition;
            curPosition.x = mapPoseOpted->points[curFrameID].x;
            curPosition.y = mapPoseOpted->points[curFrameID].y;
            curPosition.z = 0;

            float loopDistThre = 2*localmapDura;  // range to search revisited place
            int loopframeNumThre = 10;  // the minimal frame number between the loop ids
            vector<int> inds; vector<float> dists;
            kdtreePoses_xy->radiusSearch(curPosition, loopDistThre, inds, dists);
            cout << BLUE << "[ Graph ] searched revisited places " << inds.size() << RESET << endl;

            if (!inds.empty()) {
                for (int j = 0; j < inds.size(); ++j) {
                    if (curFrameID - inds[j] < loopframeNumThre || loopIsClosed)
                        continue;

                    cout << BLUE << "[ Graph ] Potential loop between " << inds[j] << " - " << curFrameID << RESET
                         << endl;
                    PointTypePose lastPose, curPose;
                    lastPose = mapPoseOpted->points[inds[j]];
                    curPose = mapPoseOpted->points[curFrameID];
                    // retrieve history point cloud map
                    pcl::PointCloud<PointType>::Ptr cloud_history(new pcl::PointCloud<PointType>());
                    if (inds[j] > 0)
                        *cloud_history = *transformPointCloud(localmapVec[inds[j] - 1],
                                                              &mapPoseOpted->points[inds[j] - 1]);
                    *cloud_history += *transformPointCloud(localmapVec[inds[j]],
                                                           &lastPose);
                    *cloud_history += *transformPointCloud(localmapVec[inds[j] + 1],
                                                           &mapPoseOpted->points[inds[j] + 1]);
                    // current cloud
                    pcl::PointCloud<PointType>::Ptr alignedCloud(new pcl::PointCloud<PointType>());
                    Eigen::Matrix4f relaT;
                    double score = globalmapFusion(lastPose, curPose, localmapVec[curFrameID], cloud_history,
                                                   alignedCloud, relaT, true);
                    cout << BOLDRED << "[ Graph ] Loop score " << score << RESET << endl;
                    if (score > 30)  // not registered well
                        continue;
                    Eigen::Matrix4d optT = relaT.cast<double>() * getTransformMatrix4d(curPose);
//                gtsamHelper.addloopFactor( curFrameID, inds[j], 1, relaT);
                    gtsamHelper.addloopFactor(curFrameID, inds[j], 1, optT,
                                              getTransformMatrix4d(lastPose));
                    loopIsClosed = true;
                }
            }
            lastID = curFrameID;
        }
    }

    // use convex hull of ground to evaluate the overlap area
    double evaluateOverlap(const pcl::PointCloud<PointType>::Ptr cloud1,
                           const pcl::PointCloud<PointType>::Ptr cloud2){

        // get convex hull
        pcl::ConvexHull<PointType>::Ptr hull(new pcl::ConvexHull<PointType>());
        std::vector<pcl::Vertices> polygon1, polygon2;
        pcl::PointCloud<PointType>::Ptr cloud1_hull(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr cloud2_hull(new pcl::PointCloud<PointType>());

        hull->setInputCloud(cloud1);
        hull->setDimension(3);
        hull->reconstruct(*cloud1_hull, polygon1);

        hull.reset(new pcl::ConvexHull<PointType>());
        hull->setInputCloud(cloud2);
        hull->setDimension(3);
        hull->reconstruct(*cloud2_hull, polygon2);

        // evaluate overlap by sampling triangles
        PolygonOverlapHelper::Point p1[300], p2[300];
        PolygonOverlapHelper polygonOverlapHelper;
        int n1 = cloud1_hull->points.size(), n2 = cloud2_hull->points.size();
        for(int i = 0; i < n1; i++) {
            p1[i].x = cloud1_hull->points[i].x;
            p1[i].y = cloud1_hull->points[i].y;
        }
        for(int i = 0; i < n2; i++) {
            p2[i].x = cloud2_hull->points[i].x;
            p2[i].y = cloud2_hull->points[i].y;
        }

        double Area = polygonOverlapHelper.SPIA(p1, p2, n1, n2);
        cout << "Area=" << Area << endl;
        double A1 = polygonOverlapHelper.PolygonArea(p1, n1);
        double A2 = polygonOverlapHelper.PolygonArea(p2, n2);
        cout << "A1 =" << A1 << ", A2=" << A2 << endl;

        double ratio = Area / (A1 < A2? A1 : A2);
        return ratio;
    }

    void fineMapFusion(){

        mapUnitNum = localmapVec.size();
        if(mapUnitNum < 2)
            return;

        pcl::PointCloud<PointType>::Ptr alignedCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr srcCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr tgtCloud(new pcl::PointCloud<PointType>());

        *tgtCloud = *localmapVec[mapUnitNum-2];
        *srcCloud = *localmapVec[mapUnitNum-1];

        PointTypePose curPose = mapPoseOri->points.back();
        PointTypePose prePose = mapPoseOri->points[mapUnitNum-2];

        Eigen::Matrix4f optT, relaT, T2, globalmapT;

        if (!badregisteredFrameInds.empty() &&
            badregisteredFrameInds.back() == mapUnitNum-2){  // last frame is bad

            globalReg = true;
            cout << BOLDMAGENTA << "[ Graph ] Bad registration at " << mapUnitNum-1 << RESET << endl;

            globalmapFusion(prePose, curPose, srcCloud, tgtCloud, alignedCloud, globalmapT, true);

            T2 = getTransformMatrix(curPose).matrix().cast<float>();  // system restart at a new frame
//           T2 = getTransformMatrix(relaPose).matrix().cast<float>();
            relaT = globalmapT * T2;
        }else
            localmapFusion(prePose, curPose, srcCloud, tgtCloud, alignedCloud, relaT);

        Eigen::Matrix4f T1 = getTransformMatrix4d(mapPoseOpted->points[mapUnitNum-2]).cast<float>();  // last frame optimized pose
        optT = T1 * relaT;
//        fastGicp.swapSourceAndTarget();

        double score = fastGicp.getFitnessScore();
        cout << BOLDYELLOW << "[ Graph ] Map " << mapUnitNum << " fused with score " << score << RESET << endl;

//            if(score > 0.0){
//                localmapVec.back()->clear();
//                pcl::copyPointCloud(*tempCloud, *localmapVec.back());
//            }
//        cout << " [ Debug ] Original transform \n" << oriT << endl;
//        cout << " [ Debug ] Original transform1 \n" << oriT1 << endl;

        cout << " [ Debug ] Aligned transform \n" << optT << endl;

        // update cur pose
        mapPoseOpted->back() = matrix4fToPose(optT, mapPoseOpted->back().time);
    }

    /// Super4PCS too slow
    void mapRegisSuper4PCS(const pcl::PointCloud<PointType>::Ptr srcCloud,
                           const pcl::PointCloud<PointType>::Ptr tgtCloud,
                           pcl::PointCloud<PointType>::Ptr destCloud,
                           Eigen::Matrix4f &trans){

        // downsamling
        pcl::PointCloud<PointType>::Ptr srcCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr tgtCloudDS(new pcl::PointCloud<PointType>());
        downSampler.setLeafSize(odom_ds_leafsize, odom_ds_leafsize, odom_ds_leafsize);
        downSampler.setInputCloud(srcCloud);
        downSampler.filter(*srcCloudDS);
        downSampler.setInputCloud(tgtCloud);
        downSampler.filter(*tgtCloudDS);

        pcl::Super4PCS<pcl::PointXYZI,pcl::PointXYZI> align;
        align.options_.configureOverlap(0.5f);
        align.options_.sample_size = 8000;  // little effect on time
        align.options_.delta = 0.05f;  // smaller -> less time

        align.setInputSource(srcCloudDS);
        align.setInputTarget(tgtCloudDS);
        align.align(*destCloud);
        if(align.converged_){

            cout << "Super4PCS converged with " << align.getFitnessScore() << endl;
            trans = align.getFinalTransformation();
        }
    }


    void mapRegisFVGICP(const pcl::PointCloud<PointType>::Ptr srcCloud,
                        const pcl::PointCloud<PointType>::Ptr tgtCloud,
                        pcl::PointCloud<PointType>::Ptr destCloud,
                        Eigen::Matrix4f &trans){

//        fast_gicp::FastVGICP<pcl::PointXYZI, pcl::PointXYZI> align;
        fast_gicp::FastGICP<pcl::PointXYZI, pcl::PointXYZI> align;
        align.setMaxCorrespondenceDistance(maxCorrDist_glo);
//        align.setResolution(0.15f);

        align.setInputSource(srcCloud);
        align.setInputTarget(tgtCloud);
        align.align(*destCloud);
        if(align.hasConverged()){

            cout << "FVGICP converged with " << align.getFitnessScore() << endl;
            trans = align.getFinalTransformation();
        }
    }

    void localmapFusion(PointTypePose prePose, PointTypePose curPose,
                        const pcl::PointCloud<PointType>::Ptr srcCloud,
                        const pcl::PointCloud<PointType>::Ptr tgtCloud,
                        pcl::PointCloud<PointType>::Ptr &alignedCloud,
                        Eigen::Matrix4f &relaT){

        cout << YELLOW << "[ Graph ] Fusing local map ..." << RESET << endl;
        PointTypePose relaPose = getRelativePose(prePose, curPose);
        pcl::PointCloud<PointType>::Ptr fineAlignedCloud(new pcl::PointCloud<PointType>());
        *fineAlignedCloud = *transformPointCloud(srcCloud, &relaPose);  // transformed by initial pose

        fastGicp.clearSource();
        fastGicp.clearTarget();
        fastGicp.setInputTarget(tgtCloud);  // in local frame
        fastGicp.setInputSource(fineAlignedCloud);
        fastGicp.setMaxCorrespondenceDistance(maxCorrDist_loc);
        fastGicp.align(*alignedCloud);

        relaT = fastGicp.getFinalTransformation() * (getTransformMatrix4d(relaPose).cast<float>());
    }

    double globalmapFusion(PointTypePose prePose, PointTypePose curPose,
                           const pcl::PointCloud<PointType>::Ptr srcCloud,
                           const pcl::PointCloud<PointType>::Ptr tgtCloud,
                           pcl::PointCloud<PointType>::Ptr &alignedCloud,
                           Eigen::Matrix4f &relaT, bool ifsave = false){

//            PointTypePose relaPose = getRelativePose(prePose, curPose);

        // global registration first
        Eigen::Matrix4f T_g;
        pcl::PointCloud<PointType>::Ptr coarseAlignedCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr fineAlignedCloud(new pcl::PointCloud<PointType>());
        *fineAlignedCloud = *transformPointCloud(srcCloud, &curPose);  // transformed by initial pose
//            globalMapFusion(localmapVec.back(),  // no initial transform
//            globalMapFusion(transformPointCloud(localmapVec.back(), &relaPose),
        mapRegisFVGICP(fineAlignedCloud,  tgtCloud, coarseAlignedCloud, T_g);
        // refine result, that's doing registration twice
        fastGicp.clearSource();
        fastGicp.clearTarget();
        fastGicp.setInputTarget(tgtCloud);  // in local frame
        fastGicp.setInputSource(coarseAlignedCloud);
        fastGicp.setMaxCorrespondenceDistance(maxCorrDist_glo_refi);
        fastGicp.align(*alignedCloud);

        relaT = fastGicp.getFinalTransformation() * T_g;

        if(!ifsave) return fastGicp.getFitnessScore();
        // only save the global registration result
        globalReg = false;
        pcl::io::savePCDFileBinaryCompressed(projPath + "testMapFusion/src_"+to_string(mapUnitNum)+".pcd",
                                             *fineAlignedCloud);
        pcl::io::savePCDFileBinaryCompressed(projPath + "testMapFusion/tgt_"+to_string(mapUnitNum)+".pcd",
                                             *tgtCloud);
        pcl::io::savePCDFileBinaryCompressed(projPath + "testMapFusion/aligned_"+to_string(mapUnitNum)+".pcd",
                                             *alignedCloud);
        ofstream ofs(projPath + "transToConnect" + to_string(mapUnitNum) + "txt");
        Eigen::Matrix4f trans = getTransformMatrix4d(prePose).cast<float>() * relaT;
//        globalmapT = T2.inverse() * trans;
        ofs << trans.matrix();
        ofs.close();
        return fastGicp.getFitnessScore();
    }

    void publishMapPose(){

        if (mapPoseOpted->empty())
            return;

        sensor_msgs::PointCloud2 cloudmsg;
        pcl::toROSMsg(*mapPoseOpted, cloudmsg);
//        cloudmsg.header.stamp = curlocalmapMsg.header.stamp;
        cloudmsg.header.stamp = ros::Time::now();
//        cout << "ros::Time::now() : " << to_string(cloudmsg.header.stamp.toSec()) << endl;
        cloudmsg.header.frame_id = "/aft_mapped_spline";
        pubposesGraph.publish(cloudmsg);

        mapUnitNum = mapPoseOpted->size();
        if (mapUnitNum < 2)
            return;
        visualization_msgs::Marker markerEdge;
        markerEdge.header.stamp = ros::Time::now();
        markerEdge.header.frame_id = "/aft_mapped_spline";
        markerEdge.type = visualization_msgs::Marker::LINE_STRIP;
        markerEdge.scale.x = 0.3f;  // line width
        markerEdge.color.b = 1;
        markerEdge.color.a = 0.7f;
//        markerEdge.lifetime = ros::Duration(1);
        markerEdge.id = mapUnitNum;
        markerEdge.pose.position.x = mapPoseOpted->points.back().x;
        markerEdge.pose.position.y = mapPoseOpted->points.back().y;
        markerEdge.pose.position.z = mapPoseOpted->points.back().z;
        markerEdge.pose.orientation.x = mapPoseOpted->points.back().roll;
        markerEdge.pose.orientation.y = mapPoseOpted->points.back().pitch;
        markerEdge.pose.orientation.z = mapPoseOpted->points.back().yaw;
        markerEdge.pose.orientation.w = mapPoseOpted->points.back().intensity;

        geometry_msgs::Point point1, point2;
        for (auto pt : mapPoseOpted->points){
            point1.x = pt.x;
            point1.y = pt.y;
            point1.z = pt.z;
            markerEdge.points.push_back(point1);
        }
        markerEdge.action = visualization_msgs::Marker::ADD;
        pubgraphEdges.publish(markerEdge);

//        ros::Duration(0.2).sleep();  // sleep 0.2s
//
//        markerEdge.color.a = 0;
//        pubgraphEdges.publish(markerEdge);
    }

    void runMapFusionWithGraph(){

        ROS_INFO("\033[1;32m---->\033[0m GraphOptimization Started.");
        ros::Rate loop_rate(5);

        while(ros::ok()){

            loop_rate.sleep();

            publishMapPose();

            if (!parseMap())
                continue;

            fineMapFusion();
            updatePoseGraph_Map();
//            saveGlobalPoseAndMap();
            continue;

            /// optimize pose in window part
            if(!parseData())
                continue;

            updatePoseInWin();

            startInd = poseInSlidingWindow.front().first;
            endInd = poseInSlidingWindow.back().first;
            cout << YELLOW << "[ mapBT ] Pose begin from " << startInd
                 << " to " << endInd << RESET << endl;

//            testOneFactor();
            updatePoseGraph();

            publishCloud();
        }
        cout << "\033[1;32m ----> GraphOptimization thread closed.\033[0m" << endl;
//        saveGlobalMap();
    }

    void runBacktracingInWin(){

        ROS_INFO("\033[1;32m---->\033[0m GraphOptimization Started.");
        ros::Rate loop_rate(5);

        while(ros::ok()){

            loop_rate.sleep();

            /// optimize pose in window part
            if(!parseData())
                continue;

            updatePoseInWin();

            startInd = poseInSlidingWindow.front().first;
            endInd = poseInSlidingWindow.back().first;
            cout << YELLOW << "[ mapBT ] Pose begin from " << startInd
                 << " to " << endInd << RESET << endl;

//            testOneFactor();
            updatePoseGraph();

            publishCloud();
        }
        cout << "\033[1;32m ----> GraphOptimization thread closed.\033[0m" << endl;
//        saveGlobalMap();
    }

    void saveGlobalPoseAndMap(bool fuseMap = false){

        if(!mapPoseOpted->empty())
            pcl::io::savePCDFileBinary(projPath + "mapPosesOpted.pcd", *mapPoseOpted);
        cout << BOLDGREEN << "[ Graph ] Map Poses saved to " << projPath << RESET << endl;

        if(fuseMap){
            pcl::PointCloud<PointType>::Ptr tempCloud(new pcl::PointCloud<PointType>());
            mapUnitNum = localmapVec.size();
            for (int j = 0; j < mapUnitNum; ++j) {

                cout << " Integrating Frame " << j << endl;
                *tempCloud += *transformPointCloud(localmapVec[j], &mapPoseOpted->at(j));
            }
            if (!tempCloud->empty())
                pcl::io::savePCDFileBinaryCompressed(projPath + "globalmap_Fused.pcd", *tempCloud);
        }
    }
};


int main(int argc, char** argv){

    ros::init(argc, argv, "graph_optimization");
    GraphOptimization graphOptimization;

    thread processor(&GraphOptimization::runMapFusionWithGraph, &graphOptimization);
    thread loopThread(&GraphOptimization::detectLoopByPosition, &graphOptimization);
//    thread bactracingThread(&GraphOptimization::runBacktracingInWin, &graphOptimization);

    ros::spin();
    processor.join();
    loopThread.join();
//    bactracingThread.join();
    graphOptimization.saveGlobalPoseAndMap(true);
    cout << BLUE << "[ ROS ] graph_optimization node is done." << RESET << endl;
    return 0;
}