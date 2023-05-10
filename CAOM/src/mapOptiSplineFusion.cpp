/////////////////////////////////////////////////////////////////////////////////////////////////////
/// Created by joe on 2020/10/6.
///
/// 20201015:
/// add Catmull Rom spline, splineT : 1;
/// result is not better
///
/// 20201019:
/// B-Spline: from data points to control points, better
///
/// 20210203:
/// not good at handle strong motion(rotation)?
///
/// 20210209：
/// 1)the map is noisy when added outliers!;
/// 2)the pose after optimization sometimes can be erroneously large such as tx > 1m,
///
/// 20210413:
/// add Sophus Derivatives for spline with splineT : 2;
/// TODO : fix the derivatives !!!
///
//////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ceresAnalyticCost.h"
#include "ceresSplineCost.h"
#include "ceresSplineCostSophus.h"
#include "tools.h"
#include "nanoflann.hpp"
#include "csignal"
#include "structural_mapping/localmapWithPoseMsg.h"

// dynamic kdtree
typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloudforKDTREE<double> >,
        PointCloudforKDTREE<double>, 3> DynamicKDtree;

class mapOptimizationSpline{

private:

    ros::NodeHandle nh;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubCurCloudInWorld;

    ros::Publisher pubOdomAftMapped;
    ros::Publisher pubKeyPoses;

    ros::Publisher pubValidfeaPoints;
    ros::Publisher pubCurLocalMapAll;
    ros::Publisher pubCurCloudOri;
    ros::Publisher pubCurCloudaft;
    ros::Publisher pubSurfelsCloud;
    ros::Publisher pubLocalmapWithPose;

    ros::Subscriber subCloudInfo;
    ros::Subscriber subLaserOdometry;
    ros::Subscriber subImu;

    ros::Subscriber subvelodyneScan;

    structural_mapping::cloud_info featureCloudinfo;
    deque<structural_mapping::cloud_infoConstPtr> featureCloudinfoQue;

    nav_msgs::Odometry odomAftMapped;
    tf::StampedTransform aftMappedTrans;
    tf::TransformBroadcaster tfBroadcaster;

    int latestFrameID;

    PointType previousRobotPosPoint;
    PointType currentRobotPosPoint;

    // 特征点intensity = row + (col/10000)
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfTotalLastDS;

    pcl::PointCloud<PointType>::Ptr cloudCurInWorld;
    pcl::PointCloud<PointType>::Ptr localMapAll;
    pcl::PointCloud<PointType>::Ptr localMapAllLabled;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;
    pcl::KdTreeFLANN<pcl::PointSurfel>::Ptr kdtreeSurfelsFromMap;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterOutlier;

    double timeCloudinfo;
    double timeLaserOdometry;

    bool saveThisKeyFrame = false;
    int scanNums = 0;
    bool saveDateTofolder = false;

    std::mutex mtx;

    double timeLastProcessing;

    PointType pointOri, pointSel;

    bool isDegenerate;

    int laserCloudCornerFromMapDSNum;
    int laserCloudSurfFromMapDSNum;
    int laserCloudCornerLastDSNum;
    int laserCloudSurfLastDSNum;
    int laserCloudOutlierLastDSNum;
    int laserCloudSurfTotalLastDSNum;

    bool potentialLoopFlag;
    bool aLoopIsClosed;

    pcl::console::TicToc timer;
    bool isOutdoor = true;

    bool enablefilterObj;
    int lastMapsize = 0;  // for save global map
    int anchorID = 0;  // where to build local map(the size of frames)
    pcl::PointCloud<PointType>::Ptr curSurfcloudall;
    pcl::PointCloud<PointType>::Ptr curCornercloudall;
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudkeyframes;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudkeyframes;

    pcl::PointCloud<PointType>::Ptr localcornerMap;
    pcl::PointCloud<PointType>::Ptr localSurfMap;
    pcl::PointCloud<pcl::PointSurfel>::Ptr localsurfels;

    pcl::PointCloud<PointType>::Ptr localcornerMapLabeled;
    pcl::PointCloud<PointType>::Ptr localSurfMapLabeled;

    pcl::PointCloud<PointTypePose>::Ptr globalposesQuat;
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointType>::Ptr laserCloudOri;

    clock_t st, et;
    double ut;
    /// pose format (double)[qx, qy, qz, qw, tx, ty, tz, timestamp]
    double* anchorPose;  // where registration is perfomed (local coordiante)
    std::vector<double* > contrlposesVec;  // relative pose for optimization
    std::vector<double* > odomposeInSpline;  // odom pose to calculate relative pose in spline
    std::deque<double* > odomposeBufVec;  // save the odom pose [transformSum]

    // withdrawed data from buffer, to construct current cloud
    std::deque<pcl::PointCloud<PointType> > curSurfCloudsTobeAdded;
    std::deque<pcl::PointCloud<PointType> > curCornerCloudsTobeAdded;
    std::deque<pcl::PointCloud<PointType> > curOutlierCloudsTobeAdded;

    int splineNum;  // number of frames to fit the spline
    const int splineControlNum = 5;  // spline control points num
    int slidingFrameNum;  // sliding the time window of spline

    bool initedsp = false, badregistered = false;
    bool visualize = true;
    double localmapResolu;  // the local map resolution
    double searchRadi;
    double annealfactor;

    double systemStartTime, splineDura;
    std::vector<int > savedmapFrameInds;  // when to filter localmap(in anchor frame)
    std::vector<int > badregisFrameInds;  // frame that didnt align well
    double localmapDura;  // seconds of localmap duration
    pcl::PassThrough<PointType>::Ptr passthroughFilter;

    unordered_map<size_t , Eigen::Matrix3d> surfelCovs;  // save for the surfel covariance
    int numOfptInitSurfel ;
    double initRadiSurfel ;

    bool mapInited = false, surfelInited = false;
    std::unique_ptr<DynamicKDtree> dynamicKDtreeSurfel;
    std::unique_ptr<PointCloudforKDTREE<double> > cloudsurfelCentros;
    double cloudoverlapRatio, timerWindowRatio;

    int lineCnt = 0, planeCnt = 0, reInitedTimes;
    bool initializeSpline = true, useSurfels, labelMap;
    int optiIterations, minLocalMapSize, maxLocalmapSize;
    PointTypePose lastglobalPoseQuat;

    int randN;  // random frame num for debug
    string resultFolder;
    double maxRatio = 1.0;

public:

    mapOptimizationSpline():
            nh("~"){

        nh.param<bool>  ("/mapOptiSpline/useSurfels", useSurfels, true);
        nh.param<bool>  ("/mapOptiSpline/labelMap", labelMap, false);
        nh.param<double>("/mapOptiSpline/cloudoverlapRatio", cloudoverlapRatio, 0.6f);
        nh.param<double>("/mapOptiSpline/initRadiSurfel", initRadiSurfel, 0.6f);
        nh.param<double>("/mapOptiSpline/annealfactor", annealfactor, 1.2f);
        nh.param<double>("/mapOptiSpline/searchRadi", searchRadi, 0.4f);
        nh.param<double>("/mapOptiSpline/localmapResolu", localmapResolu, 0.1f);
        nh.param<int>   ("/mapOptiSpline/numOfptInitSurfel", numOfptInitSurfel, 10);
        nh.param<double>("/mapOptiSpline/localmapDura", localmapDura, 20.f);
        nh.param<double>("/mapOptiSpline/timerWindowRatio", timerWindowRatio, 1.5f);
        nh.param<float> ("/mapOptiSpline/ds_leaf_mapping", ds_leaf_mapping, 0.3f);
        nh.param<int>   ("/mapOptiSpline/optiIterations", optiIterations, 6);
        nh.param<int>   ("/mapOptiSpline/minLocalMapSize", minLocalMapSize, 30000);
        nh.param<int>   ("/mapOptiSpline/maxLocalmapSize", maxLocalmapSize, 170000);
        nh.param<int>   ("/mapOptiSpline/splineT", splineT, 0);
        nh.param<int>   ("/mapOptiSpline/splineNum", splineNum, 5);
        nh.param<int>   ("/mapOptiSpline/slidingFrameNum", slidingFrameNum, 3);
        nh.param("filterObjects", enablefilterObj, false);
        nh.getParam("/projpath", projPath);  // global param add '/' before name

        resultFolder = projPath + "splineFusionRes_";
        boost::filesystem::create_directory(boost::filesystem::path(resultFolder));
        if(splineT == 2) slidingFrameNum = splineNum - (splineNum-1)/2;

        cout << "[ MapOpti ] useSurfels        " << useSurfels        << endl;
        cout << "[ MapOpti ] labelMap          " << labelMap          << endl;
        cout << "[ MapOpti ] cloudoverlapRatio " << cloudoverlapRatio << endl;
        cout << "[ MapOpti ] initRadiSurfel    " << initRadiSurfel    << endl;
        cout << "[ MapOpti ] annealfactor      " << annealfactor      << endl;
        cout << "[ MapOpti ] searchRadi        " << searchRadi        << endl;
        cout << "[ MapOpti ] localmapResolu    " << localmapResolu    << endl;
        cout << "[ MapOpti ] numOfptInitSurfel " << numOfptInitSurfel << endl;
        cout << "[ MapOpti ] localmapDura      " << localmapDura      << endl;
        cout << "[ MapOpti ] timerWindowRatio  " << timerWindowRatio  << endl;
        cout << "[ MapOpti ] optiIterations    " << optiIterations    << endl;
        cout << "[ MapOpti ] minLocalMapSize   " << minLocalMapSize   << endl;
        cout << "[ MapOpti ] maxLocalmapSize   " << maxLocalmapSize   << endl;
        cout << "[ MapOpti ] enablefilterObj : "  << enablefilterObj  << endl;
        cout << "[ MapOpti ] ds_leaf_mapping : "  << ds_leaf_mapping  << endl;
        cout << "[ MapOpti ] enableLoopClosure : "<< loopClosureEnableFlag << endl;
        cout << "[ MapOpti ] splineT : "          << splineT << endl;

        cout << "[ MapOpti ] projFolder : "  << projPath << endl;
        cout << "[ MapOpti ] projFolder : "  << resultFolder << endl;
        cout << "[ MapOpti ] weightModel : " << weightModeltype << endl;

        cout << "[ MapOpti ] splineNum : "  << splineNum << endl;
        cout << "[ MapOpti ] splineControlNum : " << splineControlNum << endl;
        cout << "[ MapOpti ] slidingFrameNum : "  << slidingFrameNum << endl;

        std::string logpath = resultFolder + "logs/logMapOptiSP-";
        google::SetLogDestination(google::INFO, static_cast<const char*>(logpath.c_str()));

        // 订阅
        // 作为current关键帧点云
        subCloudInfo = nh.subscribe<structural_mapping::cloud_info>("/feature_cloud_info", 1,
//        subCloudInfo = nh.subscribe<structural_mapping::cloud_info>("/odom_featureCloudinfo", 1,
                                                                    &mapOptimizationSpline::laserCloudInfoHandler, this);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/odom_pose", 1,
                                                            &mapOptimizationSpline::laserOdometryHandler, this);

        //        subImu = nh.subscribe<sensor_msgs::Imu> (imuTopic, 50, &mapOptimization::imuHandler, this);//topic: /imu/data

//        subvelodyneScan = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 1,
//                                                                 &mapOptimization::velodynecloudHandler, this);

        // 发布节点
        pubCurLocalMapAll = nh.advertise<sensor_msgs::PointCloud2>("/recent_cloud_mapping", 1);
        pubValidfeaPoints = nh.advertise<sensor_msgs::PointCloud2>("/validfeapoints", 1);  //
        pubKeyPoses       = nh.advertise<sensor_msgs::PointCloud2>("/key_pose_origin", 1);  // cloudkeyposes3D
        pubCurCloudOri    = nh.advertise<sensor_msgs::PointCloud2>("/curCloudOri_mapping", 1);
        pubCurCloudaft    = nh.advertise<sensor_msgs::PointCloud2>("/curCloudaft_mapping", 1);
        pubSurfelsCloud   = nh.advertise<sensor_msgs::PointCloud2> ("/map_surfels", 1);
        pubCurCloudInWorld= nh.advertise<sensor_msgs::PointCloud2> ("/curCloud_world", 10);  // current frame cloud added to map
        pubOdomAftMapped  = nh.advertise<nav_msgs::Odometry> ("/aft_mapped_spline", 1);  // tf

        pubLocalmapWithPose= nh.advertise<structural_mapping::localmapWithPoseMsg>("/localmapWithPose", 1);  // for pose graph and loop

        allocateMemory();
    }

    void allocateMemory(){

        downSizeFilterCorner.setLeafSize(ds_leaf_mapping, ds_leaf_mapping, ds_leaf_mapping/2);
        downSizeFilterSurf.setLeafSize(ds_leaf_mapping, ds_leaf_mapping, ds_leaf_mapping);
        downSizeFilterOutlier.setLeafSize(ds_leaf_mapping, ds_leaf_mapping, ds_leaf_mapping);

        cloudsurfelCentros.reset(new PointCloudforKDTREE<double>);
        dynamicKDtreeSurfel.reset(new DynamicKDtree(3, *cloudsurfelCentros));

        contrlposesVec.resize(splineNum);
        for (int k = 0; k < splineNum; ++k)
            contrlposesVec[k] = new double[8]{0};

        odomposeInSpline.resize(splineNum);
        for (int k = 0; k < splineNum; ++k)
            odomposeInSpline[k] = new double[8]{0};

        anchorPose = new double [8]{0};

        passthroughFilter.reset(new pcl::PassThrough<PointType>(true));

        localSurfMap.reset(new pcl::PointCloud<PointType>());
        localcornerMap.reset(new pcl::PointCloud<PointType>());
        localSurfMapLabeled.reset(new pcl::PointCloud<PointType>());
        localcornerMapLabeled.reset(new pcl::PointCloud<PointType>());

        curSurfcloudall.reset(new pcl::PointCloud<PointType>());
        curCornercloudall.reset(new pcl::PointCloud<PointType>());
        localsurfels.reset(new pcl::PointCloud<pcl::PointSurfel>());
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        globalposesQuat.reset(new pcl::PointCloud<PointTypePose>());


        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfelsFromMap.reset(new pcl::KdTreeFLANN<pcl::PointSurfel>());

        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudOri.reset(new pcl::PointCloud<PointType>());

        cloudCurInWorld.reset(new pcl::PointCloud<PointType>());
        localMapAll.reset(new pcl::PointCloud<PointType>());
        localMapAllLabled.reset(new pcl::PointCloud<PointType>());

        timeLaserOdometry = 0;
        timeCloudinfo = 0;

        timeLastProcessing = -1;

        isDegenerate = false;

        laserCloudCornerFromMapDSNum = 0;
        laserCloudSurfFromMapDSNum = 0;
        laserCloudCornerLastDSNum = 0;
        laserCloudSurfLastDSNum = 0;
        laserCloudOutlierLastDSNum = 0;
        laserCloudSurfTotalLastDSNum = 0;

        potentialLoopFlag = false;
        aLoopIsClosed = false;

        latestFrameID = 0;
    }

    void laserCloudInfoHandler(const structural_mapping::cloud_infoConstPtr& msgIn) {

        std::lock_guard<mutex> lockGuard(mtx);
        featureCloudinfoQue.emplace_back(msgIn);
    }
    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr& laserOdometry){

        timeLaserOdometry = laserOdometry->header.stamp.toSec();

        double* tmp = new double[8];
        tmp[0] = laserOdometry->pose.pose.orientation.x;
        tmp[1] = laserOdometry->pose.pose.orientation.y;
        tmp[2] = laserOdometry->pose.pose.orientation.z;
        tmp[3] = laserOdometry->pose.pose.orientation.w;
        tmp[4] = laserOdometry->pose.pose.position.x;
        tmp[5] = laserOdometry->pose.pose.position.y;
        tmp[6] = laserOdometry->pose.pose.position.z;
        tmp[7] = timeLaserOdometry;

        std::lock_guard<mutex> lockGuard(mtx);
        odomposeBufVec.emplace_back(tmp);
        cout << CYAN << "[ mapSP ] RECEIVED ODOM # " << to_string(odomposeBufVec.back()[7]) << RESET << endl;
    }
    void velodynecloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        if(saveThisKeyFrame && saveDateTofolder)
        {
            cout << "start saving valid /velodyne data>>>> " << endl;
            pcl::PointCloud<pcl::PointXYZI>::Ptr veloScan(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::fromROSMsg(*laserCloudMsg, *veloScan);
            string velofolder = "/home/joe/workspace/testData/veloScans/";
            pcl::io::savePCDFile(velofolder +
                                 to_string(laserCloudMsg->header.stamp.toSec())+".pcd", *veloScan);
            cout << "#Saved scan~" << endl;

            FILE *fp;
            fp = fopen( "/home/joe/workspace/testData/veloScans/timesOfscan.txt","a");
            fprintf(fp, "%lf\n", laserCloudMsg->header.stamp.toSec());
            fclose(fp);
            scanNums++;
        }
    }


    Eigen::Matrix4d getTransMatrixT(const Eigen::Quaterniond &q, const Eigen::Vector3d &t){

        Eigen::Matrix4d curT;
        curT.block(0,0,3,3) = q.matrix();
        curT.block(0,3,3,1) = t;
        curT.block(3,0,1,4) = Eigen::Vector4d(0, 0, 0, 1).transpose();
        return curT;
    }
    Eigen::Affine3d getTransformAff(const double* pose){

        Eigen::Affine3d transform = Eigen::Affine3d::Identity();
        Eigen::Quaterniond estimated_rot( pose[3], pose[0], pose[1],pose[2]);
        estimated_rot.normalize();
        transform.rotate(estimated_rot);
        transform.pretranslate(Eigen::Vector3d(pose[4], pose[5], pose[6]));
        return transform;
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po) {

        double *ptpose = new double[8]{0};
        if (splineT == 2)
            SophusSpline::CeresFactorsSP::splineFusionPoses(contrlposesVec[0], contrlposesVec[1],
                                                            contrlposesVec[2],contrlposesVec[3],
                                                            contrlposesVec[4], ptpose, double(pi->intensity), splineT);

        else
            CeresFactorsSP::splineFusionPoses(contrlposesVec[0], contrlposesVec[1],
                                              contrlposesVec[2],contrlposesVec[3],
                                              contrlposesVec[4], ptpose, double(pi->intensity), splineT);
        Eigen::Vector4d ptIn;
        Eigen::Vector3d ptOut;
        ptIn << pi->x, pi->y, pi->z, 1.0;
        ptOut = (getTransformAff(ptpose) * ptIn).head<3>();
        po->x = ptOut(0);
        po->y = ptOut(1);
        po->z = ptOut(2);
        po->intensity = pi->intensity;
        delete ptpose;
        ptpose = NULL;
    }
    // get relative transform from cur to tar
    // ! 不能将R，t分开求解 ！
    void getRelativeTransform(const double* tar, const double *cur, double* res){

        PointTypePose rela ;
        // watch the order !
        Eigen::Quaterniond curR (cur[3], cur[0], cur[1], cur[2]);
        Eigen::Quaterniond tarR (tar[3], tar[0], tar[1], tar[2]);
        curR.normalize();
        tarR.normalize();

        Eigen::Matrix4d curT, tarT, relaT;
        curT = getTransMatrixT(curR, Eigen::Vector3d(cur[4], cur[5], cur[6]) );
        tarT = getTransMatrixT(tarR, Eigen::Vector3d(tar[4], tar[5], tar[6]) );

        relaT = curT.inverse() * tarT;

        Eigen::Matrix3d rot = relaT.block(0,0,3,3);
        Eigen::Quaterniond quat(rot) ;
        quat.normalize();
        res[0] = quat.x();
        res[1] = quat.y();
        res[2] = quat.z();
        res[3] = quat.w();

        res[4] = relaT(0,3);
        res[5] = relaT(1,3);
        res[6] = relaT(2,3);

        res[7] = tar[7];
    }


    /// REMOVE points from cloud by indices.
    /// \param inCloud : output
    /// \param indices : point-indices in inCloud
    void filterOutFromCloudByIndices(pcl::PointCloud<PointType>::Ptr inCloud,
                                     pcl::PointIndicesPtr indicesPtr){

        pcl::ExtractIndices<PointType>::Ptr extractor(new pcl::ExtractIndices<PointType>());
        extractor->setInputCloud(inCloud);
        extractor->setIndices(indicesPtr);
        extractor->setNegative(true);
        extractor->filter(*inCloud);
    }


    void publishTF(){

        PointTypePose poseCur = globalposesQuat->points.back();
        odomAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
        odomAftMapped.header.frame_id = "/aft_mapped_spline";
//        odomAftMapped.child_frame_id = "/map";
        odomAftMapped.child_frame_id = "/base_link";
        odomAftMapped.pose.pose.orientation.x = poseCur.roll;
        odomAftMapped.pose.pose.orientation.y = poseCur.pitch;
        odomAftMapped.pose.pose.orientation.z = poseCur.yaw;
        odomAftMapped.pose.pose.orientation.w = poseCur.intensity;
        odomAftMapped.pose.pose.position.x = poseCur.x;
        odomAftMapped.pose.pose.position.y = poseCur.y;
        odomAftMapped.pose.pose.position.z = poseCur.z;

        //储存当前帧的transformSum即Odometry累计值
        odomAftMapped.twist.twist.angular.x = lastglobalPoseQuat.roll;
        odomAftMapped.twist.twist.angular.y = lastglobalPoseQuat.pitch;
        odomAftMapped.twist.twist.angular.z = lastglobalPoseQuat.yaw;
        odomAftMapped.twist.covariance[0]   = lastglobalPoseQuat.intensity;

        odomAftMapped.twist.twist.linear.x = lastglobalPoseQuat.x;
        odomAftMapped.twist.twist.linear.y = lastglobalPoseQuat.y;
        odomAftMapped.twist.twist.linear.z = lastglobalPoseQuat.z;
        pubOdomAftMapped.publish(odomAftMapped);

        aftMappedTrans.frame_id_ = "/aft_mapped_spline";
        aftMappedTrans.child_frame_id_ = "/base_link";
        aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
        aftMappedTrans.setRotation(tf::Quaternion(poseCur.roll, poseCur.pitch, poseCur.yaw, poseCur.intensity));
        aftMappedTrans.setOrigin(tf::Vector3(poseCur.x, poseCur.y, poseCur.z));
        tfBroadcaster.sendTransform(aftMappedTrans);
    }

    void publishKeyPosesAndFrames(){

        if (pubKeyPoses.getNumSubscribers() != 0){

            sensor_msgs::PointCloud2 cloudMsgTemp;
//            pcl::toROSMsg(*cloudKeyPoses3D, cloudMsgTemp);
            pcl::toROSMsg(*globalposesQuat, cloudMsgTemp);  // pass
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/aft_mapped_spline";
            pubKeyPoses.publish(cloudMsgTemp);
        }

        if (pubCurLocalMapAll.getNumSubscribers() != 0){

            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg((*localSurfMap)+(*localcornerMap), cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/base_link";
            pubCurLocalMapAll.publish(cloudMsgTemp);
        }

        if (pubSurfelsCloud.getNumSubscribers() != 0){

            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*localsurfels, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/base_link";
            pubSurfelsCloud.publish(cloudMsgTemp);
        }

        if (pubCurCloudInWorld.getNumSubscribers() != 0){

            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*cloudCurInWorld, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/aft_mapped_spline";
            pubCurCloudInWorld.publish(cloudMsgTemp);
        }

    }

    void printControlPose(){

        cout << "------Timestamp------- tx ------- ty -------- tz " << endl;
        for (int j = 0; j < splineNum; ++j) {

            cout << to_string(contrlposesVec[j][7]) << " : "
                 << contrlposesVec[j][4] << " " << contrlposesVec[j][5] << " " << contrlposesVec[j][6] << endl;
//            cout << to_string(odomposeBufVec[j][7]) << " : "
//                 << odomposeBufVec[j][4] << " " << odomposeBufVec[j][5] << " " << odomposeBufVec[j][6] << endl;
        }
    }
    void printPoseInSpline(){

        cout << "------Timestamp------- tx ------- ty -------- tz " << endl;

        for (int j = 0; j < splineControlNum; ++j) {

            double ratio = (contrlposesVec[j][7] - contrlposesVec.front()[7]) /
                           (contrlposesVec.back()[7] - contrlposesVec.front()[7]);

            auto* poseTemp = new double[8]{0};
            if(splineT == 2)
                SophusSpline::CeresFactorsSP::splineFusionPoses(contrlposesVec[0],
                                                                contrlposesVec[1],
                                                                contrlposesVec[2],
                                                                contrlposesVec[3],
                                                                contrlposesVec[4],
                                                                poseTemp, ratio, splineT);
            else
                CeresFactorsSP::splineFusionPoses(contrlposesVec[0], contrlposesVec[1],
                                                  contrlposesVec[2], contrlposesVec[3],
                                                  contrlposesVec[4], poseTemp, ratio, splineT);

            cout << to_string(poseTemp[7]) << " : " << poseTemp[4] << " " << poseTemp[5] << " " << poseTemp[6] << endl;
            delete poseTemp;
        }

    }

    // for motion distortion comparison
    void outputCloudbeforeSpOpti(bool ifsave = false){

        cout << "[ Control ] Before Optimization. " << endl;
        printControlPose();

        randN = rand() % 500;  // random fileNo.

        pcl::PointCloud<PointType>::Ptr frameCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr tmp(new pcl::PointCloud<PointType>());
        for (int j = 0; j < splineNum-1; ++j) {

            Eigen::Affine3d transToCurFrame = (getTransformAff(contrlposesVec[j]));

//            pcl::transformPointCloud(curSurfCloudsTobeAdded[j], *tmp, transToCurFrame);
//            *frameCloud += *tmp;
//            pcl::transformPointCloud(curCornerCloudsTobeAdded[j], *tmp, transToCurFrame);
//            *frameCloud += *tmp;
            pcl::transformPointCloud(curOutlierCloudsTobeAdded[j], *tmp, transToCurFrame);
            *frameCloud += *tmp;
        }

        if(pubCurCloudOri.getNumSubscribers()){

            sensor_msgs::PointCloud2 cloudmsg;
            pcl::toROSMsg(*frameCloud, cloudmsg);
            cloudmsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudmsg.header.frame_id = "/base_link";
            pubCurCloudOri.publish(cloudmsg);
        }
        if(ifsave)
            pcl::io::savePCDFileBinary(projPath + "localmaps/frameOri"
                                       + to_string(randN) + ".pcd", *frameCloud);
//        + to_string(anchorID) + ".pcd", (*curSurfcloudall)+(*curCornercloudall));
        frameCloud->clear();
    }
    void outputCloudAftSpOpti(bool ifsave = false){

        cout << "[ Control ] After Optimization. " << endl;
        printControlPose();
        cout << "[ Spline ] After Optimization. " << endl;
        printPoseInSpline();

        pcl::PointCloud<PointType>::Ptr frameCloud(new pcl::PointCloud<PointType>());
        PointType tmp;
        double st = contrlposesVec.front()[7] - systemStartTime;
        double totalt = contrlposesVec.back()[7] - contrlposesVec.front()[7];

        for (int j = 1; j < splineNum-2; ++j) {
//#pragma omp parallel for
//            for(auto pt : curSurfCloudsTobeAdded[j]){
//                pointAssociateToMap(&pt, &tmp);
//                tmp.intensity = pt.intensity*totalt + st;
//                frameCloud->points.emplace_back(tmp);
//            }
//#pragma omp parallel for
//            for(auto pt : curCornerCloudsTobeAdded[j]){
//                pointAssociateToMap(&pt, &tmp);
//                tmp.intensity = pt.intensity*totalt + st;
//                frameCloud->points.emplace_back(tmp);
//            }
//#pragma omp parallel for
            for(auto pt : curOutlierCloudsTobeAdded[j]){
                pointAssociateToMap(&pt, &tmp);
                tmp.intensity = pt.intensity*totalt + st;
                frameCloud->points.emplace_back(tmp);
            }
        }

        if(pubCurCloudaft.getNumSubscribers()){

            sensor_msgs::PointCloud2 cloudmsg;
            pcl::toROSMsg(*frameCloud, cloudmsg);
            cloudmsg.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudmsg.header.frame_id = "/base_link";
            pubCurCloudaft.publish(cloudmsg);
        }
        if(ifsave)
            pcl::io::savePCDFileBinary(projPath + "localmaps/frameOpt"
                                       + to_string(randN) + ".pcd", *frameCloud);
        frameCloud->clear();
    }

    void saveCurLocalmap(){

        pcl::PointCloud<PointType>::Ptr saveCloud(new pcl::PointCloud<PointType>());
//        *saveCloud = *localSurfMapLabeled + *localcornerMap;
        *localMapAllLabled = (*localSurfMapLabeled) + (*localcornerMapLabeled);
        *localMapAll = (*localSurfMap) + (*localcornerMap);
        savedmapFrameInds.emplace_back(anchorID);

//        if(strcmp(projPath.c_str(), "nan")){
//
//            cout << "[ Debug ] Saving localmap to " << projPath << " No : " << anchorID << endl;
////            pcl::io::savePCDFileBinary(projPath + "localmaps/mapCorner"
////                                       + to_string(anchorID) + ".pcd", *localcornerMap);
////            pcl::io::savePCDFileBinary(projPath + "localmaps/mapSurf"
////                                       + to_string(anchorID) + ".pcd", *localSurfMap);
//            if(!localSurfMap->empty())
//                pcl::io::savePCDFileBinary(projPath + "localmaps/localmap"
//                                           + to_string(anchorID) + ".pcd", *localMapAll);
//            if(!localMapAllLabled->empty())
//                pcl::io::savePCDFileBinary(projPath + "localmaps/mapLabeled"
//                                           + to_string(anchorID) + ".pcd", *localMapAllLabled);
//        }

    }

    // preserve the points with timestamp between t1 and t2
    // NOTE: timestamp is relative(begin from the start of mapping)
    void filterMapByTime(const double& t1,
                         const double& t2, bool save = true){

        cout << "[ Debug ] Filter Time window " << to_string(t1) << " ~ " << to_string(t2) << endl;

        pcl::PointCloud<PointType>::Ptr surfMapFiltered(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr cornerMapFiltered(new pcl::PointCloud<PointType>());

        passthroughFilter->setFilterFieldName("intensity");
        passthroughFilter->setFilterLimits(t1, t2);

        passthroughFilter->setInputCloud(localSurfMap);
        passthroughFilter->filter(*surfMapFiltered);
        if(labelMap){
            pcl::PointIndicesPtr ptIndsptr(new pcl::PointIndices());
            passthroughFilter->getRemovedIndices(*ptIndsptr);
            filterOutFromCloudByIndices(localSurfMapLabeled, ptIndsptr);
        }

        passthroughFilter->setInputCloud(localcornerMap);
        passthroughFilter->filter(*cornerMapFiltered);
        if(labelMap){
            pcl::PointIndicesPtr ptIndsptr(new pcl::PointIndices());
            passthroughFilter->getRemovedIndices(*ptIndsptr);
            filterOutFromCloudByIndices(localcornerMapLabeled, ptIndsptr);
        }

        if(save){

        }else{
            *localSurfMap = *surfMapFiltered;
            *localcornerMap = *cornerMapFiltered;
        }

        cout << "[ MAP ] Local map filtered. " << "Left " << localcornerMap->size()
             << " / " << localSurfMap->size() << " points in feature map." << endl;

        if(badregistered && (localSurfMap->size()+localcornerMap->size()) > minLocalMapSize)
            publishLocalMapWithPose(globalposesQuat->points.back(),true);  // not current estimated pose
    }

    void publishLocalMapWithPose(PointTypePose pose, bool ifbad = false){

        structural_mapping::localmapWithPoseMsg mapMsg;
        mapMsg.header.stamp = ros::Time().fromSec(pose.time);
        mapMsg.header.frame_id = "/base_link";
        mapMsg.badregistered = ifbad;

        saveCurLocalmap();
        cout << BOLDMAGENTA << "[ Debug ] Publish localmap : " << localMapAll->size() << RESET << endl;

        sensor_msgs::PointCloud2 cloudMsgTemp;
        pcl::toROSMsg(*localMapAll, cloudMsgTemp);
        cloudMsgTemp.header.stamp = ros::Time().fromSec(pose.time);
        cloudMsgTemp.header.frame_id = "/base_link";
        mapMsg.localmap = cloudMsgTemp;

        mapMsg.pose.position.x = pose.x;
        mapMsg.pose.position.y = pose.y;
        mapMsg.pose.position.z = pose.z;
        mapMsg.pose.orientation.x = pose.roll;
        mapMsg.pose.orientation.y = pose.pitch;
        mapMsg.pose.orientation.z = pose.yaw;
        mapMsg.pose.orientation.w = pose.intensity;

        pubLocalmapWithPose.publish(mapMsg);
    }

    /// transform local map to current anchor frame
    void prepareLocalMap(){

        // set origin
        if(globalposesQuat->points.empty()){

            PointTypePose origin;
            origin.x = origin.y = origin.z = origin.roll = origin.pitch = origin.yaw = 0;
            origin.intensity = 1.0;
            origin.time = timeLaserOdometry;
            systemStartTime = timeLaserOdometry;
            globalposesQuat->points.emplace_back(origin);
            cout << BOLDRED << "[ Spline ] Origin initiated at TIME " << to_string(systemStartTime) << RESET << endl;
            initedsp = true;
            reInitedTimes = 0;
            return;
        }

        // filter out old points in local map
        double endtime = localSurfMap->points.back().intensity;
        double begintime = localSurfMap->points.front().intensity;
        if(badregistered){

            localcornerMap->clear();
            localSurfMap->clear();
            cout << BOLDMAGENTA << "[ Spline ] Local map & poses cleared ! " ;
            cout << "#######################################################" << RESET << endl;

            saveposesAndframes(true);
            reInitedTimes++;

            cornerCloudkeyframes.clear();
            surfCloudkeyframes.clear();
            globalposesQuat->points.resize(1);  // save the first identity pose
            initedsp = true;

        }else if(initedsp && endtime - begintime > localmapDura){  // publish map at the start or restart of system

            publishLocalMapWithPose(globalposesQuat->points.back());
            initedsp = false;
        }else if(endtime - begintime > localmapDura*timerWindowRatio ||
                 localSurfMap->size() > maxLocalmapSize){  // the system has already run for a period

            filterMapByTime(endtime - localmapDura, endtime, false);
            publishLocalMapWithPose(globalposesQuat->points.back());
        }

        laserCloudCornerFromMapDSNum = localcornerMap->points.size();
        laserCloudSurfFromMapDSNum = localSurfMap->points.size();

        cout << " Current local CORNER map : " << laserCloudCornerFromMapDSNum << endl;
        cout << " Current local SURF map : "   << laserCloudSurfFromMapDSNum << endl;
        cout << " Current local SURFEL map : " << localsurfels->points.size() << endl;

        if(laserCloudCornerFromMapDSNum < 100)
            return;

        // transform local map to local frame
        Eigen::Affine3d transToCurFrame = (getTransformAff(anchorPose).inverse());
        pcl::transformPointCloud(*localcornerMap, *localcornerMap, transToCurFrame);
        pcl::transformPointCloud(*localSurfMap,   *localSurfMap, transToCurFrame);
        pcl::transformPointCloudWithNormals(*localsurfels, *localsurfels, transToCurFrame);

        // for dynamic kdtree search
        for (int j = 0; j < localsurfels->points.size(); ++j) {
            cloudsurfelCentros->pts[j].x = localsurfels->points[j].x;
            cloudsurfelCentros->pts[j].y = localsurfels->points[j].y;
            cloudsurfelCentros->pts[j].z = localsurfels->points[j].z;
            surfelCovs[j] = (transToCurFrame.rotation() * surfelCovs[j]).eval();

//            cout << "[ SURFEL] Transformed Normal " << localsurfels->points[j].data_n[3]
//               << " " << localsurfels->points[j].data_n[0]
//               << " " << localsurfels->points[j].data_n[1]
//               << " " << localsurfels->points[j].data_n[2] << endl;
        }

        kdtreeCornerFromMap->setInputCloud(localcornerMap);
        kdtreeSurfFromMap->setInputCloud(localSurfMap);
        if(!localsurfels->points.empty())
            kdtreeSurfelsFromMap->setInputCloud(localsurfels);

        //cout << "[ Spline ] Local map Transformed to Anchor Pose. "<< endl;

        // update labeled Map
        if(!labelMap)
            return;
        pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>());
        localSurfMapLabeled->points.resize(laserCloudSurfFromMapDSNum);
        *temp = *localSurfMapLabeled;
        *localSurfMapLabeled = *localSurfMap;
#pragma omp parallel for
        for (int i = 0; i < laserCloudSurfFromMapDSNum; ++i)
            localSurfMapLabeled->points[i].intensity = temp->points[i].intensity;

        localcornerMapLabeled->points.resize(laserCloudCornerFromMapDSNum);
        *temp = *localcornerMapLabeled;
        *localcornerMapLabeled = *localcornerMap;
#pragma omp parallel for
        for (int i = 0; i < laserCloudCornerFromMapDSNum; ++i)
            localcornerMapLabeled->points[i].intensity = temp->points[i].intensity;
    }

    void downsampleCurrentScan() {

        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(curCornercloudall);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->points.size();

        laserCloudSurfTotalLastDS->clear();
        downSizeFilterSurf.setInputCloud(curSurfcloudall);
        downSizeFilterSurf.filter(*laserCloudSurfTotalLastDS);
        laserCloudSurfTotalLastDSNum = laserCloudSurfTotalLastDS->points.size();

        cout << BLUE << "Current corner size: " << laserCloudCornerLastDSNum
             << ", Current surf size: " << laserCloudSurfTotalLastDSNum << RESET << endl;
    }

    // to see the distribution of feature points used for pose estimation
    void publishvalidfeaturepoints(bool saveToFile = false){

        if (pubValidfeaPoints.getNumSubscribers() != 0){
            sensor_msgs::PointCloud2 cloudMsgTemp;
            pcl::toROSMsg(*laserCloudOri, cloudMsgTemp);
            cloudMsgTemp.header.stamp = ros::Time().fromSec(timeLaserOdometry);
            cloudMsgTemp.header.frame_id = "/base_link";
            pubValidfeaPoints.publish(cloudMsgTemp);

            if(saveToFile && !laserCloudOri->empty())
                pcl::io::savePCDFileBinaryCompressed(resultFolder + "pointsTobeOpted_" +
                                                     to_string(anchorID) + ".pcd", *laserCloudOri);
        }
    }

    /// update global poses by estimated incremental anchor pose
    void updateGlobalPose(){

        // last global pose
        lastglobalPoseQuat = globalposesQuat->points.back();
        Eigen::Quaterniond qlast (lastglobalPoseQuat.intensity, lastglobalPoseQuat.roll, lastglobalPoseQuat.pitch, lastglobalPoseQuat.yaw);
        Eigen::Vector3d tlast(lastglobalPoseQuat.x, lastglobalPoseQuat.y, lastglobalPoseQuat.z);
        qlast.normalize();

        // estimated relative pose
        Eigen::Quaterniond quaternionIncre (anchorPose[3], anchorPose[0],
                                            anchorPose[1], anchorPose[2]);
        Eigen::Vector3d tIncre(anchorPose[4],anchorPose[5],anchorPose[6]);
        quaternionIncre.normalize();

//        Eigen::AngleAxisd angleAxisd(quaternionIncre);
//        if(angleAxisd.angle() > )

        // add up
        Eigen::Matrix4d lastT, curT, increT;
        increT = getTransMatrixT(quaternionIncre, tIncre);
        lastT = getTransMatrixT( qlast, tlast);
        curT = lastT * increT;

        // record updated global pose
        lastglobalPoseQuat = matrix4fToPose(curT.cast<float>());
        lastglobalPoseQuat.time = anchorPose[7];
        globalposesQuat->points.push_back(lastglobalPoseQuat);
        anchorID = globalposesQuat->points.size();

        PointType tmp;
        tmp.z = lastglobalPoseQuat.z;
        tmp.x = lastglobalPoseQuat.x;
        tmp.y = lastglobalPoseQuat.y;
        tmp.intensity = cloudKeyPoses3D->points.size();
        cloudKeyPoses3D->points.emplace_back(tmp);

        //cout<<"[ Ceres ] Added global pose :" << anchorID << endl;

    }

    /// main functor
    void ceressolver(int iterCounts){

        double radius = searchRadi;

        for (int iterCount = 0; iterCount < iterCounts; iterCount++){

            ceres::LossFunction *loss_function = new ceres::CauchyLoss(0.1);

            ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
            ceres::Problem::Options problem_options;
            std::unique_ptr<ceres::Problem> problem;
            std::vector<ceres::ResidualBlockId > resiIDs;

            problem.reset(new ceres::Problem(problem_options));
            if(splineT == 2){  // self-defined params

                problem->AddParameterBlock(contrlposesVec[0], 7, new SophusSpline::PoseSE3SophusParameterization());
                problem->AddParameterBlock(contrlposesVec[1], 7, new SophusSpline::PoseSE3SophusParameterization());
                problem->AddParameterBlock(contrlposesVec[2], 7, new SophusSpline::PoseSE3SophusParameterization());
                problem->AddParameterBlock(contrlposesVec[3], 7, new SophusSpline::PoseSE3SophusParameterization());
                problem->AddParameterBlock(contrlposesVec[4], 7, new SophusSpline::PoseSE3SophusParameterization());
            }else if(splineT == 0) {  // CeresAutoDiffCostFunction

                problem->AddParameterBlock(contrlposesVec[0], 4, q_parameterization);
                problem->AddParameterBlock(contrlposesVec[1], 4, q_parameterization);
                problem->AddParameterBlock(contrlposesVec[2], 4, q_parameterization);
                problem->AddParameterBlock(contrlposesVec[3], 4, q_parameterization);
                problem->AddParameterBlock(contrlposesVec[4], 4, q_parameterization);
                problem->AddParameterBlock(contrlposesVec[0] + 4, 3);
                problem->AddParameterBlock(contrlposesVec[1] + 4, 3);
                problem->AddParameterBlock(contrlposesVec[2] + 4, 3);
                problem->AddParameterBlock(contrlposesVec[3] + 4, 3);
                problem->AddParameterBlock(contrlposesVec[4] + 4, 3);
            }
//            problem->SetParameterBlockConstant(contrlposesVec[0] + 4);
//            problem->SetParameterBlockConstant(contrlposesVec[0]);

            int corner_num = 0;
            if(iterCount < std::ceil(iterCounts/2)){  // only use this in the half iterations
                std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > nearCorners(5);
                Eigen::Vector3d center, unit_direction, curr_point;
                Eigen::Matrix3d covMat;
                for (int i = 0; i < laserCloudCornerLastDSNum; i++){

                    pointOri = laserCloudCornerLastDS->points[i];
                    if (pointOri.intensity > maxRatio) continue;  // fixme : ratio larger than 1.0
                    //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
                    pointAssociateToMap(&pointOri, &pointSel);
                    kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                    if (pointSearchSqDis[4] < 1.0){

                        center = Eigen::Vector3d (0, 0, 0);
                        int lineId = -1;
                        for (int j = 0; j < 5; j++){

                            Eigen::Vector3d tmp(localcornerMap->points[pointSearchInd[j]].x,
                                                localcornerMap->points[pointSearchInd[j]].y,
                                                localcornerMap->points[pointSearchInd[j]].z);
                            center = center + tmp;
//                            nearCorners.push_back(tmp);
                            nearCorners[j] = tmp;

//                            if(labelMap && localcornerMapLabeled->points[pointSearchInd[j]].intensity > 0)
//                                lineId = localcornerMapLabeled->points[pointSearchInd[j]].intensity;
                        }
                        center = center / 5.0;

                        covMat = Eigen::Matrix3d::Zero();
                        for (int j = 0; j < 5; j++){

                            Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                            covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                        }

                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                        // if is indeed line feature
                        // note Eigen library sort eigenvalues in increasing order
                        unit_direction = saes.eigenvectors().col(2);
                        curr_point = Eigen::Vector3d (pointOri.x, pointOri.y, pointOri.z);
                        if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]){

                            // label the same line
//                            if(labelMap){
//                                if(lineId < 0) {
//                                    lineCnt++;
//                                    lineId = lineCnt;
//                                }
//                                for (int j = 0; j < 5; j++)
//                                    localcornerMapLabeled->points[pointSearchInd[j]].intensity = lineId;
//                            }

                            Eigen::Vector3d point_on_line = center;
                            Eigen::Vector3d point_a, point_b;
                            point_a = 0.1 * unit_direction + point_on_line;
                            point_b = -0.1 * unit_direction + point_on_line;

                            ceres::ResidualBlockId id;
//                        double weight = 1.0 - saes.eigenvalues()[1] / (saes.eigenvalues()[2]);
                            double weight = 1.0 ;
                            if(splineT == 0){

                                ceres::CostFunction *cost_function = CeresFactorsSP::LidarEdgeFactorSP::Create
                                        (curr_point, point_a, point_b, pointOri.intensity, splineT);
                                id = problem->AddResidualBlock(cost_function, loss_function,
                                                               contrlposesVec[0],
                                                               contrlposesVec[1],
                                                               contrlposesVec[2],
                                                               contrlposesVec[3],
                                                               contrlposesVec[4],
                                                               contrlposesVec[0]+4,
                                                               contrlposesVec[1]+4,
                                                               contrlposesVec[2]+4,
                                                               contrlposesVec[3]+4,
                                                               contrlposesVec[4]+4);
                            }else if(splineT == 2){

                                ceres::CostFunction *cost_function = new SophusSpline::CeresFactorsSP::LidarEdgeFactorSP
                                        (curr_point, point_a, point_b, pointOri.intensity, splineDura, splineT);
                                id = problem->AddResidualBlock(cost_function, loss_function,
                                                               contrlposesVec[0],
                                                               contrlposesVec[1],
                                                               contrlposesVec[2],
                                                               contrlposesVec[3],
                                                               contrlposesVec[4]);
                            }
                            resiIDs.emplace_back(id);
                            corner_num++;
                            laserCloudOri->points.emplace_back(pointSel);
                        }
                    }

                }
            }

            Eigen::Vector3d norm, centro, curr_point;
            Eigen::MatrixXd matA0, matB0;
            int surf_num = 0;
            for (int i = 0; i < laserCloudSurfTotalLastDSNum; i++) {

                pointOri = laserCloudSurfTotalLastDS->points[i];
                if (pointOri.intensity > maxRatio) continue;  // fixme : ratio larger than 1.0 ?

                double range = sqrt(pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z);
                pointAssociateToMap(&pointOri, &pointSel);
//                kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

                bool added = false;
                if(surfelInited){

                    pcl::PointSurfel tmpSurf;
//                    tmpSurf.x = pointSel.x;
//                    tmpSurf.y = pointSel.y;
//                    tmpSurf.z = pointSel.z;
//                    kdtreeSurfelsFromMap->nearestKSearch(tmpSurf, 1, pointSearchInd, pointSearchSqDis);
//                    tmpSurf = localsurfels->points[pointSearchInd.back()];

                    double kdpt[3];
                    kdpt[0] = pointSel.x;
                    kdpt[1] = pointSel.y;
                    kdpt[2] = pointSel.z;
                    size_t ind; double dist;
                    nanoflann::KNNResultSet<double> resultSet(1);
                    resultSet.init(&ind, &dist);
                    dynamicKDtreeSurfel->findNeighbors(resultSet, kdpt,
                                                       nanoflann::SearchParams(10));
                    tmpSurf = localsurfels->points[ind];

                    norm = Eigen::Vector3d (tmpSurf.normal_x, tmpSurf.normal_y, tmpSurf.normal_z);
                    centro = Eigen::Vector3d (tmpSurf.x, tmpSurf.y, tmpSurf.z);
                    double d = norm.norm();
                    norm.normalize();

                    if(pointSearchSqDis.back() < radius &&
                       fabs(norm.cast<float>().dot(pointSel.getVector3fMap()) + d) < 0.15){

                        curr_point = Eigen::Vector3d (pointOri.x, pointOri.y, pointOri.z);
                        double weight = 1.0;
//                        ceres::CostFunction *cost_function = CeresFactorsSP::LidarSurfelFactorSP::Create
//                                (curr_point, centro, surfelCovs[pointSearchInd.back()],
//                                 tmpSurf.confidence, pointOri.intensity, splineT);
//                        auto id = problem->AddResidualBlock(cost_function, loss_function,
//                                                            contrlposesVec[0],
//                                                            contrlposesVec[1],
//                                                            contrlposesVec[2],
//                                                            contrlposesVec[3],
//                                                            contrlposesVec[4],
//                                                            contrlposesVec[0] + 4,
//                                                            contrlposesVec[1] + 4,
//                                                            contrlposesVec[2] + 4,
//                                                            contrlposesVec[3] + 4,
//                                                            contrlposesVec[4] + 4);
//                        resiIDs.emplace_back(id);
                        surf_num++;
                        added = true;
                        laserCloudOri->points.emplace_back(pointSel);
                    }
                }
                if(added)
                    continue;
                kdtreeSurfFromMap->radiusSearch(pointSel, radius, pointSearchInd, pointSearchSqDis);

                int neibor = pointSearchInd.size();
                if(neibor < 5) continue;

                matA0.resize(neibor, 3);
                matB0 = -1 * Eigen::MatrixXd::Ones(neibor, 1);
                if (pointSearchSqDis.back() < 1.0) {
                    int planeId = -1;
                    for (int j = 0; j < neibor; j++) {
                        matA0(j, 0) = localSurfMap->points[pointSearchInd[j]].x;
                        matA0(j, 1) = localSurfMap->points[pointSearchInd[j]].y;
                        matA0(j, 2) = localSurfMap->points[pointSearchInd[j]].z;
                        // printf(" pts %f %f %f ", matA0(j, 0), matA0(j, 1), matA0(j, 2));

                        if(labelMap && localSurfMapLabeled->points[pointSearchInd[j]].intensity > 0)
                            planeId = localSurfMapLabeled->points[pointSearchInd[j]].intensity;
                    }
                    // find the norm of plane
                    norm = matA0.colPivHouseholderQr().solve(matB0);
                    double negative_OA_dot_norm = 1 / norm.norm();
                    norm.normalize();

                    // Here n(pa, pb, pc) is unit norm of plane
                    bool planeValid = true;
                    double avgdist = 0.0;
                    for (int j = 0; j < neibor; j++) {
                        double dist = fabs(norm(0) * localSurfMap->points[pointSearchInd[j]].x +
                                           norm(1) * localSurfMap->points[pointSearchInd[j]].y +
                                           norm(2) * localSurfMap->points[pointSearchInd[j]].z +
                                           negative_OA_dot_norm);
                        // if OX * n > 0.2, then plane is not fit well
                        if (dist > 0.2) {
                            planeValid = false;
                            break;
                        }
                        avgdist += dist;
                    }
                    curr_point = Eigen::Vector3d (pointOri.x, pointOri.y, pointOri.z);
                    if (planeValid) {

                        // label the same plane
                        if(labelMap){
                            if(planeId < 0) {
                                planeCnt++;
                                planeId = planeCnt;
                            }
                            for (int j = 0; j < 5; j++)
                                localSurfMapLabeled->points[pointSearchInd[j]].intensity = planeId;
                        }

//                        double weight = 1.0 - avgdist / 5.0 ;
                        double weight = 1.0;
                        ceres::ResidualBlockId id;
                        if(splineT == 0){

                            ceres::CostFunction *cost_function = CeresFactorsSP::LidarPlaneNormFactorSP::Create
                                    (curr_point, norm, negative_OA_dot_norm, pointOri.intensity, splineT);
                            id = problem->AddResidualBlock(cost_function, loss_function,
                                                           contrlposesVec[0],
                                                           contrlposesVec[1],
                                                           contrlposesVec[2],
                                                           contrlposesVec[3],
                                                           contrlposesVec[4],
                                                           contrlposesVec[0]+4,
                                                           contrlposesVec[1]+4,
                                                           contrlposesVec[2]+4,
                                                           contrlposesVec[3]+4,
                                                           contrlposesVec[4]+4);
                        }else if(splineT == 2){

                            ceres::CostFunction *cost_function = new SophusSpline::CeresFactorsSP::LidarPlaneNormFactorSP
                                    (curr_point, norm, negative_OA_dot_norm, pointOri.intensity, splineDura, splineT);
                            id = problem->AddResidualBlock(cost_function, loss_function,
                                                           contrlposesVec[0],
                                                           contrlposesVec[1],
                                                           contrlposesVec[2],
                                                           contrlposesVec[3],
                                                           contrlposesVec[4]);
                        }
                        resiIDs.emplace_back(id);
                        surf_num++;
                        laserCloudOri->points.emplace_back(pointSel);
                    }
                }
            }
            radius /= annealfactor;

//            printf("Problem BUILT. \n");
            cout << "[ CERES ] Founded CORNER COSTS -- " << corner_num ;
            cout << "  ///  SURF COSTS -- " << surf_num << endl;

            // vis
            publishvalidfeaturepoints();

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT; // L-M is default?
            options.max_num_iterations = 15;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;
            options.num_threads = 8;
            ceres::Solver::Summary summary;

            ceres::Solve(options, problem.get(), &summary);
            cout << summary.BriefReport() << endl;
//            LOG(INFO) << "Iter - " << iterCount << endl;
//            LOG(INFO) << summary.FullReport() << endl;
            printf("--- solver DONE \n");

//            for(auto id : resiIDs)
//                problem->RemoveResidualBlock(id);
            resiIDs.clear();
            laserCloudOri->clear();
        }

    }

    //run()-4 ↓
    void scan2MapOptimization(){

        // 地图中特征点个数限制
        if (laserCloudCornerFromMapDSNum > 30 && laserCloudSurfFromMapDSNum > 300){

            // 迭代求解相对位姿
            ceressolver(optiIterations);
            //cout << "[ Ceres ] Poses optimized. " << endl;
        }
        // get middle pose as anchor
        double ratio = (contrlposesVec[2][7] - contrlposesVec.front()[7]) /
                       (contrlposesVec.back()[7] - contrlposesVec.front()[7]);
        if (splineT == 2)
            SophusSpline::CeresFactorsSP::splineFusionPoses(contrlposesVec[0], contrlposesVec[1], contrlposesVec[2],
                                                            contrlposesVec[3], contrlposesVec[4], anchorPose, ratio, splineT);
        else
            CeresFactorsSP::splineFusionPoses(contrlposesVec[0], contrlposesVec[1], contrlposesVec[2],
                                              contrlposesVec[3], contrlposesVec[4], anchorPose, ratio, splineT);

//        updateGlobalPose();
    }

    bool initSurfels(const PointType &tmp){

//        cout << BOLDWHITE << "[ Debug ] Initializing surfel ... " ;
        pcl::PointSurfel tmpsurfel;
        Eigen::Vector4d centro;
        Eigen::Matrix3d covMat;
        Eigen::Vector3d norm, eigenValues;
        Eigen::Vector3d ptVec;

        // check new surfels
        kdtreeSurfFromMap->nearestKSearch(tmp, numOfptInitSurfel, pointSearchInd, pointSearchSqDis);
        if(pointSearchInd.size() > initRadiSurfel)
            return false;

//        kdtreeSurfFromMap->radiusSearch(tmp, initRadiSurfel, pointSearchInd, pointSearchSqDis);
//        if(pointSearchInd.size() < numOfptInitSurfel)
//            return false;

        int n = pcl::computeMeanAndCovarianceMatrix(*localSurfMap, pointSearchInd, covMat, centro);
        covMat = n / (n-1.0) * covMat.eval();  // scale?
//                    cout << "\033[1:33m " << centro << "\033[0m" << endl;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
        norm = saes.eigenvectors().col(0);
        eigenValues = saes.eigenvalues();
        norm.normalize();
        double d = -1.0*norm.dot(centro.head<3>());  // plane parameter
        bool valid = true;
        for (int k = 0; k < numOfptInitSurfel; ++k) {
            ptVec(0) = localSurfMap->points[pointSearchInd[k]].getArray3fMap()(0);
            ptVec(1) = localSurfMap->points[pointSearchInd[k]].getArray3fMap()(1);
            ptVec(2) = localSurfMap->points[pointSearchInd[k]].getArray3fMap()(2);
            if(fabs(norm.dot(ptVec) + d) > 0.06){  // point to plane dist
                valid = false;
                cout << "[ Debug ] surfel is not valid. " << endl;
                break;
            }
        }
        if(valid){

            // TODO add surfel
            tmpsurfel.x = centro(0);
            tmpsurfel.y = centro(1);
            tmpsurfel.z = centro(2);
            if(d < 0){
                norm = -1.0*norm;
                d = -1.0*d;
            }
            tmpsurfel.normal_x = norm(0)* d;
            tmpsurfel.normal_y = norm(1)* d;
            tmpsurfel.normal_z = norm(2)* d;

            tmpsurfel.radius = pointSearchSqDis.back();
            tmpsurfel.curvature = tmp.intensity;  // timestamp
            tmpsurfel.confidence = 0.5f;
            surfelCovs[localsurfels->points.size()] = covMat;
            localsurfels->points.emplace_back(tmpsurfel);

//            cout << "[ SURFEL] Added Normal " << localsurfels->points.back().data_n[3]
//                 << " " << localsurfels->points.back().data_n[0]
//                 << " " << localsurfels->points.back().data_n[1]
//                 << " " << localsurfels->points.back().data_n[2] << endl;

            // add centro to kdtree
            PointCloudforKDTREE<double >::Point cent;
            cent.x = tmpsurfel.x;
            cent.y = tmpsurfel.y;
            cent.z = tmpsurfel.z;
            cloudsurfelCentros->pts.emplace_back(cent);
            dynamicKDtreeSurfel->addPoints(cloudsurfelCentros->pts.size()-1,
                                           cloudsurfelCentros->pts.size()-1);

            cout << "[ Surfel ] New Surfel. "  << RESET << endl;

            surfelInited = true;
            return true;
        }
        cout << "  FALSE  " << RESET << endl;
        return false;
    }

    //run()-5 构建gtsam因子图
    void saveKeyFramesAndFactor(){

        if(!localcornerMap->points.empty())
            mapInited = true;
        else mapInited = false;
        if(!localsurfels->points.empty())
            surfelInited = true;
        else surfelInited = false;

        badregistered = false;
        // save undistorted cloud
        pcl::PointCloud<PointType>::Ptr curCornerAdded2map(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr curSurfAdded2map(new pcl::PointCloud<PointType>());

        static double maxRangeMap = 50;
        // transform cloud to local map
        PointType tmp;
        pcl::PointSurfel tmpsurfel;
        int newptNum = 0, curAllptNum = 0;

        double st = contrlposesVec.front()[7] - systemStartTime;
//        double st = contrlposesVec[1][7] - systemStartTime;
        double totalt = contrlposesVec.back()[7] - contrlposesVec.front()[7];

        int startInd, endInd;
        if(splineT == 2){ startInd = 0; endInd = splineNum-3; }
        else{ startInd = 1; endInd = splineNum-2; }

        for (int j = startInd; j < endInd; ++j) {  // middle frames
//#pragma omp parallel for
            // add new corner cloud
            for(auto pt : curCornerCloudsTobeAdded[j].points){
                if (splineT == 2 && pt.intensity > 0.5) continue;
                pointAssociateToMap(&pt, &tmp);
                tmp.intensity = pt.intensity*totalt + st;
//                if(mapInited){
//                    kdtreeCornerFromMap->radiusSearch(tmp, localmapResolu, pointSearchInd, pointSearchSqDis);
//                    if(pointSearchInd.empty()){
//                        newptNum ++;
//                        localcornerMap->points.emplace_back(tmp);
//                    }
//                    continue;
//                }
                if(pointRange(pt) < maxRangeMap){
                    newptNum++; curAllptNum++;
                    curCornerAdded2map->points.emplace_back(tmp);
                    localcornerMap->points.emplace_back(tmp);
                }
            }
//            curAllptNum += curCornerCloudsTobeAdded[j].points.size();

            // add new surf cloud
            for(auto pt : curSurfCloudsTobeAdded[j].points){
                if (splineT == 2 && pt.intensity > 0.5) continue;
                pointAssociateToMap(&pt, &tmp);
                tmp.intensity = pt.intensity*totalt + st;
                if(pointRange(pt) < maxRangeMap){
                    curAllptNum++;
                    curSurfAdded2map->points.emplace_back(tmp);
                }

                if(surfelInited){
                    double kdpt[3];
                    kdpt[0] = tmp.x;
                    kdpt[1] = tmp.y;
                    kdpt[2] = tmp.z;
                    size_t ind; double dist;
                    nanoflann::KNNResultSet<double> resultSet(1);
                    resultSet.init(&ind, &dist);
                    dynamicKDtreeSurfel->findNeighbors(resultSet, kdpt,
                                                       nanoflann::SearchParams(10));

//                    kdtreeSurfelsFromMap->nearestKSearch(tmpsurfel, 1, pointSearchInd, pointSearchSqDis);
                    tmpsurfel = localsurfels->points[ind];
                    double pldist = tmp.x*tmpsurfel.normal_x + tmp.y*tmpsurfel.normal_y + tmp.z*tmpsurfel.normal_z;
                    if(dist < tmpsurfel.radius){
                        // TODO update surfel
                        localsurfels->points[ind].confidence += log(0.55/0.45);
                        localsurfels->points[ind].curvature = tmp.intensity;
                        continue;
                    }
                }

                if(mapInited){
                    if(!initSurfels(tmp)){
                        kdtreeSurfFromMap->radiusSearch(tmp, localmapResolu, pointSearchInd, pointSearchSqDis);
                        if(pointSearchInd.empty()){
                            if(pointRange(pt) < maxRangeMap)
                                localSurfMap->points.emplace_back(tmp);
                        }else{
                            newptNum++;
                            localSurfMap->points[pointSearchInd.front()].intensity = tmp.intensity;   // update timestamp
                        }
                    }
                } else if(pointRange(pt) < maxRangeMap){
                    newptNum++;
                    localSurfMap->points.emplace_back(tmp);
                }
            }
//            curAllptNum += curSurfCloudsTobeAdded[j].points.size();

            // add new outlier cloud
//            for(auto pt : curOutlierCloudsTobeAdded[j].points){
//
//                pointAssociateToMap(&pt, &tmp);
//                tmp.intensity = pt.intensity*totalt + st;
//                if(pointRange(pt) > maxRangeMap)
//                    continue;
//
////                if(surfelInited){
////                    double kdpt[3];
////                    kdpt[0] = tmp.x;
////                    kdpt[1] = tmp.y;
////                    kdpt[2] = tmp.z;
////                    size_t ind; double dist;
////                    nanoflann::KNNResultSet<double> resultSet(1);
////                    resultSet.init(&ind, &dist);
////                    dynamicKDtreeSurfel->findNeighbors(resultSet, kdpt,
////                                                       nanoflann::SearchParams(10));
////
//////                    kdtreeSurfelsFromMap->nearestKSearch(tmpsurfel, 1, pointSearchInd, pointSearchSqDis);
////                    tmpsurfel = localsurfels->points[ind];
////                    double pldist = tmp.x*tmpsurfel.normal_x + tmp.y*tmpsurfel.normal_y + tmp.z*tmpsurfel.normal_z;
////                    if(dist < tmpsurfel.radius){
////                        // TODO update surfel
////                        localsurfels->points[ind].confidence += log(0.55/0.45);
////                        localsurfels->points[ind].curvature = tmp.intensity;
////                        continue;
////                    }
////                }
//
//                if(mapInited){
//                    if(!initSurfels(tmp)){
//                        kdtreeSurfFromMap->radiusSearch(tmp, localmapResolu, pointSearchInd, pointSearchSqDis);
//                        if(pointSearchInd.empty()){
////                            newptNum++;
////                            if(pointRange(pt) < maxRangeMap)
//                            curSurfAdded2map->points.emplace_back(tmp);
////                            localSurfMap->points.emplace_back(tmp);
//                        }
////                        localSurfMap->points[pointSearchInd.front()].intensity = tmp.intensity;   // update timestamp
//                    }
//                } else if(pointRange(pt) < maxRangeMap)
//                    curSurfAdded2map->points.emplace_back(tmp);
////                localSurfMap->points.emplace_back(tmp);
//            }
//            curAllptNum += curOutlierCloudsTobeAdded[j].points.size();
        }
        //cout << "[ Spline ] Added current cloud to local map. ";

        // todo : too much new inserted points -> bad registration ?
        if(mapInited && newptNum*1.0 / curAllptNum > cloudoverlapRatio){
            cout << BOLDMAGENTA << "[ Map ] Bad registration. \033[0m" << endl;
            badregistered = true;
            filterMapByTime(0, st, false);  // filter out current cloud
        }
        else updateGlobalPose();

        cout << BOLDBLACK << " Added corner : " << curCornerAdded2map->points.size()
             << " / Added surf : " << curSurfAdded2map->points.size() << RESET << endl;
        cornerCloudkeyframes.emplace_back(curCornerAdded2map);
        surfCloudkeyframes.emplace_back(curSurfAdded2map);

        cloudCurInWorld->clear();
        *cloudCurInWorld += *transformPointCloud(curCornerAdded2map,
                                                 &globalposesQuat->points.back());
        *cloudCurInWorld += *transformPointCloud(curSurfAdded2map,
                                                 &globalposesQuat->points.back());

        if(!mapInited) pcl::io::savePCDFileBinaryCompressed(resultFolder + "initMap.pcd", *cloudCurInWorld);

        // todo : unfinished
        currentRobotPosPoint.x = globalposesQuat->points.back().x;
        currentRobotPosPoint.y = globalposesQuat->points.back().y;
        currentRobotPosPoint.z = globalposesQuat->points.back().z;

        // 当前位姿点与上一位姿点的欧氏距离>0.3
        saveThisKeyFrame = true;
        if ( sqrt((previousRobotPosPoint.x-currentRobotPosPoint.x)*(previousRobotPosPoint.x-currentRobotPosPoint.x)+
                  (previousRobotPosPoint.y-currentRobotPosPoint.y)*(previousRobotPosPoint.y-currentRobotPosPoint.y)+
                  (previousRobotPosPoint.z-currentRobotPosPoint.z)*(previousRobotPosPoint.z-currentRobotPosPoint.z))
             < 0.3 ){
            saveThisKeyFrame = false;
        }

        if (!saveThisKeyFrame && !cloudKeyPoses3D->points.empty())
            return;
        previousRobotPosPoint = currentRobotPosPoint;

    }

    void clearCloud(){

    }

    // to make sure the front splineNum frames' timestamp is synchronized
    bool syncData(){

        if(odomposeBufVec.size() < splineNum || featureCloudinfoQue.size() < splineNum)
            return false;
        std::lock_guard<mutex> lockGuard(mtx);

        int num = 0;
        while(!odomposeBufVec.empty() && !featureCloudinfoQue.empty()){

            if(num == odomposeBufVec.size() || num == featureCloudinfoQue.size())
                break;
            timeLaserOdometry = odomposeBufVec[num][7];
            timeCloudinfo = featureCloudinfoQue[num]->header.stamp.toSec();

            if(timeLaserOdometry < timeCloudinfo-0.05){

                delete odomposeBufVec.front();
                odomposeBufVec.pop_front();
                cout << RED << "odomInfo is old"  << RESET << endl;
                continue;
            }else if(timeCloudinfo < timeLaserOdometry-0.05){

                featureCloudinfoQue.pop_front();
                cout << RED << "cloudInfo is old"  << RESET << endl;
                continue;
            }
            num++;
            if(num == splineNum)
                break;
        }
        return (num == splineNum);
    }

    void setInitialControlposes(){

        curSurfCloudsTobeAdded.clear();
        curCornerCloudsTobeAdded.clear();
        curOutlierCloudsTobeAdded.clear();
        pcl::PointCloud<PointType> tmpCloud;

        std::lock_guard<mutex> lockGuard(mtx);

        // withdraw front frames
        for (int m = 0; m < splineNum; ++m){

            featureCloudinfo = *featureCloudinfoQue[m];
//            pcl::fromROSMsg(featureCloudinfo.cloud_surface, tmpCloud);  // todo: compare
            pcl::fromROSMsg(featureCloudinfo.cloud_lines, tmpCloud);
            curSurfCloudsTobeAdded.emplace_back( tmpCloud);
            pcl::fromROSMsg(featureCloudinfo.cloud_corner, tmpCloud);
            curCornerCloudsTobeAdded.emplace_back( tmpCloud);
            pcl::fromROSMsg(featureCloudinfo.segmentedCloud, tmpCloud);
//            pcl::fromROSMsg(featureCloudinfo.cloud_outlier, tmpCloud);
            curOutlierCloudsTobeAdded.emplace_back( tmpCloud);

            getRelativeTransform(odomposeBufVec[m], odomposeBufVec[0], contrlposesVec[m]);
        }

        // sliding the time window by flush frames
        for (int j = splineNum; j > slidingFrameNum; --j){

            delete odomposeBufVec.front();
            odomposeBufVec.front() = NULL;
            odomposeBufVec.pop_front();

            featureCloudinfoQue.pop_front();
        }
    }

    void deriveCurCloud(){

        curCornercloudall->clear();
        curSurfcloudall->clear();

        // add points to current cloud
        // every points' intensity is the ratio in current spline trajectory
        double totaltime = contrlposesVec.back()[7] - contrlposesVec.front()[7];
        int cloudsize, cloudsizeAll, startInd, endInd;

        if(splineT == 2){ startInd = 0; endInd = splineNum-3; }
        else{ startInd = 1; endInd = splineNum-2; }
//        double maxRatio = (contrlposesVec[endInd][7] - contrlposesVec.front()[7]) / totaltime;

        // todo : front half frames
        for (int m = startInd; m < endInd; ++m) {

            double curFrameRatio = (contrlposesVec[m][7] - contrlposesVec.front()[7]) / totaltime;  // ratio

            cloudsizeAll = curSurfcloudall->points.size();
            cloudsize = curSurfCloudsTobeAdded[m].points.size();
            curSurfcloudall->points.resize(cloudsizeAll + cloudsize);
#pragma omp parallel for
            for (int i = 0; i < cloudsize; i++) {
                PointType tmp;
                tmp = curSurfCloudsTobeAdded[m].points[i];
                tmp.intensity = curFrameRatio + (tmp.intensity - (int)tmp.intensity) / totaltime;
                curSurfCloudsTobeAdded[m].points[i] = tmp;
                curSurfcloudall->points[cloudsizeAll + i] = tmp;
            }

            cloudsize = curOutlierCloudsTobeAdded[m].points.size();
#pragma omp parallel for
            for (int i = 0; i < cloudsize; i++) {
                PointType tmp;
                tmp = curOutlierCloudsTobeAdded[m].points[i];
                tmp.intensity = curFrameRatio + (tmp.intensity - (int)tmp.intensity) / totaltime;
                curOutlierCloudsTobeAdded[m].points[i] = tmp;
//                curSurfcloudall->points[cloudsizeAll + i] = tmp;
            }

            cloudsizeAll = curCornercloudall->points.size();
            cloudsize = curCornerCloudsTobeAdded[m].points.size();
            curCornercloudall->points.resize(cloudsizeAll + cloudsize);
#pragma omp parallel for
            for (int i = 0; i < cloudsize; i++) {
                PointType tmp;
                tmp = curCornerCloudsTobeAdded[m].points[i];
                tmp.intensity = curFrameRatio + (tmp.intensity - (int)tmp.intensity) / totaltime;

                curCornerCloudsTobeAdded[m].points[i] = tmp;
                curCornercloudall->points[cloudsizeAll + i] = tmp;
            }
        }
        //cout << "\033[1:36m Current cloud derived. \033[0m" << endl;
    }

    /// MAIN PROCESS
    void run(){

        ros::Rate loop_rate(5);
        ROS_INFO("\033[1;32m---->\033[0m Map Optimization [Spline] Started.");
        ofstream ofs(resultFolder + "splineTimes.txt");
        while (ros::ok()) {

            loop_rate.sleep();
            // 新点云以及时间戳判断
            if (!syncData())
                continue;
            setInitialControlposes();

            timeLaserOdometry = contrlposesVec[splineNum/2][7];  // TODO first value is wrong？
            splineDura = contrlposesVec.back()[7] - contrlposesVec.front()[7];
            cout << "[ Time ] Current Odom :" << to_string(timeLaserOdometry)
                 << ", with Spline Duration " << splineDura << endl;

            st = clock();
            ROS_INFO("mapOptimization starting after %lf s", (timeLaserOdometry - timeLastProcessing)); // ~0.4s

            timeLastProcessing = timeLaserOdometry;
            timer.tic();

//            double dist = sqrt(contrlposesVec[4][4] * contrlposesVec[4][4] +
//                               contrlposesVec[4][5] * contrlposesVec[4][5] +
//                               contrlposesVec[4][6] * contrlposesVec[4][6]);
//
//            Eigen::AngleAxisd rotVec(Eigen::Quaterniond(contrlposesVec[4][3], contrlposesVec[4][0],
//                                                        contrlposesVec[4][1], contrlposesVec[4][2]));
//            if (dist < 0.15 && rotVec.angle() < M_PI / 6) {
//
//                cout << RED << "[ Spline ] Short Spline ! " << dist << " m!" << RESET << endl;
////                clearCloud();
//                continue;
//            }
            deriveCurCloud();

            outputCloudbeforeSpOpti(0);

            if(splineT == 2){
                maxRatio = 0.5;
                SophusSpline::CeresFactorsSP::fromDataToControlpointsDynamic(contrlposesVec);
            } else if(splineT == 0)
                CeresFactorsSP::fromDataToControlpointsDynamic(contrlposesVec);
            outputCloudAftSpOpti(0);
//                cout << BLUE << " >>> The control points : \n";
//                printControlPose();
//                cout << " >>> The intelopated points : \n";
//                printPoseInSpline();
//                cout << RESET;
//                continue;

            prepareLocalMap();
            cout << RED << "[ Map ] Prepare Map Time : " << timer.toc() << " ms" << RESET << endl;
            ofs << timer.toc() << " ";

            downsampleCurrentScan();

            scan2MapOptimization();
            cout << RED << "[ Ceres ] Solver Time : " << timer.toc() << " ms" << RESET << endl;
            ofs << timer.toc() << " " ;

            saveKeyFramesAndFactor();
            cout << RED << "[ Map ] Update Map Time : " << timer.toc() << " ms" << RESET << endl;
            ofs << timer.toc() << " " ;

//            outputCloudAftSpOpti();

            publishTF();

            publishKeyPosesAndFrames();

            clearCloud();

            ROS_INFO("keyposes No. %ld \n", globalposesQuat->points.size());

            et = clock();
            ut = double(et - st) / CLOCKS_PER_SEC * 1000;
            ROS_INFO("Time used is :%f ms for Spline mapOptimization. \n", ut);
            cout << RED << "[ Total ] Time : " << timer.toc() << " ms" << RESET << endl;
            ofs << timer.toc() << endl;

            cout << "\033[3:33m =============================================================== \033[0m" << endl;
        }
        ofs.close();
        cout << "\033[1;32m ----> Map Optimization [Spline] thread closed.\033[0m" << endl;
    }

    // save per keyframe poses and cloud
    void saveposesAndframesThread(){

        ros::Rate rate(0.01);
        while(ros::ok()){
            rate.sleep();
            saveposesAndframes();
        }
    }
    void saveposesAndframes(bool reinited = false){

        cout << BLUE << "[ Mapping ] Saving map thread started ... " << RESET << endl;
        std::lock_guard<mutex> lockGuard(mtx);

        FILE *fp; string file;
        if (reinited) file = resultFolder + "keyposes6dspline" + to_string(reInitedTimes) + ".txt";
        else file = resultFolder + "keyposes6dspline.txt";
        fp = fopen(file.data(),"w");
        for(auto pose : globalposesQuat->points){

            fprintf(fp, "%lf %f %f %f %f %f %f %f\n", pose.time,
                    pose.x, pose.y, pose.z,
                    pose.roll, pose.pitch, pose.yaw,
                    pose.intensity);
        }
        fclose(fp);

        int frameN = globalposesQuat->points.size();
        cout << BOLDRED << "[ mapping ] Saving global map with frames " << frameN;

        pcl::PointCloud<PointType>::Ptr globalMap(new pcl::PointCloud<PointType>());
        for (int i = 0; i < frameN-1; ++i) {

            *globalMap += *transformPointCloud(cornerCloudkeyframes[i],
                                               &globalposesQuat->points[i]);
            *globalMap += *transformPointCloud(surfCloudkeyframes[i],
                                               &globalposesQuat->points[i]);
//            cout << "Frame" << i << " / " << globalMap->points.size();
        }
        if(!globalMap->points.empty()){
            if(reinited) pcl::io::savePCDFileBinary(resultFolder + "globalMap" + to_string(reInitedTimes) + ".pcd", *globalMap);
            else pcl::io::savePCDFileBinary(resultFolder + "globalMap.pcd", *globalMap);
            cout << " ... SUCCEED !" << RESET << endl;
        }

    }

};


void mySigintHandler(int sig);
//
void testBsplineFunction(){

    // test control points
    Eigen::Vector3d p1(0,0,0);
    Eigen::Vector3d p2(10,10,0);
    Eigen::Vector3d p3(30,-10,0);
    Eigen::Vector3d p4(50,20,0);
    Eigen::Vector3d p5(70,0,0);
    Eigen::Vector3d p;

    pcl::PointCloud<PointType> testline;
    PointType pt;
    double* coeff;
    for (float i = 0; i < 1; ) {

        CeresFactorsSP::splineBasicCoeffCalcu(i, 4, 3, coeff);
        p = coeff[0]*p1 + coeff[1]*p2 + coeff[2]*p3 + coeff[3]*p4 + coeff[4]*p5;
        pt.x = p(0);
        pt.y = p(1);
        testline.points.push_back(pt);
        i+= 0.01;
        delete coeff;
    }
    if(!testline.empty())
        pcl::io::savePCDFileBinaryCompressed("/home/cyz/workspace/testData/splineCloud.pcd", testline);
}
//
void testBsplineFunctionSophus(){

    const double eps = 1e-6;

    basalt::Se3Spline<4> se3Spline = basalt::Se3Spline<4>(100);
//    se3Spline.applyInc();

    se3Spline.genRandomTrajectory(5);
    int64_t maxT = se3Spline.maxTimeNs();
    cout << "[ DEBUG ] maxTimeNs : " << maxT << endl;
    cout << GREEN << setprecision(7) ;
    se3Spline.print_knots();
    cout << RESET;
    double* params[5];
//    double* trans[5];
    cout << " Copying Parameters ..." << endl;
    for (int i = 0; i < 5; ++i){
        params[i] = new double[7];
        memcpy(params[i], se3Spline.getKnot(i).data(), 7* sizeof(double ));
//        params[i] = se3Spline.getKnot(i).data();
        for (int j = 0; j < 7; ++j) cout << params[i][j] << " " ;
        cout << endl;
    }
    for (int i = 0; i < 5; ++i) {
        Eigen::Map<Sophus::SE3<double> > T0(params[i]);
        se3Spline.knots_push_back(T0);
    }
    cout << GREEN;
    se3Spline.print_knots();
    cout << RESET;
    for (int i = 0; i < 5; ++i) se3Spline.knots_pop_back();

    cout << "==================== Evaluating Sophus Spline Knot Jacobians ================================" << endl;
    Sophus::SE3<double> T_ori, T_aft;
    basalt::Se3Spline<4>::PosePosSO3JacobianStruct JacobsSpline;
    T_ori = se3Spline.pose(maxT * 0.3);
    cout << YELLOW << "[ JacobTest ] Pose.log() before : " << T_ori.log().transpose() << RESET << endl;

    for (int i = 0; i < 5; ++i) {

//        Sophus::SO3<double> rot = se3Spline.getKnotSO3(i);
//        Eigen::Vector3d rot_log = rot.log();
        Eigen::Map<Eigen::Quaternion<double> > quat (se3Spline.getKnotSO3(i).data());
        for (int j = 0; j < 7; ++j) {

            cout << "[ JacobTest ] Perturbation in  " << CYAN << i << ", " << j << RESET;
            cout << " Original Value " << params[i][j];
            params[i][j] += eps;
            cout << " with new value " << params[i][j] << endl;

//            cout << " Original rot Value " << rot_log.transpose();
//            rot_log(j) += eps;
//            cout << " with new rot value " << rot_log.transpose() << endl;
//            rot = Sophus::SO3<double>::exp(rot_log);
//            se3Spline.getKnotSO3(i) = rot;
            if (j>3)
                se3Spline.getKnotPos(i) = Eigen::Map<Eigen::Vector3d>(params[i] + 4);
            else
                quat = Eigen::Map<Eigen::Quaternion<double> >(params[i]);

            cout << GREEN;
            se3Spline.print_knots();
            cout << RESET;

            T_aft = se3Spline.pose(maxT * 0.3, &JacobsSpline);
            cout << "[ JacobTest ] Pose.log() after : " << T_aft.log().transpose() << endl;

            cout << RED << "[ JacobTest ] Numeric : " << (T_aft.log() - T_ori.log()).transpose() / eps
                 << " ，Analytic : \n" << JacobsSpline.d_val_d_knot[i].matrix() << RESET << endl;
            params[i][j] -= eps;
//            rot_log(j) -= eps;
            if(j == 3)
                quat = Eigen::Map<Eigen::Quaternion<double> >(params[i]);
        }
//        rot = Sophus::SO3<double>::exp(rot_log);
//        se3Spline.getKnotSO3(i) = rot;
        // recover
        se3Spline.getKnotPos(i) = Eigen::Map<Eigen::Vector3d>(params[i] + 4);
    }

    Eigen::Matrix<double, 3, 1> ptV(42, 50, 61), norm(3.5, 6.35, 9.5);
    norm.normalize();
    cout << "==================== Evaluating Sophus Cost Jacobians ================================" << endl;
    ceres::CostFunction* costFuncSophus = new SophusSpline::CeresFactorsSP::LidarPlaneNormFactorSP
            (ptV, norm, 1, 0.23, 4e-7);

    double* resOri = new double [1];
    double* resAft = new double [1];
    costFuncSophus->Evaluate(params, resOri, nullptr);
    cout << YELLOW << "[ JacobTest ] Residual before : " << resOri[0] << RESET << endl;
    double* Jacobs[5];
    double delta;
    double* se3_delta = new double[6]{0};
    for (int i = 0; i < 5; ++i) Jacobs[i] = new double[7];

    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 7; ++j) {

            cout << "[ JacobTest ] Perturbation in  knot_" << CYAN << i << ", " << j << RESET;
            cout << " Original Value " << params[i][j];
            params[i][j] += eps;
            cout << " with new value " << params[i][j] << endl;

            costFuncSophus->Evaluate(params, resAft, Jacobs);
            cout << "[ JacobTest ] Residual after : " << resAft[0] << endl;

            // multiply with self derivative?
//            Eigen::Map<Sophus::SE3<double> > T(params[i]);
//            Eigen::Map<Eigen::Matrix<double, 1, 7> > e_J(Jacobs[i]);
//            Eigen::Matrix<double, 7, 6>  T_J = T.Dx_this_mul_exp_x_at_0();
//            e_J.block<1, 6>(0,0) = e_J * T_J;

            delta = (resAft[0] - resOri[0]) / eps;
            if(j>3)
                cout << RED << "[ JacobTest ] Numeric : " << delta << "，Analytic : " << Jacobs[i][j-1] << RESET << endl;
            else
                cout << RED << "[ JacobTest ] Numeric : " << delta << "，Analytic : " << Jacobs[i][j] << RESET << endl;

            params[i][j] -= eps;
        }
    }

    cout << "=================== Evaluating Analytic Cost Jacobians ================================" << endl;
    ceres::CostFunction* costFunc = new SurfNormAnalyticCostFunction(ptV, norm, 1);
    double *paramsN[1];
    paramsN[0] = new double[7];
    for (int j = 0; j < 7; ++j) paramsN[0][j] = params[0][j];
    costFunc->Evaluate(paramsN, resOri, nullptr);
    cout << YELLOW << "[ JacobTest ] Residual before : " << resOri[0] << RESET << endl;
    for (int j = 0; j < 7; ++j) {
        cout << "[ JacobTest ] Perturbation in  " << j;
        cout << " Original Value " << paramsN[0][j];
        paramsN[0][j] += eps;
        cout << " with new value " << paramsN[0][j] << endl;

        costFunc->Evaluate(paramsN, resAft, Jacobs);
        cout << "[ JacobTest ] Residual after : " << resAft[0] << endl;

        delta = (resAft[0] - resOri[0]) / eps;
        if(j>3)
            cout << RED << "[ JacobTest ] Numeric : " << delta << "，Analytic : " << Jacobs[0][j-1] << RESET << endl;
        else
            cout << RED << "[ JacobTest ] Numeric : " << delta << "，Analytic : " << Jacobs[0][j] << RESET << endl;

        paramsN[0][j] -= eps;
    }


    pcl::PointCloud<PointType> testline;
    PointType pt;
    Sophus::SE3<double> pose;
    for (float i = 0; i < 1; ) {

        pose = se3Spline.pose(maxT*i);
        pt.x = pose.translation().x();
        pt.y = pose.translation().y();
        pt.z = pose.translation().z();
        testline.points.push_back(pt);
        i += 0.01;
    }
    if(!testline.empty())
        pcl::io::savePCDFileBinaryCompressed("/home/cyz/workspace/testData/splineCloud.pcd", testline);
}

void testSplineJacob(){


}

int main(int argc, char** argv){

    ros::init(argc, argv, "mapOptimization");

    google::InitGoogleLogging(argv[0]);

    mapOptimizationSpline MO;
    signal(SIGINT, mySigintHandler);

//    testBsplineFunctionSophus();

    std::thread saveposesAndframes(&mapOptimizationSpline::saveposesAndframesThread, &MO);
    std::thread processor(&mapOptimizationSpline::run, &MO);

    ros::spin();

    saveposesAndframes.join();
    processor.join();

    cout << BLUE << "[ ROS ] mapOptimization node is done." << RESET << endl;
    return 0;
}

// do sth before it shutdown
void mySigintHandler(int sig){

    ROS_INFO("shutting down!");
    ros::shutdown();
}