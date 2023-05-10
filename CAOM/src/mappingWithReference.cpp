/////////////////////////////////////////////////////////////////////////////////////////////////////
/// Created by cyz on 2022/2/17.
///
/// 20220223：
/// slam with reference MLS map, the start of which is near to each other;
/// register clouds with same FoV through spherical projection of range image (SRI);
///
/// 20220326:
/// add key frame based dense graph optimization: global map is not better !?
///
//////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ceresAnalyticCost.h"
#include "ceresSplineCost.h"
#include "ceresSplineCostSophus.h"
#include "tools.h"
#include "csignal"
#include "structural_mapping/localmapWithPoseMsg.h"
#include "gtsamHelper.hpp"
#include "imageProjectionManager.h"
#include "ikd_Tree.h"

#include <fast_gicp/gicp/fast_gicp.hpp>

#define KF_DIST 4
#define ScoreThre 0.15

class MappingWithReference{

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
//    KD_TREE<PointType>::Ptr increkdtree_ptr;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterOutlier;

    double timeCloudinfo;
    double timeLaserOdometry;

    bool localizeInRef = false;
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
    bool mapIsRefered;  // whether graph is anchored with prior factor

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

    pcl::PointCloud<PointType>::Ptr localcornerMapLabeled;
    pcl::PointCloud<PointType>::Ptr localSurfMapLabeled;

    pcl::PointCloud<PointTypePose>::Ptr globalposesQuat;
    pcl::PointCloud<PointTypePose>::Ptr globalkeyposesQuat;
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr hdmapAll;

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
    Eigen::Matrix4d lastIncreT;
    bool visualize = true;
    double localmapResolu;  // the local map resolution
    double searchRadi;
    double annealfactor;

    double systemStartTime, splineDura;
    std::vector<int > keyFrameInds;
    std::vector<int > savedmapFrameInds;  // when to filter localmap (in anchor frame)
    std::vector<int > badregisFrameInds;  // frame that didnt align well
    double localmapDura;  // seconds of localmap duration
    pcl::PassThrough<PointType>::Ptr passthroughFilter;

    bool mapInited = false, surfelInited = false;
    double cloudoverlapRatio, timerWindowRatio;

    int lineCnt = 0, planeCnt = 0, reInitedTimes;
    bool initializeSpline = true, useSurfels, labelMap, saveMapForKF = false;
    int optiIterations, minLocalMapSize, maxLocalmapSize;
    PointTypePose lastglobalPoseQuat;

    int randN;  // random frame num for debug
    string resultFolder;
    double maxRatio = 1.0, maxRangeMap;
    string ref_file;
    GtsamHelper gtsamHelper;
    int curFrameID = -1;
    std::vector<std::pair<Eigen::Matrix4d, float> > priorTs_withS;

public:

    MappingWithReference():
            nh("~"){

        nh.param<bool>  ("/mapOptiSpline/useSurfels", useSurfels, true);
        nh.param<bool>  ("/mapOptiSpline/labelMap", labelMap, false);
        nh.param<double>("/mapOptiSpline/cloudoverlapRatio", cloudoverlapRatio, 0.6f);
        nh.param<double>("/mapOptiSpline/maxRangeMap", maxRangeMap, 60);
        nh.param<double>("/mapOptiSpline/annealfactor", annealfactor, 1.2f);
        nh.param<double>("/mapOptiSpline/searchRadi", searchRadi, 0.4f);
        nh.param<double>("/mapOptiSpline/localmapResolu", localmapResolu, 0.1f);
        nh.param<double>("/mapOptiSpline/localmapDura", localmapDura, 20.f);
        nh.param<double>("/mapOptiSpline/timerWindowRatio", timerWindowRatio, 1.5f);
        nh.param<float> ("/mapOptiSpline/ds_leaf_mapping", ds_leaf_mapping, 0.3f);
        nh.param<int>   ("/mapOptiSpline/optiIterations", optiIterations, 6);
        nh.param<int>   ("/mapOptiSpline/minLocalMapSize", minLocalMapSize, 30000);
        nh.param<int>   ("/mapOptiSpline/maxLocalmapSize", maxLocalmapSize, 170000);
        nh.param<int>   ("/mapOptiSpline/splineT", splineT, 0);
        nh.param<int>   ("/mapOptiSpline/splineNum", splineNum, 5);
        nh.param<int>   ("/mapOptiSpline/slidingFrameNum", slidingFrameNum, 3);
        nh.param<string>("ref_file", ref_file, " ");
        nh.param("filterObjects", enablefilterObj, false);
        nh.getParam("/projpath", projPath);  // global param add '/' before name

        resultFolder = projPath + "splineFusionRes_";
        boost::filesystem::create_directory(boost::filesystem::path(resultFolder));
        resultFolder += "/";
        if(splineT == 2) slidingFrameNum = splineNum - (splineNum-1)/2;

        cout << "[ MapOpti ] ref_file: " << ref_file        << endl;
        cout << "[ MapOpti ] useSurfels        " << useSurfels        << endl;
        cout << "[ MapOpti ] labelMap          " << labelMap          << endl;
        cout << "[ MapOpti ] cloudoverlapRatio " << cloudoverlapRatio << endl;
        cout << "[ MapOpti ] maxRangeMap       " << maxRangeMap       << endl;
        cout << "[ MapOpti ] annealfactor      " << annealfactor      << endl;
        cout << "[ MapOpti ] searchRadi        " << searchRadi        << endl;
        cout << "[ MapOpti ] localmapResolu    " << localmapResolu    << endl;
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
        cout << "[ MapOpti ] resultFolder : "  << resultFolder << endl;
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
                                                                    &MappingWithReference::laserCloudInfoHandler, this);
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/odom_pose", 1,
                                                            &MappingWithReference::laserOdometryHandler, this);

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

        if (pcl::io::loadPCDFile(ref_file, *hdmapAll) == -1)
//        if (pcl::io::loadPCDFile("/home/cyz/workspace/testData/playground/mappingWithMLS/mls_roadaroundPlayground.pcd", *hdmapAll) == -1)
//        if (pcl::io::loadPCDFile("/home/cyz/workspace/testData/playground/mappingWithMLS/mls_2thCanteenCBD_DS.pcd", *hdmapAll) == -1)
            cout << BOLDYELLOW << " NO HD MAP !" << RESET << endl;
        else
            cout << BOLDGREEN << "[MLS] HD MAP: " << hdmapAll->points.size() << RESET << endl;

    }

    void allocateMemory(){

//        increkdtree_ptr.reset(new KD_TREE<PointType>(0.3, 0.6, ds_leaf_mapping));

        downSizeFilterCorner.setLeafSize(ds_leaf_mapping, ds_leaf_mapping, ds_leaf_mapping/2);
        downSizeFilterSurf.setLeafSize(ds_leaf_mapping, ds_leaf_mapping, ds_leaf_mapping);
        downSizeFilterOutlier.setLeafSize(ds_leaf_mapping, ds_leaf_mapping, ds_leaf_mapping);

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
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        globalposesQuat.reset(new pcl::PointCloud<PointTypePose>());
        globalkeyposesQuat.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfTotalLastDS.reset(new pcl::PointCloud<PointType>());
        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        hdmapAll.reset(new pcl::PointCloud<PointType>());

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
        mapIsRefered = false;

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

        if(localizeInRef && saveDateTofolder)
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
        double start = contrlposesVec.front()[7] - systemStartTime;
        double totalt = contrlposesVec.back()[7] - contrlposesVec.front()[7];

        for (int j = 1; j < splineNum-2; ++j) {
//#pragma omp parallel for
//            for(auto pt : curSurfCloudsTobeAdded[j]){
//                pointAssociateToMap(&pt, &tmp);
//                tmp.intensity = pt.intensity*totalt + start;
//                frameCloud->points.emplace_back(tmp);
//            }
//#pragma omp parallel for
//            for(auto pt : curCornerCloudsTobeAdded[j]){
//                pointAssociateToMap(&pt, &tmp);
//                tmp.intensity = pt.intensity*totalt + start;
//                frameCloud->points.emplace_back(tmp);
//            }
//#pragma omp parallel for
            for(auto pt : curOutlierCloudsTobeAdded[j]){
                pointAssociateToMap(&pt, &tmp);
                tmp.intensity = pt.intensity*totalt + start;
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

//        cornerCloudkeyframes.emplace_back(localcornerMap);
//        surfCloudkeyframes.emplace_back(localSurfMap);

//        pcl::PointCloud<PointType>::Ptr saveCloud(new pcl::PointCloud<PointType>());
////        *saveCloud = *localSurfMapLabeled + *localcornerMap;
//        *localMapAllLabled = (*localSurfMapLabeled) + (*localcornerMapLabeled);
//        *localMapAll = (*localSurfMap) + (*localcornerMap);

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
                         const double& t2, bool save = false){

        cout << "[ F ] Filter Time window " << to_string(t1) << " ~ " << to_string(t2) << endl;

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
            cornerCloudkeyframes.emplace_back(cornerMapFiltered);
            surfCloudkeyframes.emplace_back(surfMapFiltered);
//            savedmapFrameInds.emplace_back(anchorID);
        }else{
            *localSurfMap = *surfMapFiltered;
            *localcornerMap = *cornerMapFiltered;
        }

        cout << "[ MAP ] Local map filtered. Left " << localcornerMap->size()
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

    // transform local map to current anchor frame
    void prepareLocalMap(){

        // set origin
        if(globalposesQuat->points.empty()){

            PointTypePose origin;
            origin.x = origin.y = origin.z = origin.roll = origin.pitch = origin.yaw = 0;
            origin.intensity = 1.0;
            origin.time = timeLaserOdometry;
            systemStartTime = timeLaserOdometry;
            globalposesQuat->points.emplace_back(origin);
            globalkeyposesQuat->points.emplace_back(origin);
            keyFrameInds.emplace_back(0);

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

//            saveposesAndframes(true);
            reInitedTimes++;

//            cornerCloudkeyframes.clear();
//            surfCloudkeyframes.clear();
//            globalposesQuat->points.resize(1);  // save the first identity pose
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

        if(laserCloudCornerFromMapDSNum < 100)
            return;

        // transform local map to local frame
        Eigen::Affine3d transToCurFrame = (getTransformAff(anchorPose).inverse());
        pcl::transformPointCloud(*localcornerMap, *localcornerMap, transToCurFrame);
        pcl::transformPointCloud(*localSurfMap,   *localSurfMap, transToCurFrame);
        curFrameID = globalposesQuat->size()-1;
//        if (saveMapForKF){
//            double time = globalposesQuat->points[curFrameID].time - systemStartTime;
//            filterMapByTime(time- 5, time+ 3, true);
//            cout << YELLOW << "[ Debug ] save new keyframe at " << time << RESET << endl;
//            assert(globalkeyposesQuat->size() == cornerCloudkeyframes.size());
//            saveMapForKF = false;
//            localizeInRef = true;
//        }

        kdtreeCornerFromMap->setInputCloud(localcornerMap);
        kdtreeSurfFromMap->setInputCloud(localSurfMap);

//        increkdtree_ptr->Build(localSurfMap->points);
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

    // update global poses by estimated incremental anchor pose
    void updateGlobalPose(){

        // last global pose
        lastglobalPoseQuat = globalposesQuat->points.back();
        Eigen::Quaterniond qlast (lastglobalPoseQuat.intensity, lastglobalPoseQuat.roll, lastglobalPoseQuat.pitch, lastglobalPoseQuat.yaw);
        Eigen::Vector3d tlast(lastglobalPoseQuat.x, lastglobalPoseQuat.y, lastglobalPoseQuat.z);
        qlast.normalize();

        Eigen::Matrix4d lastT, curT, increT;
        lastT = getTransMatrixT(qlast, tlast);

        if (badregistered) // todo: add last incremental motion, this is not precise
            curT = lastT * lastIncreT;
        else{
            // estimated relative pose
            Eigen::Quaterniond quaternionIncre (anchorPose[3], anchorPose[0],
                                                anchorPose[1], anchorPose[2]);
            Eigen::Vector3d tIncre(anchorPose[4],anchorPose[5],anchorPose[6]);
            quaternionIncre.normalize();
            increT = getTransMatrixT(quaternionIncre, tIncre);
            curT = lastT * increT;
        }
        lastIncreT = increT;

        // record updated global pose
        lastglobalPoseQuat = matrix4fToPose(curT.cast<float>());
        lastglobalPoseQuat.time = anchorPose[7];
        globalposesQuat->points.push_back(lastglobalPoseQuat);
        anchorID = globalposesQuat->points.size();  // begin from 2 ?

        //cout<<"[ Ceres ] Added global pose :" << anchorID << endl;
    }

    // main NLS functor
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

//                    pcl::PointSurfel tmpSurf;
//                    tmpSurf.x = pointSel.x;
//                    tmpSurf.y = pointSel.y;
//                    tmpSurf.z = pointSel.z;
//                    KD_TREE<pcl::PointSurfel>::PointVector searchedPts;
//                    dynamicKDtreeSurfel->Nearest_Search(tmpSurf, 1, searchedPts, pointSearchSqDis);
//                    tmpSurf = searchedPts[0];

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

    }

    // updating map
    void saveKeyFramesAndFactor(){

        if(!localcornerMap->points.empty())
            mapInited = true;
        else mapInited = false;
        surfelInited = false;

        badregistered = false;
        // save undistorted cloud
        pcl::PointCloud<PointType>::Ptr curCornerAdded2map(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr curSurfAdded2map(new pcl::PointCloud<PointType>());

        // transform cloud to local map
        PointType tmp;
        pcl::PointSurfel tmpsurfel;
        int newptNum = 0, curAllptNum = 0;

        double startT = contrlposesVec.front()[7] - systemStartTime;
//        double startT = contrlposesVec[1][7] - systemStartTime;
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
                tmp.intensity = pt.intensity*totalt + startT;
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
                tmp.intensity = pt.intensity*totalt + startT;
                if(pointRange(pt) < maxRangeMap){
                    curAllptNum++;
                    curSurfAdded2map->points.emplace_back(tmp);
                }

                if(mapInited){
                    kdtreeSurfFromMap->radiusSearch(tmp, localmapResolu, pointSearchInd, pointSearchSqDis);
                    if(pointSearchInd.empty()){
                        if(pointRange(pt) < maxRangeMap){
                            newptNum++;
                            localSurfMap->points.emplace_back(tmp);
                        }
                    }else
                        localSurfMap->points[pointSearchInd.front()].intensity = tmp.intensity;   // update timestamp
                } else if(pointRange(pt) < maxRangeMap){
                    newptNum++;
                    localSurfMap->points.emplace_back(tmp);
                }
            }
//            curAllptNum += curSurfCloudsTobeAdded[j].points.size();

        }
        //cout << "[ Spline ] Added current cloud to local map. ";

        // todo : too much new inserted points -> bad registration ?
        float curOverlapRatio = newptNum*1.0 / curAllptNum;
        if (mapInited && curOverlapRatio > cloudoverlapRatio){
            cout << BOLDMAGENTA << "[ Map ] Bad registration Ratio: " << curOverlapRatio << RESET << endl;
            badregisFrameInds.emplace_back(globalposesQuat->points.size()); // this pose need re-optimize
//
            badregistered = true;
            filterMapByTime(0, startT, false);  // filter out current cloud from map
        }

//        ofstream ofs(resultFolder+"overlapRatios.txt", ios::app);
//        double ratio_ = newptNum*1.0 / curAllptNum;
//        ofs << ratio_ << endl;
//        ofs.close();
//        cout << BOLDYELLOW << "[ Debug ] Bad registration. Ratio: " << ratio_ << RESET << endl;

        updateGlobalPose();
        cout << BOLDBLACK << " Added corner : " << curCornerAdded2map->points.size()
             << " / Added surf : " << curSurfAdded2map->points.size() << RESET << endl;

        cloudCurInWorld->clear();
        *cloudCurInWorld += *transformPointCloud(curCornerAdded2map,
                                                 &globalposesQuat->points.back());
        *cloudCurInWorld += *transformPointCloud(curSurfAdded2map,
                                                 &globalposesQuat->points.back());

        if(!mapInited && surfCloudkeyframes.empty()){  // first frame
//            cornerCloudkeyframes.emplace_back(localcornerMap);
            surfCloudkeyframes.emplace_back(cloudCurInWorld);
            pcl::io::savePCDFileBinaryCompressed(resultFolder + "initMap.pcd", *cloudCurInWorld);
        }
//        else{
//            cornerCloudkeyframes.emplace_back(curCornerAdded2map);
//            surfCloudkeyframes.emplace_back(curSurfAdded2map);
//        }

        // the distance between last and current keyframes
        currentRobotPosPoint.x = globalposesQuat->points.back().x;
        currentRobotPosPoint.y = globalposesQuat->points.back().y;
        currentRobotPosPoint.z = globalposesQuat->points.back().z;
        double distTravel = pointDistBet(previousRobotPosPoint, currentRobotPosPoint);

        if (!badregistered && distTravel > KF_DIST) {

            int frameID = globalposesQuat->points.size()-1;
//            if (frameID - keyFrameInds.back() < kf_dist)  // error!
            PointTypePose lastglobalPose = globalposesQuat->points[frameID];
            globalkeyposesQuat->points.emplace_back(lastglobalPose);
            keyFrameInds.emplace_back(frameID);
            saveMapForKF = true;

            currentRobotPosPoint.intensity = cloudKeyPoses3D->points.size();
            cloudKeyPoses3D->points.emplace_back(currentRobotPosPoint);
            updateKeyPoseGraph_Map();

//            if (!mapIsRefered || distTravel > 2*kf_dist)
            localizeInRef = true;
            previousRobotPosPoint = currentRobotPosPoint;
        }else
            updatePoseGraph_Map();
    }

    void updatePoseGraph_Map(){

        int mapUnitNum = globalposesQuat->points.size();
        int lastKF_ID = keyFrameInds.back();
        if( mapUnitNum > lastKF_ID ){

            Eigen::Isometry3d poseBegin, poseEnd;
            poseBegin = getTransformMatrix(globalposesQuat->points[lastKF_ID]);
            poseEnd = getTransformMatrix(globalposesQuat->points[mapUnitNum-1]);
            double noise_odom = 1;
            if (badregistered) noise_odom = 50;  // todo: noise for bad registration factor?

            gtsamHelper.addBetweenFactor(poseBegin, lastKF_ID,
                                         poseEnd, mapUnitNum-1, true, noise_odom);
            cout << YELLOW << "[ Graph ] add frame edge " << lastKF_ID
                 << " to " << (mapUnitNum-1) << RESET << endl;
        }
    }

    void updateKeyPoseGraph_Map(){

        int mapUnitNum = globalkeyposesQuat->points.size();
        assert(keyFrameInds.size() == mapUnitNum);

        if( mapUnitNum < 2 ){  // fix the first pose?

            return;
//            gtsamHelper.addpriorFactor(getTransformMatrix(globalposesQuat->points.front()));
//            cout << GREEN << "[ mapUnit ] Graph Origin set. " << RESET << endl;
        }else{

            Eigen::Isometry3d poseBegin, poseEnd;
            poseBegin = getTransformMatrix(globalkeyposesQuat->points[mapUnitNum-2]);
            poseEnd = getTransformMatrix(globalkeyposesQuat->points[mapUnitNum-1]);
            float noise_odom = 1;
//            if (mapUnitNum > 1){
//                Eigen::Matrix4f trans;
//                pcl::PointCloud<PointType>::Ptr cloud_align(new pcl::PointCloud<PointType>());
//                fastGICP_Fusion(surfCloudkeyframes[mapUnitNum-1], surfCloudkeyframes[mapUnitNum-2],
//                                2., trans, noise_odom, cloud_align);
//                if (noise_odom < ScoreThre){
//                    trans = (poseEnd.matrix().cast<float>()*trans).eval();
//                    poseEnd = Eigen::Isometry3d(trans.cast<double>());
//                }
//            }
            gtsamHelper.addBetweenFactor(poseBegin, keyFrameInds[mapUnitNum-2],
                                         poseEnd, keyFrameInds[mapUnitNum-1], true, noise_odom);
            cout << BOLDYELLOW << "[ Graph ] add KEY frame edge " << keyFrameInds[mapUnitNum-2]
                 << " to " << keyFrameInds[mapUnitNum-1] << RESET << endl;
        }

        if (!mapIsRefered) return;

        gtsamHelper.optimizeGraph();
        gtsamHelper.correctPoseCloud(globalposesQuat, 0, globalposesQuat->size());

        // update keyframe poses
        for (int j = 0; j < mapUnitNum; ++j)
            globalkeyposesQuat->points[j] = globalposesQuat->points[keyFrameInds[j]];

    }

    void clearCloud(){

        cornerCloudkeyframes.clear();
        surfCloudkeyframes.clear();
        localSurfMap->clear();
        localcornerMap->clear();
        gtsamHelper.reset();
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

    // get control poses and scans
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
            curSurfCloudsTobeAdded.emplace_back(tmpCloud);
            pcl::fromROSMsg(featureCloudinfo.cloud_corner, tmpCloud);
            curCornerCloudsTobeAdded.emplace_back( tmpCloud);
            pcl::fromROSMsg(featureCloudinfo.segmentedCloud, tmpCloud);
//            pcl::fromROSMsg(featureCloudinfo.cloud_outlier, tmpCloud);
            curOutlierCloudsTobeAdded.emplace_back(tmpCloud);

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

    /// [THREAD] MAIN PROCESS
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

            deriveCurCloud();

            outputCloudbeforeSpOpti(false);

            if(splineT == 2){
                maxRatio = 0.5;
                SophusSpline::CeresFactorsSP::fromDataToControlpointsDynamic(contrlposesVec);
            } else if(splineT == 0)
                CeresFactorsSP::fromDataToControlpointsDynamic(contrlposesVec);
            outputCloudAftSpOpti(false);

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

            publishTF();

            publishKeyPosesAndFrames();

            ROS_INFO("Frame No. %ld \n", globalposesQuat->points.size());
            ROS_INFO("KeyFrame No. %ld \n", globalkeyposesQuat->points.size());

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

    inline float getProbability(const float& label){
        if (label == 2) return Prob_New;
        if (label == 3) return Prob_Change;
        else return Prob_Static;
    }

    /// [THREAD] save per keyframe poses and cloud
    void saveposesAndframesThread(){

        ros::Rate rate(0.02);
        while(ros::ok()){
            rate.sleep();
            saveposesAndframes();
        }
    }
    void saveposesAndframes(bool reinited = false){

        cout << BLUE << "[ Mapping ] Saving map thread started ... " << RESET << endl;
//        std::lock_guard<mutex> lockGuard(mtx);

        // save labelled MLS
        if (!hdmapAll->empty())
            pcl::io::savePCDFileBinary(resultFolder + "MLS_labelled.pcd", *hdmapAll);

        FILE *fp; string file;
        if (reinited) file = resultFolder + "keyposes6dspline" + to_string(reInitedTimes) + ".txt";
        else file = resultFolder + "Key_poses6dspline.txt";
        fp = fopen(file.data(),"w");
        for(auto pose : globalkeyposesQuat->points){

            fprintf(fp, "%lf %f %f %f %f %f %f %f\n", pose.time,
                    pose.x, pose.y, pose.z,
                    pose.roll, pose.pitch, pose.yaw,
                    pose.intensity);
        }
        fclose(fp);
        // write all poses
        file = resultFolder + "poses6dspline.txt";
        fp = fopen(file.data(),"w");
        for(auto pose : globalposesQuat->points){
            fprintf(fp, "%lf %f %f %f %f %f %f %f\n", pose.time,
                    pose.x, pose.y, pose.z,
                    pose.roll, pose.pitch, pose.yaw,
                    pose.intensity);
        }
        fclose(fp);

        int frameN = surfCloudkeyframes.size();
        int poseN = globalkeyposesQuat->points.size();
        cout << BOLDRED << "[ Map ] Saving global map with frames " << frameN <<", poses " << poseN;
        if (frameN > poseN) return;
        assert(frameN-1 == priorTs_withS.size());

        KD_TREE<PointType>::Ptr increkdtree_ptr(new KD_TREE<PointType>(0.3, 0.6, localmapResolu));
        pcl::KdTreeFLANN<PointType>::Ptr kdtree(new pcl::KdTreeFLANN<PointType>());

        pcl::PointCloud<PointType>::Ptr globalMap(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr cloud_transformed(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr cloud_last(new pcl::PointCloud<PointType>());
        int ptr = 0, badframeN = badregisFrameInds.size();
        float score;
        Eigen::Matrix4f trans_cumulated = Eigen::Matrix4f::Identity(), trans;
        Eigen::Matrix4d poseBegin, poseTo;
        // Fixme: re-align the key-frame clouds and graph optimize
//        GtsamHelper graph;
//        for (int j = 1; j < frameN; ++j) {
//
//            if (surfCloudkeyframes[j]->empty()) break;
//            *cloud_transformed = *transformPointCloud(surfCloudkeyframes[j],
//                                                      &globalkeyposesQuat->points[j]);
//            pcl::transformPointCloud(*cloud_transformed, *cloud_transformed, trans_cumulated);
//
//            if (cloud_last->empty()) {
//                *cloud_last = *cloud_transformed;
//            }else{
//                pcl::PointCloud<PointType>::Ptr cloud_align(new pcl::PointCloud<PointType>());
//                fastGICP_Fusion(cloud_transformed, cloud_last, 2., trans, score, cloud_align);
//                poseBegin = trans_cumulated.cast<double>() *
//                            getTransformMatrix4d(globalkeyposesQuat->points[j-1]);
//                cout << "[ Debug ] frame " << j-1 << "-" << j << " : " << score << endl;
//
////                pcl::io::savePCDFileBinary(resultFolder + to_string(j)+"globalMap_src.pcd",
////                                           *cloud_transformed);
////                pcl::io::savePCDFileBinary(resultFolder + to_string(j)+"globalMap_tgt.pcd",
////                                           *cloud_last);
////                pcl::io::savePCDFileBinary(resultFolder + to_string(j)+"globalMap_ali.pcd",
////                                           *cloud_align);
//
//                if (score < ScoreThre){
//                    trans_cumulated = (trans * trans_cumulated).eval();
//                    *cloud_last = *cloud_align;
//                }else
//                    *cloud_last = *cloud_transformed;
//
//                poseTo = trans_cumulated.cast<double>() *
//                         getTransformMatrix4d(globalkeyposesQuat->points[j]);
//                graph.addBetweenFactor(poseBegin, j-1, poseTo, j, true, score);
//                graph.addpriorFactor(priorTs_withS[j-2].first, j-1, priorTs_withS[j-2].second);
//
////                graph.optimizeGraph();
////                graph.correctPoseCloud(globalkeyposesQuat, 0, j+1);
//            }
//        }
//        graph.optimizeGraph(false);  // no isam
//        graph.correctPoseCloud(globalkeyposesQuat, 1, globalkeyposesQuat->size());

        for (int i = 1; i < frameN; ++i) {

            if (surfCloudkeyframes[i]->empty()) continue;
            *cloud_transformed = *transformPointCloud(surfCloudkeyframes[i],
//                                                      &globalposesQuat->points[i]);
                                                      &globalkeyposesQuat->points[i]);
            pcl::transformPointCloud(*cloud_transformed, *cloud_transformed, trans_cumulated);

//            pcl::io::savePCDFileBinary(resultFolder+to_string(i)+"globalMap.pcd", *cloud_transformed);
            ptr = cloud_transformed->points.size();

            if (globalMap->empty()) {  // first frame
                increkdtree_ptr->Build(cloud_transformed->points);
//                *globalMap = *cloud_transformed;
                globalMap->points.resize(ptr);
                cloud_last->points.resize(ptr);
#pragma omp parallel for
                for (int j = 0; j < ptr; ++j) {
                    cloud_transformed->points[j].intensity = 0.5;
                    globalMap->points[j] = cloud_transformed->points[j];
                    cloud_last->points[j] = cloud_transformed->points[j];
                }
                continue;
            }

            // fuse the cloud
//            pcl::PointCloud<PointType>::Ptr cloud_align(new pcl::PointCloud<PointType>());
//            fastGICP_Fusion(cloud_transformed, cloud_last, 2., trans, score, cloud_align);
//            if (score < ScoreThre){
//                trans_cumulated = (trans * trans_cumulated).eval();
//                *cloud_last = *cloud_align;
//            }else
//                *cloud_last = *cloud_transformed;

            kdtree->setInputCloud(globalMap);
            pcl::PointCloud<PointType>::Ptr cloud_inliers(new pcl::PointCloud<PointType>());
#pragma omp parallel for
            for (int j = 0; j < ptr; ++j) {

//                PointType pt = cloud_last->points[j];
                PointType pt = cloud_transformed->points[j];
                KD_TREE<PointType >::PointVector searchedPts;
                increkdtree_ptr->Radius_Search(pt, localmapResolu, searchedPts);
                if (searchedPts.empty()){
                    searchedPts.emplace_back(pt);
                    increkdtree_ptr->Add_Points(searchedPts, true);
                    cloud_inliers->points.emplace_back(pt);
                }else {  // fusion the changing-status with map points

                    for (auto npt: searchedPts){
                        kdtree->nearestKSearch(npt, 1, pointSearchInd, pointSearchSqDis);
                        float prob = globalMap->points[pointSearchInd[0]].intensity;
                        prob += log(pt.intensity/(1-pt.intensity));
                        globalMap->points[pointSearchInd[0]].intensity = min<float>(1.0, max<float>(0.0, prob));
                    }
                }
            }
            float newptRatio = cloud_inliers->size()*1.0 / cloud_transformed->points.size();
            if (newptRatio < cloudoverlapRatio)
                *globalMap += *cloud_inliers;
        }

        if(!globalMap->points.empty()){
            if(reinited) pcl::io::savePCDFileBinary(resultFolder + "globalMap" + to_string(reInitedTimes) + ".pcd", *globalMap);
            else pcl::io::savePCDFileBinary(resultFolder + "globalMap.pcd", *globalMap);
            cout << " ... SUCCEED !" << RESET << endl;
        }
    }

    void fastGICP_Fusion(const pcl::PointCloud<PointType>::Ptr& SRC_cloud,
                         const pcl::PointCloud<PointType>::Ptr& TGT_cloud,
                         const float& corrDist,
                         Eigen::Matrix4f& T, float& score,
                         pcl::PointCloud<PointType>::Ptr& alignedCloud_){

        fast_gicp::FastGICP<PointType , PointType> fastGicp;
        fastGicp.setInputSource(SRC_cloud);
        fastGicp.setInputTarget(TGT_cloud);
        fastGicp.setMaxCorrespondenceDistance(corrDist);
        fastGicp.align(*alignedCloud_);
        T = fastGicp.getFinalTransformation();

        score = fastGicp.getFitnessScore(1.0);
    }

    /// [THREAD] register with reference map
    void localizeInRef_thread(){

        ros::Rate rate(0.5);
        int curAnchorID, lastID = -1, idInKf = 1;
        while (ros::ok()){

            rate.sleep();
            if (!localizeInRef) continue;
            if (hdmapAll->empty()) continue;
            if (globalkeyposesQuat->points.size() - idInKf < 1.1) continue;

            curAnchorID = keyFrameInds[idInKf];
            if (curAnchorID == lastID) continue;
//            lastID = curAnchorID;
            cout << BOLDBLUE << "[ MLS ] Localizing in HD map at KF " << curAnchorID << RESET << endl;

            pcl::PointCloud<PointType>::Ptr curMapCloud_slam(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr curMapCloud_slam_SRI(new pcl::PointCloud<PointType>());
            PointTypePose poseCur, poseAnchor, poseRela;
            {  // in case of stuck
                std::lock_guard<std::mutex> lock(mtx);
                int localmapSize = localSurfMap->points.size() + localcornerMap->points.size();
                cout << MAGENTA << "[ MLS ] curlocalmapSize : " << localmapSize << RESET << endl;
                curMapCloud_slam->points.reserve(localmapSize);
                *curMapCloud_slam = (*localSurfMap) + (*localcornerMap);
                poseCur = globalposesQuat->points[curFrameID];
                poseAnchor = globalposesQuat->points[curAnchorID];

//                *curMapCloud_slam += *transformPointCloud(cornerCloudkeyframes[idInKf], &poseAnchor);
//                *curMapCloud_slam += *transformPointCloud(surfCloudkeyframes[idInKf], &poseAnchor);
            }
            if(curMapCloud_slam->size() < 50000) continue;

            poseRela = getRelativePose(poseAnchor, poseCur);
            Eigen::Matrix4d transformRela = getTransformMatrix4d(poseRela);
            pcl::transformPointCloud(*curMapCloud_slam, *curMapCloud_slam,  transformRela);

            ImageProjectionManager imagePM_SLAM(0.2, 0.2, 60);
            imagePM_SLAM.fromPointCloud(curMapCloud_slam);
            imagePM_SLAM.toPointCloud(curMapCloud_slam, curMapCloud_slam_SRI);
//            if (imagePM_SLAM.writeRangeImage(resultFolder+"rangeMat_slam"+to_string(curAnchorID)+".png"))
//                cout << MAGENTA << "curMapCloud_slam : " << curMapCloud_slam->size() << RESET << endl;

//            if(!curMapCloud_slam->empty())
//                pcl::io::savePCDFileBinaryCompressed(resultFolder + "curMapCloud_" +
//                                                     to_string(curAnchorID)+".pcd", *curMapCloud_slam);

            pcl::console::TicToc timer_L;
            timer_L.tic();
            // withdraw HDmap
            pcl::PointCloud<PointType>::Ptr curMapCloud_HD(new pcl::PointCloud<PointType>());
            vector<int> originalInds;
            {
                pcl::PassThrough<PointType>::Ptr fieldfilter(new pcl::PassThrough<PointType>());
                std::lock_guard<std::mutex> lock(mtx);
                fieldfilter->setFilterFieldName("x");
                fieldfilter->setFilterLimits(poseAnchor.x-60, poseAnchor.x+60);
                fieldfilter->setInputCloud(hdmapAll);
                fieldfilter->filter(*curMapCloud_HD);
                fieldfilter->filter(originalInds);
//                cout << "curMapCloud_HD: " << curMapCloud_HD->size() << ", originalInds: " << originalInds.size() << endl;
            }
            if(curMapCloud_HD->size() < 50000) continue;

            Eigen::Matrix4d transformAnchor = getTransformMatrix4d(poseAnchor);
            Eigen::Matrix4d transformToLocal = transformAnchor.inverse();
            //            cout << RED << " transformToLocal: \n " << transformToLocal << RESET << endl;
            pcl::transformPointCloud(*curMapCloud_HD, *curMapCloud_HD, transformToLocal);

            pcl::PointCloud<PointType>::Ptr curMapCloud_HD_SRI(new pcl::PointCloud<PointType>());
            ImageProjectionManager imagePM_MLS(0.2, 0.2, 60);
            imagePM_MLS.fromPointCloud(curMapCloud_HD);
            // imagePM_MLS.fromPointCloud(curMapCloud_HD, poseAnchor.getVector3fMap());
            imagePM_MLS.toPointCloud(curMapCloud_HD, curMapCloud_HD_SRI);
//            if (imagePM_MLS.writeRangeImage(resultFolder+"rangeMat_HD"+to_string(curAnchorID)+".png"))
//                cout << MAGENTA << "curMapCloud_HD : " << curMapCloud_HD->size() << RESET << endl;
            cout << RED << "[ MLS ] Assemble Cloud Time " << timer_L.toc() << " ms " << RESET << endl;

            // registration
            Eigen::Matrix4f T_g, T_final;
            pcl::PointCloud<PointType>::Ptr coarseAlignedCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr alignedCloud(new pcl::PointCloud<PointType>());
            float noiseScore = -1;
            if (!mapIsRefered)
                fastGICP_Fusion(curMapCloud_slam_SRI, curMapCloud_HD_SRI, 5.,
                                T_final, noiseScore, alignedCloud);
            else
                fastGICP_Fusion(curMapCloud_slam_SRI, curMapCloud_HD_SRI, 2.,
                                T_final, noiseScore, alignedCloud);

            if(noiseScore > ScoreThre) continue;
            cout << BOLDGREEN << "[ MLS ] converged with " << noiseScore << RESET << endl;

            // test for range difference
            pcl::transformPointCloud(*curMapCloud_slam, *curMapCloud_slam, T_final);
            imagePM_SLAM.fromPointCloud(curMapCloud_slam);
            ImageProjectionManager::rangeImageDiff(imagePM_MLS.rangeMat, imagePM_SLAM.rangeMat,
                                                   imagePM_MLS.labelMat, imagePM_SLAM.labelMat);
            imagePM_SLAM.toPointCloud(curMapCloud_slam, curMapCloud_slam_SRI);
            imagePM_MLS.toPointCloud(curMapCloud_HD, curMapCloud_HD_SRI);
            imagePM_MLS.updateCloudStatus(hdmapAll, originalInds);

            surfCloudkeyframes.emplace_back(curMapCloud_slam_SRI);
//            imagePM_SLAM.updateCloudStatus(curMapCloud_slam);
            cout << RED << "[ MLS ] Update Cloud Time " << timer_L.toc() << " ms " << RESET << endl;

//            if(!alignedCloud->empty())
//                pcl::io::savePCDFileBinaryCompressed(resultFolder + "alignedCloud_" +
//                                                     to_string(curAnchorID)+".pcd", *alignedCloud);
//            if(!curMapCloud_slam_SRI->empty())
//                pcl::io::savePCDFileBinaryCompressed(resultFolder + "alignedCloud_new_" +
//                                                     to_string(curAnchorID)+".pcd", *curMapCloud_slam_SRI);
//            if(!curMapCloud_HD_SRI->empty())
//                pcl::io::savePCDFileBinaryCompressed(resultFolder + "curMapCloud_HD_" +
//                                                     to_string(curAnchorID)+".pcd", *curMapCloud_HD_SRI);
//            ofs_oriPose << curFrameId << " " << setprecision(8) << poseCur.translation().transpose() << endl;

            Eigen::Matrix4d pose3new = transformAnchor * (T_final.cast<double>());
            priorTs_withS.emplace_back(pose3new, noiseScore);
//            pcl::transformPointCloud(*curMapCloud_slam, *curMapCloud_slam, pose3new);
//            if(!alignedCloud->empty())
//                pcl::io::savePCDFileBinaryCompressed(resultFolder + "alignedCloud_g" +
//                                                     to_string(noiseScore)+".pcd", *curMapCloud_slam);

            // add factor
            {
                cout << BOLDGREEN << "[ MLS ] ADD prior factor in Frame_" << curAnchorID << RESET << endl;
                std::lock_guard<std::mutex> lock(mtx);

//                if (!mapIsRefered)
                gtsamHelper.addpriorFactor(pose3new, curAnchorID, noiseScore);  // only once?
//                gtsamHelper.addGPSFactor(pose3new.block<3,1>(0,3), curAnchorID, noiseScore);
                mapIsRefered = true;
                localizeInRef = false;
                lastID = curAnchorID;
                idInKf++;
            }
        }

    }
};


void mySigintHandler(int sig);

int main(int argc, char** argv){

    ros::init(argc, argv, "MappingWithReference");

    google::InitGoogleLogging(argv[0]);

    MappingWithReference MO;
    signal(SIGINT, mySigintHandler);

//    std::thread saveposesAndframes_thread(&MappingWithReference::saveposesAndframesThread, &MO);
    std::thread processor(&MappingWithReference::run, &MO);
    std::thread localizeInRef_thread(&MappingWithReference::localizeInRef_thread, &MO);

    ros::spin();

//    saveposesAndframes_thread.join();
    localizeInRef_thread.join();
    processor.join();

    cout << BLUE << "[ ROS ] MappingWithReference node is done." << RESET << endl;
    MO.saveposesAndframes();
    return 0;
}

// do sth before it shutdown
void mySigintHandler(int sig){

    ROS_INFO("shutting down!");
    ros::shutdown();
}