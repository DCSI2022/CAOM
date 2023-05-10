////////////////////////////////////////////////////////////////////////////////////////
// Created by joe on 2020/10/5.
//
// pose tends to drift upwards along the z direction.
//
///////////////////////////////////////////////////////////////////////////////////////

#include "tools.h"
#include "poseEstimationLib.hpp"

class OdomEstimation{

    std::unique_ptr<PoseEstimationManager> poseEstimator;
    ros::NodeHandle nh;

    ros::Subscriber subCloudinfo;

    ros::Publisher pubOdometry;
    ros::Publisher pubLocalMapCorner;
    ros::Publisher pubLocalMapSurf;
    ros::Publisher pubLocalMapAll;
    ros::Publisher pubOdomPath;
    ros::Publisher pubOctoMap;
    ros::Publisher pubOdomPose;
    ros::Publisher pubOdomFeatureCloudinfo;

    pcl::CropBox<PointType>::Ptr cropBoxFilter;

    pcl::PointCloud<PointTypePose>::Ptr odomPosesCloud;
    pcl::PointCloud<PointType>::Ptr localmap_surf;
    pcl::PointCloud<PointType>::Ptr localmap_corner;
    // for localization mode
    pcl::PointCloud<PointType>::Ptr localmap_all;
    pcl::PointCloud<PointType>::Ptr globalmap_localize;
    pcl::PointCloud<PointType>::Ptr globalmap_build;

    pcl::KdTreeFLANN<PointType>::Ptr kdtree_map_surf;
    pcl::KdTreeFLANN<PointType>::Ptr kdtree_map_corner;

    pcl::PointCloud<PointType>::Ptr curCloud_surf;
    pcl::PointCloud<PointType>::Ptr curCloud_corner;
    pcl::PointCloud<PointType>::Ptr curCloud_others;
    pcl::PointCloud<PointType>::Ptr curCloud_segmented;
    pcl::PointCloud<PointType>::Ptr lastCloud_segmented;  // for motion distortion

    // save for reconstruction
    vector<pcl::PointCloud<PointType>::Ptr> surfcloudsVec;
    vector<pcl::PointCloud<PointType>::Ptr> cornercloudsVec;

    pcl::VoxelGrid<PointType> downSampler;

    std::deque<structural_mapping::cloud_infoConstPtr> cloudinfomsgQue;
    structural_mapping::cloud_info featureCloudinfo;

    std::mutex locker;

    double timeCur, timeLast;
    bool odom_init, useConstraintSphere;
    int curCornerNum, curSurfNum;
    int mapCornerNum, mapSurfNum;
    int frameNum;
    std::vector<double> poseVars;

    OctomapManager octomapManager;

    // rotation first
    double transformation[7] = {0, 0, 0, 1, 0, 0, 0};
    Eigen::Map<Eigen::Quaterniond> q_w_curr = Eigen::Map<Eigen::Quaterniond>(transformation);
    Eigen::Map<Eigen::Vector3d> t_w_curr = Eigen::Map<Eigen::Vector3d>(transformation + 4);
    // translation first
//    double transformation[7] = {0, 0, 0, 0, 0, 0, 1};
//    Eigen::Map<Eigen::Quaterniond> q_w_curr = Eigen::Map<Eigen::Quaterniond>(transformation+3);
//    Eigen::Map<Eigen::Vector3d> t_w_curr = Eigen::Map<Eigen::Vector3d>(transformation);

public:
    OdomEstimation():nh(){

        nh.param<bool>("/odom/odom_useOctomap", odom_useOctomap, false);
        nh.param<bool>("/odom/odom_octoFilterMap", odom_octoFilterMap, false);
        nh.param<bool>("/odom/useConstraintSphere", useConstraintSphere, true);
        nh.param<bool>("/odom/odom_localize_mode", odom_localize_mode, false);
        nh.param<float>("/odom/odom_ds_leafsize", odom_ds_leafsize, 0.25f);
        nh.param<float>("/odom/odom_map_ds_leafsize", odom_map_ds_leafsize, 0.25f);
        nh.getParam("/projpath", projPath);  // global param add '/' before name

        cout << "[ odom ] projFolder : "  << projPath << endl;
        cout << "[ odom ] odom_useOctomap : "  << odom_useOctomap << endl;
        cout << "[ odom ] odom_octoFilterMap : "  << odom_octoFilterMap << endl;
        cout << "[ odom ] odom_localize_mode : "  << odom_localize_mode << endl;
        cout << "[ odom ] useConstraintSphere : "  << useConstraintSphere << endl;
        cout << "[ odom ] odom_ds_leafsize : "  << odom_ds_leafsize << endl;
        cout << "[ odom ] odom_map_ds_leafsize : "  << odom_map_ds_leafsize << endl;

        subCloudinfo = nh.subscribe<structural_mapping::cloud_info>("/feature_cloud_info", 1, &OdomEstimation::cloudinfoHandler, this);

        pubLocalMapSurf   = nh.advertise<sensor_msgs::PointCloud2>("/localmapSurf_odom", 1);
        pubLocalMapCorner = nh.advertise<sensor_msgs::PointCloud2>("/localmapCorner_odom", 1);
        pubOdometry       = nh.advertise<nav_msgs::Odometry>("/odom_pose", 1);

        pubOdomPath = nh.advertise<nav_msgs::Path>("/odom_path", 1);

        pubOctoMap = nh.advertise<octomap_msgs::Octomap>("/localOctomap_odom", 1);

        pubLocalMapAll = nh.advertise<sensor_msgs::PointCloud2>("/localmapAll_odom", 1);
        pubOdomPose = nh.advertise<sensor_msgs::PointCloud2>("/poseCloud_odom", 1);
        pubOdomFeatureCloudinfo = nh.advertise<structural_mapping::cloud_info>("/odom_featureCloudinfo", 1);

        poseEstimator.reset(new PoseEstimationManager(odom_map_ds_leafsize, 1, useConstraintSphere));
        allocateMemo();
    }

    void allocateMemo(){

        cropBoxFilter.reset(new pcl::CropBox<PointType>());

        localmap_surf  .reset(new pcl::PointCloud<PointType>()) ;
        localmap_corner.reset(new pcl::PointCloud<PointType>()) ;
        odomPosesCloud.reset(new pcl::PointCloud<PointTypePose>()) ;

        localmap_all.reset(new pcl::PointCloud<PointType>()) ;
        globalmap_localize.reset(new pcl::PointCloud<PointType>()) ;
        globalmap_build.reset(new pcl::PointCloud<PointType>()) ;

        kdtree_map_corner.reset(new pcl::KdTreeFLANN<PointType>());
        kdtree_map_surf.reset(new pcl::KdTreeFLANN<PointType>());

        curCloud_corner.reset(new pcl::PointCloud<PointType>()) ;
        curCloud_surf  .reset(new pcl::PointCloud<PointType>()) ;
        curCloud_others.reset(new pcl::PointCloud<PointType>()) ;
        curCloud_segmented  .reset(new pcl::PointCloud<PointType>()) ;
        lastCloud_segmented .reset(new pcl::PointCloud<PointType>()) ;

        odom_init = false;
        frameNum = 0;

        downSampler.setLeafSize(odom_ds_leafsize, odom_ds_leafsize, odom_ds_leafsize);
    }

    void cloudinfoHandler(const structural_mapping::cloud_infoConstPtr& msgIn){

        std::lock_guard<std::mutex> lockGuard(locker);
        cloudinfomsgQue.emplace_back(msgIn);
    }

    bool parseData(){

        {
            std::lock_guard<std::mutex> lockGuard(locker);
            if(cloudinfomsgQue.empty())
                return false;
            featureCloudinfo = *cloudinfomsgQue.front();
            cloudinfomsgQue.pop_front();
        }

        // todo : which kind of features ?
        pcl::PointCloud<PointType>::Ptr tmpo(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(featureCloudinfo.segmentedCloud, *curCloud_others);
        pcl::fromROSMsg(featureCloudinfo.cloud_outlier, *tmpo);
//        *curCloud_others = *curCloud_surf;
        // for registration
        pcl::fromROSMsg(featureCloudinfo.cloud_lines, *curCloud_surf);
        pcl::fromROSMsg(featureCloudinfo.cloud_corner, *curCloud_corner);

        if(odom_localize_mode)
            pcl::fromROSMsg(featureCloudinfo.segmentedCloud, *curCloud_segmented);

        return true;
    }


    void initlocalMap(){

        // relocalization in map mode
        poseEstimator->init(curCloud_corner, curCloud_surf,
                            odom_useOctomap, projPath + "globalmap.pcd", odom_localize_mode);

        odom_init = true;
        cout << BOLDRED << "[ odom ] Odom local map inited. " << RESET << endl;
//        publishOdomInfo();
    }

    void saveOctoMap(string filename){

        cout << MAGENTA << "[ odom ] Saving final map info" << RESET << endl;
        pcl::PointCloud<PointType>::Ptr tmpCloud(new pcl::PointCloud<PointType>());
        *tmpCloud += *localmap_surf;
        *tmpCloud += *localmap_corner;
        if(!tmpCloud->points.empty())
            pcl::io::savePCDFileBinary(projPath + "odomFeatureMap.pcd", *tmpCloud);

        if(!globalmap_build->points.empty())
            pcl::io::savePCDFileBinary(projPath + "odomGlobalMap.pcd", *globalmap_build);

        if(!odom_useOctomap)
            return;
        cout << RED << "[ odom ] Saving octomap..." << RESET << endl;

        poseEstimator->octomapManager.saveOctoMap(filename);
    }

    // 依据对应位姿transform点云
    pcl::PointCloud<PointType>::Ptr transformPointCloud( pcl::PointCloud<PointType>::Ptr cloudIn,
                                                         const PointTypePose* transformIn){

        if(cloudIn->empty())
            cout << BOLDMAGENTA << "NO CLOUD DATA" << RESET << endl;

        Eigen::Quaternionf quat(transformIn->intensity, transformIn->roll, transformIn->pitch, transformIn->yaw);
        quat.normalize();
        Eigen::Vector3f trans(transformIn->x, transformIn->y, transformIn->z);
//        Eigen::Vector3f ptVec;

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

    void buildGlobalMap(){

        if(lastCloud_segmented->points.empty()){
            *lastCloud_segmented = *curCloud_segmented;
            return;
        }

        // add last undistorted cloud to globalmap
        int cloudsize = lastCloud_segmented->points.size();
        int mapsize = globalmap_build->points.size();
        globalmap_build->points.resize(mapsize + cloudsize);
//        Eigen::Isometry3d incre = odom_last.inverse() * odom;
//#pragma omp parallel for
//        for (int i = 0; i < cloudsize; ++i) {
//            PointType tmp = lastCloud_segmented->points[i];
//            Eigen::Vector4d tmpV(tmp.x, tmp.y, tmp.z, 1.0);
//            double ratio = (double)(tmp.intensity - (int)tmp.intensity) / scanPeriod;
////            Eigen::Vector3d resV = (odom_last *
////                                    Eigen::Isometry3d(incre.matrix()* ratio)
////                                    * tmpV).head<3>();
//            Eigen::Vector3d resV = (odom_last * tmpV).head<3>();
//            globalmap_build->points[mapsize + i].x = resV(0);
//            globalmap_build->points[mapsize + i].y = resV(1);
//            globalmap_build->points[mapsize + i].z = resV(2);
//            globalmap_build->points[mapsize + i].intensity = frameNum;
//        }
        *lastCloud_segmented = *curCloud_segmented;
        cout << MAGENTA << "[ odom ] Global map size " << mapsize << RESET << endl;
    }

    void updateLocalMap(){

        poseEstimator->updatelocalMap(curCloud_corner, curCloud_surf);
//        poseEstimator->updatelocalMap(curCloud_corner, curCloud_others);

        if(!odom_useOctomap)
            return;
        octomap_msgs::Octomap octoMsg;
        octoMsg.header.stamp = featureCloudinfo.header.stamp;
        octoMsg.header.frame_id = "/odom";
        if(poseEstimator->octomapManager.OctoMaptoRosMsg(octoMsg) && pubOctoMap.getNumSubscribers())
            pubOctoMap.publish(octoMsg);

    }

    void publishOdomInfo(){

        timeCur = featureCloudinfo.header.stamp.toSec();
        // publish odometry
        nav_msgs::Odometry laserOdometry;
        laserOdometry.header.frame_id = "/odom";
        laserOdometry.child_frame_id = "/base_link";
        laserOdometry.header.stamp = featureCloudinfo.header.stamp;
        laserOdometry.pose.pose.orientation.x = q_w_curr.x();
        laserOdometry.pose.pose.orientation.y = q_w_curr.y();
        laserOdometry.pose.pose.orientation.z = q_w_curr.z();
        laserOdometry.pose.pose.orientation.w = q_w_curr.w();
        laserOdometry.pose.pose.position.x = t_w_curr.x();
        laserOdometry.pose.pose.position.y = t_w_curr.y();
        laserOdometry.pose.pose.position.z = t_w_curr.z();
        pubOdometry.publish(laserOdometry);

        // for rviz view
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::poseMsgToTF(laserOdometry.pose.pose, transform);
//        transform.setOrigin( tf::Vector3(t_w_curr.x(), t_w_curr.y(), t_w_curr.z()) );
//        transform.setRotation(tf::Quaternion(q_w_curr.x(),q_w_curr.y(),q_w_curr.z(),q_w_curr.w()));
//        br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "base_link", "odom"));
        br.sendTransform(tf::StampedTransform(transform, featureCloudinfo.header.stamp, "odom", "base_link"));

//        geometry_msgs::PoseStamped odom;
//        odom.header.frame_id = "/odom";
//        odom.header.stamp = featureCloudinfo.header.stamp;
//        odom.pose.orientation.x = q_w_curr.x();
//        odom.pose.orientation.y = q_w_curr.y();
//        odom.pose.orientation.z = q_w_curr.z();
//        odom.pose.orientation.w = q_w_curr.w();
//        odom.pose.position.x = t_w_curr.x();
//        odom.pose.position.y = t_w_curr.y();
//        odom.pose.position.z = t_w_curr.z();
//
//        nav_msgs::Path path;
//        path.header.frame_id = "/odom";
//        path.header.stamp = featureCloudinfo.header.stamp;
//        path.poses.emplace_back(odom);
//        pubOdomPath.publish(path);

        if(odom_save_pose){

            ofstream ofs(projPath + "odom_poses.txt", ios::app);
            ofs << to_string(timeCur) << " "
                << t_w_curr.x() << " " << t_w_curr.y() << " " << t_w_curr.z() << " "
                << q_w_curr.x() << " " << q_w_curr.y() << " " << q_w_curr.z() << " "
                << q_w_curr.w() << endl;
//                << q_w_curr.w() << " " << poseVars.back() << endl;
            ofs.close();
        }

        PointTypePose odompose;
        odompose.roll  = q_w_curr.x();
        odompose.pitch = q_w_curr.y();
        odompose.yaw   = q_w_curr.z();
        odompose.intensity = q_w_curr.w();
        odompose.x = t_w_curr.x();
        odompose.y = t_w_curr.y();
        odompose.z = t_w_curr.z();
        odompose.time = featureCloudinfo.header.stamp.toSec();
        odomPosesCloud->points.emplace_back(odompose);

        sensor_msgs::PointCloud2 cloudmsg;
        pcl::toROSMsg(*odomPosesCloud, cloudmsg);
        cloudmsg.header.stamp = featureCloudinfo.header.stamp;
        cloudmsg.header.frame_id = "/odom";
        pubOdomPose.publish(cloudmsg);
    }

    void pubClouds(){

        poseEstimator->deriveMap(localmap_corner, localmap_surf);

        sensor_msgs::PointCloud2 pointCloud2msg;
        pcl::toROSMsg(*localmap_surf, pointCloud2msg);
        pointCloud2msg.header.stamp = featureCloudinfo.header.stamp;
        pointCloud2msg.header.frame_id = "/odom";
        pubLocalMapSurf.publish(pointCloud2msg);

        pcl::toROSMsg(*localmap_corner, pointCloud2msg);
        pointCloud2msg.header.stamp = featureCloudinfo.header.stamp;
        pointCloud2msg.header.frame_id = "/odom";
        pubLocalMapCorner.publish(pointCloud2msg);

//        poseEstimator->deriveSelectedSurf(curCloud_surf);  // fixme
        pcl::toROSMsg(*curCloud_surf, pointCloud2msg);
        featureCloudinfo.cloud_surface = pointCloud2msg;
        pubOdomFeatureCloudinfo.publish(featureCloudinfo);

        if(odom_localize_mode){
            pcl::toROSMsg(*localmap_surf, pointCloud2msg);
            pointCloud2msg.header.stamp = featureCloudinfo.header.stamp;
            pointCloud2msg.header.frame_id = "/odom";
            pubLocalMapAll.publish(pointCloud2msg);
        }

    }


    bool prepareCurCloud(){

        downSampler.setLeafSize(odom_ds_leafsize, odom_ds_leafsize, odom_ds_leafsize);
        downSampler.setInputCloud(curCloud_corner);
        downSampler.filter(*curCloud_corner);

//        downSampler.setLeafSize(2*odom_ds_leafsize, 2*odom_ds_leafsize, 2*odom_ds_leafsize);
        downSampler.setLeafSize(odom_ds_leafsize, odom_ds_leafsize, odom_ds_leafsize);
        downSampler.setInputCloud(curCloud_surf);
        downSampler.filter(*curCloud_surf);

        curSurfNum = curCloud_surf->points.size();
        curCornerNum = curCloud_corner->points.size();

        if(curCornerNum < 30 || curSurfNum < 100)
            return false;

        pcl::PointCloud<PointType>::Ptr tmp1(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr tmp2(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*curCloud_surf, *tmp1);
        pcl::copyPointCloud(*curCloud_corner, *tmp2);
        surfcloudsVec.emplace_back(tmp1);
        cornercloudsVec.emplace_back(tmp2);

        return true;
    }
    void run(){

        ROS_INFO("\033[1;32m---->\033[0m Odom Estimation Started.");
        ros::Rate loop_rate(10);

        while(ros::ok()){
            loop_rate.sleep();

            clock_t st, et;
            double ut;
            st = clock();

            if(!parseData())
                continue;
            if(!odom_init){
                initlocalMap();
                continue;
            }

            if (!prepareCurCloud()) continue;
            if(poseEstimator->ceresSolver(curCloud_corner, curCloud_surf))
                poseVars.emplace_back(poseEstimator->getCurPose(transformation));
            else continue;

            updateLocalMap();

            if(odom_localize_mode) buildGlobalMap();

            publishOdomInfo();

            pubClouds();

//            for (int i = 0; i < surfcloudsVec.size(); ++i) {
//                cout << BOLDRED << "surf clouds / " << surfcloudsVec[i]->points.size() ;
//                cout << BOLDRED << " , corner clouds / " << cornercloudsVec[i]->points.size() << RESET << endl;
//            }

            et = clock();
            ut = double(et - st) / CLOCKS_PER_SEC;
            ROS_INFO("time used is :%f s for OdomEstimation. \n", ut);
        }
        ROS_INFO("\033[1;32m---->\033[0m Odom Estimation thread Ended .");
    }
};

int main(int argc, char** argv){

    ros::init(argc, argv, "odom_estimation");
    OdomEstimation odometer;

    thread processor(&OdomEstimation::run, &odometer);

    ros::spin();

    processor.join();
    odometer.saveOctoMap("");
    cout << BLUE << "[ ROS ] odom_estimation node is done." << RESET << endl;

    return 0;
}