//
// Created by joe on 2020/10/4.
//

#include "tools.h"
#include "featureExtractor_livox.hpp"

//#include <livox_ros_driver/CustomMsg.h>

class FeatureExtraction{

private:

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subOutlierCloud;
    ros::Subscriber subImu;
    ros::Subscriber subLivoxCloud;
    ros::Subscriber subLineCloud;
    ros::Subscriber subEdgeCloud;

    ros::Publisher pubCornerPointsSharp;
    ros::Publisher pubCornerPointsLessSharp;
    ros::Publisher pubSurfPointsFlat;
    ros::Publisher pubSurfPointsLessFlat;
    ros::Publisher pubLivoxCloud;

    pcl::PointCloud<PointType>::Ptr segmentedCloud; // 从imageProjection得到的分割好的较大物体点云
    pcl::PointCloud<PointType>::Ptr outlierCloud; // 从imageProjection得到的散乱点云
    pcl::PointCloud<PointType>::Ptr livoxCloud; //
    pcl::PointCloud<PointType>::Ptr lineCloud; //

    // 特征点都来源于segmentedCloud
    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan; // 每一条扫面线中平面点
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatAll;

    pcl::VoxelGrid<PointType> downSizeFilter;

    double timeScanCur;   // 当前帧点云起始点的接收时间, 这四个时间是一样的
    double timeNewSegmentedCloud;
    double timeNewSegmentedCloudInfo;
    double timeNewOutlierCloud;
    double timeNewLivoxCloud;

    structural_mapping::cloud_info segInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float* cloudCurvature;
    int*  cloudNeighborPicked;
    int*  cloudLabel;

    bool isOutdoor;
    ros::Subscriber subGroundcloud;
    ros::Publisher pubLaserCloudground;
    ros::Publisher pubFeatureCloudInfo;
    double timeNewGroundcloud;
    pcl::PointCloud<PointType>::Ptr groundcloud;
    bool undistort = false;

    std::deque<sensor_msgs::PointCloud2ConstPtr> livoxCloudmsgQue;
    std::deque<sensor_msgs::PointCloud2ConstPtr> lineCloudmsgQue;
    std::deque<sensor_msgs::PointCloud2ConstPtr> edgeCloudmsgQue;
    std::deque<structural_mapping::cloud_infoConstPtr> segmentedCloudInfomsgQue;

    std::mutex locker;
    bool addlivoxFea = true;

public:

    FeatureExtraction():
            nh("~"){

        nh.param<float>  ("/featureExtraction/edgeThreshold", edgeThreshold, 0.1);
        nh.param<bool> ("/linefitting/filterEdge", filterEdge, false);

        std::cout << YELLOW << "[ feaExt ] edgeThreshold : " << edgeThreshold << RESET << endl;
        std::cout << YELLOW << "[ feaExt ] filterEdge : " << filterEdge << RESET << endl;

        // 订阅
        subLaserCloudInfo = nh.subscribe<structural_mapping::cloud_info>("/segmented_cloud_info", 1,
                                                                         &FeatureExtraction::laserCloudInfoHandler, this);
//        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 50, &FeatureExtraction::imuHandler, this);
        subGroundcloud = nh.subscribe<sensor_msgs::PointCloud2>("/ground_cloud", 1,
                                                                &FeatureExtraction::groundcloudHandler, this);
        subLivoxCloud = nh.subscribe<sensor_msgs::PointCloud2>("/livox/lidar", 1,
                                                               &FeatureExtraction::livoxCloudHandler, this);

        subLineCloud = nh.subscribe<sensor_msgs::PointCloud2>("/curLineCloud", 1,
                                                              &FeatureExtraction::lineCloudHandler, this);
        subEdgeCloud = nh.subscribe<sensor_msgs::PointCloud2>("/curEdgeCloud", 1,
                                                              &FeatureExtraction::edgeCloudHandler, this);
        // 发布
        pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
        pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
        pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);  // ground points
        pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);

        pubLaserCloudground = nh.advertise<sensor_msgs::PointCloud2>("/ground_cloud", 1);
        pubFeatureCloudInfo = nh.advertise<structural_mapping::cloud_info>("/feature_cloud_info", 1);

        pubLivoxCloud = nh.advertise<sensor_msgs::PointCloud2>("/livox_cloud", 1);

        initializationValue();
    }

    void initializationValue() {

        cloudSmoothness.resize(N_SCAN * Horizon_SCAN);

        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);

        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        groundcloud.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());
        livoxCloud.reset(new pcl::PointCloud<PointType>());
        lineCloud.reset(new pcl::PointCloud<PointType>());

        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

        surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlatAll.reset(new pcl::PointCloud<PointType>());

        timeScanCur = 0;
        timeNewSegmentedCloud = 0;
        timeNewSegmentedCloudInfo = 0;
        timeNewOutlierCloud = 0;

        const int SIZE = N_SCAN * Horizon_SCAN;
        cloudCurvature = new float [SIZE];
        cloudNeighborPicked = new int [SIZE];
        cloudLabel = new int [SIZE];
    }

    void groundcloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){

        timeNewGroundcloud = msgIn->header.stamp.toSec();

        groundcloud->clear();
        pcl::fromROSMsg(*msgIn, *groundcloud);
    }
    void laserCloudInfoHandler(const structural_mapping::cloud_infoConstPtr& msgIn) {

        locker.lock();
        segmentedCloudInfomsgQue.emplace_back(msgIn);
        locker.unlock();
    }
    void livoxCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){

        livoxCloudmsgQue.emplace_back(msgIn);
        timeNewLivoxCloud = msgIn->header.stamp.toSec();
//        double diff = ros::Time::now().toSec() - timeNewLivoxCloud;
//        cout << BOLDRED << "[!!!!!!!!!!] " <<  to_string(timeNewLivoxCloud) << endl;  // start from 0
//        cout << BOLDRED << "[!!!!!!!!!!] " <<  to_string(ros::Time::now().toSec()) << endl;  // start from 0
//        cout << BOLDRED << "[!!!!!!!!!!] " <<  to_string(diff) << endl;  // start from 0
        livoxCloud->clear();
        pcl::fromROSMsg(*msgIn, *livoxCloud);

        if (!livoxCloud->empty())
            FeatureExtractorLivox::extract(*livoxCloud, cornerPointsLessSharp, surfPointsLessFlat);
//        publishCloud();
    }
    void livoxCustomCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){

        livoxCloudmsgQue.emplace_back(msgIn);
        timeNewLivoxCloud = msgIn->header.stamp.toSec();
//        double diff = ros::Time::now().toSec() - timeNewLivoxCloud;
//        cout << BOLDRED << "[!!!!!!!!!!] " <<  to_string(timeNewLivoxCloud) << endl;  // start from 0
//        cout << BOLDRED << "[!!!!!!!!!!] " <<  to_string(ros::Time::now().toSec()) << endl;  // start from 0
//        cout << BOLDRED << "[!!!!!!!!!!] " <<  to_string(diff) << endl;  // start from 0
        livoxCloud->clear();
        pcl::fromROSMsg(*msgIn, *livoxCloud);
    }
    void lineCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){

        std::lock_guard<mutex> lockGuard(locker);
        lineCloudmsgQue.emplace_back(msgIn);
    }
    void edgeCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){

        std::lock_guard<mutex> lockGuard(locker);
        edgeCloudmsgQue.emplace_back(msgIn);
    }

    bool syncData(){

        {
            std::lock_guard<mutex> lockGuard(locker);

            if(segmentedCloudInfomsgQue.empty())
                return false;
            segInfo = *segmentedCloudInfomsgQue.front();
            segmentedCloudInfomsgQue.pop_front();
        }

        cloudHeader = segInfo.header;
        pcl::fromROSMsg(segInfo.segmentedCloud, *segmentedCloud);
        pcl::fromROSMsg(segInfo.cloud_outlier, *outlierCloud);

        return true;
    }

    //1 mark the relative time of points within a scan by intensity label
    void adjustDistortion(){

        bool halfPassed = false;
        int cloudSize = segmentedCloud->points.size();
        cout << GREEN << "Segmented Cloud : " << cloudSize << RESET << endl;
        float oridiff, ori, relTime;
        PointType point;

        for (int i = 0; i < cloudSize; i++){
            point = segmentedCloud->points[i];
            // 该点的方位角 horizontal = -atan(y,x)
            ori = atan2(point.y, point.x);
            oridiff = ori - segInfo.startOrientation;
            if(oridiff < 0)
                oridiff = -oridiff;
            else
                oridiff = 2*M_PI - oridiff;
            // 时间比例
            relTime = oridiff / segInfo.orientationDiff;
            // 点强度 = 扫描线号 + 在单根扫描线中的相对时间(0, 0.1)
            point.intensity = int(segmentedCloud->points[i].intensity) + scanPeriod * relTime;

            segmentedCloud->points[i] = point;
        }

    }

    // 2
    void calculateSmoothness()
    {
        int cloudSize = segmentedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++){

            // 利用前后十个点的range差异计算curvature
            float diffRange = segInfo.segmentedCloudRange[i-5] + segInfo.segmentedCloudRange[i-4]
                              + segInfo.segmentedCloudRange[i-3] + segInfo.segmentedCloudRange[i-2]
                              + segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i] * 10
                              + segInfo.segmentedCloudRange[i+1] + segInfo.segmentedCloudRange[i+2]
                              + segInfo.segmentedCloudRange[i+3] + segInfo.segmentedCloudRange[i+4]
                              + segInfo.segmentedCloudRange[i+5];

            cloudCurvature[i] = diffRange*diffRange;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;

            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    // 3
    void markOccludedPoints()
    {
        int cloudSize = segmentedCloud->points.size();

        for (int i = 5; i < cloudSize - 6; ++i)
        {

            float depth1 = segInfo.segmentedCloudRange[i];
            float depth2 = segInfo.segmentedCloudRange[i+1];
            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[i+1] - segInfo.segmentedCloudColInd[i]));

            if (columnDiff < 10){

                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }

            float diff1 = std::abs(segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i]);
            float diff2 = std::abs(segInfo.segmentedCloudRange[i+1] - segInfo.segmentedCloudRange[i]);

            if (diff1 > 0.02 * segInfo.segmentedCloudRange[i] && diff2 > 0.02 * segInfo.segmentedCloudRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    // 4-from segmentedCloud
    void extractFeatures(){

        cornerPointsSharp->clear();
        cornerPointsLessSharp->clear();
        surfPointsFlat->clear();
        surfPointsLessFlat->clear();

        surfPointsLessFlatAll->clear();

        // 遍历每一根扫面线
        for (int i = 0; i < N_SCAN; i++){

            surfPointsLessFlatScan->clear();

            // divide each row into 6 blocks(sub-images)
            for (int j = 0; j < 6; j++){

                int sp = (segInfo.startRingIndex[i] * (6 - j)    + segInfo.endRingIndex[i] * j) / 6;
                int ep = (segInfo.startRingIndex[i] * (5 - j)    + segInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

//                cout << YELLOW << "from " << sp << " to " << ep << RESET << endl;
                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value()); // ascend order

                std::ofstream txtor1("/home/joe/workspace/testData/timing/sharpcurvature.txt", std::ios::app);
                ///1. extract cornerPoints
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--){

                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] > edgeThreshold &&
                        segInfo.segmentedCloudGroundFlag[ind] == false){

                        largestPickedNum++;
                        if (largestPickedNum <= 2) {
                            cloudLabel[ind] = 2;   // sharp
                            cornerPointsSharp->push_back(segmentedCloud->points[ind]);
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);

                        } else if (largestPickedNum <= 20) {
                            cloudLabel[ind] = 1;   // less sharp (include sharp)
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);

                        } else   // largestPickedNum >20
                            break;

//                        txtor1<<cloudCurvature[ind]<<"\n";

                        // 标记当前ind前后各5个点
                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                txtor1.close();

                ///2. extract surfPointsFlat(only from ground points)
                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 &&
                        cloudCurvature[ind] < surfThreshold &&
                        segInfo.segmentedCloudGroundFlag[ind] == true){

                        cloudLabel[ind] = -1;   // on the ground
                        surfPointsFlat->push_back(segmentedCloud->points[ind]);

                        smallestPickedNum++;
//                        if (smallestPickedNum >= 8)  //TODO make sure there is at most 4 points
//                            break;

                        // same as before
                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++) {

                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {

                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                std::ofstream txtor("/home/joe/workspace/testData/timing/surfcurvature.txt",std::ios::app);

                // 剩下的点全部作为less flat (include ground points)
                for (int k = sp; k <= ep; k++)
                    if (cloudLabel[k] <= 0){
                        surfPointsLessFlatScan->push_back(segmentedCloud->points[k]);

//                        int ind = cloudSmoothness[k].ind;
//                        txtor<< cloudCurvature[ind]<<"\n"; // for curvature test
                    }
                txtor.close();

            }
            *surfPointsLessFlatAll += *surfPointsLessFlatScan;
            //  对整根扫描线中less flat的点进行进行抽稀
            surfPointsLessFlatScanDS->clear();
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.filter(*surfPointsLessFlatScanDS);


//            *surfPointsLessFlat += *surfPointsLessFlatScan;
            *surfPointsLessFlat += *surfPointsLessFlatScanDS;
        }
        cout << GREEN << "[ feature ] Features: surf / " << surfPointsLessFlat->points.size() <<
             " ; corner / " << cornerPointsLessSharp->points.size() << RESET << endl;

    }

    // 5-publish corner & surf Points (4 kinds)
    void publishCloud(){

        sensor_msgs::PointCloud2 laserCloudOutMsg;

        if (pubCornerPointsSharp.getNumSubscribers() != 0){
            pcl::toROSMsg(*cornerPointsSharp, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = cloudHeader.stamp;
            laserCloudOutMsg.header.frame_id = "/base_link";
            pubCornerPointsSharp.publish(laserCloudOutMsg);
        }
        if (pubSurfPointsFlat.getNumSubscribers() != 0){
            pcl::toROSMsg(*surfPointsFlat, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = cloudHeader.stamp;
            laserCloudOutMsg.header.frame_id = "/base_link";
            pubSurfPointsFlat.publish(laserCloudOutMsg);
        }

        if (!filterEdge){
            pcl::toROSMsg(*cornerPointsLessSharp, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = cloudHeader.stamp;
            laserCloudOutMsg.header.frame_id = "/base_link";
            segInfo.cloud_corner = laserCloudOutMsg;
            pubCornerPointsLessSharp.publish(laserCloudOutMsg);
        }

        pcl::toROSMsg(*surfPointsLessFlat, laserCloudOutMsg);
//        pcl::toROSMsg(*surfPointsFlat, laserCloudOutMsg);
        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
        laserCloudOutMsg.header.frame_id = "/base_link";
        segInfo.cloud_surface = laserCloudOutMsg;
        pubSurfPointsLessFlat.publish(laserCloudOutMsg);

        pcl::toROSMsg(*segmentedCloud, laserCloudOutMsg);
        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
        laserCloudOutMsg.header.frame_id = "/base_link";
        segInfo.segmentedCloud = laserCloudOutMsg;

        pubFeatureCloudInfo.publish(segInfo);

        if(pubLivoxCloud.getNumSubscribers()){
            pcl::toROSMsg(*livoxCloud, laserCloudOutMsg);
            laserCloudOutMsg.header.stamp = ros::Time().fromSec(timeNewLivoxCloud);
            laserCloudOutMsg.header.frame_id = "/base_link";
            pubLivoxCloud.publish(laserCloudOutMsg);
        }

        // ground cloud
        if(abs(timeNewGroundcloud - timeNewSegmentedCloud) < 0.05){

            sensor_msgs::PointCloud2 laserCloudgroundLast2;
            pcl::toROSMsg(*groundcloud, laserCloudgroundLast2);
            laserCloudgroundLast2.header.stamp = cloudHeader.stamp;
            laserCloudgroundLast2.header.frame_id = "/camera";
            pubLaserCloudground.publish(laserCloudgroundLast2);
        }

        cornerPointsSharp->clear();
        cornerPointsLessSharp->clear();
        surfPointsFlat->clear();
        surfPointsLessFlat->clear();
    }

    // dispose feature cloud from lineFitting node
    bool calcuRelativeTimeInScan(){

        sensor_msgs::PointCloud2 lineMsg, edgeMsg;
        double timeCur = cloudHeader.stamp.toSec();

        {
            std::lock_guard<mutex> lockGuard(locker);
            while (!lineCloudmsgQue.empty() &&
                   timeCur - lineCloudmsgQue.front()->header.stamp.toSec() > 0.05 )
                lineCloudmsgQue.pop_front();

            while ( !edgeCloudmsgQue.empty() &&
                    timeCur - edgeCloudmsgQue.front()->header.stamp.toSec() > 0.05){
                edgeCloudmsgQue.pop_front();
            }

            if(!lineCloudmsgQue.empty()) lineMsg = *lineCloudmsgQue.front();
            if(!edgeCloudmsgQue.empty()) edgeMsg = *edgeCloudmsgQue.front();
            else return false;
        }

        if(abs(timeCur - lineMsg.header.stamp.toSec()) < 0.05){

            pcl::fromROSMsg(lineMsg, *lineCloud);

            int cloudSize = lineCloud->points.size();
            cout << GREEN << "[ feat ] Syned Line Cloud : " << cloudSize << RESET << endl;
            float oridiff, ori, relTime;
            PointType point;

            for (int i = 0; i < cloudSize; i++){

                point = lineCloud->points[i];
                // 该点的方位角 horizontal = -atan(y,x)
                ori = atan2(point.y, point.x);
                oridiff = ori - segInfo.startOrientation;
                if(oridiff < 0)
                    oridiff = -oridiff;
                else
                    oridiff = 2*M_PI - oridiff;
                // 时间比例
                relTime = oridiff / segInfo.orientationDiff;
                // 点强度 = 扫描线号 + 在单根扫描线中的相对时间(0, 0.1)
                point.intensity = int(lineCloud->points[i].intensity) + scanPeriod * relTime;

                lineCloud->points[i] = point;
            }

            pcl::toROSMsg(*lineCloud, lineMsg);
            lineMsg.header = segInfo.header;
            segInfo.cloud_lines = lineMsg;
        }

        if(filterEdge && abs(timeCur - edgeMsg.header.stamp.toSec()) < 0.05){

            pcl::fromROSMsg(edgeMsg, *lineCloud);

            int cloudSize = lineCloud->points.size();
            cout << GREEN << "[ feat ] Syned Filtered Edge Cloud : " << cloudSize << RESET << endl;
            float oridiff, ori, relTime;
            PointType point;

            for (int i = 0; i < cloudSize; i++){

                point = lineCloud->points[i];
                // 该点的方位角 horizontal = -atan(y,x)
                ori = atan2(point.y, point.x);
                oridiff = ori - segInfo.startOrientation;
                if(oridiff < 0)
                    oridiff = -oridiff;
                else
                    oridiff = 2*M_PI - oridiff;
                // 时间比例
                relTime = oridiff / segInfo.orientationDiff;
                // 点强度 = 扫描线号 + 在单根扫描线中的相对时间(0, 0.1)
                point.intensity = int(lineCloud->points[i].intensity) + scanPeriod * relTime;

                lineCloud->points[i] = point;
            }

            pcl::toROSMsg(*lineCloud, edgeMsg);
            edgeMsg.header = segInfo.header;
            segInfo.cloud_corner = edgeMsg;
        }
        return true;
    }

    double rad2deg(double radians) {
        return radians * 180.0 / M_PI;
    }
    double deg2rad(double degrees) {
        return degrees * M_PI / 180.0;
    }

    void run(){

        ROS_INFO("\033[1;32m---->\033[0m Feature Extraction Started.");
        ros::Rate loop_rate(10);

        while(ros::ok()) {

            loop_rate.sleep();
            clock_t st, et;
            double ut;
            st = clock();

            if (!syncData())
                continue;

            // 运动畸变纠正
            adjustDistortion();

            if(!filterEdge){
                // 特征提取
                calculateSmoothness();  // for VLP
                markOccludedPoints();
                extractFeatures();
            }

            while (1)  // waiting
                if(calcuRelativeTimeInScan()) break;

            publishCloud();

            et = clock();
            ut = double(et - st) / CLOCKS_PER_SEC;
            ROS_INFO("time used is :%f s for Feature Extraction\n", ut);
        }
    }

};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "feature_extraction");

    FeatureExtraction featureExtraction;

    std::thread processor(&FeatureExtraction::run, &featureExtraction);

    ros::spin();
    processor.join();
    cout << BLUE << "[ ROS ] featureExtraction node is done." << RESET << endl;

    return 0;
}