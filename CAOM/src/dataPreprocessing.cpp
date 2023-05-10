//
// Created by joe on 2020/10/3.
//

#include "tools.h"

class DataPreprocessing{

    std::mutex locker;

    ros::NodeHandle nh;

    ros::Subscriber subCloud;

    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubNonGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointTypeRing>::Ptr laserCloudringIn;  // cloud with ring info
    deque<sensor_msgs::PointCloud2ConstPtr> laserCloudmsgQue;

    pcl::PointCloud<PointType>::Ptr fullCloud;   // 距离图像对应的点云, intensity=row + (col/10000)
    pcl::PointCloud<PointType>::Ptr fullInfoCloud;  // 投影得到的距离图像 16*1800（index为行列号，无坐标值，intensity为range）

    pcl::PointCloud<PointType>::Ptr groundCloud; // 地面点, intensity为点的ind
    pcl::PointCloud<PointType>::Ptr nongroundCloud; // 非地面点
    pcl::PointCloud<PointType>::Ptr segmentedCloud;  // 代表较大物体的点，其中地面点抽稀过,与fullCloud点类型相同
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;  // 所有代表较大物体的点（不含地面点）,intensity为标签
    pcl::PointCloud<PointType>::Ptr outlierCloud;  // 地面点中因抽稀而滤掉的点以及散乱点

    ros::Publisher pubCloudaboveGround;
    pcl::PointCloud<PointType>::Ptr cloudaboveGround;  //位于地面上的点，不同于非地面点
    double *maxGroundrangeOfcol;  // 每一列中地面点的最远range

    PointType nanPoint;

    cv::Mat rangeMat;  // 距离图像对应的距离矩阵
    cv::Mat labelMat;  // 距离图像对应的点的label, -1为地面点
    cv::Mat groundMat; // 距离图像对应的点是否为地面点 1则为地面点
    int labelCount;

    float startOrientation;
    float endOrientation;

    structural_mapping::cloud_info segMsg;  // 分割点云的信息
    std_msgs::Header cloudHeader;

    std::vector<std::pair<uint8_t, uint8_t> > neighborIterator;

    uint16_t *allPushedIndX;
    uint16_t *allPushedIndY;

    uint16_t *queueIndX;
    uint16_t *queueIndY;

    bool isOutdoor;
//    bool labelGround;
//    string cloudTopicName;


public:
    DataPreprocessing():nh("~"){

        nh.param<string>("/dataPreprocessing/cloudTopicName", cloudTopicName, "/velodyne_points");
        nh.param<bool>  ("/dataPreprocessing/labelGround", labelGround, 1);
        nh.param<bool>  ("/dataPreprocessing/segmentCloud", segmentCloud, 1);

        std::cout << YELLOW << "[pre] cloudTopicName : " << cloudTopicName << RESET << endl;
        std::cout << YELLOW << "[pre] labelGround : " << labelGround << RESET << endl;
        std::cout << YELLOW << "[pre] segmentCloud : " << segmentCloud << RESET << endl;
        std::cout << YELLOW << "[pre] saveImg : " << saveImg << RESET << endl;

        subCloud = nh.subscribe<sensor_msgs::PointCloud2>(cloudTopicName, 1,
                                                          &DataPreprocessing::cloudHandler, this);

        //发布
        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);
        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<structural_mapping::cloud_info> ("/segmented_cloud_info", 1);
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);

        pubCloudaboveGround = nh.advertise<sensor_msgs::PointCloud2> ("/cloudaboveGround", 1);
        pubNonGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/nonGroundCloud", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();
        resetParameters();
    }

    void allocateMemory(){

        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudringIn.reset(new pcl::PointCloud<PointTypeRing>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        nongroundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        cloudaboveGround.reset(new pcl::PointCloud<PointType>());
        maxGroundrangeOfcol = new double[Horizon_SCAN];//初始化为0


        fullCloud->points.resize(N_SCAN*Horizon_SCAN);
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);
        segMsg.endRingIndex.assign(N_SCAN, 0);

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false);
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        std::pair<int8_t, int8_t> neighbor;
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }
    void resetParameters(){

        laserCloudIn->clear();
        groundCloud->clear();
        nongroundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        cloudaboveGround->clear();
        maxGroundrangeOfcol = new double[Horizon_SCAN];

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& cloudmsg) {

        std::cerr << "+ New cloud msg." << endl;
//        cout << BOLDRED << "[!!!!!!!!!!] VLP " <<  to_string(cloudmsg->header.stamp.toSec()) << endl;

        std::lock_guard<mutex> lg(locker);
        laserCloudmsgQue.emplace_back(cloudmsg);
    }

    void run(){

        ROS_INFO("\033[1;32m---->\033[0m Running Data Preprocessing Node..");
        ros::Rate loop_rate(10);
        while(ros::ok()){

            loop_rate.sleep();

            clock_t st, et;
            double ut;
            st = clock();

            if(laserCloudmsgQue.empty())
                continue;

            locker.lock();
            cloudHeader = laserCloudmsgQue.front()->header;
            pcl::fromROSMsg(*laserCloudmsgQue.front(), *laserCloudIn);
            if (useRingInfo)
                pcl::fromROSMsg(*laserCloudmsgQue.front(), *laserCloudringIn);

            laserCloudmsgQue.pop_front();
            locker.unlock();

            findStartEndAngle();

            projectPointCloud();
//            projectPointCloud_ouster();
            if(saveImg) showScanlineRangeDistriRand();

            if(labelGround)
                groundRemoval();

            cloudSegmentation();

            publishCloud();

            resetParameters();

            et = clock();
            ut = double(et - st)/CLOCKS_PER_SEC;
            ROS_INFO("time used is : %f s for imageProjection.\n",ut);
        }
    }

    //2
    void findStartEndAngle(){
        // atan2 E [-pi, pi]
        segMsg.startOrientation = atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
        segMsg.endOrientation   = atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                        laserCloudIn->points[laserCloudIn->points.size() - 1].x);
        // VLP16 rotate clockwisely
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
        segMsg.orientationDiff = -segMsg.orientationDiff + 2 * M_PI;

        cout << MAGENTA << "[ dataPre ] Scan start ori : " << segMsg.startOrientation <<
             " / end ori : " << segMsg.endOrientation <<
             " / total range : " << segMsg.orientationDiff << RESET << endl;
    }

    //3
    void projectPointCloud(){

        int longrangeCounts = 0; // 用来判断室内室外
        int cloudSize = laserCloudIn->points.size();

#pragma omp parallel for
        for (size_t i = 0; i < cloudSize; ++i){

            float verticalAngle, horizonAngle, range;
            size_t rowIdn, columnIdn, index;
            PointType thisPoint;

            laserCloudIn->points[i].z = laserCloudIn->points[i].z - dataDriftZ;  // fixme
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;

            if (useRingInfo)
                rowIdn = laserCloudringIn->points[i].ring;
            else {
                verticalAngle =
                        atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
            }
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            columnIdn = -round( (horizonAngle-90.0)/ang_res_x ) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

            index = columnIdn  + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;

            fullInfoCloud->points[index].intensity = range;
            fullInfoCloud->points[index].x = i;  // 将该点在原始点云中的indice记录下来

            // 根据range判断室内外场景
            if(range > 60)
                longrangeCounts++;
        }

        cout<<"[ dataPre ] Long range measurements : "<<longrangeCounts<<endl;
//        if( (longrangeCounts*1.0)/cloudSize < 0.1)//百分比
        if( longrangeCounts < 20){
            cout<<"@@@   indoor  --------"<<endl;
            isOutdoor = false;
        }else
            cout<<"@@@   Outdoor ++++++++"<<endl;

        //cyz>> save depth images
//        string depthimgfolder = "/home/cyz/Data/legoloam/poses/depthimgs/";
//        cv::imwrite(depthimgfolder+to_string(cloudHeader.stamp.toSec())+".jpg",rangeMat);
    }
    /// for ouster LiDAR
    void projectPointCloud_ouster(){

        int longrangeCounts = 0; // 用来判断室内室外
        int cloudSize = laserCloudIn->points.size();

#pragma omp parallel for
        for(int i = 0; i < N_SCAN; i++)
            for (int columnIdn = 0; columnIdn < Horizon_SCAN; ++columnIdn) {

                float range;
                size_t rowIdn, index, ind2;
                PointType thisPoint;

                rowIdn = i;
                index = rowIdn*Horizon_SCAN + (columnIdn + Horizon_SCAN - int(ang_y_Vec[rowIdn])) % Horizon_SCAN;
                if(index > cloudSize || index < 0) continue;

//                laserCloudIn->points[index].z = laserCloudIn->points[index].z + dataDriftZ;
                thisPoint.x = laserCloudIn->points[index].x;
                thisPoint.y = laserCloudIn->points[index].y;
                thisPoint.z = laserCloudIn->points[index].z;

                range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
                rangeMat.at<float>(rowIdn, columnIdn) = range;

                thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

                ind2 = columnIdn  + rowIdn * Horizon_SCAN;
                fullCloud->points[ind2] = thisPoint;
                fullInfoCloud->points[ind2].intensity = range;
                fullInfoCloud->points[ind2].x = index;  // 将该点在原始点云中的indice记录下来

                // 根据range判断室内外场景
                if(range > 60)
                    longrangeCounts++;
            }

        cout<<"[ dataPre ] Long range measurements : "<<longrangeCounts << endl;
//        if( (longrangeCounts*1.0)/cloudSize < 0.1)//百分比
        if( longrangeCounts < 20){
            cout<<"@@@   indoor  --------"<<endl;
            isOutdoor = false;
        }else
            cout<<"@@@   Outdoor ++++++++"<<endl;

        //cyz>> save depth images
//        string depthimgfolder = "/home/cyz/Data/legoloam/poses/depthimgs/";
//        cv::imwrite(depthimgfolder+to_string(cloudHeader.stamp.toSec())+".jpg",rangeMat);
    }

    void showScanlineRangeDistriRand(){

        int cloudsize, ind = 0;
        int n = rand() % N_SCAN ;  // random select one ring
        float r, avg = 0;
        std::string filename = "/home/joe/workspace/catkin_ws/src/structuralMapping_ROS/data/" +
                               to_string(cloudHeader.stamp.toSec()) ;
        std::ofstream writer(filename + ".txt");

        pcl::PointCloud<PointType>::Ptr lineCloud(new pcl::PointCloud<PointType>);
        for (int i = 0; i < Horizon_SCAN; ++i) {

            ind = i + n*Horizon_SCAN;
            r = rangeMat.at<float>(n, i);
            if(r > 100)
                continue;
            writer << r << "\n";
            avg += r;
            lineCloud->points.emplace_back(fullCloud->points[ind]);
            lineCloud->points.back().intensity = i;
        }
        writer.close();

        cloudsize = lineCloud->points.size();
        if(!cloudsize)
            return;
        pcl::io::savePCDFileBinary(filename + ".pcd", *lineCloud);

        avg /= cloudsize; // average range for the points in this ring
        float res = (avg * ang_res_x*PI/180.0);
        avg *= 1.5;  // larger
        int grids = 2* (int)(avg / res);
        cout << BLUE << "Scanline " << n <<
             " Image info : resolution / " << res <<", size / " << grids << RESET << endl;

        // grey image
        cv::Mat scanlineImg(grids, grids, CV_8UC1, cv::Scalar::all(0));
#pragma omp parallel for
        for (int j = 0; j < cloudsize; ++j) {
            int x = round(lineCloud->points[j].x / res);
            int y = round(lineCloud->points[j].y / res);
            if(abs(x) > grids/2 || abs(y) > grids/2)
                continue;
            scanlineImg.at<uchar >(grids/2 - y, grids/2 + x) = 255;
        }
//        cv::imshow(filename, scanlineImg);
        if(!cv::imwrite(filename+".jpg", scanlineImg))  // check
            return;
//        cv::imshow(filename, rangeMat);
//        cv::waitKey(50);
    }

    // 4
    void groundRemoval(){

        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;

        for (size_t j = 0; j < Horizon_SCAN; ++j)
            for (size_t i = 0; i < groundScanInd; ++i)  // row
            {

                lowerInd = j + ( i )*Horizon_SCAN; //indice
                upperInd = j + (i+1)*Horizon_SCAN;

                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1)  // which is initialized as nanPoint
                {
                    groundMat.at<int8_t>(i,j) = -1;
                    continue;
                }

                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                if (abs(angle - sensorMountAngle) <= 10){
                    groundMat.at<int8_t>(i,j) = 1;  // 地面点标记为1
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }

        // for dynaObj
        // 地面点和无效range点标签-1
        // 计算每一列地面点的最大range
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j){

                if( groundMat.at<int8_t>(i,j) == 1 ){
                    labelMat.at<int>(i,j) = -1;
                    if(rangeMat.at<float>(i,j) > maxGroundrangeOfcol[j])
                        maxGroundrangeOfcol[j] = rangeMat.at<float>(i,j);
                }
                if (rangeMat.at<float>(i,j) == FLT_MAX)
                    labelMat.at<int>(i,j) = -1;
            }

        // 订阅激活
        if (pubGroundCloud.getNumSubscribers() != 0)
        {
            for (size_t i = 0; i < N_SCAN; ++i)
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (i <= groundScanInd && groundMat.at<int8_t>(i,j) == 1){
                        groundCloud->push_back(fullCloud->points[j+i*Horizon_SCAN]);
                        groundCloud->back().intensity = fullInfoCloud->points[j+i*Horizon_SCAN].x;
                    } else
                        nongroundCloud->points.emplace_back(fullCloud->points[j+i*Horizon_SCAN]);
                }

//            pcl::io::savePCDFileBinary("/home/joe/workspace/testData/groundcloud/"
//                                       +to_string(cloudHeader.stamp.toSec())+".pcd", *groundCloud);
        }

    }

    // 5
    void cloudSegmentation(){

        if(segmentCloud && labelGround) {

            for (size_t i = 0; i < N_SCAN; ++i)
                for (size_t j = 0; j < Horizon_SCAN; ++j) {
                    if (rangeMat.at<float>(i, j) == FLT_MAX)
                        continue;
                    if (labelMat.at<int>(i, j) == 0)  // 所有非地面点
                        labelComponents(i, j);
                }

            int sizeOfSegCloud = 0;
            for (size_t i = 0; i < N_SCAN; ++i) {
                segMsg.startRingIndex[i] = sizeOfSegCloud - 1 + 5;

                for (size_t j = 0; j < Horizon_SCAN; ++j) {
                    if (labelMat.at<int>(i, j) > 0 || groundMat.at<int8_t>(i, j) == 1) {

                        if (labelMat.at<int>(i, j) == 999999) {  // 杂乱点
//                        if (i > groundScanInd && j % 5 == 0){   // 行号>groundScanInd且列号为5的倍数的点加入outlier
                            outlierCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                            continue;
                        }

                        if (groundMat.at<int8_t>(i, j) == 1)
                            if (j % 5 != 0 && j > 5 && j < Horizon_SCAN - 5)  // 5的倍数序号的地面点加入segmentedCloud
                                continue;

                        //labelMat中有效标记点(segments)加入segmentedCloud
                        segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i, j) == 1);
                        segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                        segMsg.segmentedCloudRange[sizeOfSegCloud] = rangeMat.at<float>(i, j);
                        segmentedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
//                    segmentedCloud->points.back().intensity = labelMat.at<int>(i,j);//cyz

                        ++sizeOfSegCloud;
                    }
                }
                segMsg.endRingIndex[i] = sizeOfSegCloud - 1 - 5;
            }

            // for dynaObjModule
            for (size_t i = 0; i < N_SCAN; ++i)
                for (size_t j = 0; j < Horizon_SCAN; ++j) {
                    if (labelMat.at<int>(i, j) > 0 && labelMat.at<int>(i, j) != 999999) {
                        segmentedCloudPure->push_back(fullCloud->points[j + i * Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i, j);

                        if (groundMat.at<int8_t>(i, j) != 1 && rangeMat.at<float>(i, j) < maxGroundrangeOfcol[j]) {
                            cloudaboveGround->points.push_back(
                                    laserCloudIn->points[fullInfoCloud->points[j + i * Horizon_SCAN].x]);
                        }
                    }
                }
        }else{  // dont segment cloud

            int sizeOfSegCloud = 0;
            for (size_t i = 0; i < N_SCAN; ++i) {
                segMsg.startRingIndex[i] = sizeOfSegCloud - 1 + 5;

                for (size_t j = 0; j < Horizon_SCAN; ++j) {

                    if (rangeMat.at<float>(i, j) == FLT_MAX)
                        continue;

                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    segMsg.segmentedCloudRange[sizeOfSegCloud] = rangeMat.at<float>(i, j);
                    segmentedCloud->push_back(fullCloud->points[j + i * Horizon_SCAN]);
//                    segmentedCloud->points.back().intensity = labelMat.at<int>(i,j);  //cyz

                    ++sizeOfSegCloud;
                }
                segMsg.endRingIndex[i] = sizeOfSegCloud - 1 - 5;
            }
        }
    }
    void labelComponents(int row, int col){

        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY;
        bool lineCountFlag[N_SCAN] = {false};

        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;

        while(queueSize > 0) {
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;

            //neighborIterator 上下左右四个邻域
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){

                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;

                if (thisIndX < 0 || thisIndX >= N_SCAN)//row出界
                    continue;

                if (thisIndY < 0)//begin col of row
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)//end col of row
                    thisIndY = 0;

                if (labelMat.at<int>(thisIndX, thisIndY) != 0)//已标记
                    continue;

                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),
                              rangeMat.at<float>(thisIndX, thisIndY));

                if ((*iter).first == 0)
                    alpha = segmentAlphaX;//0.2°→rad
                else
                    alpha = segmentAlphaY;//2°→rad

                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                if (angle > segmentTheta)  // 同一类生长
                {

                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;
                }
            }
        }


        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)
            feasibleSegment = true;
        else if (allPushedIndSize >= segmentValidPointNum)//点数大于5且占据连续三行row以上
        {
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i])
                    ++lineCount;
            if (lineCount >= segmentValidLineNum)
                feasibleSegment = true;
        }

        if (feasibleSegment){
            ++labelCount;
        }else{
            for (size_t i = 0; i < allPushedIndSize; ++i)
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;//无效标记
        }
    }

    void publishCloud(){

        sensor_msgs::PointCloud2 laserCloudTemp;
        if (outlierCloud->points.size() > 0){
            pcl::toROSMsg(*outlierCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            laserCloudTemp.is_dense = isOutdoor;  //
            segMsg.cloud_outlier = laserCloudTemp;
            pubOutlierCloud.publish(laserCloudTemp);
        }

        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        segMsg.segmentedCloud = laserCloudTemp;
        pubSegmentedCloud.publish(laserCloudTemp);

        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);

        // for dynaObjModule
        if (pubCloudaboveGround.getNumSubscribers() != 0) {

            cout << "[ dataPre ] There are " << cloudaboveGround->points.size() << " points above ground." << endl;
            pcl::toROSMsg(*cloudaboveGround, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubCloudaboveGround.publish(laserCloudTemp);
        }

        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);

        }

        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);

        }

        if (pubNonGroundCloud.getNumSubscribers() != 0){

            pcl::toROSMsg(*nongroundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubNonGroundCloud.publish(laserCloudTemp);

        }

        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);

        }

        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);

        }
    }

};

int main(int argc, char** argv){

    ros::init(argc, argv, "data_preprocessing");

    DataPreprocessing dataPreprocessor;

    std::thread processor(&DataPreprocessing::run, &dataPreprocessor);

    ros::spin();
    processor.join();

//    while(ros::ok()){
////        ros::spinOnce();
//        dataPreprocessor.run();
//    }

    return 0;
}