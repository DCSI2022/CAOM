////////////////////////////////////////////////////////////////////////////////////////////////////
/// Created by joe on 2020/10/9.
///
/// 20220219:
/// extract structural features including edges
///
/// 20220311:
/// filtering features based on ring cluster (range image)
///
/// 20220328:
/// add ground extraction;
///
////////////////////////////////////////////////////////////////////////////////////////////////

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl/filters/radius_outlier_removal.h>

#include "tools.h"
#include "lineStructure.h"
#include "poseEstimationLib.hpp"

#include "pclomp/ndt_omp.h"
#include "pclomp/gicp_omp.h"

#define LinearityThre 20
#define PlanarityThre 0
#define NeighborSize 3
#define MIN_RANGE 0.5
#define MAX_RANGE 100
#define EDGE_LABLE 1
#define SURF_LABLE 2
#define GROUND_LABLE 3

class LineFitting{

    pcl::console::TicToc timer;

    ros::NodeHandle nh;
    ros::Subscriber subcloud;
    ros::Subscriber subframePose;
    ros::Subscriber subsegmentedInfo;

    ros::Publisher publineCloud, pubEdgeCloud, pubCurCloud, pubGroundcloud;
    ros::Publisher publineCloudglobal;
    ros::Publisher publineMsgs;
    image_transport::Publisher pubRangeImage;

    pcl::VoxelGrid<PointType> downSampler;

    deque<sensor_msgs::PointCloud2ConstPtr> cloudmsgQue;
    deque<nav_msgs::OdometryConstPtr> poseMsgQue;
    mutex mtx;

    pcl::PointCloud<PointTypePose>::Ptr framePosesCloud;
    pcl::PointCloud<PointType>::Ptr curCloud;
    pcl::PointCloud<PointType>::Ptr curCloudDS;
    pcl::PointCloud<PointTypeRing>::Ptr curCloudRing;
    pcl::PointCloud<PointType>::Ptr curCloudEdge;
    pcl::PointCloud<PointType>::Ptr curCloudRange;  // only save range, no point

    sensor_msgs::PointCloud2 curCloudmsg;
    pcl::PointCloud<PointType>::Ptr curLineCloud;
    pcl::PointCloud<PointType>::Ptr curGroundCloud;
    pcl::PointCloud<PointType>::Ptr globalLineCloud;
    vector<pcl::PointCloud<PointType>::Ptr > globalsubLineCloudVec;

    vector<pcl::PointCloud<PointType>::Ptr > lineCloudVec;
    vector<double > lineCloudTimeVec;  // cloud timestamp w.r.t lineCloudVec
    vector<vector<int> > scanlineIndices_;
    vector<pcl::PointCloud<PointType>::Ptr > scanlineCloudVec;

    string poseTopic;
    std::unique_ptr<PoseEstimationManager> poseEstimator;
    int maxLineCloudSize;
    double leafsize_perScanline, breakThre, clusterThre_row, clusterThre_col;

    structural_mapping::cloud_info segInfo;
    cv::Mat rangeMat, indMat, labelMat;
    std::vector<float> ang_y_ordered = ang_y_Vec;
    visualization_msgs::MarkerArray lineMsgs;

public:
    LineFitting():nh("~"){

        poseEstimator.reset(new PoseEstimationManager());
        // read parameters
        nh.param<string>("/linefitting/poseTopic", poseTopic, "/aft_mapped_to_init");
        nh.param<int>   ("/linefitting/maxLineCloudSize", maxLineCloudSize, 100000);
        nh.param<string>("/projpath", projPath, "/home/cyz/workspace/testData/");  // global param add '/' before name
        nh.param<double>("/linefitting/leafsize_perScanline", leafsize_perScanline, 0.1);
        nh.param<double>("/linefitting/breakThre", breakThre, 1.1);
        nh.param<double>("/linefitting/clusterThre_row", clusterThre_row, 0.2);
        nh.param<double>("/linefitting/clusterThre_col", clusterThre_col, 1.20);  // Fixme: too sensitive?
        nh.param<int>   ("/linefitting/minlinePtNum",  minlinePtNum, 6);
        nh.param<float> ("/linefitting/minlineLength", minlineLength, 0.5);
        nh.param<bool>  ("/linefitting/filterEdge",    filterEdge, true);
        // print parameters
        cout << "[ LINE ] projFolder : "           << projPath << endl;
        cout << "[ LINE ] poseTopic : "            << poseTopic << endl;
        cout << "[ LINE ] maxLineCloudSize : "     << maxLineCloudSize << endl;
        cout << "[ LINE ] leafsize_perScanline : " << leafsize_perScanline << endl;
        cout << "[ LINE ] minlinePtNum : "    << minlinePtNum << endl;
        cout << "[ LINE ] minlineLength : "   << minlineLength << endl;
        cout << "[ LINE ] filterEdge : "      << filterEdge << endl;
        cout << "[ LINE ] breakThre : "       << breakThre << endl;
        cout << "[ LINE ] clusterThre_row : " << clusterThre_row << endl;
        cout << "[ LINE ] clusterThre_col : " << clusterThre_col << endl;

        subcloud = nh.subscribe<sensor_msgs::PointCloud2>(cloudTopicName, 1, &LineFitting::cloudHandler, this);
        subframePose = nh.subscribe<nav_msgs::Odometry>(poseTopic, 1, &LineFitting::framePoseHandler, this);
//        subsegmentedInfo = nh.subscribe<structural_mapping::cloud_info>("/segmented_cloud_info", 1,
//                                                                        &LineFitting::segmentedCloudInfoHandler, this);

        publineCloud = nh.advertise<sensor_msgs::PointCloud2>("/curLineCloud", 1);
        pubGroundcloud = nh.advertise<sensor_msgs::PointCloud2>("/curGroundCloud", 1);
        pubEdgeCloud = nh.advertise<sensor_msgs::PointCloud2>("/curEdgeCloud", 1);
        pubCurCloud = nh.advertise<sensor_msgs::PointCloud2>("/curCloudAll", 1);
        publineCloudglobal = nh.advertise<sensor_msgs::PointCloud2>("/globalLineCloud", 1);
        publineMsgs = nh.advertise<visualization_msgs::MarkerArray>("/lineVectors", 1);

        image_transport::ImageTransport imageTransport(nh);
        pubRangeImage = imageTransport.advertise("/rangeImage",1);

        allocateMemo();
    }

    void allocateMemo(){

//        scanlineCloudVec.resize(N_SCAN);
        for (int i = 0; i < N_SCAN; ++i)
            scanlineCloudVec.emplace_back(new pcl::PointCloud<PointType>());

        curCloud.reset(new pcl::PointCloud<PointType>());
        curCloudDS.reset(new pcl::PointCloud<PointType>());
        curCloudEdge.reset(new pcl::PointCloud<PointType>());
        curCloudRing.reset(new pcl::PointCloud<PointTypeRing>());
        curCloudRange.reset(new pcl::PointCloud<PointType>());
        curLineCloud.reset(new pcl::PointCloud<PointType>());
        curGroundCloud.reset(new pcl::PointCloud<PointType>());
        globalLineCloud.reset(new pcl::PointCloud<PointType>());
        framePosesCloud.reset(new pcl::PointCloud<PointTypePose>());
    }

    void framePoseHandler(const nav_msgs::OdometryConstPtr& msg){
        lock_guard<mutex> lockGuard(mtx);
        poseMsgQue.push_back(msg);
    }
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& msg){

        lock_guard<mutex> lockGuard(mtx);
        cloudmsgQue.push_back(msg);
    }
    void segmentedCloudInfoHandler(const structural_mapping::cloud_infoConstPtr& msgIn){

        segInfo = *msgIn;
//        segmentedCloudInfomsgQue.emplace_back(msgIn);
    }

    bool parseData(){

        if(cloudmsgQue.empty())
            return false;
        {
            lock_guard <mutex> lockGuard(mtx);
            curCloudmsg = *cloudmsgQue.front();
            cloudmsgQue.pop_front();
        }

        cout << "[ Line ] Current scan #" << to_string(curCloudmsg.header.stamp.toSec()) << endl;
        pcl::fromROSMsg(curCloudmsg, *curCloud);

        if(useRingInfo)
            pcl::fromROSMsg(curCloudmsg, *curCloudRing);

        return true;
    }

    // 计算扫面线上的range，返回平面range（xOy平面）的平均值
    /// \param inCloud 单条扫描线点云
    double fillrangeMatperScanline(const pcl::PointCloud<PointType>::Ptr &inCloud ) {

        double sumRange_XY = 0, avgRange_XY = 0;
        int cloudsize = inCloud->points.size();
        double range_XY;

        for (int i = 0; i < cloudsize; ++i) {

            pcl::PointXYZI thisPoint = inCloud->points[i];
            range_XY = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y);

            sumRange_XY += range_XY;
        }
        avgRange_XY = sumRange_XY / cloudsize;
        return avgRange_XY;
    }

    /// 利用DP递归获取最短线元
    /// \param inpCloud ：输入扫描线点云
    /// \param curlineInd ：线元的inds
    /// \param vecline ：线元首尾点向量
    void douglasPeukerLineFitting(const pcl::PointCloud<PointType>::Ptr &inpCloud,
                                  const vector<int > &curlineInd,
                                  const Eigen::Vector3f &vecline,
                                  vector< pair<int, int>> &lineSegInd) {

        int curlineIndnum = curlineInd.size();

        if(curlineIndnum < minlinePtNum) // 线元最少点数
            return;
        if(vecline.norm() < minlineLength) // 线元最短距离
            return;

        Eigen::Vector3f pt2start, pt2line, newlineVec;
        double maxdist = 0;
        int ind = 0;  // 离当前线元距离最远点的ind
        for (int i = 0; i < curlineIndnum ; ++i) {

            pt2start = inpCloud->points[curlineInd[i]].getVector3fMap() -
                       inpCloud->points[curlineInd[0]].getVector3fMap();
            pt2line = pt2start - (pt2start.dot(vecline)/vecline.norm() * vecline.normalized());
            if(pt2line.norm() > maxdist){  // 点到线元的距离
                ind = i;
                maxdist = pt2line.norm();
            }
        }
        vector<int> newlineInds;
        if(maxdist > maxDPdist){  // 最远点距离判断，一分为二,迭代回归

            if(ind > minlinePtNum ){
                newlineVec = inpCloud->points[curlineInd[ind]].getVector3fMap() -
                             inpCloud->points[curlineInd[0]].getVector3fMap();
                newlineInds.assign(curlineInd.begin(), curlineInd.begin()+ind);
                douglasPeukerLineFitting(inpCloud, newlineInds, newlineVec, lineSegInd);
            }

            if(curlineIndnum - ind > minlinePtNum){
                newlineVec = inpCloud->points[curlineInd[curlineIndnum-1]].getVector3fMap() -
                             inpCloud->points[curlineInd[ind]].getVector3fMap();
                newlineInds.assign(curlineInd.begin()+ind, curlineInd.end());
                douglasPeukerLineFitting(inpCloud, newlineInds, newlineVec, lineSegInd);
            }
        }
        else{
            lineSegInd.emplace_back(pair<int, int>(curlineInd[0], curlineInd[curlineIndnum-1]));
//        cout<<"NOW there is "<<tmplineSegsInd.size()<<" line segs on this scanline."<<endl;
        }
    }


    // sort function according to the azimuth of point
    struct pointOrientationCompare{
        bool operator ()(const PointType& p1, const PointType& p2){
            return atan2(p1.y, p1.x) < atan2(p2.y, p2.x);
        }
    };
    // downsample each scanline uniformly
    void getlinesByVectorOri(){

        int cloudsizeALL = curCloud->points.size();
        bool deriveAllstructFea = true;
        int lineCount = 0;
//#pragma omp parallel for num_threads(numofCore) shared(cloudsizeALL, deriveAllstructFea)
        // according to the similarity of the tangential direction from consecutive points
        for (int j = 0; j < N_SCAN; ++j) {

            int cloudSize = scanlineCloudVec[j]->points.size();  // 该扫描线点数

            pcl::PointCloud<PointType>::Ptr cloud_ds(new pcl::PointCloud<PointType>());
//            pcl::VoxelGrid<PointType> ds;  // TODO downsample by voxel(not the original point)
//            ds.setLeafSize(leafsize_perScanline, leafsize_perScanline, leafsize_perScanline);
            pcl::UniformSampling<PointType> ds;  // TODO downsample by sphere
            ds.setRadiusSearch(leafsize_perScanline);
            ds.setInputCloud(scanlineCloudVec[j]);
            ds.filter(*cloud_ds);  // the order of points index is changed

            // reorder points
            std::sort(cloud_ds->points.begin(), cloud_ds->points.end(),
                      pointOrientationCompare());

            int cloudSize_DS = cloud_ds->points.size();
            scanlineCloudVec[j]->clear();  //

//            *scanlineCloudVec[j] = *cloud_ds;
//            continue;

            Eigen::Vector3f vec1, vec2;
            int curlineSize = 0;
            pcl::PointIndicesPtr curlineindices(new pcl::PointIndices());
            double anglrad, len = 0;
            bool ifbreak = false;

            for (int i = 0; i < cloudSize_DS-2 ; ++i) {

                curlineindices->indices.emplace_back(i);
                curlineSize++;

                vec1 = cloud_ds->points[i + 1].getVector3fMap()
                       - cloud_ds->points[i].getVector3fMap();
                vec2 = cloud_ds->points[i + 2].getVector3fMap()
                       - cloud_ds->points[i + 1].getVector3fMap();
                anglrad = acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm()));

                if (vec1.norm() > 5*vec2.norm()){
                    ifbreak = true;
                }else if(vec2.norm() > 5*vec1.norm()){
                    curlineindices->indices.emplace_back(i+1);
                    curlineSize++;
                    len += vec1.norm();

                    ifbreak = true;
                    i++;
                }else if( fabs(anglrad) > M_PI/4) {

                    curlineindices->indices.emplace_back(i+1);
                    curlineSize++;
                    len += vec1.norm();

                    ifbreak = true;
                }

                if(ifbreak) {  //
                    ifbreak = false;
                    if(curlineSize > minlinePtNum && len > minlineLength) {  // if this curve is valid
                        if(deriveAllstructFea){  // withdraw all points within the idx interval

                            int idx_begin = curlineindices->indices.front();
                            assert(j == int(cloud_ds->points[idx_begin].intensity));
//                            cout << YELLOW << "[" << idx_begin << ", ";
                            idx_begin = int((cloud_ds->points[idx_begin].intensity - j)*cloudsizeALL);  // ind in one scanline
                            int idx_end = curlineindices->indices.back();
                            assert(j == int(cloud_ds->points[idx_end].intensity));
//                            cout << idx_end << "]; " ;
                            idx_end = int((cloud_ds->points[idx_end].intensity - j)*cloudsizeALL);

//                            cout << "[" << idx_begin - idx_end << "]" << RESET << endl;
                            if(idx_begin > idx_end) { // fixme: why >?
                                int temp = idx_begin;
                                idx_begin = idx_end;
                                idx_end = temp;
                            }
                            curlineindices->indices.resize(0);
                            for (int k = idx_begin; k < idx_end; ++k){
                                curlineindices->indices.emplace_back(scanlineIndices_[j][k]);
                                PointType pt = curCloud->points[scanlineIndices_[j][k]];
//                                pt.intensity = lineCount;
                                scanlineCloudVec[j]->points.emplace_back(pt);
                                labelMat.at<uint16_t>(int(curCloudRange->points[scanlineIndices_[j][k]].x),
                                                      int(curCloudRange->points[scanlineIndices_[j][k]].y)) = SURF_LABLE;
                            }
                        }else  // only withdraw the downsampled indices
                            for(auto id : curlineindices->indices){
                                PointType pt = cloud_ds->points[id];
//                                pt.intensity = lineCount;
                                scanlineCloudVec[j]->points.emplace_back(pt);

                                int id_inLine = int((cloud_ds->points[id].intensity - j)*cloudsizeALL);
                                int id_inCloud = scanlineIndices_[j][id_inLine];
                                labelMat.at<uint16_t>(int(curCloudRange->points[id_inCloud].x),
                                                      int(curCloudRange->points[id_inCloud].y)) = SURF_LABLE;
                            }
                        lineCount++;
                    }
//                    extractEdgePointsByCurvature(curlineindices->indices);  // label edge features

                    curlineindices->indices.clear();
                    curlineSize = 0;
                    len = 0;
                }
                len += vec1.norm();
            }
        }

        curLineCloud->clear();
        int step = -1 ;
        if(N_SCAN > 16) // test HDL-64E, from 64 to 32
            step = 1;
        else step = 1;
        for (int k = 0; k < N_SCAN;) {
            *curLineCloud += *scanlineCloudVec[k];
            scanlineCloudVec[k]->clear();
            k = k + step;
        }
        // test
        if(!curLineCloud->empty()){
            pcl::io::savePCDFileBinary(projPath + "scanCloud.pcd", *curCloud);
            pcl::io::savePCDFileBinary(projPath + "lineCloud.pcd", *curLineCloud);
        }
        if(!curCloudEdge->empty())
            pcl::io::savePCDFileBinary(projPath + "edgeCloud_Ori.pcd", *curCloudEdge);
    }

    // based on range image
    void getlinesByVector(){

        int cloudsizeALL = curCloud->points.size();
        bool deriveAllstructFea = true;
        int lineCount = 0, step = 3;
        const double rad = ang_res_x / 180.0 * M_PI, resolution = 0.1;
//#pragma omp parallel for num_threads(numofCore) shared(cloudsizeALL, deriveAllstructFea)
        // according to the similarity of the tangential direction from consecutive points
        for (int j = 0; j < N_SCAN; ++j) {  // scanline(row)

            scanlineCloudVec[j]->clear();  //
            Eigen::Vector3f vec1, vec2;
            int curlineSize = 0, ind1, ind2, ind3, n_pixNoPt=0, ind=0;
            pcl::PointIndicesPtr curlineindices(new pcl::PointIndices());
            double anglrad, len = 0;
            bool ifbreak = false;

            for ( ; ind < Horizon_SCAN ; ) {  // row

                ind1 = indMat.at<uint16_t>(j, ind);
                if (ind1==0 || curCloudRange->points[ind1].intensity < MIN_RANGE){
                    ind++; continue;
                }
                step = floor(resolution / (curCloudRange->points[ind1].intensity * rad));
//                cout << RED << "STEP : " << step << RESET << endl;
                if (ind + 2*step >= Horizon_SCAN) break;

                ind2 = indMat.at<uint16_t>(j, ind+step);
                ind3 = indMat.at<uint16_t>(j, ind+2*step);
                if (step==0 || ind2==0 || ind3==0){
                    n_pixNoPt++;
                    ind++; continue;
                }

//                scanlineCloudVec[j]->points.emplace_back(curCloud->points[ind1]);
//                ind += step;
//                continue;

//                cout << YELLOW << "[ " << ind1 << ", " << ind2 << ", " << ind3 << "]" <<RESET << endl;
                curlineindices->indices.emplace_back(ind1);
                curlineSize++;

                vec1 = curCloud->points[ind2].getVector3fMap()
                       - curCloud->points[ind1].getVector3fMap();
                vec2 = curCloud->points[ind3].getVector3fMap()
                       - curCloud->points[ind2].getVector3fMap();
                anglrad = acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm()));

                if (vec1.norm() > 3*vec2.norm()){

                    ifbreak = true;
                }else if(vec2.norm() > 3*vec1.norm()){
                    curlineindices->indices.emplace_back(ind2);
                    curlineSize++;
                    len += vec1.norm();

                    ifbreak = true;
                    ind+=step;
                }else if(fabs(anglrad) > M_PI/4) {

                    curlineindices->indices.emplace_back(ind2);
                    curlineSize++;
                    len += vec1.norm();

                    ifbreak = true;
                }

                if(ifbreak) {
                    ifbreak = false;
                    if(curlineSize > minlinePtNum && len > minlineLength) {  // if this curve is valid
                        if(deriveAllstructFea){  // withdraw all points within the idx interval

                            int idx_begin = curlineindices->indices.front();
                            assert(j == int(curCloud->points[idx_begin].intensity));
//                            cout << YELLOW << "[" << idx_begin << ", ";
                            idx_begin = int((curCloud->points[idx_begin].intensity - j)*cloudsizeALL);  // ind in one scanline
                            int idx_end = curlineindices->indices.back();
                            assert(j == int(curCloud->points[idx_end].intensity));
//                            cout << idx_end << "]; " ;
                            idx_end = int((curCloud->points[idx_end].intensity - j)*cloudsizeALL);

//                            cout << "[" << idx_begin - idx_end << "]" << RESET << endl;
                            if(idx_begin > idx_end) { // fixme: why >?
                                int temp = idx_begin;
                                idx_begin = idx_end;
                                idx_end = temp;
                            }
                            curlineindices->indices.resize(0);
                            for (int k = idx_begin; k < idx_end; ++k){
                                curlineindices->indices.emplace_back(scanlineIndices_[j][k]);
                                PointType pt = curCloud->points[scanlineIndices_[j][k]];
//                                pt.intensity = lineCount;
                                scanlineCloudVec[j]->points.emplace_back(pt);
                                labelMat.at<uint16_t>(int(curCloudRange->points[scanlineIndices_[j][k]].x),
                                                      int(curCloudRange->points[scanlineIndices_[j][k]].y)) = SURF_LABLE;
                            }
                        }else  // only withdraw the downsampled indices
                            for(auto id : curlineindices->indices){
                                PointType pt = curCloud->points[id];
//                                pt.intensity = lineCount;
                                scanlineCloudVec[j]->points.emplace_back(pt);
                                labelMat.at<uint16_t>(int(curCloudRange->points[id].x),
                                                      int(curCloudRange->points[id].y)) = SURF_LABLE;
                            }
                        lineCount++;
                    }
//                    extractEdgePointsByCurvature(curlineindices->indices);  // label edge features

                    curlineindices->indices.clear();
                    curlineSize = 0;
                    len = 0;
                }
                len += vec1.norm();
                ind += step;
            }
//            cout << RED << "[ Line ] scanline "<< j << ", pixel without points: " << n_pixNoPt << RESET << endl ;
        }

        curLineCloud->clear();
        step = -1 ;
        if(N_SCAN > 16) // test HDL-64E, from 64 to 32
            step = 2;
        else step = 1;
        for (int k = 0; k < N_SCAN; ) {
            *curLineCloud += *scanlineCloudVec[k];
            scanlineCloudVec[k]->clear();
            k = k + step;
        }
        // test
        if(!curLineCloud->empty()){
            pcl::io::savePCDFileBinary(projPath + "scanCloud.pcd", *curCloud);
            pcl::io::savePCDFileBinary(projPath + "lineCloud.pcd", *curLineCloud);
        }
        if(!curCloudEdge->empty())
            pcl::io::savePCDFileBinary(projPath + "edgeCloud_Ori.pcd", *curCloudEdge);
    }

    void getlinesBasedOnDirecVector(const vector<int> &ptInds_, vector<int> &ptInds_DS){

        if (ptInds_.empty()) return;
        const int cloudsizeALL = curCloud->points.size();
        const int n_pts = ptInds_.size();

        const int lineNo = int(curCloud->points[ptInds_.front()].intensity);
        const double rad = ang_res_x / 180.0 * M_PI, resolution = 0.08;

        int ind = int(curCloudRange->points[ptInds_.front()].y);
        int endcol = int(curCloudRange->points[ptInds_.back()].y);
//        cout << YELLOW << "[ Debug ] line segment: " << ind << " ~ " << endcol << " | " << n_pts << RESET << endl;

        if(ind >= endcol) return;  // Fixme : in the start/end of the ring?
//        assert(lineNo == int(curCloudRange->points[ptInds_.front()].x));
        if (lineNo != int(curCloudRange->points[ptInds_.front()].x)) return;

        int curlineSize = 0, ind1, ind2, ind3, n_pixNoPt=0, step1, step2;
        bool ifbreak = false;
        ptInds_DS.clear();

        for ( ; ind <= endcol ; ) {

            ind1 = indMat.at<uint16_t>(lineNo, ind);
            if (ind1 == 0 || curCloudRange->points[ind1].intensity < MIN_RANGE)
            { ind++; ifbreak = false; continue; }

            /// note: no need for downsampling for UAV cloud!
//            ptInds_DS.emplace_back(ind1);
//            ind++; continue;

            step1 = floor(resolution / (curCloudRange->points[ind1].intensity * rad));
            if (step1 < 1) step1 = 1;
            if (ind + step1 > endcol) {
                ptInds_DS.emplace_back(indMat.at<uint16_t>(lineNo, endcol));
                break;
            }
            ind2 = indMat.at<uint16_t>(lineNo, ind+step1);
            while (ind2 == 0 || curCloudRange->points[ind2].intensity < MIN_RANGE )
            { step1++; ind2 = indMat.at<uint16_t>(lineNo, ind+step1); }

            step2 = floor(resolution / (curCloudRange->points[ind2].intensity * rad));
            if (step2 < 1) step2 = 1;
            if (ind + step1 + step2 > endcol) {

                ptInds_DS.emplace_back(ind2);
                ptInds_DS.emplace_back(indMat.at<uint16_t>(lineNo, endcol));
                break;
            }
            ind3 = indMat.at<uint16_t>(lineNo, ind + step1 + step2);
            while (ind3 == 0) { step2++; ind3 = indMat.at<uint16_t>(lineNo, ind + step1 + step2); }
//            cout << YELLOW << "[ Debug ] step  : " << step1 << " / " << step2 << RESET << endl;

            if (!ifbreak) { ptInds_DS.emplace_back(ind1); ifbreak = true;}
            ptInds_DS.emplace_back(ind2);
            ptInds_DS.emplace_back(ind3);
            ind = ind + step1 + step2;
        }

//        publishScanlineVectorMsg(ptInds_DS, lineMsgs);
        for(auto id : ptInds_DS)
            curCloudDS->points.emplace_back(curCloud->points[id]);

        Eigen::Vector3f vec1, vec2;
        pcl::PointIndicesPtr curlineindices(new pcl::PointIndices());
        double anglrad, len = 0;
        vector<int> turningPtInds;

        ifbreak = false;
        // extract line features
        int n_pts_DS = ptInds_DS.size();
//        cout << YELLOW << "[ Debug ] ptInds_DS.size(): " << n_pts_DS << RESET << endl;
        if (n_pts_DS < minlinePtNum) return;
        for (int i = 0; i < n_pts_DS-2; ++i) {

            ind1 = ptInds_DS[i];
            ind2 = ptInds_DS[i+1];
            ind3 = ptInds_DS[i+2];

            curlineindices->indices.emplace_back(ind1);
            curlineSize++;

            vec1 = curCloud->points[ind2].getVector3fMap()
                   - curCloud->points[ind1].getVector3fMap();
            vec2 = curCloud->points[ind3].getVector3fMap()
                   - curCloud->points[ind2].getVector3fMap();
            anglrad = acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm()));

            if(fabs(anglrad) > M_PI/4) {

                curlineindices->indices.emplace_back(ind2);
                curlineSize++;
                len += vec1.norm();
                ifbreak = true;
                turningPtInds.emplace_back(ind2);
            }

            if ( !ifbreak && i+2==n_pts_DS-1 ) {  // in the end
                curlineindices->indices.emplace_back(ind2);
                curlineSize++;
                curlineindices->indices.emplace_back(ind3);
                curlineSize++;
                len += vec1.norm();
                len += vec2.norm();
                ifbreak = true;
            }

            if(ifbreak) {
                ifbreak = false;
                if(curlineSize > minlinePtNum && len > minlineLength) {  // if this curve is valid

                    int idx_begin = int(curCloudRange->points[curlineindices->indices.front()].y);
                    int idx_end = int(curCloudRange->points[curlineindices->indices.back()].y);
                    if(idx_begin < idx_end) {
                        for (int j = idx_begin; j < idx_end; ++j) {  // label all points between
                            if (indMat.at<uint16_t>(lineNo, j) != 0)
                                labelMat.at<uint16_t>(lineNo, j) = SURF_LABLE;
                        }
                    }
                }
                curlineindices->indices.clear();
                curlineSize = 0;
                len = 0;
            }
            len += vec1.norm();
        }

        for(auto id: turningPtInds)  // label turning points
            labelMat.at<uint16_t>(int(curCloudRange->points[id].x),
                                  int(curCloudRange->points[id].y)) = EDGE_LABLE;
    }

    /// 利用道格拉斯-普克的思想在单根扫描线上提取连续线元 TODO: 直线约束太强
    /// \param inCloud ： 单根扫描线点云
    /// \param id ：扫描线ID
    vector< pair<int, int> > increlinefittingByDP(const pcl::PointCloud<PointType>::Ptr &inCloud,
                                                  int id, int method = 0) {

        vector< pair<int, int> > lineSegInd;

        int cloudSize = inCloud->points.size();  // 该扫描线点数
        double avgRangeofScanline = fillrangeMatperScanline(inCloud);
        double avgPtdist = (2*M_PI*avgRangeofScanline) / cloudSize;  // 估计扫描线平均点密度

        Eigen::Vector3f vecPre, vecBac, vecMid;
        int curlineSize = 0, totalLineNum;
        std::vector<int> curlineindices;
        vecPre = inCloud->points[1].getVector3fMap() - inCloud->points[0].getVector3fMap();

        // method 1
        if(method == 0)
            for (int i = 0; i < cloudSize-1 ; ++i) {

                curlineindices.push_back(i);
                curlineSize++;

                vecBac = inCloud->points[i+1].getVector3fMap() - inCloud->points[i].getVector3fMap();  // 连续2点
                if( vecBac.norm() > avgPtdist*5 || i==cloudSize-2 ){  // 断开位置
                    if(curlineSize < minlinePtNum)
                        continue;
//                cout<<"NEW BREAK --> DouglasPeuck iterative..."<<endl;

                    vecMid = inCloud->points[curlineindices[curlineSize-1]].getVector3fMap() -
                             inCloud->points[curlineindices[0]].getVector3fMap();  //
                    douglasPeukerLineFitting(inCloud, curlineindices, vecMid, lineSegInd);
//                douglasPeukerLineFittingTEST(inCloud, vecMid,cnt);

                    curlineindices.clear();
                    curlineSize = 0;
                }
            }

        // method 2
        if(method == 1)
            for (int i = 0; i < cloudSize-1 ; ++i) {

                curlineindices.push_back(i);
                curlineSize++;

                vecBac = inCloud->points[i+1].getVector3fMap() - inCloud->points[i].getVector3fMap();
                if(vecBac.norm() > 3*vecPre.norm()){

                    if(i+2 < cloudSize){
                        vecMid = inCloud->points[i+2].getVector3fMap() - inCloud->points[i].getVector3fMap();
                        if(vecMid.norm() < 2*vecPre.norm()){  // i+1 为一个噪点
                            curlineindices.push_back(i+2);
                            curlineSize++;
                            i++;  // skip i+1
                            vecPre = vecMid;
                            continue;
                        }
                    }
                    if(curlineSize > minlinePtNum){
//                cout<<"NEW BREAK --> DouglasPeuck iterative..."<<endl;

                        vecMid = inCloud->points[curlineindices[curlineSize-1]].getVector3fMap() -
                                 inCloud->points[curlineindices[0]].getVector3fMap();
                        douglasPeukerLineFitting(inCloud, curlineindices, vecMid, lineSegInd);
//                douglasPeukerLineFittingTEST(inCloud, vecMid,cnt);

                        curlineindices.clear();
                        curlineSize = 0;
                        vecPre = vecBac;
                        continue;
                    }
                }
                if(vecPre.norm() > 3*vecBac.norm()) {
                    curlineindices.clear();
                    curlineSize = 0;
                }
                vecPre = vecBac;
            }

        return lineSegInd;
    }
    // 10 ms
    void getlines() {

        vector<pcl::PointCloud<PointType>> tmplineCloud;
        tmplineCloud.resize(N_SCAN);
#pragma omp parallel for
        // 遍历每根扫描线上的点，提取直线点
        for (int j = 0; j < N_SCAN; ++j) {  // 扫描线ID

            vector< pair<int, int> > lineSegs = increlinefittingByDP(scanlineCloudVec[j], j, 1);
            for (int i = 0; i < lineSegs.size(); ++i)
                for (int k = lineSegs[i].first; k < lineSegs[i].second; ++k){
                    tmplineCloud[j].points.push_back(scanlineCloudVec[j]->points[k]);
                    tmplineCloud[j].points.back().intensity = i;
                }
            scanlineCloudVec[j]->clear();
        }

        curLineCloud->clear();
        for (int j = 0; j < N_SCAN; ++j) {  // 扫描线ID
            *curLineCloud += tmplineCloud[j];
            tmplineCloud[j].clear();
        }
        vector<pcl::PointCloud<PointType> >().swap(tmplineCloud);
    }


    bool extractScanLineCloud() {

        cout << "[ line ] Extracting every scan line . . ." << endl;

        float verticalAngle = 0, horizonAngle = 0;
        size_t rowIdn = 0, columnIdn = 0;
        float range = 0;
        pcl::PointXYZI thisPoint;


//        if (N_SCAN == 32) {  // for HDL-32E of NCLT dataset
////            for (int i = 0; i < N_SCAN; ++i)
////                ang_y_ordered[i] = ang_y_Vec[i];  //
//            std::sort(ang_y_ordered.begin(), ang_y_ordered.end());
//            std::reverse(ang_y_ordered.begin(), ang_y_ordered.end());  // descending
//        }else
        std::sort(ang_y_Vec.begin(), ang_y_Vec.end());

        // init
        scanlineIndices_.resize(N_SCAN);
        for (int j = 0; j < N_SCAN; ++j)
            scanlineIndices_[j].resize(0);
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32FC1, cv::Scalar::all(0.0));
        indMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_16UC1, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_16UC1, cv::Scalar::all(0));

        int cloudsize = curCloud->points.size(), n_overlap = 0, n_invalidRow = 0, n_invalidCol = 0;
        curCloudRange->points.resize(cloudsize);

        for (int i = 0; i < cloudsize ; ++i) {

            thisPoint = curCloud->points[i];

            if (useRingInfo)
                rowIdn = curCloudRing->points[i].ring;
//                rowIdn = int(thisPoint.intensity);
            else {
                // atan2~(-PI, PI]
                verticalAngle = atan2(thisPoint.z,
                                      sqrt(thisPoint.x*thisPoint.x + thisPoint.y*thisPoint.y)) *180.0/M_PI;
                rowIdn = findClosest<float>(ang_y_Vec, verticalAngle);
//                rowIdn = std::round((verticalAngle + ang_bottom) / ang_res_y);

//                if (N_SCAN == 32){  // for HDL-32E of NCLT dataset
//                    rowIdn = int(thisPoint.intensity);
//                    verticalAngle = ang_y_Vec[rowIdn];
////                    cout << YELLOW << "[ debug ] row/ " << rowIdn << ", angle: " << verticalAngle ;
//
//                    rowIdn = findClosest<float>(ang_y_ordered, verticalAngle - 0.0001);
////                    cout << "; renewed row: " << rowIdn << RESET << endl;
//                }
            }

            if (rowIdn < 0 || rowIdn >= N_SCAN){
                n_invalidRow++;
                continue;
            }
            int indxInline = scanlineIndices_[rowIdn].size();
            scanlineIndices_[rowIdn].push_back(i);  // should be ordered
            thisPoint.intensity = rowIdn + (indxInline * 1.0/cloudsize);  // encode point index to decimal
//            scanlineCloudVec[rowIdn]->points.push_back(thisPoint);

            curCloud->points[i].intensity = thisPoint.intensity;

            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180.0 / M_PI;
            columnIdn = -round( (horizonAngle-90.0)/ang_res_x ) + Horizon_SCAN/2;
            if (columnIdn >= Horizon_SCAN) columnIdn -= Horizon_SCAN;
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN){
                n_invalidCol++;
                continue;
            }

            range = pointRange(thisPoint);
            curCloudRange->points[i].intensity = range;
            curCloudRange->points[i].x = rowIdn; // save image coord
            curCloudRange->points[i].y = columnIdn;

            if (indMat.at<uint16_t >(rowIdn, columnIdn) != 0){  // keep the closer one
                n_overlap++;
                int idori = indMat.at<uint16_t >(rowIdn, columnIdn);
                indMat.at<uint16_t >(rowIdn, columnIdn) = (range < rangeMat.at<float>(rowIdn, columnIdn)? i : idori);
                idori = indMat.at<uint16_t >(rowIdn, columnIdn);
                rangeMat.at<float>(rowIdn, columnIdn) = curCloudRange->points[idori].intensity;
            }else{
                indMat.at<uint16_t >(rowIdn, columnIdn) = i; // save point ind
                rangeMat.at<float>(rowIdn, columnIdn) = range;
            }
        }

        cout << RED << " [ Line ] converting to range image with overlap points: " << n_overlap ;
        cout << RED << " ; invalid row points: " << n_invalidRow;
        cout << RED << " ; invalid column points: " << n_invalidCol << RESET << endl;
        return true;
    }

    /// for ouster LiDAR!
    bool extractScanLineCloud_Ouster() {

        cout << "[ line ] Extracting every scan line . . ." << endl;

        float verticalAngle = 0, horizonAngle = 0;
        size_t rowIdn = 0;
        float range = 0;
        pcl::PointXYZI thisPoint;

        // init
        scanlineIndices_.resize(N_SCAN);
        for (int j = 0; j < N_SCAN; ++j)
            scanlineIndices_[j].resize(0);
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32FC1, cv::Scalar::all(0.0));
        indMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_16UC1, cv::Scalar::all(0));
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_16UC1, cv::Scalar::all(0));

        int cloudsize = curCloud->points.size(), n_overlap = 0, n_invalidRow = 0, n_invalidCol = 0;
        curCloudRange->points.resize(cloudsize);

        for(int i = 0; i < N_SCAN; i++)
            for (int columnIdn = 0; columnIdn < Horizon_SCAN; ++columnIdn) {

                rowIdn = i;
                int index = rowIdn*Horizon_SCAN + (columnIdn + Horizon_SCAN - int(ang_y_Vec[rowIdn])) % Horizon_SCAN;
                if(index > cloudsize || index < 0) continue;

                thisPoint = curCloud->points[index];

                int indxInline = scanlineIndices_[rowIdn].size();
                scanlineIndices_[rowIdn].push_back(i);  // should be ordered
                thisPoint.intensity = rowIdn + (indxInline * 1.0/cloudsize);  // encode point index to decimal

                curCloud->points[i].intensity = thisPoint.intensity;

                range = pointRange(thisPoint);
                curCloudRange->points[i].intensity = range;
                curCloudRange->points[i].x = rowIdn; // save image coord
                curCloudRange->points[i].y = columnIdn;

                if (indMat.at<uint16_t >(rowIdn, columnIdn) != 0){  // keep the closer one
                    n_overlap++;
                    int idori = indMat.at<uint16_t >(rowIdn, columnIdn);
                    indMat.at<uint16_t >(rowIdn, columnIdn) = (range < rangeMat.at<float>(rowIdn, columnIdn)? i : idori);
                    idori = indMat.at<uint16_t >(rowIdn, columnIdn);
                    rangeMat.at<float>(rowIdn, columnIdn) = curCloudRange->points[idori].intensity;
                }else{
                    indMat.at<uint16_t >(rowIdn, columnIdn) = i; // save point ind
                    rangeMat.at<float>(rowIdn, columnIdn) = range;
                }
            }

        cout << RED << " [ Line ] converting to range image with overlap points: " << n_overlap ;
        cout << " ; invalid row points: " << n_invalidRow;
        cout << " ; invalid column points: " << n_invalidCol << RESET << endl;
        return true;
    }

    void extractStructuralFeatures(){

        lineMsgs.markers.clear();
        curCloudDS->clear();

        int curInd, nextInd, curPtInd, nextPtInd;
        float curDist;
        vector<int> segmentsInds;
        const double rad = ang_res_x /180.0*M_PI;

        curCloudEdge->points.resize(0);
        curLineCloud->points.resize(0);
        for (int i = 0; i < N_SCAN; ++i) {

            curInd = 0;
            while (curInd < Horizon_SCAN-1){
                segmentsInds.resize(0);
                float lastDistT = 10;
                for ( ; curInd < Horizon_SCAN-1; curInd++) {
//                    cout<< "[ Debug for ]  curInd: " << curInd << endl;

                    if (indMat.at<uint16_t>(i, curInd) == 0)  // no point
                        continue;
                    curPtInd = indMat.at<uint16_t>(i, curInd);
                    if (curCloudRange->points[curPtInd].intensity < MIN_RANGE) continue;

                    segmentsInds.emplace_back(curPtInd);
                    nextInd = curInd+1;
                    while (indMat.at<uint16_t>(i, nextInd) == 0 && nextInd < Horizon_SCAN) nextInd++;
                    if (nextInd == Horizon_SCAN) break;
                    nextPtInd = indMat.at<uint16_t>(i, nextInd);

                    double distT = getAdaptiveDistThre(curCloud->points[curPtInd].getVector3fMap(),
                                                       curCloud->points[nextPtInd].getVector3fMap(),
                                                       (nextInd - curInd)*rad);

                    curDist = pointDistBet(curCloud->points[nextPtInd], curCloud->points[curPtInd]);
                    if ( curDist > breakThre*distT || curDist > 9*lastDistT) // TODO: break threshold
                        break;

                    curInd = nextInd-1;
                    lastDistT = distT;
                }
                extractStructuralFeaturesInSeg(segmentsInds);
                if (curInd >= nextInd) curInd++;
                else curInd = nextInd;

                publishScanlineVectorMsg(segmentsInds);
            }
        }

        // debug
//        pcl::io::savePCDFileBinary(projPath + "lineSegmentsCloud.pcd", *curLineCloud);
//        curLineCloud->points.resize(0);
//        pcl::io::savePCDFileBinary(projPath + "edgeCloudwithCurv.pcd", *curCloudEdge);
//        curCloudEdge->points.resize(0);

        // derive feature clouds
        for (int i = 0; i < N_SCAN; ++i) {
            for (int j = 0; j < Horizon_SCAN; ++j) {

                if(labelMat.at<uint16_t>(i,j) == EDGE_LABLE)
                    curCloudEdge->points.emplace_back(curCloud->points[indMat.at<uint16_t>(i,j)]);
                else if(labelMat.at<uint16_t>(i,j) == SURF_LABLE)
                    curLineCloud->points.emplace_back(curCloud->points[indMat.at<uint16_t>(i,j)]);
            }
        }

        // save
//        if(!curCloudEdge->empty() && !curLineCloud->empty()) {
//            pcl::io::savePCDFileBinary(projPath + "edgeCloud_Ori.pcd", *curCloudEdge);
//            pcl::io::savePCDFileBinary(projPath + "lineCloud_ori.pcd", *curLineCloud);
//            pcl::io::savePCDFileBinary(projPath + "scanCloud.pcd", *curCloud);
//            pcl::io::savePCDFileBinary(projPath + "scanCloudDS.pcd", *curCloudDS);
//        }

        pcl::PointCloud<PointType>::Ptr edgeCloudOri(new pcl::PointCloud<PointType>());
        *edgeCloudOri = *curCloudEdge;
        curCloudEdge->points.resize(0);
        curLineCloud->points.resize(0);
        curGroundCloud->points.resize(0);

        if (filterEdge) filteringFeatures(EDGE_LABLE);
//        if (1) filteringFeatures(SURF_LABLE);
//        labelGround = true;
//        if (labelGround) {
//            labelGroundcloud();
//            cout << "[ Debug ] Ground labelled. " << endl;
//        }

        // save
        if(!curCloudEdge->empty() && !curLineCloud->empty()) {
            pcl::io::savePCDFileBinary(projPath + "edgeCloud_ver.pcd", *curCloudEdge);
            pcl::io::savePCDFileBinary(projPath + "lineCloud_ver.pcd", *curLineCloud);
        }
        evalVerticalConnectivity();

        cv::Mat matNormlized;
        cv::normalize(labelMat, matNormlized, 0, 255, CV_MINMAX, CV_8U);
        cv::resize(matNormlized, matNormlized, cv::Size(360, 32));
        if(!cv::imwrite(projPath + "labelMat_trans.png", matNormlized))  return;

        // derive feature clouds
        for (int j = 0; j < Horizon_SCAN; ++j) {
            for (int i = 0; i < N_SCAN; ++i) {
                curPtInd = indMat.at<uint16_t>(i, j);
                if (labelMat.at<uint16_t>(i, j) == EDGE_LABLE) {
                    curCloudEdge->points.emplace_back(curCloud->points[curPtInd]);
//                    curCloudEdge->points.back().intensity = curCloudRange->points[curPtInd].intensity;  // range val
                }
                else if (labelMat.at<uint16_t>(i, j) == SURF_LABLE) {
                    curLineCloud->points.emplace_back(curCloud->points[curPtInd]);
//                    curLineCloud->points.back().intensity = curCloudRange->points[curPtInd].intensity;
                }
                else if (labelGround && labelMat.at<uint16_t>(i, j) == GROUND_LABLE) {
                    curGroundCloud->points.emplace_back(curCloud->points[curPtInd]);
                    curGroundCloud->points.back().intensity = -1;
                }
            }
        }

        // save
        if(!curCloudEdge->empty() && !curLineCloud->empty()) {
            pcl::io::savePCDFileBinary(projPath + "edgeCloud.pcd", *curCloudEdge);
            pcl::io::savePCDFileBinary(projPath + "lineCloud.pcd", *curLineCloud);
        }
    }

    void extractStructuralFeaturesInSeg(const vector<int> &ptInds){

        int n_curSeg = ptInds.size();
        if (n_curSeg < 1) return;

        for (auto id : ptInds) {
            curLineCloud->points.emplace_back(curCloud->points[id]);
            curLineCloud->points.back().intensity = int(curCloud->points[id].intensity)+n_curSeg/1000.0;
        }
//        return;

        if (n_curSeg < minlinePtNum &&
            pointDistBet(curCloud->points[ptInds.front()],
                         curCloud->points[ptInds.back()]) < minlineLength) { // small cluster

            labelMat.at<uint16_t>(int(curCloudRange->points[ptInds.front()].x),
                                  int(curCloudRange->points[ptInds.front()].y)) = EDGE_LABLE;
            labelMat.at<uint16_t>(int(curCloudRange->points[ptInds.back()].x),
                                  int(curCloudRange->points[ptInds.back()].y)) = EDGE_LABLE;

        }else {  // extract line features

            vector<int> ptInds_DS;
            getlinesBasedOnDirecVector(ptInds, ptInds_DS);

            float w_factor = 4;
            n_curSeg = ptInds_DS.size();
            for (int k = NeighborSize; k < n_curSeg-NeighborSize; ++k) {

                uint16_t &label = labelMat.at<uint16_t>(int(curCloudRange->points[ptInds_DS[k]].x),
                                                        int(curCloudRange->points[ptInds_DS[k]].y));
                if (label != EDGE_LABLE) continue;

                float sumRange = 0, sumWeights = 0, smoothness, dist, weight;
                float curRange = curCloudRange->points[ptInds_DS[k]].intensity, neighRange;
                int n_neighbor = 0;
                PointType currPt = curCloud->points[ptInds_DS[k]], neighPt;
                for (int j = -NeighborSize; j <= NeighborSize; ++j) {  // neighbors

                    neighPt = curCloud->points[ptInds_DS[k+j]];
                    neighRange = curCloudRange->points[ptInds_DS[k+j]].intensity;

                    dist = pointDistBet(neighPt, currPt);
                    weight = exp(-w_factor * dist*dist);
                    sumWeights += weight;
                    sumRange += weight*(curRange - neighRange);  // add weights
                    n_neighbor++;
                }
//                sumRange -= n_neighbor*curCloudRange->points[scanlineIndices_[i][k]].intensity;
                smoothness = pow(sumRange/sumWeights, 2);

//                if(smoothness > 0.002){
                PointType edgePoint = curCloud->points[ptInds_DS[k]];
                edgePoint.intensity = smoothness;
                curCloudEdge->points.push_back(edgePoint);
//                     = EDGE_LABLE;
//                }
            }

            // todo: label two endpoints?
            labelMat.at<uint16_t>(int(curCloudRange->points[ptInds.front()].x),
                                  int(curCloudRange->points[ptInds.front()].y)) = EDGE_LABLE;
            labelMat.at<uint16_t>(int(curCloudRange->points[ptInds.back()].x),
                                  int(curCloudRange->points[ptInds.back()].y)) = EDGE_LABLE;
        }
    }

    double getAdaptiveDistThre(const Eigen::Vector3f p1, const Eigen::Vector3f p2, const double rad){

        Eigen::Vector3f p1p2 = p2 - p1;
        double ang_p1p2 = acos((p1.dot(p1p2)) / (p1.norm() * p1p2.norm()));  // [0, Pi]
//        double range = min(p1.norm(), p2.norm());
        double range = p1.norm();
//        if (ang_p1p2 < M_PI/2.0)
//            range = max(p1.norm(), p2.norm());

        double extremeAng = 10.0 / 180.0 * M_PI;  // todo: 10 degree
//        double dist = range * sin(rad) / sin(ang_p1p2);
        double dist = range * sin(rad) / sin(extremeAng - rad);
        return (dist + 3*0.03);
    }
    // by image region growing based on ring clustering
    void filteringFeatures(uint16_t feaType) {

        cv::Mat matNormlized;
        cv::normalize(labelMat, matNormlized, 0, 255, CV_MINMAX, CV_8U);
        if(!cv::imwrite(projPath + "labelMat.png", matNormlized))  return;

        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> visitMat(N_SCAN, Horizon_SCAN);
        visitMat.fill(false);
        cv::Point2i seed;
        vector<cv::Point2i> seeds;
        vector<int> clusterInds, scanlineIDs;
        int grow_direction[8][2] = {{0, -1}, {0, 1},  // row-wise
                                    {1, -1}, {1, 1}, {1, 0},  // col-wise
                                    {2, -1}, {2, 1}, {2, 0}};  // 2 rows
        int clusterID = 0, minThre;
        double radVer;
        if (feaType == EDGE_LABLE) minThre = 4;
        if (feaType == SURF_LABLE) minThre = 2;

        for (int i = 0; i < N_SCAN; ++i) {
            for (int j = 0; j < Horizon_SCAN; ++j) {

                if (visitMat(i, j)) continue;
                if (labelMat.at<uint16_t>(i, j) == feaType) { // region growing
                    seed.x = i;
                    seed.y = j;
                    scanlineIDs.resize(0);
                    clusterInds.resize(0);
                    seeds.resize(0);
                    seeds.emplace_back(seed);

                    PointType curPt, neiPt;
                    Eigen::Matrix<double, 9, 1> cumulants = Eigen::Matrix<double, 9, 1>::Zero();
                    Eigen::Matrix3d cov_temp;

                    while (!seeds.empty()) {
                        seed = seeds.back();
                        seeds.pop_back();
                        int ind = indMat.at<uint16_t>(seed.x, seed.y);
                        visitMat(seed.x, seed.y) = true;
                        clusterInds.emplace_back(ind);

                        curPt = curCloud->points[ind];
                        cumulants(0) += curPt.x;
                        cumulants(1) += curPt.y;
                        cumulants(2) += curPt.z;
                        cumulants(3) += curPt.x * curPt.x;
                        cumulants(4) += curPt.x * curPt.y;
                        cumulants(5) += curPt.x * curPt.z;
                        cumulants(6) += curPt.y * curPt.y;
                        cumulants(7) += curPt.y * curPt.z;
                        cumulants(8) += curPt.z * curPt.z;
                        double distThre;
                        for (int k = 0; k < 8; ++k) {

                            cv::Point2i neighbor_seed(seed.x + grow_direction[k][0],
                                                      seed.y + grow_direction[k][1]);
                            // check whether in image or visited
                            if ( neighbor_seed.x < 0 || neighbor_seed.x > (N_SCAN - 1) ||
                                 neighbor_seed.y < 0 || neighbor_seed.y > (Horizon_SCAN - 1) ||
                                 visitMat(neighbor_seed.x, neighbor_seed.y))
                                continue;
                            if(labelMat.at<uint16_t>(neighbor_seed.x, neighbor_seed.y) != feaType)
                                continue;
                            ind = indMat.at<uint16_t>(neighbor_seed.x, neighbor_seed.y);
                            neiPt = curCloud->points[ind];
                            if (k > 1) { // between rows
//                                radVer = abs(ang_y_ordered[neighbor_seed.x] - ang_y_ordered[seed.x]) / 180.0 * M_PI;
                                radVer = abs(ang_y_Vec[neighbor_seed.x] - ang_y_Vec[seed.x]) / 180.0 * M_PI;
                                if (feaType == EDGE_LABLE)
                                    distThre = clusterThre_col*(pointRange(curPt)+pointRange(neiPt))/2.0 * radVer;

                                if (feaType == SURF_LABLE)  // ground
                                    distThre = clusterThre_col*getAdaptiveDistThre(curPt.getVector3fMap(),
                                                                                   neiPt.getVector3fMap(),
                                                                                   radVer);
                            }else { // between cols
                                if (feaType == EDGE_LABLE)
                                    distThre = (pointRange(curPt)+pointRange(neiPt))/2.0 * (ang_res_x/180.0*M_PI);  // different threshold for row/col clustering
                                else
                                    distThre = 100;
                            }

//                            if (feaType == SURF_LABLE) distThre = 100; // todo: cluster all for surf?

                            if (pointDistBet(curPt, neiPt) < distThre){
                                seeds.push_back(neighbor_seed);
                                if (k > 1)  { // different row
                                    scanlineIDs.emplace_back(neighbor_seed.x);
//                                    if (feaType == SURF_LABLE){
//                                        Eigen::Vector3f p1p2 = neiPt.getVector3fMap() - curPt.getVector3fMap();
//                                        radVer = atan2(p1p2.z(), sqrt(p1p2.x()*p1p2.x() + p1p2.y()*p1p2.y())) *180/M_PI;
//                                        if (radVer < 10){
//                                            labelMat.at<uint16_t>(seed.x, seed.y) = GROUND_LABLE;
//                                            labelMat.at<uint16_t>(neighbor_seed.x, neighbor_seed.y) = GROUND_LABLE;
//                                        }
//                                    }
                                }
                            }else if (k < 2 && feaType == EDGE_LABLE) {  // in case of occlusion
                                cv::Point2i occlu = pointRange(curPt) < pointRange(neiPt)? neighbor_seed:seed;
                                labelMat.at<uint16_t>(occlu.x, occlu.y) = 0;
//                                curCloudRange->points[indMat.at<uint16_t>(occlu.x, occlu.y)].intensity = -100;
                            }
                        }
                    }
                    if (clusterInds.size() < minThre) {  // small cluster
                        for(auto id : clusterInds){
                            labelMat.at<uint16_t>(int(curCloudRange->points[id].x),
                                                  int(curCloudRange->points[id].y)) = 0;
                            curCloudRange->points[id].intensity = 0;
                        }
                        continue;
                    }
                    auto it = std::unique(scanlineIDs.begin(), scanlineIDs.end());
                    scanlineIDs.erase(it, scanlineIDs.end());
                    if (scanlineIDs.size() < minThre) {  // cluster not covering enough vertically
                        for(auto id : clusterInds){
                            labelMat.at<uint16_t>(int(curCloudRange->points[id].x),
                                                  int(curCloudRange->points[id].y)) = 0;
                            curCloudRange->points[id].intensity = 0;
                        }
                        continue;
                    }
                    // PCA
                    cumulants /= (double)clusterInds.size();
                    cov_temp(0, 0) = cumulants(3) - cumulants(0) * cumulants(0);
                    cov_temp(1, 1) = cumulants(6) - cumulants(1) * cumulants(1);
                    cov_temp(2, 2) = cumulants(8) - cumulants(2) * cumulants(2);
                    cov_temp(0, 1) = cumulants(4) - cumulants(0) * cumulants(1);
                    cov_temp(1, 0) = cov_temp(0, 1);
                    cov_temp(0, 2) = cumulants(5) - cumulants(0) * cumulants(2);
                    cov_temp(2, 0) = cov_temp(0, 2);
                    cov_temp(1, 2) = cumulants(7) - cumulants(1) * cumulants(2);
                    cov_temp(2, 1) = cov_temp(1, 2);
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver;
                    solver.compute(cov_temp, Eigen::ComputeEigenvectors);
                    Eigen::Vector3d eigen_value = solver.eigenvalues();      // in ascending order
//                    pca_info_[i].normal_dir = solver.eigenvectors().col(0);  // 特征向量按照列方向一次排列最后一个方向是主方向, 第一个方向是法向量的方向

                    float featurity, feaThre;
                    if (feaType == EDGE_LABLE){
                        featurity = (eigen_value[2] - eigen_value[1])/eigen_value[2];
                        featurity = (eigen_value[2] / eigen_value[1]);
                        feaThre = LinearityThre;
                    }else if(feaType == SURF_LABLE){
                        featurity = (eigen_value[1] - eigen_value[0])/eigen_value[2];
                        feaThre = PlanarityThre;
                    }
                    uint16_t label = 0;
                    if (featurity > feaThre){
                        label = feaType;
                        clusterID++;
                    }
                    for(auto id : clusterInds){
                        uint16_t &pt_L = labelMat.at<uint16_t>(int(curCloudRange->points[id].x),
                                                               int(curCloudRange->points[id].y));
                        pt_L = (pt_L == 0)? 0 : label;  // 0 means occlusion
                    }
                }
            }
        }

        cv::normalize(labelMat, matNormlized, 0, 255, CV_MINMAX, CV_8U);
        cv::resize(matNormlized, matNormlized, cv::Size(360, 32));
        if(!cv::imwrite(projPath + "labelMat_Filtered"+to_string(feaType)+".png", matNormlized))  return;
    }

    void labelGroundcloud(){

        for (int i = 0; i < groundScanInd; ++i) {
            for (int j = 0; j < Horizon_SCAN; ++j) {

                int l1 = labelMat.at<uint16_t>(i,j);
                int l2 = labelMat.at<uint16_t>(i+1,j);

                if ((l1 == SURF_LABLE || l1 == GROUND_LABLE) && l2 == SURF_LABLE){

                    PointType neiPt = curCloud->points[indMat.at<uint16_t>(i+1, j)];
                    PointType curPt = curCloud->points[indMat.at<uint16_t>(i, j)];
                    Eigen::Vector3f p1p2 = neiPt.getVector3fMap() - curPt.getVector3fMap();
                    double radVer = atan2(p1p2.z(), sqrt(p1p2.x()*p1p2.x() + p1p2.y()*p1p2.y())) *180/M_PI;
                    if (radVer < 10){
                        labelMat.at<uint16_t>(i, j) = GROUND_LABLE;
                        labelMat.at<uint16_t>(i+1, j) = GROUND_LABLE;
                    }
                }
            }
        }
    }

    void evalVerticalConnectivity(){

        int curInd, nextInd, endInd, endPtInd, curPtInd, nextPtInd;
        float curDist;
        vector<int> segmentsInds;
        const double rad = ang_res_y /180.0*M_PI;
        Eigen::Vector3f vec1, vec2;
        double vecAngl_rad, groundAngl_rad;
        bool lastState = false;

        for (int i = 0; i < Horizon_SCAN; ++i) {

            segmentsInds.resize(0);
            curInd = nextInd = endInd = 0;
            while (curInd < N_SCAN-2){

                while (indMat.at<uint16_t>(curInd, i) == 0 && curInd < N_SCAN-3) curInd++;
                nextInd = curInd+1;
                while (indMat.at<uint16_t>(nextInd, i) == 0 && nextInd < N_SCAN-2) nextInd++;
                endInd = nextInd+1;
                while (indMat.at<uint16_t>(endInd, i) == 0 && endInd < N_SCAN) endInd++;
                if (endInd == N_SCAN) break;

                curPtInd  = indMat.at<uint16_t>(curInd, i);
                nextPtInd = indMat.at<uint16_t>(nextInd, i);
                endPtInd  = indMat.at<uint16_t>(endInd, i);

                vec1 = curCloud->points[nextPtInd].getVector3fMap() - curCloud->points[curPtInd].getVector3fMap();
                vec2 = curCloud->points[endPtInd].getVector3fMap() - curCloud->points[nextPtInd].getVector3fMap();

                vecAngl_rad = acos(vec1.dot(vec2) / (vec1.norm() * vec2.norm())) *180/M_PI; // vector angle
                groundAngl_rad = atan2(vec1.z(), sqrt(vec1.x()*vec1.x() + vec1.y()*vec1.y())) *180/M_PI; // azimuth

                if (labelGround && nextInd < groundScanInd && groundAngl_rad < 9) {
                    labelMat.at<uint16_t>(curInd, i) = GROUND_LABLE;
                    labelMat.at<uint16_t>(nextInd, i) = GROUND_LABLE;
                    curInd = nextInd;
                    continue;
                }
                curInd = nextInd;

                if (groundAngl_rad < 45) continue;

                if (vecAngl_rad < 30){
                    if (lastState) segmentsInds.emplace_back(endPtInd);
                    else {
                        segmentsInds.emplace_back(curPtInd);
                        segmentsInds.emplace_back(nextPtInd);
                        segmentsInds.emplace_back(endPtInd);
                    }
                    lastState = true;
                }else{
//                    publishScanlineVectorMsg(segmentsInds);
                    int label = -1;
                    for(auto& ind : segmentsInds){
                        if (labelMat.at<uint16_t>(int(curCloudRange->points[ind].x),
                                                  int(curCloudRange->points[ind].y)) == SURF_LABLE)
                            label = SURF_LABLE;
                        else if (labelMat.at<uint16_t>(int(curCloudRange->points[ind].x),
                                                       int(curCloudRange->points[ind].y)) == EDGE_LABLE)
                            label = EDGE_LABLE;
                    }
                    if (label > 0)
                        for(auto& ind : segmentsInds)
                            labelMat.at<uint16_t>(int(curCloudRange->points[ind].x),
                                                  int(curCloudRange->points[ind].y)) = label;
                    segmentsInds.clear();
                    lastState = false;
                }
            }
        }

    }


    // transform line cloud by pose
    void congregateCloud(){

        lineCloudTimeVec.emplace_back(curCloudmsg.header.stamp.toSec());
        lineCloudVec.emplace_back(curLineCloud);

        lock_guard<mutex> lockGuard(mtx);
        static nav_msgs::OdometryConstPtr curPoseMsg;
        static int lastInd = 0;
        PointTypePose pose;

        // withdraw the front pose
        if (!poseMsgQue.empty()) {
            curPoseMsg = poseMsgQue.front();

            // pose in camera frame [ aft_mapped_to_init ]
            pose.x = curPoseMsg->pose.pose.position.z;
            pose.y = curPoseMsg->pose.pose.position.x;
            pose.z = curPoseMsg->pose.pose.position.y;
            pose.roll  = curPoseMsg->pose.pose.orientation.z;
            pose.pitch = curPoseMsg->pose.pose.orientation.x;
            pose.yaw   = curPoseMsg->pose.pose.orientation.y;
            pose.intensity = curPoseMsg->pose.pose.orientation.w;

            // pose in lidar frame
//            pose.x = curPoseMsg->pose.pose.position.x;
//            pose.y = curPoseMsg->pose.pose.position.y;
//            pose.z = curPoseMsg->pose.pose.position.z;
//            pose.roll  = curPoseMsg->pose.pose.orientation.x;
//            pose.pitch = curPoseMsg->pose.pose.orientation.y;
//            pose.yaw   = curPoseMsg->pose.pose.orientation.z;
//            pose.intensity = curPoseMsg->pose.pose.orientation.w;

            pose.time = curPoseMsg->header.stamp.toSec();
            framePosesCloud->points.push_back(pose);

            poseMsgQue.pop_front();
        }else return;

        cout << MAGENTA << "[ line ] Lastframe ID "<< lastInd << RESET << endl;
        for (int i = lastInd; i < lineCloudTimeVec.size(); ++i) {

            if (abs(pose.time - lineCloudTimeVec[i]) > 0.05)
                continue;

            pcl::PointCloud<PointType>::Ptr emptyCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>());

            // 1. only using point-to-plane cost
//            poseEstimator->setlocalMap(globalLineCloud, globalLineCloud);
//            if(poseEstimator->ceresSolver(emptyCloud, lineCloudVec[i],
//                                          true, getTransformMatrix(pose))){
//                double* poseA = new double[7];
//                poseEstimator->getCurPose(poseA);
//                pose.roll  = poseA[0];
//                pose.pitch = poseA[1];
//                pose.yaw   = poseA[2];
//                pose.intensity = poseA[3];
//                pose.x = poseA[4];
//                pose.y = poseA[5];
//                pose.z = poseA[6];
//                delete poseA;
//            }
            *globalLineCloud += *transformPointCloud(lineCloudVec[i], &pose);

            // 2. NDT-OMP
//            *emptyCloud = *transformPointCloud(lineCloudVec[i], &pose);
//            if(globalLineCloud->empty())
//                *globalLineCloud += *emptyCloud;
//            else{
////                pclomp::NormalDistributionsTransform<PointType, PointType>::Ptr
////                        ndt_omp(new pclomp::NormalDistributionsTransform<PointType, PointType>());
//                pcl::NormalDistributionsTransform<PointType, PointType>::Ptr
//                        ndt_omp(new pcl::NormalDistributionsTransform<PointType, PointType>());
//
//                ndt_omp->setResolution(5*ds_map_leaf);
////                ndt_omp->setNumThreads(numofCore);
////                ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
//                ndt_omp->setInputTarget(globalLineCloud); // FixMe SIGSEGV
//                ndt_omp->setInputSource(emptyCloud);
//                ndt_omp->align(*aligned);
//                cout << MAGENTA << "[ Line ] NDT_OMP score : " << ndt_omp->getFitnessScore() <<  RESET << endl;
//                *globalLineCloud += *aligned;
//            }

            downSampler.setLeafSize(ds_map_leaf, ds_map_leaf, ds_map_leaf);
            downSampler.setInputCloud(globalLineCloud);
            downSampler.filter(*globalLineCloud);

            int size = globalLineCloud->points.size();
            cout << MAGENTA << "[ Line ] Global line cloud " <<  size <<  RESET << endl;

            if(size > maxLineCloudSize){

                emptyCloud->clear();
                pcl::copyPointCloud(*globalLineCloud, *emptyCloud);
                globalsubLineCloudVec.emplace_back(emptyCloud);
                globalLineCloud->clear();
            }
            lastInd = i+1;
            break;
        }

    }

    void saveGlobalCloud(){

        cout << YELLOW << "[ line ] Saving line cloud ..." << RESET << endl;

        int n = globalsubLineCloudVec.size();
        for (int i = 0; i < n; ++i)
            *globalLineCloud += *globalsubLineCloudVec[i];

        if (!globalLineCloud->empty()){
            pcl::io::savePCDFileBinary(projPath + "globalLineCloud.pcd", *globalLineCloud);
            cout << YELLOW << "[ line ] SUCCEED." << RESET << endl;
        }

    }

    void pubCloud(){

        if (!lineMsgs.markers.empty())
            publineMsgs.publish(lineMsgs);

        sensor_msgs::PointCloud2 cloudmsg;
        pcl::toROSMsg(*curLineCloud, cloudmsg);
        cloudmsg.header.stamp = curCloudmsg.header.stamp;
        cloudmsg.header.frame_id = "/base_link";
        publineCloud.publish(cloudmsg);

        pcl::toROSMsg(*curGroundCloud, cloudmsg);
        cloudmsg.header.stamp = curCloudmsg.header.stamp;
        cloudmsg.header.frame_id = "/base_link";
        pubGroundcloud.publish(cloudmsg);

        pcl::toROSMsg(*curCloudEdge, cloudmsg);
        cloudmsg.header.stamp = curCloudmsg.header.stamp;
        cloudmsg.header.frame_id = "/base_link";
        pubEdgeCloud.publish(cloudmsg);

        pcl::toROSMsg(*curCloud, cloudmsg);
        cloudmsg.header.stamp = curCloudmsg.header.stamp;
        cloudmsg.header.frame_id = "/base_link";
        pubCurCloud.publish(cloudmsg);

        /// save structural feature-cloud
//        pcl::PointCloud<PointType>::Ptr structuralCloud(new pcl::PointCloud<PointType>());
//        structuralCloud->points.reserve(curLineCloud->points.size()+curCloudEdge->points.size());
//        *structuralCloud = *curLineCloud;
//        *structuralCloud += *curCloudEdge;
//        structuralCloud->height = 1;
//        structuralCloud->width = structuralCloud->points.size();
        if (!curLineCloud->empty())
            pcl::io::savePCDFileBinary("/media/cyz/Seagate Basic/openDatasets/GRACO/ugv-01-fea-pcds/"
                                       + to_string(curCloudmsg.header.stamp.toSec()) + ".pcd", *curLineCloud);
//        if (!curCloudEdge->empty())
//            pcl::io::savePCDFileBinary("/media/cyz/Seagate Basic/myPassportBackup/DATA/kylin-data/prj1_playground/laser_edges/"
//                                       + to_string(curCloudmsg.header.stamp.toSec()) + ".pcd", *curCloudEdge);

        if(publineCloudglobal.getNumSubscribers() && !globalLineCloud->points.empty()){

            pcl::toROSMsg(*globalLineCloud, cloudmsg);
            cloudmsg.header.stamp = curCloudmsg.header.stamp;
            cloudmsg.header.frame_id = "/camera_init";
            publineCloudglobal.publish(cloudmsg);
        }

        cv::Mat matNormlized;
        cv::normalize(rangeMat, matNormlized, 0, 255, CV_MINMAX, CV_8U);
        cv::resize(matNormlized, matNormlized, cv::Size(360, 32));
        if(!cv::imwrite(projPath + "rangeMat.png", matNormlized))  return;
        sensor_msgs::ImagePtr imgMsg = cv_bridge::CvImage(curCloudmsg.header, "mono16", matNormlized).toImageMsg();
        pubRangeImage.publish(imgMsg);

    }

    void publishScanlineVectorMsg(const vector<int> &inds){

        if (inds.size() < 2) return;
        visualization_msgs::Marker lineStripMsg;
        lineStripMsg.header.frame_id = "/base_link";
        lineStripMsg.header.stamp = curCloudmsg.header.stamp;
        lineStripMsg.ns = "scanlineVectors";
        lineStripMsg.action = visualization_msgs::Marker::ADD;
        lineStripMsg.pose.orientation.w = 1.0;
        lineStripMsg.id = lineMsgs.markers.size();
        lineStripMsg.type = visualization_msgs::Marker ::LINE_STRIP;
        lineStripMsg.scale.x = 0.03;
        lineStripMsg.color.g = 1.0;
        lineStripMsg.color.a = 0.6;

//        for (int col = 0; col < Horizon_SCAN; ++col) {
//            if (indMat.at<uint16_t>(row, col) == 0)
//                continue;
        for(auto& id : inds){
//            PointType pt = curCloud->points[indMat.at<uint16_t>(row, col)];
            PointType pt = curCloud->points[id];
            geometry_msgs::Point ptMsg;
            ptMsg.x = pt.x;
            ptMsg.y = pt.y;
            ptMsg.z = pt.z;
            lineStripMsg.points.push_back(ptMsg);
        }
//        if (lineStripMsg.points.size() >= minlinePtNum)
        lineMsgs.markers.push_back(lineStripMsg);
//        }
//        publineMsgs.publish(lineMsgs);

    }

    void run(){
        ROS_INFO("\033[1;32m---->\033[0m Line Fitting Started.");
        ros::Rate loop_rate(10);

        while(ros::ok()){
            loop_rate.sleep();

            if(!parseData()) continue;

            timer.tic();

            extractScanLineCloud();
//            extractScanLineCloud_Ouster();

            /// option 1
            extractStructuralFeatures();

            /// option 2
//            getlinesByVectorOri();

            pubCloud();
            cout << BOLDWHITE << "[ line ] Line Fitting time : " << timer.toc() << " ms" << RESET << endl;
        }
        ROS_INFO("\033[1;32m---->\033[0m Line Fitting thread Ended.");
    }
};

int main(int argc, char** argv){

    ros::init(argc, argv, "line_fitting");
    LineFitting linefitter;

    thread processor(&LineFitting::run, &linefitter);

    ros::spin();

    processor.join();
    linefitter.saveGlobalCloud();
    cout << BLUE << "[ ROS ] lineFitting node is done." << RESET << endl;

    return 0;
}