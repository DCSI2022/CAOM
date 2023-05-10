
cmake_minimum_required(VERSION 2.8.3)
project(structural_mapping)
#message("PROJECT_SOURCE_DIR = " ${PROJECT_SOURCE_DIR})

# check c++14 / c++0x
#include(CheckCXXCompilerFlag)
#CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
#CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
#if(COMPILER_SUPPORTS_CXX14)
#	set(CMAKE_CXX_FLAGS "-std=c++14")
#elseif(COMPILER_SUPPORTS_CXX0X)
#	set(CMAKE_CXX_FLAGS "-std=c++0x")
#else()
#	message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
#endif()

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

message( "CMAKE_CXX_STANDARD = " ${CMAKE_CXX_FLAGS})

#find_package(Eigen3 3.3.4 REQUIRED)
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3")
message("EIGEN3_INCLUDE_DIR = " ${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(octomap REQUIRED)
include_directories(${OCTOMAP_INCLUDE_DIRS})

find_package(OpenMP)

find_package(GTSAM REQUIRED PATHS "/home/cyz/environments/gtsam-4.0.0-alpha2/makeinstalled/lib/cmake/GTSAM" NO_DEFAULT_PATH)
# self-defined directory for gtsam
############################## GTSAM 4.0 ################################################
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/thirdparty/cmake_modules)
#list(APPEND CMAKE_MODULE_PATH /home/cyz/environments/gtsam-4.0.0-alpha2/makeinstalled)
#set(GTSAM_DIR /home/cyz/environments/gtsam-4.0.0-alpha2)
#MESSAGE("GTSAM_DIR = " ${GTSAM_DIR})
#
#find_package(GTSAM REQUIRED)
#MESSAGE("GTSAM_INCLUDE_DIR = " ${GTSAM_INCLUDE_DIR})
#MESSAGE("GTSAM_LIB_DIR = " ${GTSAM_LIBS})
#
include_directories("/home/cyz/environments/gtsam-4.0.0-alpha2/makeinstalled/include")
#link_directories("/home/cyz/environments/gtsam-4.0.0-alpha2/makeinstalled/lib")

############################ GTSAM 4.1 ################################################
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/thirdparty/cmake_modules)
#set(GTSAM_DIR ${PROJECT_SOURCE_DIR}/thirdparty/gtsam)
#MESSAGE("GTSAM_DIR = " ${GTSAM_DIR})
#
#find_package(GTSAM REQUIRED)
#MESSAGE("GTSAM_INCLUDE_DIR = " ${GTSAM_INCLUDE_DIR})
#MESSAGE("GTSAM_LIB_DIR = " ${GTSAM_LIBS})
#
#include_directories("${PROJECT_SOURCE_DIR}/thirdparty/gtsam_lib/usr/local/include")
#link_directories("${PROJECT_SOURCE_DIR}/thirdparty/gtsam_lib/usr/local/lib")

# self-defined opencv
find_package(OpenCV REQUIRED PATHS /home/cyz/environments/opencv-3.4.13/makeinstalled/share/OpenCV NO_DEFAULT_PATH)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
#add_definitions(${PCL_DEFINITIONS})

# self-defined directory for ceres
# 定义自己编译的Ceres的库目录
set(Ceres_LIBS_DIR /home/cyz/environments/ceres-solver-1.14.0/makeinstalled/usr/local/lib)
# 定义CeresConfig.cmake所在文件目录
set(Ceres_DIR ${Ceres_LIBS_DIR}/cmake/Ceres)
# 找Ceres库
find_package(Ceres REQUIRED)
message(STATUS "Found Ceres: ${CERES_FOUND}")
message(STATUS "    Ceres Include Dirs: ${CERES_INCLUDE_DIRS}")
message(STATUS "    Ceres Libs: ${CERES_LIBRARIES}")

# ROS PACKAGES
find_package(catkin REQUIRED COMPONENTS
		tf
		roscpp
		rospy
		cv_bridge
		image_transport

		pcl_ros
		pcl_conversions

		std_msgs
		sensor_msgs
		geometry_msgs
		nav_msgs
		#		cloud_msgs
		ndt_omp
		fast_gicp
		)

include_directories(
		include
		#/home/joe/workspace/catkin_ws/src/ndt_omp/include #ndt_omp
		${catkin_INCLUDE_DIRS}
		${CERES_INCLUDE_DIRS}
		${GTSAM_INCLUDE_DIR}
		../../devel/include
)

add_message_files(DIRECTORY msg
		FILES poseInSlidingWin.msg
		cloud_info.msg
		localmapWithPoseMsg.msg)
generate_messages(DEPENDENCIES std_msgs geometry_msgs nav_msgs sensor_msgs)

catkin_package(
		INCLUDE_DIRS include
		CATKIN_DEPENDS message_generation message_runtime std_msgs geometry_msgs nav_msgs sensor_msgs
		#DEPENDS PCL
)

set( COMMON_LIB
		${catkin_LIBRARIES}
		${PCL_LIBRARIES}
		${CERES_LIBRARIES}
		${OCTOMAP_LIBRARIES}
		${OpenCV_LIBRARIES}
		)
# libraries
add_library(ceresSplineCost src/ceresSplineCost.cpp src/tools.cpp)
add_dependencies(ceresSplineCost ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(ceresSplineCost ${COMMON_LIB})

add_executable(dataPreprocessing src/dataPreprocessing.cpp src/tools.cpp)
add_dependencies(dataPreprocessing ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(dataPreprocessing  ${COMMON_LIB})

add_executable(featureExtraction src/featureExtraction.cpp src/tools.cpp)
add_dependencies(featureExtraction ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(featureExtraction ${COMMON_LIB})

add_executable(odomEstimation src/odomEstimation.cpp src/tools.cpp)
add_dependencies(odomEstimation ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(odomEstimation  ${COMMON_LIB})

# Sophus
#find_package(Sophus REQUIRED PATHS /home/cyz/environments/Sophus-master/build/make_installed/usr/local/share/sophus/cmake NO_DEFAULT_PATH)
find_package(Sophus)
include_directories(${Sophus_INCLUDE_DIRS})
include_directories(/home/cyz/environments/Sophus-master/build/make_installed/usr/local/include)

## mapping with Spline Fusion
add_executable(mapOptimizeSplineFusion src/mapOptiSplineFusion.cpp src/tools.cpp)
add_dependencies(mapOptimizeSplineFusion ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(mapOptimizeSplineFusion  ${CERES_LIBRARIES} ${COMMON_LIB} ceresSplineCost)

## mapping with HD map as reference
add_executable(mappingWithReference src/mappingWithReference.cpp src/ikd_Tree.cpp src/tools.cpp)
add_dependencies(mappingWithReference ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(mappingWithReference  ${CERES_LIBRARIES} ${COMMON_LIB} ceresSplineCost gtsam tbb)


# structural features extraction based on the continuity of line-vector
add_executable(lineFitting src/lineFitting.cpp src/tools.cpp)
add_dependencies(lineFitting ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(lineFitting  ${CERES_LIBRARIES} ${COMMON_LIB} )


# Super4PCS
find_package(Super4PCS REQUIRED PATHS /home/cyz/workspace/Super4PCS-master/build/install/lib/cmake NO_DEFAULT_PATH)
include_directories(${Super4PCS_INCLUDE_DIR})
link_directories(${Super4PCS_LIB_DIR})

add_executable(graphOptimization src/graphOptimization.cpp src/tools.cpp)
add_dependencies(graphOptimization ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_generate_messages_cpp ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(graphOptimization ${COMMON_LIB} ${Super4PCS_LIBRARIES} gtsam tbb)


