//
// Created by joe on 2020/10/6.
//

#ifndef STRUCTURAL_MAPPING_CERESSPLINECOST_H
#define STRUCTURAL_MAPPING_CERESSPLINECOST_H

#include "tools.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct CeresCostFactor{

    CeresCostFactor(){
        weight_ = new ceres::LossFunctionWrapper(  // reweighted
                new ceres::ScaledLoss(new ceres::CauchyLoss(0.1), 1, ceres::TAKE_OWNERSHIP),
                ceres::TAKE_OWNERSHIP);
    }

//    virtual bool operator()(const double *q, const double *t, double *residual) ;

    void updateWeight(double w){
//        if(weight_){
//            delete weight_;
////            weight_ = NULL;
//        }
        weight_->Reset(
                new ceres::ScaledLoss(NULL, w, ceres::TAKE_OWNERSHIP),
                ceres::TAKE_OWNERSHIP);
    }

    ceres::LossFunctionWrapper* weight() { return weight_; }

    ceres::LossFunctionWrapper* weight_;

};

// estimate the control poses of spline
class CeresFactorsSP{

public:

    /// Calculate the coefficients of catmull rom spline which pass the control points except first and last ones
    /// \param u : ratio between two control points, [0, 1]
    /// \param coeff : in order to include all 5 control points, we manually build two CR spline individually
    /// through the first four points and last four points
    static void catmullRomSplineCoeff(double u, double* &coeff);
    /// use 1,2,4,5 control points to build CR spline
    static void catmullRomSplineCoeff_One(double u, double* &coeff);

    /// Calculate the coefficients of clamped basic spline which pass the first and last control point
    /// \param u : target position in [0,1]
    /// \param n : control points number, from 0 to n
    /// \param p : degree of spline
    /// \return coeff : size==n and initialize as {0}
    static void splineBasicCoeffCalcu(double u, int n, int p, double* &coeff);
    // half clamped spline
    static void splineBasicCoeffCalcuHalf(double u, int n, int p, double* &coeff);

    // Use the inverse of coefficients matrix to calculate the control points from data points
    // NOTE: For B-spline; The number of data points is same to control points
    static void fromDataToControlpoints(std::vector<double *> &datapts);
    // calculate N control points from dynamic data points
    static void fromDataToControlpointsDynamic(std::vector<double *> &datapts, int N = 5);

    /// Fuse multiple poses into spline in Rotation and Translation
    /// \tparam T : data type
    /// \param p0 p1 p2 p3 p4 : poses in [qx, qy, qz, w, x, y, z, t]
    /// \param u : [0, 1]
    /// \return pose at u
    template <class T>
    static bool splineFusionPoses(const T* p0, const T* p1, const T* p2, const T* p3, const T* p4,
                                T* newpose, T u, int spT = 0){

        Eigen::Quaternion<T> q0 {p0[3], p0[0], p0[1], p0[2]};
        Eigen::Quaternion<T> q1 {p1[3], p1[0], p1[1], p1[2]};
        Eigen::Quaternion<T> q2 {p2[3], p2[0], p2[1], p2[2]};
        Eigen::Quaternion<T> q3 {p3[3], p3[0], p3[1], p3[2]};
        Eigen::Quaternion<T> q4 {p4[3], p4[0], p4[1], p4[2]};

        Eigen::Matrix<T, 3, 1> t0 {p0[4], p0[5], p0[6]};
        Eigen::Matrix<T, 3, 1> t1 {p1[4], p1[5], p1[6]};
        Eigen::Matrix<T, 3, 1> t2 {p2[4], p2[5], p2[6]};
        Eigen::Matrix<T, 3, 1> t3 {p3[4], p3[5], p3[6]};
        Eigen::Matrix<T, 3, 1> t4 {p4[4], p4[5], p4[6]};

        double* coeffs;
        if(spT == 0)
            splineBasicCoeffCalcu(u, 4, 3, coeffs);
        else
            catmullRomSplineCoeff_One(u, coeffs);
//                    cout << BLACK << "Spline Coeffs " << RESET << endl;
//        for (int i = 0; i < 5; ++i) {
//            cout << coeffs[i] << endl;
//        }
        Eigen::Matrix<T, 3, 1> t = coeffs[0] * t0 + coeffs[1] * t1 + coeffs[2] * t2 + coeffs[3] * t3 + coeffs[4] * t4;
//        cout << t << endl;

        Eigen::Quaternion<T> q = Eigen::Quaternion<T>::Identity();
        // TODO nearly no difference for these two formula/way ???
        // 1. incremental
//        q1 = q0.inverse() * q1;
//        q2 = q0.inverse() * q2;
//        q3 = q0.inverse() * q3;
//        q4 = q0.inverse() * q4;
//        q = q0 * (q.slerp(coeffs[1], q1)) * (q.slerp(coeffs[2], q2)) * (q.slerp(coeffs[3], q3)) * (q.slerp(coeffs[4], q4));
        // 2. direct intepolate
        q = (q.slerp(coeffs[0], q0)) * (q.slerp(coeffs[1], q1)) * (q.slerp(coeffs[2], q2)) * (q.slerp(coeffs[3], q3)) * (q.slerp(coeffs[4], q4));

        newpose[0] = q.x();
        newpose[1] = q.y();
        newpose[2] = q.z();
        newpose[3] = q.w();

        newpose[4] = t(0);
        newpose[5] = t(1);
        newpose[6] = t(2);

        newpose[7] = p0[7] + u * (p4[7] - p0[7]);

        delete coeffs;
        coeffs = NULL;
    }


    // for spline
    struct LidarEdgeFactorSP : CeresCostFactor{

        LidarEdgeFactorSP(const Eigen::Vector3d &curr_point_,
                          const Eigen::Vector3d &last_point_a_,
                          const Eigen::Vector3d &last_point_b_,
                          double s_, int spType = 0): curr_point(curr_point_),
                                                      last_point_a(last_point_a_),
                                                      last_point_b(last_point_b_),
                                                      u(s_),
                                                      splineType(spType){

            weight_ = new ceres::LossFunctionWrapper(  // reweighted
                    new ceres::ScaledLoss(NULL, 1, ceres::TAKE_OWNERSHIP),
                    ceres::TAKE_OWNERSHIP);

            if(splineType == 0)
                splineBasicCoeffCalcu(u, 4, 3, coeffs);
            else
                catmullRomSplineCoeff_One(u, coeffs);

//            cout << BLACK << "Spline Coeffs : " << RESET << endl;
//            for (int i = 0; i < 5; ++i)
//             cout << coeffs[i] << endl;

        }

        ~LidarEdgeFactorSP(){
            if(coeffs!=NULL)
                delete coeffs;
        }


        template <typename T>
        bool operator()(const T* quat0,const T* quat1,const T* quat2,const T* quat3,const T* quat4,
                        const T* trans0,const T* trans1,const T* trans2,const T* trans3,const T* trans4,
                        T* residual) const {

            Eigen::Quaternion<T> q0 {quat0[3], quat0[0], quat0[1], quat0[2]};
            Eigen::Quaternion<T> q1 {quat1[3], quat1[0], quat1[1], quat1[2]};
            Eigen::Quaternion<T> q2 {quat2[3], quat2[0], quat2[1], quat2[2]};
            Eigen::Quaternion<T> q3 {quat3[3], quat3[0], quat3[1], quat3[2]};
            Eigen::Quaternion<T> q4 {quat4[3], quat4[0], quat4[1], quat4[2]};

            Eigen::Matrix<T, 3, 1> t0 {trans0[0], trans0[1], trans0[2]};
            Eigen::Matrix<T, 3, 1> t1 {trans1[0], trans1[1], trans1[2]};
            Eigen::Matrix<T, 3, 1> t2 {trans2[0], trans2[1], trans2[2]};
            Eigen::Matrix<T, 3, 1> t3 {trans3[0], trans3[1], trans3[2]};
            Eigen::Matrix<T, 3, 1> t4 {trans4[0], trans4[1], trans4[2]};

            Eigen::Matrix<T, 3, 1> t = T(coeffs[0]) * t0 +
                                       T(coeffs[1]) * t1 + T(coeffs[2]) * t2 +
                                       T(coeffs[3]) * t3 + T(coeffs[4]) * t4;
//        cout << t << endl;

            Eigen::Quaternion<T> q = Eigen::Quaternion<T>::Identity();
            // TODO nearly no difference for these two formula/way ???
            // 1. incremental
//            q1 = q0.inverse() * q1;
//            q2 = q0.inverse() * q2;
//            q3 = q0.inverse() * q3;
//            q4 = q0.inverse() * q4;
//            q = q0 * (q.slerp(coeffs[1], q1)) *
//                (q.slerp(coeffs[2], q2)) *
//                (q.slerp(coeffs[3], q3)) *
//                (q.slerp(coeffs[4], q4));
            // 2. direct intepolate
            q = (q.slerp( T(coeffs[0]), q0)) *
                (q.slerp( T(coeffs[1]), q1)) *
                (q.slerp( T(coeffs[2]), q2)) *
                (q.slerp( T(coeffs[3]), q3)) *
                (q.slerp( T(coeffs[4]), q4));
//            ceres::QuaternionProduct(p0, p1, p2);
//            ceres::QuaternionRotatePoint(p0, point_x, transformed_point);

            Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
            Eigen::Matrix<T, 3, 1> lpa{T(last_point_a.x()), T(last_point_a.y()), T(last_point_a.z())};
            Eigen::Matrix<T, 3, 1> lpb{T(last_point_b.x()), T(last_point_b.y()), T(last_point_b.z())};

            Eigen::Matrix<T, 3, 1> lp;
            lp = q * cp + t;

            Eigen::Matrix<T, 3, 1> nu = (lp - lpa).cross(lp - lpb);
            Eigen::Matrix<T, 3, 1> de = lpa - lpb;

            residual[0] = nu.x() / de.norm();
            residual[1] = nu.y() / de.norm();
            residual[2] = nu.z() / de.norm();

            return true;
        }

        static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &last_point_a_,
                                           const Eigen::Vector3d &last_point_b_, const double s_, int spType_ = 0)
        {
            return (new ceres::AutoDiffCostFunction<LidarEdgeFactorSP, 3, 4,4,4,4,4, 3,3,3,3,3>(
                    new LidarEdgeFactorSP(curr_point_, last_point_a_, last_point_b_, s_, spType_)));
        }

        ceres::CostFunction *costFunc()
        {
            return (new ceres::AutoDiffCostFunction<LidarEdgeFactorSP, 3, 4,4,4,4,4, 3,3,3,3,3>(
                    new LidarEdgeFactorSP(curr_point, last_point_a, last_point_b, u, splineType)));
        }

        Eigen::Vector3d curr_point, last_point_a, last_point_b;
        double u;
        double *coeffs;
        int splineType;

    };
    struct LidarPlaneNormFactorSP : CeresCostFactor
    {

        LidarPlaneNormFactorSP(const Eigen::Vector3d &curr_point_,
                               const Eigen::Vector3d &plane_unit_norm_,
                               double negative_OA_dot_norm_,
                               double s_,
                               int spType = 0) : curr_point(curr_point_),
                                                 plane_unit_norm(plane_unit_norm_),
                                                 negative_OA_dot_norm(negative_OA_dot_norm_),
                                                 u(s_),
                                                 splineType(spType){
            weight_ = new ceres::LossFunctionWrapper(  // reweighted
                    new ceres::ScaledLoss(NULL, 1, ceres::TAKE_OWNERSHIP),
                    ceres::TAKE_OWNERSHIP);

            if(splineType == 0)
                splineBasicCoeffCalcu(u, 4, 3, coeffs);
            else
                catmullRomSplineCoeff_One(u, coeffs);

        }

        ~LidarPlaneNormFactorSP(){
            if(coeffs!=NULL)
                delete coeffs;
        }

        template <typename T>
        bool operator()(const T* quat0,const T* quat1,const T* quat2,const T* quat3,const T* quat4,
                        const T* trans0,const T* trans1,const T* trans2,const T* trans3,const T* trans4,
                        T* residual) const {

            Eigen::Quaternion<T> q0 {quat0[3], quat0[0], quat0[1], quat0[2]};
            Eigen::Quaternion<T> q1 {quat1[3], quat1[0], quat1[1], quat1[2]};
            Eigen::Quaternion<T> q2 {quat2[3], quat2[0], quat2[1], quat2[2]};
            Eigen::Quaternion<T> q3 {quat3[3], quat3[0], quat3[1], quat3[2]};
            Eigen::Quaternion<T> q4 {quat4[3], quat4[0], quat4[1], quat4[2]};

            Eigen::Matrix<T, 3, 1> t0 {trans0[0], trans0[1], trans0[2]};
            Eigen::Matrix<T, 3, 1> t1 {trans1[0], trans1[1], trans1[2]};
            Eigen::Matrix<T, 3, 1> t2 {trans2[0], trans2[1], trans2[2]};
            Eigen::Matrix<T, 3, 1> t3 {trans3[0], trans3[1], trans3[2]};
            Eigen::Matrix<T, 3, 1> t4 {trans4[0], trans4[1], trans4[2]};


            Eigen::Matrix<T, 3, 1> t = T(coeffs[0]) * t0 +
                                       T(coeffs[1]) * t1 + T(coeffs[2]) * t2 +
                                       T(coeffs[3]) * t3 + T(coeffs[4]) * t4;
//        cout << t << endl;

            Eigen::Quaternion<T> q = Eigen::Quaternion<T>::Identity();
            // TODO nearly no difference for these two formula/way ???
            // 1. incremental
//            q1 = q0.inverse() * q1;
//            q2 = q0.inverse() * q2;
//            q3 = q0.inverse() * q3;
//            q4 = q0.inverse() * q4;
//            q = q0 * (q.slerp(coeffs[1], q1)) *
//                (q.slerp(coeffs[2], q2)) *
//                (q.slerp(coeffs[3], q3)) *
//                (q.slerp(coeffs[4], q4));
            // 2. direct interpolate
            q = (q.slerp( T(coeffs[0]), q0)) *
                (q.slerp( T(coeffs[1]), q1)) *
                (q.slerp( T(coeffs[2]), q2)) *
                (q.slerp( T(coeffs[3]), q3)) *
                (q.slerp( T(coeffs[4]), q4));

            Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
            Eigen::Matrix<T, 3, 1> point_w;
            point_w = q * cp + t;

            Eigen::Matrix<T, 3, 1> norm(T(plane_unit_norm.x()), T(plane_unit_norm.y()), T(plane_unit_norm.z()));
            residual[0] = (norm.dot(point_w) + T(negative_OA_dot_norm)) ;

            return true;
        }

        static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &plane_unit_norm_,
                                           const double negative_OA_dot_norm_, double s_, int spType_ = 0)
        {
            return (new ceres::AutoDiffCostFunction<LidarPlaneNormFactorSP, 1, 4,4,4,4,4, 3,3,3,3,3>(
                    new LidarPlaneNormFactorSP(curr_point_, plane_unit_norm_, negative_OA_dot_norm_, s_, spType_)));
        }

        ceres::CostFunction *costFunc()
        {
            return (new ceres::AutoDiffCostFunction<LidarPlaneNormFactorSP, 1, 4,4,4,4,4, 3,3,3,3,3>(
                    new LidarPlaneNormFactorSP(curr_point, plane_unit_norm, negative_OA_dot_norm, u, splineType)));
        }

        Eigen::Vector3d curr_point;
        Eigen::Vector3d plane_unit_norm;
        double negative_OA_dot_norm ,u;
        double* coeffs;
        int splineType;

    };
    struct LidarSurfelFactorSP : CeresCostFactor
    {

        LidarSurfelFactorSP(const Eigen::Vector3d &curr_point_,
                            const Eigen::Vector3d &centro_,
                            const Eigen::Matrix3d &cov_,
                            double confi_,
                            double s_,
                            int spType = 0) : curr_point(curr_point_),
                                              centro(centro_),
                                              confi(confi_),  // confidence
                                              u(s_),
                                              splineType(spType){
            weight_ = new ceres::LossFunctionWrapper(  // reweighted
                    new ceres::ScaledLoss(NULL, 1, ceres::TAKE_OWNERSHIP),
                    ceres::TAKE_OWNERSHIP);

            info = Eigen::LLT<Eigen::Matrix<double , 3,3> >(cov_.inverse()).matrixL().transpose();
//            cout << "[ FACTOR ] INFO : " << info << endl;

            if(splineType == 0)
                splineBasicCoeffCalcu(u, 4, 3, coeffs);
            else
                catmullRomSplineCoeff_One(u, coeffs);

        }  // time offest

        template <typename T>
        bool operator()(const T* quat0,const T* quat1,const T* quat2,const T* quat3,const T* quat4,
                        const T* trans0,const T* trans1,const T* trans2,const T* trans3,const T* trans4,
                        T* residual) const {

            Eigen::Quaternion<T> q0 {quat0[3], quat0[0], quat0[1], quat0[2]};
            Eigen::Quaternion<T> q1 {quat1[3], quat1[0], quat1[1], quat1[2]};
            Eigen::Quaternion<T> q2 {quat2[3], quat2[0], quat2[1], quat2[2]};
            Eigen::Quaternion<T> q3 {quat3[3], quat3[0], quat3[1], quat3[2]};
            Eigen::Quaternion<T> q4 {quat4[3], quat4[0], quat4[1], quat4[2]};

            Eigen::Matrix<T, 3, 1> t0 {trans0[0], trans0[1], trans0[2]};
            Eigen::Matrix<T, 3, 1> t1 {trans1[0], trans1[1], trans1[2]};
            Eigen::Matrix<T, 3, 1> t2 {trans2[0], trans2[1], trans2[2]};
            Eigen::Matrix<T, 3, 1> t3 {trans3[0], trans3[1], trans3[2]};
            Eigen::Matrix<T, 3, 1> t4 {trans4[0], trans4[1], trans4[2]};

            Eigen::Matrix<T, 3, 1> t = T(coeffs[0]) * t0 +
                                       T(coeffs[1]) * t1 + T(coeffs[2]) * t2 +
                                       T(coeffs[3]) * t3 + T(coeffs[4]) * t4;
//        cout << t << endl;

            Eigen::Quaternion<T> q = Eigen::Quaternion<T>::Identity();
            // TODO nearly no difference for these two formula/way ???
            // 1. incremental
//            q1 = q0.inverse() * q1;
//            q2 = q0.inverse() * q2;
//            q3 = q0.inverse() * q3;
//            q4 = q0.inverse() * q4;
//            q = q0 * (q.slerp(coeffs[1], q1)) *
//                (q.slerp(coeffs[2], q2)) *
//                (q.slerp(coeffs[3], q3)) *
//                (q.slerp(coeffs[4], q4));
            // 2. direct interpolate
            q = (q.slerp( T(coeffs[0]), q0)) *
                (q.slerp( T(coeffs[1]), q1)) *
                (q.slerp( T(coeffs[2]), q2)) *
                (q.slerp( T(coeffs[3]), q3)) *
                (q.slerp( T(coeffs[4]), q4));

            Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
            Eigen::Matrix<T, 3, 1> point_w;
            point_w = q * cp + t;

//            Eigen::Matrix<T, 3, 3> info = Eigen::LLT<Eigen::Matrix<T, 3,3> >(cov.cast<T>().inverse()).matrixL().transpose();

            Eigen::Matrix<T, 3, 1> cen(T(centro.x()), T(centro.y()), T(centro.z()));
            Eigen::Map<Eigen::Matrix<T, 3, 1> > res (residual);
            res = T(confi) * info.cast<T>() * (point_w - cen) ;

//            residual[0] = dist(0) *T(confi);
//            residual[1] = dist(1) *T(confi);
//            residual[2] = dist(2) *T(confi);

            return true;
        }

        static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &centro_,
                                           const Eigen::Matrix3d &cov_, double confi_, double s_, int spType_ = 0)
        {
            return (new ceres::AutoDiffCostFunction<LidarSurfelFactorSP, 3, 4,4,4,4,4, 3,3,3,3,3>(
                    new LidarSurfelFactorSP(curr_point_, centro_, cov_, confi_, s_, spType_)));
        }

        ceres::CostFunction *costFunc()
        {
            return (new ceres::AutoDiffCostFunction<LidarSurfelFactorSP, 3, 4,4,4,4,4, 3,3,3,3,3>(
                    new LidarSurfelFactorSP(curr_point, centro, info, confi, u, splineType)));
        }

        Eigen::Vector3d curr_point;
        Eigen::Vector3d centro;
        Eigen::Matrix3d info;
        double u, confi;
        double* coeffs;
        int splineType;

    };

};

#endif //STRUCTURAL_MAPPING_CERESSPLINECOST_H
