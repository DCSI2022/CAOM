//
// Created by cyz on 2021/4/12.
//

#ifndef STRUCTURAL_MAPPING_CERESSPLINECOSTS03_H
#define STRUCTURAL_MAPPING_CERESSPLINECOSTS03_H

#include "tools.h"
#include "so3spline.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace SophusSpline {

    template<int N_>
    struct CeresCostFactor : public ceres::SizedCostFunction<N_, 7, 7, 7, 7, 7> {

        CeresCostFactor() {
            weight_ = new ceres::LossFunctionWrapper(  // reweighted
                    new ceres::ScaledLoss(new ceres::CauchyLoss(0.1), 1, ceres::TAKE_OWNERSHIP),
                    ceres::TAKE_OWNERSHIP);
        }

//    virtual bool operator()(const double *q, const double *t, double *residual);
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {}

        void updateWeight(double w) {
//        if(weight_){
//            delete weight_;
////            weight_ = NULL;
//        }
            weight_->Reset(
                    new ceres::ScaledLoss(NULL, w, ceres::TAKE_OWNERSHIP),
                    ceres::TAKE_OWNERSHIP);
        }

        ceres::LossFunctionWrapper *weight() { return weight_; }

        ceres::LossFunctionWrapper *weight_;

    };

    void getTransformFromSe3(const Eigen::Matrix<double,6,1>& se3, Eigen::Quaterniond& q, Eigen::Vector3d& t){

        Eigen::Vector3d omega(se3.data()+3);  // rotation
        Eigen::Vector3d upsilon(se3.data());  // translation
        Eigen::Matrix3d Omega = skew(omega);

        double theta = omega.norm();
        double half_theta = 0.5*theta;

        double imag_factor;
        double real_factor = cos(half_theta);
        if(theta < 1e-10)
        {
            double theta_sq = theta*theta;
            double theta_po4 = theta_sq*theta_sq;
            imag_factor = 0.5-0.0208333*theta_sq+0.000260417*theta_po4;
        }
        else
        {
            double sin_half_theta = sin(half_theta);
            imag_factor = sin_half_theta/theta;
        }

        q = Eigen::Quaterniond(real_factor, imag_factor*omega.x(), imag_factor*omega.y(), imag_factor*omega.z());

        Eigen::Matrix3d J;
        if (theta<1e-10)
            J = q.matrix();
        else{
            Eigen::Matrix3d Omega2 = Omega*Omega;
            J = (Eigen::Matrix3d::Identity() + (1-cos(theta))/(theta*theta)*Omega + (theta-sin(theta))/(pow(theta,3))*Omega2);
        }

        t = J*upsilon;
    }

    class PoseSE3SophusParameterization : public ceres::LocalParameterization {
    public:

        PoseSE3SophusParameterization() {}

        virtual ~PoseSE3SophusParameterization() {}

        virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const {

            // from Quaternion
//            Eigen::Map<const Eigen::Quaterniond> quater(x);
//            Eigen::Map<const Eigen::Vector3d> trans(x + 4);
//            Eigen::Map<Eigen::Quaterniond> quater_plus(x_plus_delta);
//            Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 4);
//
//            Eigen::Quaterniond delta_q;
//            Eigen::Vector3d delta_t;
//            getTransformFromSe3(Eigen::Map<const Eigen::Matrix<double,6,1>>(delta), delta_q, delta_t);
//            quater_plus = delta_q * quater;
//            trans_plus = delta_q * trans + delta_t;

            // Sophus
            Eigen::Map< const Sophus::SE3<double> > T0(x);
            Eigen::Map< Sophus::SE3<double> > T1(x_plus_delta);

            Sophus::SE3<double> se3_delta = Sophus::SE3<double>::exp(
                    Eigen::Map<const Eigen::Matrix<double, 6, 1>>(delta));
            T1 = se3_delta * T0;

            return true;
        }

        // (X + deltaX) derivative to X, from manifold to tangent Space
        virtual bool ComputeJacobian(const double *x, double *jacobian) const {

            Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
            (j.topRows(6)).setIdentity();
            (j.bottomRows(1)).setZero();

//        Eigen::Map<const Eigen::Quaterniond> quater(x);
//        Eigen::Map<const Eigen::Vector3d> trans(x + 4);
//        Sophus::SE3<double> T(quater, trans);
//        j = T.Dx_exp_x_at_0();

//            Eigen::Map<const Sophus::SE3<double> > T(x);
//            j = T.Dx_this_mul_exp_x_at_0();
            return true;
        }

        virtual int GlobalSize() const { return 7; }

        virtual int LocalSize() const { return 6; }  // in tangent space
    };

    // estimate the control poses of spline
    class CeresFactorsSP {

    public:

        static void interpolatePoseSE3(const double ratio,
                                       const Eigen::Quaterniond &q1,
                                       const Eigen::Quaterniond &q2,
                                       const Eigen::Vector3d &t1,
                                       const Eigen::Vector3d &t2,
                                       Eigen::Quaterniond &q, Eigen::Vector3d &t) ;

        // interpolate to get initial control points
        static void fromDataToControlpointsDynamic(std::vector<double *> &datapts, int N = 5) ;

        /// Fuse multiple poses into spline in Rotation and Translation
        /// \tparam T : data type
        /// \param p0 p1 p2 p3 p4 : poses in [qx, qy, qz, w, x, y, z, t]
        /// \param u : [0, 1]
        /// \return pose at u
        template<class T>
        static bool splineFusionPoses(const T *p0, const T *p1, const T *p2, const T *p3, const T *p4,
                                      T* newpose, T u, int spT = 0) {

            int64_t totalT = (p4[7] - p0[7]) * 1e9;
            int64_t time_interval_ns = totalT / 4;  // 2 = 5(the num of control points) - 4(order of spline) + 1
            basalt::Se3Spline<4> se3splineHelper = basalt::Se3Spline<4>(time_interval_ns);

//            if(u > 0.5) return false;
            u = (u >= 0.5 ? 0.5-1e-5 : u);

            Eigen::Map< const Sophus::SE3<T> > T0 (p0);
            Eigen::Map< const Sophus::SE3<T> > T1 (p1);
            Eigen::Map< const Sophus::SE3<T> > T2 (p2);
            Eigen::Map< const Sophus::SE3<T> > T3 (p3);
            Eigen::Map< const Sophus::SE3<T> > T4 (p4);

            se3splineHelper.knots_push_back(T0);
            se3splineHelper.knots_push_back(T1);
            se3splineHelper.knots_push_back(T2);
            se3splineHelper.knots_push_back(T3);
            se3splineHelper.knots_push_back(T4);

            Sophus::SE3<T> pose = se3splineHelper.pose(u*totalT);  // s -> ns
            Eigen::Matrix<T, 3, 1> t = pose.translation();
            Sophus::SO3<T> rot = pose.so3();
            Eigen::Quaternion<T> q = rot.unit_quaternion();

            newpose[0] = q.x();
            newpose[1] = q.y();
            newpose[2] = q.z();
            newpose[3] = q.w();

            newpose[4] = t(0);
            newpose[5] = t(1);
            newpose[6] = t(2);

            newpose[7] = p0[7] + u * (p4[7] - p0[7]);
        }

        // for spline
//    struct LidarEdgeFactorSP : CeresCostFactor<3>{
        struct LidarEdgeFactorSP : public ceres::SizedCostFunction<3, 7, 7, 7, 7, 7> {

            LidarEdgeFactorSP(const Eigen::Vector3d &curr_point_,
                              const Eigen::Vector3d &last_point_a_,
                              const Eigen::Vector3d &last_point_b_,
                              double s_, double dura, int spType = 0) : curr_point(curr_point_),
                                                                        last_point_a(last_point_a_),
                                                                        last_point_b(last_point_b_),
                                                                        u(s_),
                                                                        splineType(spType) {

//            weight_ = new ceres::LossFunctionWrapper(  // reweighted
//                    new ceres::ScaledLoss(NULL, 1, ceres::TAKE_OWNERSHIP),
//                    ceres::TAKE_OWNERSHIP);

                u = (u >= 0.5 ? 0.5-1e-5 : u);
                int64_t totalT = (dura) * 1e9;
                time_interval_ns = totalT / 4;
//                cout << "[ DEBUG ] Uniform spline time interval in ns : " << time_interval_ns << endl;
                eval_time_ns = u * totalT;
            }

            virtual ~LidarEdgeFactorSP() {}

            virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

            Eigen::Vector3d curr_point, last_point_a, last_point_b;
            double u;
            int splineType;
            int64_t time_interval_ns, eval_time_ns;
        };

//    struct LidarPlaneNormFactorSP : CeresCostFactor<1>{
        struct LidarPlaneNormFactorSP : public ceres::SizedCostFunction<1, 7, 7, 7, 7, 7> {

            LidarPlaneNormFactorSP(const Eigen::Vector3d &curr_point_,
                                   const Eigen::Vector3d &plane_unit_norm_,
                                   double negative_OA_dot_norm_,
                                   double s_, double dura,
                                   int spType = 0) : curr_point(curr_point_),
                                                     plane_unit_norm(plane_unit_norm_),
                                                     negative_OA_dot_norm(negative_OA_dot_norm_),
                                                     u(s_),
                                                     splineType(spType) {
//            weight_ = new ceres::LossFunctionWrapper(  // reweighted
//                    new ceres::ScaledLoss(NULL, 1, ceres::TAKE_OWNERSHIP),
//                    ceres::TAKE_OWNERSHIP);

                u = (u >= 0.5 ? 0.5-1e-5 : u);
                int64_t totalT = (dura) * 1e9;
                time_interval_ns = totalT / 4;
//                cout << "[ DEBUG ] Uniform spline time interval in ns : " << time_interval_ns << endl;
                eval_time_ns = u * totalT;
            }

            virtual ~LidarPlaneNormFactorSP() {
            }

            virtual bool Evaluate(double const *const *parameters,
                                  double *residuals, double **jacobians) const ;

            Eigen::Vector3d curr_point;
            Eigen::Vector3d plane_unit_norm;
            double negative_OA_dot_norm, u;
            int splineType;
            int64_t time_interval_ns, eval_time_ns;

        };


//        // todo : surfel cost
//    struct LidarSurfelFactorSP : CeresCostFactor<3>{
//
//        LidarSurfelFactorSP(const Eigen::Vector3d &curr_point_,
//                            const Eigen::Vector3d &centro_,
//                            const Eigen::Matrix3d &cov_,
//                            double confi_,
//                            double s_,
//                            int spType = 0) : curr_point(curr_point_),
//                                              centro(centro_),
//                                              confi(confi_),  // confidence
//                                              u(s_),
//                                              splineType(spType){
//            weight_ = new ceres::LossFunctionWrapper(  // reweighted
//                    new ceres::ScaledLoss(NULL, 1, ceres::TAKE_OWNERSHIP),
//                    ceres::TAKE_OWNERSHIP);
//
//            info = Eigen::LLT<Eigen::Matrix<double , 3,3> >(cov_.inverse()).matrixL().transpose();
////            cout << "[ FACTOR ] INFO : " << info << endl;
//
//        }  // time offest
//
//        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const{
//
//            basalt::Se3Spline<4> se3splineHepler = basalt::Se3Spline<4>(1e8);  // 0.1s -> ns
//
//            Eigen::Quaternion<double> q0 {parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]};
//            Eigen::Quaternion<double> q1 {parameters[1][3], parameters[1][0], parameters[1][1], parameters[1][2]};
//            Eigen::Quaternion<double> q2 {parameters[2][3], parameters[2][0], parameters[2][1], parameters[2][2]};
//            Eigen::Quaternion<double> q3 {parameters[3][3], parameters[3][0], parameters[3][1], parameters[3][2]};
//            Eigen::Quaternion<double> q4 {parameters[4][3], parameters[4][0], parameters[4][1], parameters[4][2]};
//
//            Eigen::Matrix<double, 3, 1> t0 {parameters[0][4], parameters[0][5], parameters[0][6]};
//            Eigen::Matrix<double, 3, 1> t1 {parameters[1][4], parameters[1][5], parameters[1][6]};
//            Eigen::Matrix<double, 3, 1> t2 {parameters[2][4], parameters[2][5], parameters[2][6]};
//            Eigen::Matrix<double, 3, 1> t3 {parameters[3][4], parameters[3][5], parameters[3][6]};
//            Eigen::Matrix<double, 3, 1> t4 {parameters[4][4], parameters[4][5], parameters[4][6]};
//
//            se3splineHepler.knots_push_back(Sophus::SE3<double>(q0, t0));
//            se3splineHepler.knots_push_back(Sophus::SE3<double>(q1, t1));
//            se3splineHepler.knots_push_back(Sophus::SE3<double>(q2, t2));
//            se3splineHepler.knots_push_back(Sophus::SE3<double>(q3, t3));
//            se3splineHepler.knots_push_back(Sophus::SE3<double>(q4, t4));
//
//            basalt::Se3Spline<4>::PosePosSO3JacobianStruct* J_spline;
//            Sophus::SE3<double> pose = se3splineHepler.pose(u*(parameters[4][7]-parameters[0][7])*1e9, J_spline);  // s -> ns
//            Sophus::SO3<double> rot = pose.so3();
//
//            Eigen::Quaterniond q_last_curr(rot.unit_quaternion());
//            Eigen::Vector3d t_last_curr = pose.translation();
//
//            Eigen::Matrix<double, 3, 1> cp{double(curr_point.x()), double(curr_point.y()), double(curr_point.z())};
//            Eigen::Matrix<double, 3, 1> point_w;
//            point_w = q_last_curr * cp + t_last_curr;
//
////            Eigen::Matrix<double, 3, 3> info = Eigen::LLT<Eigen::Matrix<double, 3,3> >(cov.cast<double>().inverse()).matrixL().transpose();
//
//            Eigen::Matrix<double, 3, 1> cen(double(centro.x()), double(centro.y()), double(centro.z()));
//            Eigen::Map<Eigen::Matrix<double, 3, 1> > res (residuals);
//            res = double(confi) * info.cast<double>() * (point_w - cen) ;
//
////            residual[0] = dist(0) *double(confi);
////            residual[1] = dist(1) *double(confi);
////            residual[2] = dist(2) *double(confi);
//
//            if(jacobians != NULL){
//
//                Eigen::Matrix3d skew_lp = skew(lp);
//                Eigen::Matrix<double, 3, 6> dp_by_so3;
//                dp_by_so3.block<3,3>(0,0) = -skew_lp;
//                (dp_by_so3.block<3,3>(0, 3)).setIdentity();
//                Eigen::Matrix<double, 3, 7, Eigen::RowMajor> J_se3;
//                J_se3.setZero();
//                Eigen::Vector3d re = last_point_b - last_point_a;
//                Eigen::Matrix3d skew_re = skew(re);
//                J_se3.block<3,6>(0,0) = skew_re * dp_by_so3/de.norm();
//
//                if(u < 0.5){  // spline controlled by the first four points
//                    if(jacobians[0] != NULL){
//
//                        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_0(jacobians[0]);
//                        J_se3_0.setZero();
//                        J_se3_0.block<3,6>(0,0) = J_se3.block<3,6>(0,0) * J_spline.d_val_d_knot[0];
//                    }
//                    if(jacobians[1] != NULL){
//
//                        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_1(jacobians[1]);
//                        J_se3_1.setZero();
//                        J_se3_1.block<3,6>(0,0) = J_se3.block<3,6>(0,0) * J_spline.d_val_d_knot[1];
//                    }
//                    if(jacobians[2] != NULL){
//
//                        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_2(jacobians[2]);
//                        J_se3_2.setZero();
//                        J_se3_2.block<3,6>(0,0) = J_se3.block<3,6>(0,0) * J_spline.d_val_d_knot[2];
//                    }
//                    if(jacobians[3] != NULL){
//
//                        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_3(jacobians[3]);
//                        J_se3_3.setZero();
//                        J_se3_3.block<3,6>(0,0) = J_se3.block<3,6>(0,0) * J_spline.d_val_d_knot[3];
//                    }
//                    if(jacobians[4] != NULL){
//                        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_4(jacobians[4]);
//                        J_se3_4.setZero();
//                    }
//                }else{  // spline controlled by the last four points
//                    if(jacobians[0] != NULL){
//
//                        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_0(jacobians[0]);
//                        J_se3_0.setZero();
//                    }
//                    if(jacobians[1] != NULL){
//
//                        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_1(jacobians[1]);
//                        J_se3_1.setZero();
//                        J_se3_1.block<3,6>(0,0) = J_se3.block<3,6>(0,0) * J_spline.d_val_d_knot[0];
//                    }
//                    if(jacobians[2] != NULL){
//
//                        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_2(jacobians[2]);
//                        J_se3_2.setZero();
//                        J_se3_2.block<3,6>(0,0) = J_se3.block<3,6>(0,0) * J_spline.d_val_d_knot[1];
//                    }
//                    if(jacobians[3] != NULL){
//
//                        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_3(jacobians[3]);
//                        J_se3_3.setZero();
//                        J_se3_3.block<3,6>(0,0) = J_se3.block<3,6>(0,0) * J_spline.d_val_d_knot[2];
//                    }
//                    if(jacobians[4] != NULL){
//
//                        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_4(jacobians[4]);
//                        J_se3_4.setZero();
//                        J_se3_4.block<3,6>(0,0) = J_se3.block<3,6>(0,0) * J_spline.d_val_d_knot[3];
//                    }
//                }
//
//            }
//
//            return true;
//        }
//
//
//        static ceres::CostFunction *Create(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &centro_,
//                                           const Eigen::Matrix3d &cov_, double confi_, double s_, int spType_ = 0)
//        {
//            return (new LidarSurfelFactorSP(curr_point_, centro_, cov_, confi_, s_, spType_));
//        }
//
//        ceres::CostFunction *costFunc()
//        {
//            return (new LidarSurfelFactorSP(curr_point, centro, info, confi, u, splineType));
//        }
//
//        Eigen::Vector3d curr_point;
//        Eigen::Vector3d centro;
//        Eigen::Matrix3d info;
//        double u, confi;
//        double* coeffs;
//        int splineType;
//    };

    };
}

#endif //STRUCTURAL_MAPPING_CERESSPLINECOSTS03_H
