//
// Created by joe on 2020/10/5.
//

#ifndef STRUCTURAL_MAPPING_CERESANALYTICCOST_H
#define STRUCTURAL_MAPPING_CERESANALYTICCOST_H

#include "tools.h"

#include <ceres/ceres.h>
#include <ceres/rotation.h>

class EdgeAnalyticCostFunction : public ceres::SizedCostFunction<3, 7> {
public:

    EdgeAnalyticCostFunction(Eigen::Vector3d &curr_point_,
                             Eigen::Vector3d &last_point_a_,
                             Eigen::Vector3d &last_point_b_,
                             double weight_ = 1.0):curr_point(curr_point_) ,
                                                            last_point_a(last_point_a_),
                                                            last_point_b(last_point_b_),
                                                            weight(weight_){}

    virtual ~EdgeAnalyticCostFunction() {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const{

//        Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0] + 3);
//        Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[0]);
        Eigen::Map<const Eigen::Quaterniond> q_last_curr(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t_last_curr(parameters[0] + 4);
        Eigen::Vector3d lp;
        lp = q_last_curr * curr_point + t_last_curr; //new point
        Eigen::Vector3d nu = (lp - last_point_a).cross(lp - last_point_b);
        Eigen::Vector3d de = last_point_a - last_point_b;

        residuals[0] = weight * (nu.x() / de.norm());
        residuals[1] = weight * (nu.y() / de.norm());
        residuals[2] = weight * (nu.z() / de.norm());

        if(jacobians != NULL)
        {
            if(jacobians[0] != NULL)
            {
                Eigen::Matrix3d skew_lp = skew(lp);
                Eigen::Matrix<double, 3, 6> dp_by_so3;
                dp_by_so3.block<3,3>(0,0) = -skew_lp * weight;
                (dp_by_so3.block<3,3>(0, 3)) = Eigen::Matrix3d::Identity() * weight;
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3(jacobians[0]);
                J_se3.setZero();
                Eigen::Vector3d re = last_point_b - last_point_a;
                Eigen::Matrix3d skew_re = skew(re);

                J_se3.block<3,6>(0,0) = skew_re * dp_by_so3/de.norm();
            }
        }

        return true;
    }

    Eigen::Vector3d curr_point;
    Eigen::Vector3d last_point_a;
    Eigen::Vector3d last_point_b;
    double weight;
};

class SurfNormAnalyticCostFunction : public ceres::SizedCostFunction<1, 7> {
public:
    SurfNormAnalyticCostFunction(Eigen::Vector3d& curr_point_,
                                 Eigen::Vector3d& plane_unit_norm_,
                                 double negative_OA_dot_norm_,
                                 double weight_ = 1.0) : curr_point(curr_point_),
                                                                 plane_unit_norm(plane_unit_norm_),
                                                                 negative_OA_dot_norm(negative_OA_dot_norm_),
                                                                 weight(weight_){}
    virtual ~SurfNormAnalyticCostFunction() {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const{

//        Eigen::Map<const Eigen::Quaterniond> q_w_curr(parameters[0] + 3);
//        Eigen::Map<const Eigen::Vector3d> t_w_curr(parameters[0]);
        Eigen::Map<const Eigen::Quaterniond> q_w_curr(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> t_w_curr(parameters[0] + 4);
        Eigen::Vector3d point_w = q_w_curr * curr_point + t_w_curr;

        residuals[0] = weight * (plane_unit_norm.dot(point_w) + negative_OA_dot_norm);

        if(jacobians != NULL)
        {
            if(jacobians[0] != NULL)
            {
                Eigen::Matrix3d skew_point_w = skew(point_w);

                Eigen::Matrix<double, 3, 6> dp_by_so3;
                dp_by_so3.block<3,3>(0,0) = -skew_point_w * weight;
                (dp_by_so3.block<3,3>(0, 3)) = Eigen::Matrix3d::Identity() * weight;
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3(jacobians[0]);
                J_se3.setZero();
                J_se3.block<1,6>(0,0) = plane_unit_norm.transpose() * dp_by_so3;
            }
        }
        return true;
    }

    Eigen::Vector3d curr_point;
    Eigen::Vector3d plane_unit_norm;
    double negative_OA_dot_norm, weight;
};

/// \param se3 : [ phi, upsilon ]
void getTransformFromSe3(const Eigen::Matrix<double,6,1>& se3, Eigen::Quaterniond& q, Eigen::Vector3d& t){

    Eigen::Vector3d omega(se3.data());  // rotation
    Eigen::Vector3d upsilon(se3.data()+3);  // translation
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

Eigen::Quaterniond getDeltaQ(const Eigen::Matrix<double, 3, 1> &theta){

    Eigen::Quaterniond dq;
    Eigen::Matrix<double, 3, 1> half_theta = theta;
    half_theta /= 2.0;
    dq.w() = 1.0;
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

class PoseSE3Parameterization : public ceres::LocalParameterization {
public:

    PoseSE3Parameterization() {}
    virtual ~PoseSE3Parameterization() {}

    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const{

        Eigen::Map<const Eigen::Quaterniond> quater(x);
        Eigen::Map<const Eigen::Vector3d> trans(x + 4);

        Eigen::Quaterniond delta_q;
        Eigen::Vector3d delta_t;
        getTransformFromSe3(Eigen::Map<const Eigen::Matrix<double,6,1>>(delta), delta_q, delta_t);

        Eigen::Map<Eigen::Quaterniond> quater_plus(x_plus_delta);
        Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta + 4);
        quater_plus = delta_q * quater;
        trans_plus = delta_q * trans + delta_t;

        return true;
    }

    // translation first
//    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const{
//
//        Eigen::Map<const Eigen::Quaterniond> quater(x+3);
//        Eigen::Map<const Eigen::Vector3d> trans(x);
//        Eigen::Map<const Eigen::Matrix<double,6,1>> delta_Matrix(delta);
//
//        Eigen::Quaterniond delta_q = getDeltaQ(delta_Matrix.tail<3>());
//        Eigen::Vector3d delta_t(delta_Matrix.head<3>());
//
//        Eigen::Map<Eigen::Quaterniond> quater_plus(x_plus_delta+3);
//        Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta);
//        quater_plus = (quater * delta_q).normalized();
//        trans_plus = trans + delta_t;
//
//        return true;
//    }

    virtual bool ComputeJacobian(const double* x, double* jacobian) const{

        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
        (j.topRows(6)).setIdentity();
        (j.bottomRows(1)).setZero();

        return true;
    }
    virtual int GlobalSize() const { return 7; }
    virtual int LocalSize() const { return 6; }  // in tangent space
};


// ****************************************************************
// calculate distrance from point to plane (using normal)
// from ESKF_LIO(gf) 2023.4.2
class LidarMapPlaneNormFactor : public ceres::SizedCostFunction<1, 7>
{
public:
    LidarMapPlaneNormFactor(const Eigen::Vector3d &point,
                            const Eigen::Vector4d &coeff,
                            const Eigen::Matrix3d &cov_matrix = Eigen::Matrix3d::Identity())
            : point_(point),
              coeff_(coeff),
              sqrt_info_(sqrt(1 / cov_matrix.trace()))
    {
        // std::cout << "init lidarmapplanenormFactor, point: " << point_.transpose() << ", coeff: " << coeff_.transpose() << ",cov_trace: " << cov_matrix.trace() << std::endl;
        // is a upper triangular matrix
        // std::cout << sqrt_info_ << std::endl << std::endl;
        // Eigen::matrix_sqrt_triangular(sqrt_info_, sqrt_info_);
        // std::cout << sqrt_info_ << std::endl;
        // exit(EXIT_FAILURE);
        sqrt_info_ = sqrt_info_ >= 3.0 ? 1.0 : sqrt_info_ / 3.0; // 1 / trace, 20m
    }

    bool Evaluate(double const *const *param, double *residuals, double **jacobians) const
    {
        Eigen::Quaterniond q_w_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_w_curr(param[0][0], param[0][1], param[0][2]);

        Eigen::Vector3d w(coeff_(0), coeff_(1), coeff_(2));
        double d = coeff_(3);
        double a = w.dot(q_w_curr * point_ + t_w_curr) + d;
        residuals[0] = sqrt_info_ * a;

        if (jacobians)
        {
            Eigen::Matrix3d R = q_w_curr.toRotationMatrix();
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                Eigen::Matrix<double, 1, 6> jaco; // [dy/dt, dy/dq, 1]

                jaco.leftCols<3>() = w.transpose();
                jaco.rightCols<3>() = -w.transpose() * R * skew(point_);

                jacobian_pose.setZero();
                jacobian_pose.leftCols<6>() = sqrt_info_ * jaco;
                // std::cout << "jac pose: " << jacobian_pose << std::endl;
            }
        }
        return true;
    }

    void check(double **param)
    {
        double *res = new double[1];
        double **jaco = new double *[1];
        jaco[0] = new double[1 * 7];
        Evaluate(param, res, jaco);
        std::cout << "[LidarMapPlaneNormFactor] check begins" << std::endl;
        std::cout << "analytical:" << std::endl;
        std::cout << res[0] << std::endl;
        std::cout << Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>>(jaco[0]) << std::endl;

        delete[] jaco[0];
        delete[] jaco;
        delete[] res;

        Eigen::Quaterniond q_w_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
        Eigen::Vector3d t_w_curr(param[0][0], param[0][1], param[0][2]);

        Eigen::Vector3d w(coeff_(0), coeff_(1), coeff_(2));
        double d = coeff_(3);
        double a = w.dot(q_w_curr * point_ + t_w_curr) + d;
        double r = sqrt_info_ * a;

        std::cout << "perturbation:" << std::endl;
        std::cout << r << std::endl;

        const double eps = 1e-6;
        Eigen::Matrix<double, 1, 6> num_jacobian;

        // add random perturbation
        for (int k = 0; k < 6; k++)
        {
            Eigen::Quaterniond q_w_curr(param[0][6], param[0][3], param[0][4], param[0][5]);
            Eigen::Vector3d t_w_curr(param[0][0], param[0][1], param[0][2]);
            int a = k / 3, b = k % 3;
            Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
            if (a == 0)
                t_w_curr += delta;
            else if (a == 1)
                q_w_curr = q_w_curr * skew(delta);

            Eigen::Vector3d w(coeff_(0), coeff_(1), coeff_(2));
            double d = coeff_(3);
            double tmp_r = w.dot(q_w_curr * point_ + t_w_curr) + d;
            tmp_r *= sqrt_info_;
            num_jacobian(k) = (tmp_r - r) / eps;
        }
        std::cout << num_jacobian.block<1, 6>(0, 0) << std::endl;
    }

private:
    const Eigen::Vector3d point_;
    const Eigen::Vector4d coeff_;
    double sqrt_info_;
};

#endif //STRUCTURAL_MAPPING_CERESANALYTICCOST_H
