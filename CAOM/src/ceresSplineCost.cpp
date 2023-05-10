//
// Created by cyz on 2022/2/18.
//

#include "ceresSplineCost.h"

void CeresFactorsSP::catmullRomSplineCoeff(double u, double* &coeff){

    static Eigen::Matrix<double , 4, 4> coeffs_matrix;
//        coeffs_matrix(0,0) =  2; coeffs_matrix(0,1) = -3; coeffs_matrix(0,2) = 0; coeffs_matrix(0,3) = 1;
//        coeffs_matrix(1,0) =  1; coeffs_matrix(1,1) = -2; coeffs_matrix(1,2) = 1; coeffs_matrix(1,3) = 0;
//        coeffs_matrix(2,0) = -2; coeffs_matrix(2,1) =  3; coeffs_matrix(2,2) = 0; coeffs_matrix(2,3) = 0;
//        coeffs_matrix(3,0) =  1; coeffs_matrix(3,1) = -1; coeffs_matrix(3,2) = 0; coeffs_matrix(3,3) = 0;

    coeffs_matrix(0,0) = -1; coeffs_matrix(0,1) =  3; coeffs_matrix(0,2) = -3; coeffs_matrix(0,3) =  1;
    coeffs_matrix(1,0) =  2; coeffs_matrix(1,1) = -5; coeffs_matrix(1,2) =  4; coeffs_matrix(1,3) = -1;
    coeffs_matrix(2,0) = -1; coeffs_matrix(2,1) =  0; coeffs_matrix(2,2) =  1; coeffs_matrix(2,3) =  0;
    coeffs_matrix(3,0) =  0; coeffs_matrix(3,1) =  2; coeffs_matrix(3,2) =  0; coeffs_matrix(3,3) =  0;

    static Eigen::Matrix<double, 1, 4> timeCoeff;

    coeff = new double[5];
    if(u > 0.5) {
        if(u > 0.75) u = 0.75;
        double nt = (u - 0.5) / 0.25;

        timeCoeff(0,3) = 1.0;
        timeCoeff(0,2) = nt;
        timeCoeff(0,1) = timeCoeff(0, 2) * nt;
        timeCoeff(0,0) = timeCoeff(0, 1) * nt;
        Eigen::Matrix<double, 1, 4> coeffM = timeCoeff * coeffs_matrix;

        coeff[0] = 0;
        coeff[1] = coeffM(0,0);
        coeff[2] = coeffM(0,1);
        coeff[3] = coeffM(0,2);
        coeff[4] = coeffM(0,3);
    }else{
        if(u < 0.25) u = 0.25;
        double nt = (u - 0.25) / 0.25;

        timeCoeff(0,3) = 1.0;
        timeCoeff(0,2) = nt;
        timeCoeff(0,1) = timeCoeff(0, 2) * nt;
        timeCoeff(0,0) = timeCoeff(0, 1) * nt;
        Eigen::Matrix<double, 1, 4> coeffM = timeCoeff * coeffs_matrix;

        coeff[0] = coeffM(0,0);
        coeff[1] = coeffM(0,1);
        coeff[2] = coeffM(0,2);
        coeff[3] = coeffM(0,3);
        coeff[4] = 0;
    }
}

void CeresFactorsSP::catmullRomSplineCoeff_One(double u, double* &coeff){

    static Eigen::Matrix<double , 4, 4> coeffs_matrix;
//        coeffs_matrix(0,0) =  2; coeffs_matrix(0,1) = -3; coeffs_matrix(0,2) = 0; coeffs_matrix(0,3) = 1;
//        coeffs_matrix(1,0) =  1; coeffs_matrix(1,1) = -2; coeffs_matrix(1,2) = 1; coeffs_matrix(1,3) = 0;
//        coeffs_matrix(2,0) = -2; coeffs_matrix(2,1) =  3; coeffs_matrix(2,2) = 0; coeffs_matrix(2,3) = 0;
//        coeffs_matrix(3,0) =  1; coeffs_matrix(3,1) = -1; coeffs_matrix(3,2) = 0; coeffs_matrix(3,3) = 0;

    coeffs_matrix(0,0) = -1; coeffs_matrix(0,1) =  3; coeffs_matrix(0,2) = -3; coeffs_matrix(0,3) =  1;
    coeffs_matrix(1,0) =  2; coeffs_matrix(1,1) = -5; coeffs_matrix(1,2) =  4; coeffs_matrix(1,3) = -1;
    coeffs_matrix(2,0) = -1; coeffs_matrix(2,1) =  0; coeffs_matrix(2,2) =  1; coeffs_matrix(2,3) =  0;
    coeffs_matrix(3,0) =  0; coeffs_matrix(3,1) =  2; coeffs_matrix(3,2) =  0; coeffs_matrix(3,3) =  0;

    static Eigen::Matrix<double, 1, 4> timeCoeff;
    coeff = new double[5];

    double nt = 0;
    if(u > 0.5) {
        if(u > 0.75) u = 0.75;
        nt = (u - 0.5) / 0.25;
    }else{
        if(u < 0.25) u = 0.25;
        nt = (u - 0.25) / 0.25;
    }

    timeCoeff(0,3) = 1.0;
    timeCoeff(0,2) = nt;
    timeCoeff(0,1) = timeCoeff(0, 2) * nt;
    timeCoeff(0,0) = timeCoeff(0, 1) * nt;
    Eigen::Matrix<double, 1, 4> coeffM = timeCoeff * coeffs_matrix;

    coeff[0] = coeffM(0,0);
    coeff[1] = coeffM(0,1);
    coeff[3] = coeffM(0,2);
    coeff[4] = coeffM(0,3);
    coeff[2] = 0; // we dont use the middle control point
}

void CeresFactorsSP::splineBasicCoeffCalcu(double u, int n, int p, double* &coeff){

    coeff = new double[n+1]{0};  // init
    if(u == 0){  // pass through the start control point
        coeff[0] = 1;
        return;
    }
    if(u == 1){  // pass through the end control point
        coeff[n] = 1;
        return;
    }

    int m = n + p + 1;
    double interval = 1.0 / (m - 2*p);
    auto* knots = new double[m+1]{0};  // knots
    knots[m] = 1;
    for (int i = 1; i <= p; ++i) {  // clamped spline
        knots[i] = knots[i-1];
        knots[m-i] = knots[m];
    }
    for (int j = p + 1 ; j < m - p; ++j) {
        knots[j] = interval * (j-p);  // uniformly sampling
    }
//        for (int l = 0; l <= m; ++l) {
//            cout << knots[l] << " / ";
//        }
//        cout << endl << u << endl ;

    double t1, t2;
    for (int k = p; k < m ; ++k) {
        if(u >= knots[k] && u < knots[k+1]){  // u ~ [ u|k, u|k+1 )

            coeff[k] = 1;
            for (int d = 1; d <= p; ++d) {  //

                t1 = knots[k+1] - knots[k-d+1];
                if (t1)  // t1 != 0
                    t1 = (knots[k+1] - u) / t1;
                coeff[k-d] = coeff[k-d+1] *  t1 ;

                for (int i = k-d+1; i <= k-1 ; ++i) {
                    t1 = knots[i+d] - knots[i];
                    t2 = knots[i+d+1] - knots[i+1];
                    if(t1)
                        t1 = (u - knots[i]) / t1;
                    if(t2)
                        t2 = (knots[i+d+1] - u) / t2;
                    coeff[i] = t1 * coeff[i] + t2 * coeff[i+1];
                }

                t1 = knots[k+d] - knots[k];
                if(t1)
                    t1 = (u - knots[k]) / t1;
                coeff[k] *= t1;
            }
        }
    }

    delete [] knots;
}

void CeresFactorsSP::splineBasicCoeffCalcuHalf(double u, int n, int p, double* &coeff){

    coeff = new double[n+1]{0};  // init
    if(u == 0){  // pass through the start control point
        coeff[0] = 1;
        return;
    }
    if(u == 1.){
        coeff[n] = 1.;
        return;
    }

    int m = n + p + 1;
    double interval = 1.0 / (m - p);
    auto* knots = new double[m+1]{0};  // knots
    knots[m] = 1.0f;
    for (int i = 1; i <= p; ++i)  // clamped spline
        knots[i] = knots[i-1];

    for (int j = p + 1 ; j < m; ++j)
        knots[j] = interval * (j-p);  // uniformly sampling

    double t1, t2;
    for (int k = p; k < m ; ++k) {
        if(u >= knots[k] && u < knots[k+1]){  // u ~ [ u|k, u|k+1 )

            coeff[k] = 1;
            for (int d = 1; d <= p; ++d) {  //

                t1 = knots[k+1] - knots[k-d+1];
                if (t1)  // t1 != 0
                    t1 = (knots[k+1] - u) / t1;
                coeff[k-d] = coeff[k-d+1] * t1 ;

                for (int i = k-d+1; i <= k-1 ; ++i) {
                    t1 = knots[i+d] - knots[i];
                    t2 = knots[i+d+1] - knots[i+1];
                    if(t1)
                        t1 = (u - knots[i]) / t1;
                    if(t2)
                        t2 = (knots[i+d+1] - u) / t2;
                    coeff[i] = t1 * coeff[i] + t2 * coeff[i+1];
                }

                t1 = knots[k+d] - knots[k];
                if(t1)
                    t1 = (u - knots[k]) / t1;
                coeff[k] *= t1;
            }
        }
    }

    delete [] knots;
}

void CeresFactorsSP::fromDataToControlpoints(std::vector<double *> &datapts){

    const int n = datapts.size();
    Eigen::MatrixXd coeffmatrix(n, n) ;
    Eigen::MatrixXd coeffmatrix_inv(n, n);
    double u;
    double* curCoeff;
    double total = datapts.back()[7] - datapts.front()[7];
    // 系数矩阵
    for (int i = 0; i < n; ++i) {
        u = (datapts[i][7] - datapts.front()[7]) / total;
        splineBasicCoeffCalcu(u, n-1, 3, curCoeff);
        for (int j = 0; j < n; ++j)
            coeffmatrix(i,j) = curCoeff[j];
        delete curCoeff;
    }
    coeffmatrix_inv = coeffmatrix.inverse();  // TODO

    // data points
    Eigen::Quaternion<double > q0 {datapts[0][3], datapts[0][0], datapts[0][1], datapts[0][2]};
    Eigen::Quaternion<double > q1 {datapts[1][3], datapts[1][0], datapts[1][1], datapts[1][2]};
    Eigen::Quaternion<double > q2 {datapts[2][3], datapts[2][0], datapts[2][1], datapts[2][2]};
    Eigen::Quaternion<double > q3 {datapts[3][3], datapts[3][0], datapts[3][1], datapts[3][2]};
    Eigen::Quaternion<double > q4 {datapts[4][3], datapts[4][0], datapts[4][1], datapts[4][2]};

    Eigen::Matrix<double , 3, 1> t0 {datapts[0][4], datapts[0][5], datapts[0][6]};
    Eigen::Matrix<double , 3, 1> t1 {datapts[1][4], datapts[1][5], datapts[1][6]};
    Eigen::Matrix<double , 3, 1> t2 {datapts[2][4], datapts[2][5], datapts[2][6]};
    Eigen::Matrix<double , 3, 1> t3 {datapts[3][4], datapts[3][5], datapts[3][6]};
    Eigen::Matrix<double , 3, 1> t4 {datapts[4][4], datapts[4][5], datapts[4][6]};

    for (int k = 0; k < n; ++k) {
        // rotation curve
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        q = (q.slerp(coeffmatrix_inv(k,0), q0)) *
            (q.slerp(coeffmatrix_inv(k,1), q1)) *
            (q.slerp(coeffmatrix_inv(k,2), q2)) *
            (q.slerp(coeffmatrix_inv(k,3), q3)) *
            (q.slerp(coeffmatrix_inv(k,4), q4));

        datapts[k][0] = q.x();
        datapts[k][1] = q.y();
        datapts[k][2] = q.z();
        datapts[k][3] = q.w();

        // translation curve
        Eigen::Matrix<double, 3, 1> t = t0 * coeffmatrix_inv(k,0) +
                                        t1 * coeffmatrix_inv(k,1) +
                                        t2 * coeffmatrix_inv(k,2) +
                                        t3 * coeffmatrix_inv(k,3) +
                                        t4 * coeffmatrix_inv(k,4);
        datapts[k][4] = t(0);
        datapts[k][5] = t(1);
        datapts[k][6] = t(2);
    }

}

void CeresFactorsSP::fromDataToControlpointsDynamic(std::vector<double *> &datapts, int N){

    const int n = datapts.size();
    Eigen::MatrixXd A(n, N), AT(N, n);
    Eigen::MatrixXd ATA(N, N), ATb(N, 1), ATA_inv(N,N);
    double u;
    double* curCoeff;
    double totalT = datapts.back()[7] - datapts.front()[7];
    // construct coefficients matrix
    for (int i = 0; i < n; ++i) {
        u = (datapts[i][7] - datapts.front()[7]) / totalT;
        splineBasicCoeffCalcu(u, N-1, 3, curCoeff);
        for (int j = 0; j < N; ++j)
            A(i,j) = curCoeff[j];
        delete curCoeff;
    }
    AT = A.transpose();
    ATA = AT * A;
    ATA_inv = ATA.inverse();
    Eigen::MatrixXd coeff(N, n);
    coeff = ATA_inv * AT;

    // data points
    vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > rots;
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > trans;
    for (int i = 0; i < n; ++i) {
        Eigen::Quaternion<double > q {datapts[i][3], datapts[i][0], datapts[i][1], datapts[i][2]};
        rots.emplace_back(q);
        Eigen::Vector3d t {datapts[i][4], datapts[i][5], datapts[i][6]};
        trans.emplace_back(t);
    }

    Eigen::Quaterniond I = Eigen::Quaterniond::Identity();
    double startT = datapts.front()[7];
    for (int k = 0; k < N; ++k) {
        // rotation curve
        Eigen::Quaterniond rot = Eigen::Quaterniond::Identity();
        for (int i = 0; i < n; ++i)
            rot = rot * (I.slerp(coeff(k,i), rots[i]));

        datapts[k][0] = rot.x();
        datapts[k][1] = rot.y();
        datapts[k][2] = rot.z();
        datapts[k][3] = rot.w();

        // translation curve
        Eigen::Vector3d transla;
        for (int i = 0; i < n; ++i)
            transla = transla + (coeff(k,i) * trans[i]);

        datapts[k][4] = transla(0);
        datapts[k][5] = transla(1);
        datapts[k][6] = transla(2);

        datapts[k][7] = startT + (k*1.0/(N-1)) * totalT;  //
    }

}


#include "ceresSplineCostSophus.h"

bool SophusSpline::CeresFactorsSP::LidarPlaneNormFactorSP::Evaluate(const double *const *parameters,
                                                                    double *residuals,
                                                                    double **jacobians) const {

    Eigen::Map< const Sophus::SE3<double> > T0(parameters[0]);
    Eigen::Map< const Sophus::SE3<double> > T1(parameters[1]);
    Eigen::Map< const Sophus::SE3<double> > T2(parameters[2]);
    Eigen::Map< const Sophus::SE3<double> > T3(parameters[3]);
    Eigen::Map< const Sophus::SE3<double> > T4(parameters[4]);

    basalt::Se3Spline<4> se3splineHelper = basalt::Se3Spline<4>(time_interval_ns);  // 0.1s -> ns
    se3splineHelper.knots_push_back(T0);
    se3splineHelper.knots_push_back(T1);
    se3splineHelper.knots_push_back(T2);
    se3splineHelper.knots_push_back(T3);
    se3splineHelper.knots_push_back(T4);

    basalt::Se3Spline<4>::PosePosSO3JacobianStruct J_spline;
    Sophus::SE3<double> pose = se3splineHelper.pose(eval_time_ns, &J_spline);  // s -> ns
    Sophus::SO3<double> rot = pose.so3();

    Eigen::Quaterniond q_last_curr(rot.unit_quaternion());
    Eigen::Vector3d t_last_curr = pose.translation();

    Eigen::Matrix<double, 3, 1> cp{double(curr_point.x()), double(curr_point.y()), double(curr_point.z())};
    Eigen::Matrix<double, 3, 1> point_w;
    point_w = q_last_curr * cp + t_last_curr;

    Eigen::Matrix<double, 3, 1> norm(double(plane_unit_norm.x()), double(plane_unit_norm.y()),
                                     double(plane_unit_norm.z()));
    residuals[0] = (norm.dot(point_w) + double(negative_OA_dot_norm));

    if (jacobians != NULL) {

        Eigen::Matrix3d skew_point_w = skew(point_w);
        Eigen::Matrix<double, 3, 6> dp_by_so3;
        dp_by_so3.block<3, 3>(0, 3) = -skew_point_w;
        (dp_by_so3.block<3, 3>(0, 0)).setIdentity();
        Eigen::Matrix<double, 1, 6, Eigen::RowMajor> J_se3;
        J_se3.setZero();
        J_se3 = plane_unit_norm.transpose() * dp_by_so3;

        Eigen::Matrix<double, 6, 6> J_spline_Knot;
        if (u < 0.25) {  /// spline controlled by the first four points
            if (jacobians[0] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3_0(jacobians[0]);
                J_se3_0.setZero();
//              J_spline_Knot.setZero();
//              J_spline_Knot.topLeftCorner<3,3>() = J_spline.d_val_d_knot[0].bottomRightCorner<3,3>();
//              J_spline_Knot.bottomRightCorner<3,3>() =  J_spline.d_val_d_knot[0].topLeftCorner<3,3>();
//              J_se3_0.block<1, 6>(0, 0) = J_se3 * J_spline_Knot;
                J_se3_0.block<1, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[0];
            }
            if (jacobians[1] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3_1(jacobians[1]);
                J_se3_1.setZero();
                J_se3_1.block<1, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[1];
            }
            if (jacobians[2] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3_2(jacobians[2]);
                J_se3_2.setZero();
                J_se3_2.block<1, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[2];
            }
            if (jacobians[3] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3_3(jacobians[3]);
                J_se3_3.setZero();
                J_se3_3.block<1, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[3];
            }
            if (jacobians[4] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3_4(jacobians[4]);
                J_se3_4.setZero();
            }
        } else {  /// spline controlled by the last four points
            if (jacobians[0] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3_0(jacobians[0]);
                J_se3_0.setZero();
            }
            if (jacobians[1] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3_1(jacobians[1]);
                J_se3_1.setZero();
                J_se3_1.block<1, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[0];
            }
            if (jacobians[2] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3_2(jacobians[2]);
                J_se3_2.setZero();
                J_se3_2.block<1, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[1];
            }
            if (jacobians[3] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3_3(jacobians[3]);
                J_se3_3.setZero();
                J_se3_3.block<1, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[2];
            }
            if (jacobians[4] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > J_se3_4(jacobians[4]);
                J_se3_4.setZero();
                J_se3_4.block<1, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[3];
            }
        }
    }

    return true;
}

bool SophusSpline::CeresFactorsSP::LidarEdgeFactorSP::Evaluate(const double *const *parameters,
                                                               double *residuals,
                                                               double **jacobians) const {

    Eigen::Map< const Sophus::SE3<double> > T0(parameters[0]);
    Eigen::Map< const Sophus::SE3<double> > T1(parameters[1]);
    Eigen::Map< const Sophus::SE3<double> > T2(parameters[2]);
    Eigen::Map< const Sophus::SE3<double> > T3(parameters[3]);
    Eigen::Map< const Sophus::SE3<double> > T4(parameters[4]);

    basalt::Se3Spline<4> se3splineHelper = basalt::Se3Spline<4>(time_interval_ns);  // 0.1s -> ns
    se3splineHelper.knots_push_back(T0);
    se3splineHelper.knots_push_back(T1);
    se3splineHelper.knots_push_back(T2);
    se3splineHelper.knots_push_back(T3);
    se3splineHelper.knots_push_back(T4);

    basalt::Se3Spline<4>::PosePosSO3JacobianStruct J_spline;
    Sophus::SE3<double> pose = se3splineHelper.pose(eval_time_ns, &J_spline);  // s -> ns

//                for (int i = 0; i < J_spline.d_val_d_knot.size(); ++i)
//    cout << "[ Jacob ] SE3 T0 " << T0.matrix() << endl;
//    cout << "[ Jacob ] SE3 T1 " << T1.matrix() << endl;
//    cout << "[ Jacob ] SE3 T2 " << T2.matrix() << endl;
//    cout << "[ Jacob ] SE3 T3 " << T3.matrix() << endl;
//    cout << "[ Jacob ] SE3 T4 " << T4.matrix() << endl;
//    cout << "[ Jacob ] SE3 pose " << pose.matrix() << endl;

    Sophus::SO3<double> rot = pose.so3();
    Eigen::Quaterniond q_last_curr(rot.unit_quaternion());
    Eigen::Vector3d t_last_curr = pose.translation();

    Eigen::Vector3d lp;
    lp = q_last_curr * curr_point + t_last_curr;  // new point
    Eigen::Vector3d nu = (lp - last_point_a).cross(lp - last_point_b);
    Eigen::Vector3d de = last_point_a - last_point_b;

    residuals[0] = nu.x() / de.norm();
    residuals[1] = nu.y() / de.norm();
    residuals[2] = nu.z() / de.norm();

    if (jacobians != NULL) {

        Eigen::Matrix3d skew_lp = skew(lp);
        Eigen::Matrix<double, 3, 6> dp_by_so3;
        dp_by_so3.block<3, 3>(0, 3) = -skew_lp;
        (dp_by_so3.block<3, 3>(0, 0)).setIdentity();
        Eigen::Matrix<double, 3, 6, Eigen::RowMajor> J_se3;
        J_se3.setZero();
        Eigen::Vector3d re = last_point_b - last_point_a;
        Eigen::Matrix3d skew_re = skew(re);
        J_se3 = skew_re * dp_by_so3 / de.norm();
//                    cout << "[ Jacob ] J_se3 with u "<< u << " \n" << J_se3 << endl;

        if (u < 0.25) {  // spline controlled by the first four points
            if (jacobians[0] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_0(jacobians[0]);
                J_se3_0.setZero();
                J_se3_0.block<3, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[0];
            }
            if (jacobians[1] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_1(jacobians[1]);
                J_se3_1.setZero();
                J_se3_1.block<3, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[1];
            }
            if (jacobians[2] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_2(jacobians[2]);
                J_se3_2.setZero();
                J_se3_2.block<3, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[2];
            }
            if (jacobians[3] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_3(jacobians[3]);
                J_se3_3.setZero();
                J_se3_3.block<3, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[3];
            }
            if (jacobians[4] != NULL) {
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_4(jacobians[4]);
                J_se3_4.setZero();
            }
//                        cout << "[ Jacob ] Jacob with u < 0.25 "<< u << " DONE " << endl;
        } else {  // spline controlled by the last four points
            if (jacobians[0] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_0(jacobians[0]);
                J_se3_0.setZero();
            }
            if (jacobians[1] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_1(jacobians[1]);
                J_se3_1.setZero();
                J_se3_1.block<3, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[0];
            }
            if (jacobians[2] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_2(jacobians[2]);
                J_se3_2.setZero();
                J_se3_2.block<3, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[1];
            }
            if (jacobians[3] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_3(jacobians[3]);
                J_se3_3.setZero();
                J_se3_3.block<3, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[2];
            }
            if (jacobians[4] != NULL) {

                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J_se3_4(jacobians[4]);
                J_se3_4.setZero();
                J_se3_4.block<3, 6>(0, 0) = J_se3 * J_spline.d_val_d_knot[3];
            }
//                        cout << "[ Jacob ] Jacob with u > 0.25 "<< u << " DONE " << endl;
        }

    }

    return true;
}

void SophusSpline::CeresFactorsSP::interpolatePoseSE3(const double ratio, const Eigen::Quaterniond &q1,
                                                      const Eigen::Quaterniond &q2, const Eigen::Vector3d &t1,
                                                      const Eigen::Vector3d &t2, Eigen::Quaterniond &q,
                                                      Eigen::Vector3d &t) {

    Sophus::SE3<double> T1(q1, t1);
    Sophus::SE3<double> T2(q2, t2);
    Sophus::SE3<double> deltT, T;
    deltT = T1.inverse() * T2;
    T = T1 * Sophus::SE3<double>::exp(ratio * deltT.log());
    q = T.unit_quaternion();
    t = T.translation();
//            cout << "[ DEBUG ] Sophus Data : " << T.data()[0] << " "
//                 << T.data()[1] << " " << T.data()[2] << " " << T.data()[3]<< endl;
}

void SophusSpline::CeresFactorsSP::fromDataToControlpointsDynamic(std::vector<double *> &datapts, int N) {

    const int n = datapts.size();
    double totalT = datapts.back()[7] - datapts.front()[7];

    // data points
    vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond> > rots;
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > trans;
    for (int i = 0; i < n; ++i) {
        Eigen::Quaternion<double> q{datapts[i][3], datapts[i][0], datapts[i][1], datapts[i][2]};
        rots.emplace_back(q);
        Eigen::Vector3d t{datapts[i][4], datapts[i][5], datapts[i][6]};
        trans.emplace_back(t);
    }

    /// DEBUG
//            Eigen::Map<const Sophus::SE3<double> > testT(datapts[0]);
//            Sophus::SE3<double> testT_rt(rots[0], trans[0]);
//            cout << YELLOW << "[ DEBUG Sophus ] T_Rt: " << testT_rt.matrix() << endl;
//            cout << "[ DEBUG Sophus ] T_map: " << testT.matrix() << RESET << endl;

    Eigen::Quaterniond rot = Eigen::Quaterniond::Identity();
    Eigen::Vector3d transla;
    double startT = datapts.front()[7], inT, ratio;
    int i = 0;
    for (int k = 0; k < N; ++k) {

        inT = startT + (k * 1.0 / (N - 1)) * totalT;
        for (; i < n - 1; ++i) {
            if (inT >= datapts[i][7] && inT <= datapts[i + 1][7]) {
                ratio = (inT - datapts[i][7]) / (datapts[i + 1][7] - datapts[i][7]);
                interpolatePoseSE3(ratio, rots[i], rots[i + 1], trans[i], trans[i + 1], rot, transla);
                break;
            }
        }

        datapts[k][0] = rot.x();
        datapts[k][1] = rot.y();
        datapts[k][2] = rot.z();
        datapts[k][3] = rot.w();

        datapts[k][4] = transla(0);
        datapts[k][5] = transla(1);
        datapts[k][6] = transla(2);

        datapts[k][7] = inT;  //
    }
}