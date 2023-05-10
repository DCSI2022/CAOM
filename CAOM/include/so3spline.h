////////////////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Created by cyz on 2021/4/12.
///
/// License is from basalt ETH.
///
///////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STRUCTURAL_MAPPING_SO3SPLINE_H
#define STRUCTURAL_MAPPING_SO3SPLINE_H

#include <sophus/so3.hpp>
#include <sophus/se3.hpp>

#define BASALT_ASSERT(expr) ((void)0)

#define BASALT_ASSERT_MSG(expr, msg) ((void)0)

#define BASALT_ASSERT_STREAM(expr, msg) ((void)0)

namespace basalt {

    /// @brief Left Jacobian for SO(3)
    ///
    /// For \f$ \exp(x) \in SO(3) \f$ provides a Jacobian that approximates the sum
    /// under expmap with a left multiplication of expmap for small \f$ \epsilon
    /// \f$.  Can be used to compute:  \f$ \exp(\phi + \epsilon) \approx
    /// \exp(J_{\phi} \epsilon) \exp(\phi) \f$
    /// @param[in] phi (3x1 vector)
    /// @param[out] J_phi (3x3 matrix)
    template <typename Derived1, typename Derived2>
    inline void leftJacobianSO3(const Eigen::MatrixBase<Derived1> &phi,
                                const Eigen::MatrixBase<Derived2> &J_phi) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

        using Scalar = typename Derived1::Scalar;

        Eigen::MatrixBase<Derived2> &J =
                const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

        Scalar phi_norm2 = phi.squaredNorm();
        Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J.setIdentity();

        if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
            Scalar phi_norm = std::sqrt(phi_norm2);
            Scalar phi_norm3 = phi_norm2 * phi_norm;

            J += phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
            J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
        } else {
            // Taylor expansion around 0
            J += phi_hat / 2;
            J += phi_hat2 / 6;
        }
    }

    /// @brief Left Inverse Jacobian for SO(3)
    ///
    /// For \f$ \exp(x) \in SO(3) \f$ provides an inverse Jacobian that approximates
    /// the logmap of the left multiplication of expmap of the arguments with a sum
    /// for small \f$ \epsilon \f$.  Can be used to compute:  \f$ \log
    /// (\exp(\epsilon) \exp(\phi)) \approx \phi + J_{\phi} \epsilon\f$
    /// @param[in] phi (3x1 vector)
    /// @param[out] J_phi (3x3 matrix)
    template <typename Derived1, typename Derived2>
    inline void leftJacobianInvSO3(const Eigen::MatrixBase<Derived1> &phi,
                                   const Eigen::MatrixBase<Derived2> &J_phi) {
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived1);
        EIGEN_STATIC_ASSERT_FIXED_SIZE(Derived2);
        EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived1, 3);
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived2, 3, 3);

        using Scalar = typename Derived1::Scalar;

        Eigen::MatrixBase<Derived2> &J =
                const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

        Scalar phi_norm2 = phi.squaredNorm();
        Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
        Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

        J.setIdentity();
        J -= phi_hat / 2;

        if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
            Scalar phi_norm = std::sqrt(phi_norm2);

            // We require that the angle is in range [0, pi]. We check if we are close
            // to pi and apply a Taylor expansion to scalar multiplier of phi_hat2.
            // Technically, log(exp(phi)exp(epsilon)) is not continuous / differentiable
            // at phi=pi, but we still aim to return a reasonable value for all valid
            // inputs.
            BASALT_ASSERT(phi_norm <= M_PI + Sophus::Constants<Scalar>::epsilon());

            if (phi_norm < M_PI - Sophus::Constants<Scalar>::epsilonSqrt()) {
                // regular case for range (0,pi)
                J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) /
                                                 (2 * phi_norm * std::sin(phi_norm)));
            } else {
                // 0th-order Taylor expansion around pi
                J += phi_hat2 / (M_PI * M_PI);
            }
        } else {
            // Taylor expansion around 0
            J += phi_hat2 / 12;
        }
    }

    /// @brief Compute binomial coefficient.
    ///
    /// Computes number of combinations that include k objects out of n.
    /// @param[in] n
    /// @param[in] k
    /// @return binomial coefficient
    inline uint64_t C_n_k(uint64_t n, uint64_t k) {
        if (k > n)
            return 0;

        uint64_t r = 1;
        for (uint64_t d = 1; d <= k; ++d) {
            r *= n--;
            r /= d;
        }
        return r;
    }

    /// @brief Compute blending matrix for uniform B-spline evaluation.
    ///
    /// @param _N order of the spline
    /// @param _Scalar scalar type to use
    /// @param _Cumulative if the spline should be cumulative
    template <int _N, typename _Scalar = double, bool _Cumulative = false>
    Eigen::Matrix<_Scalar, _N, _N> computeBlendingMatrix() {
        Eigen::Matrix<double, _N, _N> m;
        m.setZero();

        for (int i = 0; i < _N; ++i) {
            for (int j = 0; j < _N; ++j) {
                double sum = 0;

                for (int s = j; s < _N; ++s) {
                    sum += std::pow(-1.0, s - j) * C_n_k(_N, s - j) *
                           std::pow(_N - s - 1.0, _N - 1.0 - i);
                }
                m(j, i) = C_n_k(_N - 1, _N - 1 - i) * sum;
            }
        }

        if (_Cumulative) {
            for (int i = 0; i < _N; i++) {
                for (int j = i + 1; j < _N; j++) {
                    m.row(i) += m.row(j);
                }
            }
        }

        uint64_t factorial = 1;
        for (int i = 2; i < _N; ++i) {
            factorial *= i;
        }

        return (m / factorial).template cast<_Scalar>();
    }

    /// @brief Compute base coefficient matrix for polynomials of size N.
    ///
    /// In each row starting from 0 contains the derivative coefficients of the
    /// polynomial. For _N=5 we get the following matrix: \f[ \begin{bmatrix}
    ///   1 & 1 & 1 & 1 & 1
    /// \\0 & 1 & 2 & 3 & 4
    /// \\0 & 0 & 2 & 6 & 12
    /// \\0 & 0 & 0 & 6 & 24
    /// \\0 & 0 & 0 & 0 & 24
    /// \\ \end{bmatrix}
    /// \f]
    /// Functions \ref RdSpline::baseCoeffsWithTime and \ref
    /// So3Spline::baseCoeffsWithTime use this matrix to compute derivatives of the
    /// time polynomial.
    ///
    /// @param _N order of the polynomial
    /// @param _Scalar scalar type to use
    template <int _N, typename _Scalar = double>
    Eigen::Matrix<_Scalar, _N, _N> computeBaseCoefficients() {
        Eigen::Matrix<double, _N, _N> base_coefficients;

        base_coefficients.setZero();
        base_coefficients.row(0).setOnes();

        const int DEG = _N - 1;
        int order = DEG;
        for (int n = 1; n < _N; n++) {
            for (int i = DEG - order; i < _N; i++) {
                base_coefficients(n, i) = (order - DEG + i) * base_coefficients(n - 1, i);
            }
            order--;
        }
        return base_coefficients.template cast<_Scalar>();
    }

    template <int _N, typename _Scalar = double>
    class So3Spline {
    public:
        static constexpr int N = _N;        ///< Order of the spline.
        static constexpr int DEG = _N - 1;  ///< Degree of the spline.

        static constexpr _Scalar ns_to_s = 1e-9;  ///< Nanosecond to second conversion
        static constexpr _Scalar s_to_ns = 1e9;   ///< Second to nanosecond conversion

        using MatN = Eigen::Matrix<_Scalar, _N, _N>;
        using VecN = Eigen::Matrix<_Scalar, _N, 1>;

        using Vec3 = Eigen::Matrix<_Scalar, 3, 1>;
        using Mat3 = Eigen::Matrix<_Scalar, 3, 3>;

        using SO3 = Sophus::SO3<_Scalar>;

        /// @brief Struct to store the Jacobian of the spline
        ///
        /// Since B-spline of order N has local support (only N knots infuence the
        /// value) the Jacobian is zero for all knots except maximum N for value and
        /// all derivatives.
        struct JacobianStruct {
            size_t start_idx;
            std::array<Mat3, _N> d_val_d_knot;
        };

        /// @brief Constructor with knot interval and start time
        ///
        /// @param[in] time_interval_ns knot time interval in nanoseconds
        /// @param[in] start_time_ns start time of the spline in nanoseconds
        So3Spline(int64_t time_interval_ns, int64_t start_time_ns = 0)
                : dt_ns(time_interval_ns), start_t_ns(start_time_ns) {
            pow_inv_dt[0] = 1.0;
            pow_inv_dt[1] = s_to_ns / dt_ns;
            pow_inv_dt[2] = pow_inv_dt[1] * pow_inv_dt[1];
            pow_inv_dt[3] = pow_inv_dt[2] * pow_inv_dt[1];
        }

        /// @brief Maximum time represented by spline
        ///
        /// @return maximum time represented by spline in nanoseconds
        int64_t maxTimeNs() const {
            return start_t_ns + (knots.size() - N + 1) * dt_ns - 1;
        }

        /// @brief Minimum time represented by spline
        ///
        /// @return minimum time represented by spline in nanoseconds
        int64_t minTimeNs() const { return start_t_ns; }

        /// @brief Gererate random trajectory
        ///
        /// @param[in] n number of knots to generate
        /// @param[in] static_init if true the first N knots will be the same
        /// resulting in static initial condition
        void genRandomTrajectory(int n, bool static_init = false) {
            if (static_init) {
                Vec3 rnd = Vec3::Random() * M_PI;

                for (int i = 0; i < N; i++) knots.push_back(SO3::exp(rnd));

                for (int i = 0; i < n - N; i++)
                    knots.push_back(knots.back() * SO3::exp(Vec3::Random() * M_PI / 2));

            } else {
                knots.push_back(SO3::exp(Vec3::Random() * M_PI));

                for (int i = 1; i < n; i++)
                    knots.push_back(knots.back() * SO3::exp(Vec3::Random() * M_PI / 2));
            }
        }

        /// @brief Set start time for spline
        ///
        /// @param[in] start_time_ns start time of the spline in nanoseconds
        inline void setStartTimeNs(int64_t s) { start_t_ns = s; }

        /// @brief Add knot to the end of the spline
        ///
        /// @param[in] knot knot to add
        inline void knots_push_back(const SO3& knot) { knots.push_back(knot); }

        /// @brief Remove knot from the back of the spline
        inline void knots_pop_back() { knots.pop_back(); }

        /// @brief Return the first knot of the spline
        ///
        /// @return first knot of the spline
        inline const SO3& knots_front() const { return knots.front(); }

        /// @brief Remove first knot of the spline and increase the start time
        inline void knots_pop_front() {
            start_t_ns += dt_ns;
            knots.pop_front();
        }

        /// @brief Resize containter with knots
        ///
        /// @param[in] n number of knots
        inline void resize(size_t n) { knots.resize(n); }

        /// @brief Return reference to the knot with index i
        ///
        /// @param i index of the knot
        /// @return reference to the knot
        inline SO3& getKnot(int i) { return knots[i]; }

        /// @brief Return const reference to the knot with index i
        ///
        /// @param i index of the knot
        /// @return const reference to the knot
        inline const SO3& getKnot(int i) const { return knots[i]; }

        /// @brief Return const reference to deque with knots
        ///
        /// @return const reference to deque with knots
        const std::deque<SO3, Eigen::aligned_allocator<SO3> >& getKnots() const { return knots; }

        /// @brief Return time interval in nanoseconds
        ///
        /// @return time interval in nanoseconds
        int64_t getTimeIntervalNs() const { return dt_ns; }

        /// @brief Evaluate SO(3) B-spline
        ///
        /// @param[in] time_ns time for evaluating the value of the spline in
        /// nanoseconds
        /// @param[out] J if not nullptr, return the Jacobian of the value with
        /// respect to knots
        /// @return SO(3) value of the spline
        SO3 evaluate(int64_t time_ns, JacobianStruct* J = nullptr) const {
            int64_t st_ns = (time_ns - start_t_ns);

            BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                                      << " start_t_ns " << start_t_ns);

            int64_t s = st_ns / dt_ns;
            double u = double(st_ns % dt_ns) / double(dt_ns);

            BASALT_ASSERT_STREAM(s >= 0, "s " << s);
            BASALT_ASSERT_STREAM(size_t(s + N) <= knots.size(), "s " << s << " N " << N
                                                                     << " knots.size() "
                                                                     << knots.size());
            assert(size_t(s + N) <= knots.size());

            VecN p;
            baseCoeffsWithTime<0>(p, u);

            VecN coeff = blending_matrix_ * p;

            SO3 res = knots[s];

            Mat3 J_helper;

            if (J) {
                J->start_idx = s;
                J_helper.setIdentity();
            }

            for (int i = 0; i < DEG; i++) {
                const SO3& p0 = knots[s + i];
                const SO3& p1 = knots[s + i + 1];

                SO3 r01 = p0.inverse() * p1;
                Vec3 delta = r01.log();
                Vec3 kdelta = delta * coeff[i + 1];

                if (J) {
                    Mat3 Jl_inv_delta, Jl_k_delta;

                    leftJacobianInvSO3(delta, Jl_inv_delta);  
                    leftJacobianSO3(kdelta, Jl_k_delta);

                    J->d_val_d_knot[i] = J_helper;
                    J_helper = coeff[i + 1] * res.matrix() * Jl_k_delta * Jl_inv_delta *
                               p0.inverse().matrix();
                    J->d_val_d_knot[i] -= J_helper;
                }
                res *= SO3::exp(kdelta);
            }

            if (J) J->d_val_d_knot[DEG] = J_helper;

            return res;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    protected:
        /// @brief Vector of derivatives of time polynomial.
        ///
        /// Computes a derivative of \f$ \begin{bmatrix}1 & t & t^2 & \dots &
        /// t^{N-1}\end{bmatrix} \f$ with repect to time. For example, the first
        /// derivative would be \f$ \begin{bmatrix}0 & 1 & 2 t & \dots & (N-1)
        /// t^{N-2}\end{bmatrix} \f$.
        /// @param Derivative derivative to evaluate
        /// @param[out] res_const vector to store the result
        /// @param[in] t
        template <int Derivative, class Derived>
        static void baseCoeffsWithTime(const Eigen::MatrixBase<Derived>& res_const,
                                       _Scalar t) {
            EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, N);
            Eigen::MatrixBase<Derived>& res =
                    const_cast<Eigen::MatrixBase<Derived>&>(res_const);

            res.setZero();

            if (Derivative < N) {
                res[Derivative] = base_coefficients_(Derivative, Derivative);

                _Scalar _t = t;
                for (int j = Derivative + 1; j < N; j++) {
                    res[j] = base_coefficients_(Derivative, j) * _t;
                    _t = _t * t;
                }
            }
        }

        static const MatN
                blending_matrix_;  ///< Blending matrix. See \ref computeBlendingMatrix.

        static const MatN base_coefficients_;  ///< Base coefficients matrix.
        ///< See \ref computeBaseCoefficients.

        std::deque<SO3, Eigen::aligned_allocator<SO3> > knots;    ///< Knots
        int64_t dt_ns;                      ///< Knot interval in nanoseconds
        int64_t start_t_ns;                 ///< Start time in nanoseconds
        std::array<_Scalar, 4> pow_inv_dt;  ///< Array with inverse powers of dt
    };

    template <int _N, typename _Scalar>
    const typename So3Spline<_N, _Scalar>::MatN
            So3Spline<_N, _Scalar>::base_coefficients_ =
            computeBaseCoefficients<_N, _Scalar>();

    template <int _N, typename _Scalar>
    const typename So3Spline<_N, _Scalar>::MatN
            So3Spline<_N, _Scalar>::blending_matrix_ =
            computeBlendingMatrix<_N, _Scalar, true>();


    template <int _DIM, int _N, typename _Scalar = double>
    class RdSpline {
    public:
        static constexpr int N = _N;        ///< Order of the spline.
        static constexpr int DEG = _N - 1;  ///< Degree of the spline.

        static constexpr int DIM = _DIM;  ///< Dimension of euclidean vector space.

        static constexpr _Scalar ns_to_s = 1e-9;  ///< Nanosecond to second conversion
        static constexpr _Scalar s_to_ns = 1e9;   ///< Second to nanosecond conversion

        using MatN = Eigen::Matrix<_Scalar, _N, _N>;
        using VecN = Eigen::Matrix<_Scalar, _N, 1>;

        using VecD = Eigen::Matrix<_Scalar, _DIM, 1>;
        using MatD = Eigen::Matrix<_Scalar, _DIM, _DIM>;

        /// @brief Struct to store the Jacobian of the spline
        ///
        /// Since B-spline of order N has local support (only N knots infuence the
        /// value) the Jacobian is zero for all knots except maximum N for value and
        /// all derivatives.
        struct JacobianStruct {
            size_t
                    start_idx;  ///< Start index of the non-zero elements of the Jacobian.
            std::array<_Scalar, N> d_val_d_knot;  ///< Value of nonzero Jacobians.
        };

        /// @brief Default constructor
        RdSpline() : dt_ns(0), start_t_ns(0) {}

        /// @brief Constructor with knot interval and start time
        ///
        /// @param[in] time_interval_ns knot time interval in nanoseconds
        /// @param[in] start_time_ns start time of the spline in nanoseconds
        RdSpline(int64_t time_interval_ns, int64_t start_time_ns = 0)
                : dt_ns(time_interval_ns), start_t_ns(start_time_ns) {
            pow_inv_dt[0] = 1.0;
            pow_inv_dt[1] = s_to_ns / dt_ns;

            for (size_t i = 2; i < N; i++) {
                pow_inv_dt[i] = pow_inv_dt[i - 1] * pow_inv_dt[1];
            }
        }

        /// @brief Cast to different scalar type
        template <typename Scalar2>
        inline RdSpline<_DIM, _N, Scalar2> cast() const {
            RdSpline<_DIM, _N, Scalar2> res;

            res.dt_ns = dt_ns;
            res.start_t_ns = start_t_ns;

            for (int i = 0; i < _N; i++) res.pow_inv_dt[i] = pow_inv_dt[i];

            for (const auto& k : knots)
                res.knots.emplace_back(k.template cast<Scalar2>());

            return res;
        }

        /// @brief Set start time for spline
        ///
        /// @param[in] start_time_ns start time of the spline in nanoseconds
        inline void setStartTimeNs(int64_t start_time_ns) {
            start_t_ns = start_time_ns;
        }

        /// @brief Maximum time represented by spline
        ///
        /// @return maximum time represented by spline in nanoseconds
        int64_t maxTimeNs() const {
            return start_t_ns + (knots.size() - N + 1) * dt_ns - 1;
        }

        /// @brief Minimum time represented by spline
        ///
        /// @return minimum time represented by spline in nanoseconds
        int64_t minTimeNs() const { return start_t_ns; }

        /// @brief Gererate random trajectory
        ///
        /// @param[in] n number of knots to generate
        /// @param[in] static_init if true the first N knots will be the same
        /// resulting in static initial condition
        void genRandomTrajectory(int n, bool static_init = false) {
            if (static_init) {
                VecD rnd = VecD::Random() * 5;

                for (int i = 0; i < N; i++) knots.push_back(rnd);
                for (int i = 0; i < n - N; i++) knots.push_back(VecD::Random() * 5);
            } else {
                for (int i = 0; i < n; i++) knots.push_back(VecD::Random() * 5);
            }
        }

        /// @brief Add knot to the end of the spline
        ///
        /// @param[in] knot knot to add
        inline void knots_push_back(const VecD& knot) { knots.push_back(knot); }

        /// @brief Remove knot from the back of the spline
        inline void knots_pop_back() { knots.pop_back(); }

        /// @brief Return the first knot of the spline
        ///
        /// @return first knot of the spline
        inline const VecD& knots_front() const { return knots.front(); }

        /// @brief Remove first knot of the spline and increase the start time
        inline void knots_pop_front() {
            start_t_ns += dt_ns;
            knots.pop_front();
        }

        /// @brief Resize containter with knots
        ///
        /// @param[in] n number of knots
        inline void resize(size_t n) { knots.resize(n); }

        /// @brief Return reference to the knot with index i
        ///
        /// @param i index of the knot
        /// @return reference to the knot
        inline VecD& getKnot(int i) { return knots[i]; }

        /// @brief Return const reference to the knot with index i
        ///
        /// @param i index of the knot
        /// @return const reference to the knot
        inline const VecD& getKnot(int i) const { return knots[i]; }

        /// @brief Return const reference to deque with knots
        ///
        /// @return const reference to deque with knots
        const std::deque<VecD, Eigen::aligned_allocator<VecD> >& getKnots() const { return knots; }

        /// @brief Return time interval in nanoseconds
        ///
        /// @return time interval in nanoseconds
        int64_t getTimeIntervalNs() const { return dt_ns; }

        /// @brief Evaluate value or derivative of the spline
        ///
        /// @param Derivative derivative to evaluate (0 for value)
        /// @param[in] time_ns time for evaluating of the spline in nanoseconds
        /// @param[out] J if not nullptr, return the Jacobian of the value with
        /// respect to knots
        /// @return value of the spline or derivative. Euclidean vector of dimention
        /// DIM.
        template <int Derivative = 0>
        VecD evaluate(int64_t time_ns, JacobianStruct* J = nullptr) const {
            int64_t st_ns = (time_ns - start_t_ns);

            BASALT_ASSERT_STREAM(st_ns >= 0, "st_ns " << st_ns << " time_ns " << time_ns
                                                      << " start_t_ns " << start_t_ns);

            int64_t s = st_ns / dt_ns;
            double u = double(st_ns % dt_ns) / double(dt_ns);

            BASALT_ASSERT_STREAM(s >= 0, "s " << s);
            BASALT_ASSERT_STREAM(size_t(s + N) <= knots.size(), "s " << s << " N " << N
                                                                     << " knots.size() "
                                                                     << knots.size());

            VecN p;
            baseCoeffsWithTime<Derivative>(p, u);

            VecN coeff = pow_inv_dt[Derivative] * (blending_matrix_ * p);

            // std::cerr << "p " << p.transpose() << std::endl;
            // std::cerr << "coeff " << coeff.transpose() << std::endl;

            VecD res;
            res.setZero();

            for (int i = 0; i < N; i++) {
                res += coeff[i] * knots[s + i];

                if (J) J->d_val_d_knot[i] = coeff[i];
            }

            if (J) J->start_idx = s;

            return res;
        }

        /// @brief Alias for first derivative of spline. See \ref evaluate.
        inline VecD velocity(int64_t time_ns, JacobianStruct* J = nullptr) const {
            return evaluate<1>(time_ns, J);
        }

        /// @brief Alias for second derivative of spline. See \ref evaluate.
        inline VecD acceleration(int64_t time_ns, JacobianStruct* J = nullptr) const {
            return evaluate<2>(time_ns, J);
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    protected:
        /// @brief Vector of derivatives of time polynomial.
        ///
        /// Computes a derivative of \f$ \begin{bmatrix}1 & t & t^2 & \dots &
        /// t^{N-1}\end{bmatrix} \f$ with repect to time. For example, the first
        /// derivative would be \f$ \begin{bmatrix}0 & 1 & 2 t & \dots & (N-1)
        /// t^{N-2}\end{bmatrix} \f$.
        /// @param Derivative derivative to evaluate
        /// @param[out] res_const vector to store the result
        /// @param[in] t
        template <int Derivative, class Derived>
        static void baseCoeffsWithTime(const Eigen::MatrixBase<Derived>& res_const,
                                       _Scalar t) {
            EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, N);
            Eigen::MatrixBase<Derived>& res =
                    const_cast<Eigen::MatrixBase<Derived>&>(res_const);

            res.setZero();

            if (Derivative < N) {
                res[Derivative] = base_coefficients_(Derivative, Derivative);

                _Scalar _t = t;
                for (int j = Derivative + 1; j < N; j++) {
                    res[j] = base_coefficients_(Derivative, j) * _t;
                    _t = _t * t;
                }
            }
        }

        template <int, int, typename>
        friend class RdSpline;

        static const MatN
                blending_matrix_;  ///< Blending matrix. See \ref computeBlendingMatrix.

        static const MatN base_coefficients_;  ///< Base coefficients matrix.
        ///< See \ref computeBaseCoefficients.

        std::deque<VecD, Eigen::aligned_allocator<VecD> >  knots;    ///< Knots
        int64_t dt_ns;                       ///< Knot interval in nanoseconds
        int64_t start_t_ns;                  ///< Start time in nanoseconds
        std::array<_Scalar, _N> pow_inv_dt;  ///< Array with inverse powers of dt
    };

    template <int _DIM, int _N, typename _Scalar>
    const typename RdSpline<_DIM, _N, _Scalar>::MatN
            RdSpline<_DIM, _N, _Scalar>::base_coefficients_ =
            computeBaseCoefficients<_N, _Scalar>();

    template <int _DIM, int _N, typename _Scalar>
    const typename RdSpline<_DIM, _N, _Scalar>::MatN
            RdSpline<_DIM, _N, _Scalar>::blending_matrix_ =
            computeBlendingMatrix<_N, _Scalar, false>();


    /// @brief Uniform B-spline for SE(3) of order N. Internally uses an SO(3) (\ref
    /// So3Spline) spline for rotation and 3D Euclidean spline (\ref RdSpline) for
    /// translation (split representaion).
    ///
    /// See [[arXiv:1911.08860]](https://arxiv.org/abs/1911.08860) for more details.
    template <int _N, typename _Scalar = double>
    class Se3Spline {
    public:
        static constexpr int N = _N;        ///< Order of the spline.
        static constexpr int DEG = _N - 1;  ///< Degree of the spline.

        using MatN = Eigen::Matrix<_Scalar, _N, _N>;
        using VecN = Eigen::Matrix<_Scalar, _N, 1>;
        using VecNp1 = Eigen::Matrix<_Scalar, _N + 1, 1>;

        using Vec3 = Eigen::Matrix<_Scalar, 3, 1>;
        using Vec6 = Eigen::Matrix<_Scalar, 6, 1>;
        using Vec9 = Eigen::Matrix<_Scalar, 9, 1>;
        using Vec12 = Eigen::Matrix<_Scalar, 12, 1>;

        using Mat3 = Eigen::Matrix<_Scalar, 3, 3>;
        using Mat6 = Eigen::Matrix<_Scalar, 6, 6>;

        using Mat36 = Eigen::Matrix<_Scalar, 3, 6>;
        using Mat39 = Eigen::Matrix<_Scalar, 3, 9>;
        using Mat312 = Eigen::Matrix<_Scalar, 3, 12>;

        using Matrix3Array = std::array<Mat3, N>;
        using Matrix36Array = std::array<Mat36, N>;
        using Matrix6Array = std::array<Mat6, N>;

        using SO3 = Sophus::SO3<_Scalar>;
        using SE3 = Sophus::SE3<_Scalar>;

        using PosJacobianStruct = typename RdSpline<3, N, _Scalar>::JacobianStruct;
        using SO3JacobianStruct = typename So3Spline<N, _Scalar>::JacobianStruct;

        /// @brief Struct to store the accelerometer residual Jacobian with
        /// respect to knots
        struct AccelPosSO3JacobianStruct {
            size_t start_idx;
            std::array<Mat36, N> d_val_d_knot;
        };

        /// @brief Struct to store the pose Jacobian with respect to knots
        struct PosePosSO3JacobianStruct {
            size_t start_idx;
            std::array<Mat6, N> d_val_d_knot;
        };

        /// @brief Constructor with knot interval and start time
        ///
        /// @param[in] time_interval_ns knot time interval in nanoseconds
        /// @param[in] start_time_ns start time of the spline in nanoseconds
        Se3Spline(int64_t time_interval_ns, int64_t start_time_ns = 0)
                : pos_spline(time_interval_ns, start_time_ns),
                  so3_spline(time_interval_ns, start_time_ns),
                  dt_ns(time_interval_ns) {}

        /// @brief Gererate random trajectory
        ///
        /// @param[in] n number of knots to generate
        /// @param[in] static_init if true the first N knots will be the same
        /// resulting in static initial condition
        void genRandomTrajectory(int n, bool static_init = false) {
            so3_spline.genRandomTrajectory(n, static_init);
            pos_spline.genRandomTrajectory(n, static_init);
        }

        /// @brief Set the knot to particular SE(3) pose
        ///
        /// @param[in] pose SE(3) pose
        /// @param[in] i index of the knot
        void setKnot(const Sophus::SE3d &pose, int i) {
            so3_spline.getKnot(i) = pose.so3();
            pos_spline.getKnot(i) = pose.translation();
        }

        /// @brief Reset spline to have num_knots initialized at pose
        ///
        /// @param[in] pose SE(3) pose
        /// @param[in] num_knots number of knots to initialize
        void setKnots(const Sophus::SE3d &pose, int num_knots) {
            so3_spline.resize(num_knots);
            pos_spline.resize(num_knots);

            for (int i = 0; i < num_knots; i++) {
                so3_spline.getKnot(i) = pose.so3();
                pos_spline.getKnot(i) = pose.translation();
            }
        }

        /// @brief Reset spline to the knots from other spline
        ///
        /// @param[in] other spline to copy knots from
        void setKnots(const Se3Spline<N, _Scalar> &other) {
            BASALT_ASSERT(other.dt_ns == dt_ns);
            BASALT_ASSERT(other.pos_spline.getKnots().size() ==
                          other.pos_spline.getKnots().size());

            size_t num_knots = other.pos_spline.getKnots().size();

            so3_spline.resize(num_knots);
            pos_spline.resize(num_knots);

            for (size_t i = 0; i < num_knots; i++) {
                so3_spline.getKnot(i) = other.so3_spline.getKnot(i);
                pos_spline.getKnot(i) = other.pos_spline.getKnot(i);
            }
        }

        /// @brief Add knot to the end of the spline
        ///
        /// @param[in] knot knot to add
        inline void knots_push_back(const SE3 &knot) {
            so3_spline.knots_push_back(knot.so3());
            pos_spline.knots_push_back(knot.translation());
        }

        /// @brief Remove knot from the back of the spline
        inline void knots_pop_back() {
            so3_spline.knots_pop_back();
            pos_spline.knots_pop_back();
        }

        /// @brief Return the first knot of the spline
        ///
        /// @return first knot of the spline
        inline SE3 knots_front() const {
            SE3 res(so3_spline.knots_front(), pos_spline.knots_front());

            return res;
        }

        /// @brief Remove first knot of the spline and increase the start time
        inline void knots_pop_front() {
            so3_spline.knots_pop_front();
            pos_spline.knots_pop_front();

            BASALT_ASSERT(so3_spline.minTimeNs() == pos_spline.minTimeNs());
            BASALT_ASSERT(so3_spline.getKnots().size() == pos_spline.getKnots().size());
        }

        /// @brief Return the last knot of the spline
        ///
        /// @return last knot of the spline
        SE3 getLastKnot() {
            BASALT_ASSERT(so3_spline.getKnots().size() == pos_spline.getKnots().size());

            SE3 res(so3_spline.getKnots().back(), pos_spline.getKnots().back());

            return res;
        }

        /// @brief Return knot with index i
        ///
        /// @param i index of the knot
        /// @return knot
        SE3 getKnot(size_t i) const {
            SE3 res(getKnotSO3(i), getKnotPos(i));
            return res;
        }

        /// @brief Return reference to the SO(3) knot with index i
        ///
        /// @param i index of the knot
        /// @return reference to the SO(3) knot
        inline SO3 &getKnotSO3(size_t i) { return so3_spline.getKnot(i); }

        /// @brief Return const reference to the SO(3) knot with index i
        ///
        /// @param i index of the knot
        /// @return const reference to the SO(3) knot
        inline const SO3 &getKnotSO3(size_t i) const { return so3_spline.getKnot(i); }

        /// @brief Return reference to the position knot with index i
        ///
        /// @param i index of the knot
        /// @return reference to the position knot
        inline Vec3 &getKnotPos(size_t i) { return pos_spline.getKnot(i); }

        /// @brief Return const reference to the position knot with index i
        ///
        /// @param i index of the knot
        /// @return const reference to the position knot
        inline const Vec3 &getKnotPos(size_t i) const {
            return pos_spline.getKnot(i);
        }

        /// @brief Set start time for spline
        ///
        /// @param[in] start_time_ns start time of the spline in nanoseconds
        inline void setStartTimeNs(int64_t s) {
            so3_spline.setStartTimeNs(s);
            pos_spline.setStartTimeNs(s);
        }

        /// @brief Apply increment to the knot
        ///
        /// The incremernt vector consists of translational and rotational parts \f$
        /// [\upsilon, \omega]^T \f$. Given the current pose of the knot \f$ R \in
        /// SO(3), p \in \mathbb{R}^3\f$ the updated pose is: \f{align}{ R' &=
        /// \exp(\omega) R
        /// \\ p' &= p + \upsilon
        /// \f}
        ///  The increment is consistent with \ref
        /// PoseState::applyInc.
        ///
        /// @param[in] i index of the knot
        /// @param[in] inc 6x1 increment vector
        template <typename Derived>
        void applyInc(int i, const Eigen::MatrixBase<Derived> &inc) {
            EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 6);

            pos_spline.getKnot(i) += inc.template head<3>();
            so3_spline.getKnot(i) =
                    SO3::exp(inc.template tail<3>()) * so3_spline.getKnot(i);
        }

        /// @brief Maximum time represented by spline
        ///
        /// @return maximum time represented by spline in nanoseconds
        int64_t maxTimeNs() const {
            BASALT_ASSERT_STREAM(so3_spline.maxTimeNs() == pos_spline.maxTimeNs(),
                                 "so3_spline.maxTimeNs() " << so3_spline.maxTimeNs()
                                                           << " pos_spline.maxTimeNs() "
                                                           << pos_spline.maxTimeNs());
            return pos_spline.maxTimeNs();
        }

        /// @brief Minimum time represented by spline
        ///
        /// @return minimum time represented by spline in nanoseconds
        int64_t minTimeNs() const {
            BASALT_ASSERT_STREAM(so3_spline.minTimeNs() == pos_spline.minTimeNs(),
                                 "so3_spline.minTimeNs() " << so3_spline.minTimeNs()
                                                           << " pos_spline.minTimeNs() "
                                                           << pos_spline.minTimeNs());
            return pos_spline.minTimeNs();
        }

        /// @brief Number of knots in the spline
        size_t numKnots() const { return pos_spline.getKnots().size(); }

        /// @brief Linear acceleration in the world frame.
        ///
        /// @param[in] time_ns time to evaluate linear acceleration in nanoseconds
        inline Vec3 transAccelWorld(int64_t time_ns) const {
            return pos_spline.acceleration(time_ns);
        }

        /// @brief Linear velocity in the world frame.
        ///
        /// @param[in] time_ns time to evaluate linear velocity in nanoseconds
        inline Vec3 transVelWorld(int64_t time_ns) const {
            return pos_spline.velocity(time_ns);
        }

        /// @brief Rotational velocity in the body frame.
        ///
        /// @param[in] time_ns time to evaluate rotational velocity in nanoseconds
        inline Vec3 rotVelBody(int64_t time_ns) const {
            return so3_spline.velocityBody(time_ns);
        }

        /// @brief Evaluate pose.
        ///
        /// @param[in] time_ns time to evaluate pose in nanoseconds
        /// @return SE(3) pose at time_ns
        SE3 pose(int64_t time_ns) const {
            SE3 res;

            res.so3() = so3_spline.evaluate(time_ns);
            res.translation() = pos_spline.evaluate(time_ns);

            return res;
        }

        /// @brief Evaluate pose and compute Jacobian.
        ///
        /// @param[in] time_ns time to evaluate pose in nanoseconds
        /// @param[out] J Jacobian of the pose with respect to knots
        /// @return SE(3) pose at time_ns
        Sophus::SE3d pose(int64_t time_ns, PosePosSO3JacobianStruct *J) const {
            Sophus::SE3d res;

            typename So3Spline<_N, _Scalar>::JacobianStruct Jr;
            typename RdSpline<3, N, _Scalar>::JacobianStruct Jp;

            res.so3() = so3_spline.evaluate(time_ns, &Jr);
            res.translation() = pos_spline.evaluate(time_ns, &Jp);

            if (J) {
                Eigen::Matrix3d RT = res.so3().inverse().matrix();

                J->start_idx = Jr.start_idx;
                for (int i = 0; i < N; i++) {
                    J->d_val_d_knot[i].setZero();
                    J->d_val_d_knot[i].template topLeftCorner<3, 3>() =
                            RT * Jp.d_val_d_knot[i];
                    J->d_val_d_knot[i].template bottomRightCorner<3, 3>() =
                            RT * Jr.d_val_d_knot[i];
                }
            }

            return res;
        }

        /// @brief Evaluate pose and compute time Jacobian.
        ///
        /// @param[in] time_ns time to evaluate pose in nanoseconds
        /// @param[out] J Jacobian of the pose with time
        void d_pose_d_t(int64_t time_ns, Vec6 &J) const {
            J.template head<3>() =
                    so3_spline.evaluate(time_ns).inverse() * transVelWorld(time_ns);
            J.template tail<3>() = rotVelBody(time_ns);
        }

        /// @brief Evaluate position residual.
        ///
        /// @param[in] time_ns time of the measurement
        /// @param[in] measured_position position measurement
        /// @param[out] Jp if not nullptr, Jacobian with respect to knos of the
        /// position spline
        /// @return position residual
        Sophus::Vector3d positionResidual(int64_t time_ns,
                                          const Vec3 &measured_position,
                                          PosJacobianStruct *Jp = nullptr) const {
            return pos_spline.evaluate(time_ns, Jp) - measured_position;
        }

        /// @brief Evaluate orientation residual.
        ///
        /// @param[in] time_ns time of the measurement
        /// @param[in] measured_orientation orientation measurement
        /// @param[out] Jr if not nullptr, Jacobian with respect to knos of the
        /// SO(3) spline
        /// @return orientation residual
        Sophus::Vector3d orientationResidual(int64_t time_ns,
                                             const SO3 &measured_orientation,
                                             SO3JacobianStruct *Jr = nullptr) const {
            Sophus::Vector3d res =
                    (so3_spline.evaluate(time_ns, Jr) * measured_orientation.inverse())
                            .log();

            if (Jr) {
                Eigen::Matrix3d Jrot;
                leftJacobianSO3(res, Jrot);

                for (int i = 0; i < N; i++) {
                    Jr->d_val_d_knot[i] = Jrot * Jr->d_val_d_knot[i];
                }
            }

            return res;
        }

        /// @brief Print knots for debugging.
        inline void print_knots() const {
            for (size_t i = 0; i < pos_spline.getKnots().size(); i++) {
                std::cout << i << ": p:" << pos_spline.getKnot(i).transpose() << " q: "
                          << so3_spline.getKnot(i).unit_quaternion().coeffs().transpose()
                          << std::endl;
            }
        }

        /// @brief Print position knots for debugging.
        inline void print_pos_knots() const {
            for (size_t i = 0; i < pos_spline.getKnots().size(); i++) {
                std::cout << pos_spline.getKnot(i).transpose() << std::endl;
            }
        }

        /// @brief Knot time interval in nanoseconds.
        inline int64_t getDtNs() const { return dt_ns; }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
        RdSpline<3, _N, _Scalar> pos_spline;  ///< Position spline
        So3Spline<_N, _Scalar> so3_spline;    ///< Orientation spline

        int64_t dt_ns;  ///< Knot interval in nanoseconds
    };
}

#endif //STRUCTURAL_MAPPING_SO3SPLINE_H
