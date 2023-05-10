//
// Created by joe on 2020/11/18.
//

#ifndef STRUCTURAL_MAPPING_CONSTRAINTSPHERE_HPP
#define STRUCTURAL_MAPPING_CONSTRAINTSPHERE_HPP

#include "tools.h"

//! A class to manage the constraint matrix, which tracks how well the ICP system is constrained,
//! and can report on how much the addition of a new point would help the constraints
class ConstraintSphere
{
public:
    //! Constructor
    /*! @param threshold \sa setThreshold(double threshold) */
    ConstraintSphere(double threshold):threshold_(threshold),
                                       m_(decltype(m_)::Zero()),
                                       dirty_(true){}

    //! Set a new threshold
    /*! @param threshold The threshold at which a dimension is considered 'fully constrained'. This number
        is the equivalent to the the square-root of the number of vectors in required each dimension.
        For instance, if you need 100 unit vectors in each dimension, then the threshold should be 10. */
    void setThreshold(double threshold){threshold_ = threshold;}

    //! Get the threshold
    double getThreshold(){ return threshold_; }

    //! Add a constraint to the system in the form of a normal, and the weight.
    /*! @param normal The normal to insert. This is normalized internally.
        @param weight The weight of the constraint, e.g. the number of points this normal represents */
    void addConstraint(Eigen::Vector3f const & normal, double const weight){

        Eigen::Vector3f const n = normal.normalized();

        m_(0,0) += n.x()*n.x() * weight;
        m_(1,1) += n.y()*n.y() * weight;
        m_(2,2) += n.z()*n.z() * weight;

        m_(1,0) += n.y()*n.x() * weight;
        m_(2,0) += n.z()*n.x() * weight;
        m_(2,1) += n.z()*n.y() * weight;

        dirty_ = true;
    }

    //! Determine how useful a given normal would be to insert into the system
    /*! The higher the score, the more the given normal will help constrain the system.
        @param normal The normal to test
        @returns The score, which varies between 0.0 to 1.0*/
    double getScore(Eigen::Vector3f const & normal){

        computeSVD();

        Eigen::Vector3f const n = normal.normalized();

        // Solve the polar-coordinate ellipsoid equation to find the radius in the direction
        // of the query normal
        auto qr = R_t_ * n.cast<double>();

        auto const theta = std::atan(qr[1] / qr[0]);
        auto const phi   = std::acos(qr[2] / qr.norm());

        auto const stheta = std::sin(theta);
        auto const ctheta = std::cos(theta);
        auto const sphi   = std::sin(phi);
        auto const cphi   = std::cos(phi);

//        auto const r = std::sqrt( 1.0 /
//                                  (ctheta*ctheta * sphi*sphi / (radii_[0]*radii_[0]) +
//                                   stheta*stheta * sphi*sphi / (radii_[1]*radii_[1]) +
//                                   cphi*cphi / (radii_[2]*radii_[2])));
        auto const r = std::sqrt(ctheta*ctheta * sphi*sphi *(radii_[0]*radii_[0]) +
                                   stheta*stheta * sphi*sphi *(radii_[1]*radii_[1]) +
                                   cphi*cphi *(radii_[2]*radii_[2]));  // radius

        return 1.0 - std::min(r/threshold_, 1.0);
    }

    //! Get the 3 radii of the constraint ellipsoid.
    /*! To get the equivalent number of normals, the elements of the vector must be squared.
        e.g. `constraint.radii().cwiseProduct(constraint.radii())` */
    Eigen::Vector3d radii(){
        computeSVD();
        return radii_;
    }

    //! Get the rotation of the constraint ellipsoid.
    /*! This is generally not needed, but can be useful for plotting the constraint ellipsoid in 3D.*/
    Eigen::Matrix3d rotation(){
        computeSVD();
        return R_t_.transpose();
    }

    //! Get the constraints matrix
    /*! For debugging only */
    Eigen::Matrix3d constraints(){return m_;}

private:
    //! Compute R_t_ and radii_ with SVD
    /*! This will check and set the dirty_ flag as appropriate */
    void computeSVD(){

        if(dirty_){

            m_(0,1) = m_(1,0);
            m_(0,2) = m_(2,0);
            m_(1,2) = m_(2,1);

            //std::cout << "Constraints: " << std::endl << m_ << std::endl;

            Eigen::JacobiSVD<decltype(m_)> svd(m_, Eigen::ComputeFullV);

            auto s = svd.singularValues();

            R_t_ = svd.matrixV().transpose();
            radii_ = s.cwiseSqrt();

            dirty_ = false;
        }
    }

    double threshold_;      //! The minimum threshold to consider as "constrainted" in any given direction

    Eigen::Matrix3d m_;     //! The constraint matrix

    Eigen::Matrix3d R_t_;     //! The rotation matrix (computed from SVD) transposed
    Eigen::Vector3d radii_; //! The radii (computed from SVD)

    bool dirty_; //! Do we need to recompute SVD on the constraint ellipsoid?
};


#endif //STRUCTURAL_MAPPING_CONSTRAINTSPHERE_HPP
