//
// Created by cyz on 2021/3/10.
// Modified from PCL trimmed_icp.h
//

#ifndef STRUCTURAL_MAPPING_TRIMMEDICP_H
#define STRUCTURAL_MAPPING_TRIMMEDICP_H

#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/correspondence.h>
#include <pcl/point_cloud.h>
#include <pcl/pcl_exports.h>
#include <limits>
// bug in PCL
#include <pcl/recognition/ransac_based/auxiliary.h>

namespace pcl
{
    namespace recognition
    {
        template<typename PointT, typename Scalar>
        class PCL_EXPORTS TrimmedICP: public pcl::registration::TransformationEstimationSVD<PointT, PointT, Scalar>
        {
        public:
            typedef pcl::PointCloud<PointT> PointCloud;
            typedef typename PointCloud::ConstPtr PointCloudConstPtr;

            typedef typename Eigen::Matrix<Scalar, 4, 4> Matrix4;

        public:
            TrimmedICP ()
                    : new_to_old_energy_ratio_ (0.99f)
            {}

            virtual ~TrimmedICP ()
            {}

            /** \brief Call this method before calling align().
              *
              * \param[in] target is target point cloud. The method builds a kd-tree based on 'target' for performing fast closest point search.
              *            The source point cloud will be registered to 'target' (see align() method).
              * */
            inline void
            init (const PointCloudConstPtr& target)
            {
                target_points_ = target;
                kdtree_.setInputCloud (target);
            }

            /** \brief The method performs trimmed ICP, i.e., it rigidly registers the source to the target (passed to the init() method).
              *
              * \param[in] source_points is the point cloud to be registered to the target.
              * \param[in] num_source_points_to_use gives the number of closest source points taken into account for registration. By closest
              * source points we mean the source points closest to the target. These points are computed anew at each iteration.
              * \param[in,out] guess_and_result is the estimated rigid transform. IMPORTANT: this matrix is also taken as the initial guess
              * for the alignment. If there is no guess, set the matrix to identity!
              * */
            inline void
            align (const PointCloud& source_points, int num_source_points_to_use, Matrix4& guess_and_result) const
            {
                int num_trimmed_source_points = num_source_points_to_use, num_source_points = static_cast<int> (source_points.size ());

                if ( num_trimmed_source_points >= num_source_points )
                {
                    printf ("WARNING in 'TrimmedICP::%s()': the user-defined number of source points of interest is greater or equal to "
                            "the total number of source points. Trimmed ICP will work correctly but won't be very efficient. Either set "
                            "the number of source points to use to a lower value or use standard ICP.\n", __func__);
                    num_trimmed_source_points = num_source_points;
                }

                // These are vectors containing source to target correspondences
                pcl::Correspondences full_src_to_tgt (num_source_points), trimmed_src_to_tgt (num_trimmed_source_points);

                // Some variables for the closest point search
                pcl::PointXYZ transformed_source_point;
                std::vector<int> target_index (1);
                std::vector<float> sqr_dist_to_target (1);
                float old_energy, energy = std::numeric_limits<float>::max ();

//          printf ("\nalign\n");

                do
                {
                    // Update the correspondences
                    for ( int i = 0 ; i < num_source_points ; ++i )
                    {
                        // Transform the i-th source point based on the current transform matrix
                        aux::transform (guess_and_result, source_points.points[i], transformed_source_point);

                        // Perform the closest point search
                        kdtree_.nearestKSearch (transformed_source_point, 1, target_index, sqr_dist_to_target);

                        // Update the i-th correspondence
                        full_src_to_tgt[i].index_query = i;
                        full_src_to_tgt[i].index_match = target_index[0];
                        full_src_to_tgt[i].distance = sqr_dist_to_target[0];
                    }

                    // Sort in ascending order according to the squared distance
                    std::sort (full_src_to_tgt.begin (), full_src_to_tgt.end (), TrimmedICP::compareCorrespondences);

                    old_energy = energy;
                    energy = 0.0f;

                    // Now, setup the trimmed correspondences used for the transform estimation
                    for ( int i = 0 ; i < num_trimmed_source_points ; ++i )
                    {
                        trimmed_src_to_tgt[i].index_query = full_src_to_tgt[i].index_query;
                        trimmed_src_to_tgt[i].index_match = full_src_to_tgt[i].index_match;
                        energy += full_src_to_tgt[i].distance;
                    }

                    this->estimateRigidTransformation (source_points, *target_points_, trimmed_src_to_tgt, guess_and_result);

//            printf ("energy = %f, energy diff. = %f, ratio = %f\n", energy, old_energy - energy, energy/old_energy);
                }
                while ( energy/old_energy < new_to_old_energy_ratio_ ); // iterate if enough progress

//          printf ("\n");
            }

            inline void
            setNewToOldEnergyRatio (float ratio)
            {
                if ( ratio >= 1 )
                    new_to_old_energy_ratio_ = 0.99f;
                else
                    new_to_old_energy_ratio_ = ratio;
            }

        protected:
            static inline bool
            compareCorrespondences (const pcl::Correspondence& a, const pcl::Correspondence& b)
            {
                return static_cast<bool> (a.distance < b.distance);
            }

        protected:
            PointCloudConstPtr target_points_;
            pcl::KdTreeFLANN<PointT> kdtree_;
            float new_to_old_energy_ratio_;
        };
    } // namespace recognition
} // namespace pcl

#endif //STRUCTURAL_MAPPING_TRIMMEDICP_H
