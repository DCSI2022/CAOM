//
// Created by cyz on 2023/1/30.
//

#ifndef STRUCTURAL_MAPPING_VOXELHASH_H
#define STRUCTURAL_MAPPING_VOXELHASH_H

#pragma once

#include <Eigen/Core>
#include <array>
#include <vector>

#include <tsl/robin_map.h>  // TODO

namespace kiss_icp {

    struct Voxel {
        Voxel(int32_t x, int32_t y, int32_t z) : ijk({x, y, z}) {}
        Voxel(const Eigen::Vector3d &point, double voxel_size)
                : ijk({static_cast<int32_t>(point.x() / voxel_size),
                       static_cast<int32_t>(point.y() / voxel_size),
                       static_cast<int32_t>(point.z() / voxel_size)}) {}
        inline bool operator==(const Voxel &vox) const {
            return ijk[0] == vox.ijk[0] && ijk[1] == vox.ijk[1] && ijk[2] == vox.ijk[2];
        }

        std::array<int32_t, 3> ijk;
    };

    struct VoxelHashMap {

        using Vector3dVector = std::vector<Eigen::Vector3d>;
        using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;

        explicit VoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel)
                : voxel_size_(voxel_size),
                  max_distance_(max_distance),
                  max_points_per_voxel_(max_points_per_voxel) {}

        Eigen::Matrix4d RegisterPoinCloud(const Vector3dVector &points,
                                          const Eigen::Matrix4d &initial_guess,
                                          double max_correspondence_distance,
                                          double kernel);

        Vector3dVectorTuple GetCorrespondences(const Vector3dVector &points,
                                               double max_correspondance_distance) const;
        inline void Clear() { map_.clear(); }
        inline bool Empty() const { return map_.empty(); }
        void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin);
        void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Matrix4d &pose);
        void AddPoints(const std::vector<Eigen::Vector3d> &points);
        void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
        std::vector<Eigen::Vector3d> Pointcloud() const;

        struct VoxelBlock {
            // buffer of points with a max limit of n_points
            std::vector<Eigen::Vector3d> points;
            int num_points_;
            inline void AddPoint(const Eigen::Vector3d &point) {
                if (points.size() < static_cast<size_t>(num_points_)) points.push_back(point);
            }
        };

        double voxel_size_;
        double max_distance_;
        int max_points_per_voxel_;
        tsl::robin_map<kiss_icp::Voxel, VoxelHashMap::VoxelBlock> map_;
    };

}  // namespace kiss_icp

// Specialization of std::hash for our custom type Voxel
namespace std {

    template <>
    struct hash<kiss_icp::Voxel> {
        std::size_t operator()(const kiss_icp::Voxel &vox) const {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(vox.ijk.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
        }
    };
}  // namespace std

#endif //STRUCTURAL_MAPPING_VOXELHASH_H
