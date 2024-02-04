import copy
import open3d as od
import numpy as np


def find_transformation(source, target, voxel_size=0.05, num_iterations=1000000, thresh=0.999):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    global_transformation = global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, num_iterations=num_iterations, thresh=thresh)
    estimated_cow_body = get_result(source_down, global_transformation)
    cow_body_bbox = estimated_cow_body.get_axis_aligned_bounding_box()
    return (global_transformation, cow_body_bbox, source_down, source_fpfh, target_down, target_fpfh)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    od.visualization.draw_geometries([source_temp, target_temp])

def point_to_point_icp(source, target, threshold, trans_init):
    # print("Running...")
    reg_p2p = od.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        od.pipelines.registration.TransformationEstimationPointToPoint()
    )
    # print("Done.")
    return(reg_p2p)

def point_to_plane_icp(source, target, threshold, trans_init):
    # print("Running...")
    reg_p2l = od.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        od.pipelines.registration.TransformationEstimationPointToPlane()
    )
    # print("Done.")
    return reg_p2l

def preprocess_point_cloud(pcd, voxel_size):
    # print("Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print("Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        od.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    # print("Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = od.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        od.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, num_iterations=1000000, thresh=0.999):
    distance_threshold = voxel_size * 1.5
    # print("RANSAC registration on downsampled point clouds.")
    # print("Since the downsampling voxel size is %.3f," % voxel_size)
    # print("We use a liberal distance threshold %.3f." % distance_threshold)
    result = od.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        od.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, 
        [
            od.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            od.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], 
        od.pipelines.registration.RANSACConvergenceCriteria(num_iterations, thresh)
    )

    return result

def fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    # print("Apply fast global registration with distance threshold %.3f" % distance_threshold)
    result = od.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        od.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold)
    )

    return result

def get_result(source, transformation):
    index = np.asarray(transformation.correspondence_set)[:, 0]
    source.transform(transformation.transformation)
    source_transformed = source.select_by_index(index)
    return source_transformed



