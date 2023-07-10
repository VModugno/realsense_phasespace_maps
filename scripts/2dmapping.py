# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer

This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.

Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.

Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import copy
import rospy
from std_msgs.msg import String
from phasespace.msg import Markers as markers_msg


class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)


state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

 # change presets
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
# Set the visual preset option to 3 (High Accuracy) and 5 (medium density) to optimize for the environment
depth_sensor.set_option(rs.option.visual_preset, 5)
# Get the laser power value
laser_power = depth_sensor.get_option(rs.option.laser_power)
print(f"Laser Power: {laser_power}")

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The appplcation requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)


# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc_visualization = rs.pointcloud()
cur_pc = rs.pointcloud()
align_to = rs.stream.color
align = rs.align(align_to)
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate) # decimmation filter
threshold_filter = rs.threshold_filter()             # Threshold filter
spatial_filter = rs.spatial_filter()                 # Spatial filter
temporal_filter = rs.temporal_filter()               # Temporal filter
depth_to_disparity = rs.disparity_transform(True)    # Depth to disparity transform
disparity_to_depth = rs.disparity_transform(False)   # Disparity to depth transform



def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))

    if state.scale:
        proj *= 0.5**state.decimate

    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]


def pointTotest(out, verts, color_list, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if len(verts) > 0:
        if painter:
            # Painter's algo, sort points from back to front

            # get reverse sorted indices by z (in view-space)
            # https://gist.github.com/stevenvo/e3dad127598842459b68
            v = view(verts)
            s = v[:, 2].argsort()[::-1]
            proj = project(v[s])
        else:
            proj = project(view(verts))

        if state.scale:
            proj *= 0.5**state.decimate

        h, w = out.shape[:2]

        # proj now contains 2d image coordinates
        j, i = proj.astype(np.uint32).T

        # create a mask to ignore out-of-bound indices
        im = (i >= 0) & (i < h)
        jm = (j >= 0) & (j < w)
        m = im & jm


        # perform uv-coloring in a small center around the desired point
        out[i[m], j[m]] = np.array(color_list)[m]
        out[i[m]-1, j[m]-1] = np.array(color_list)[m]
        out[i[m]+1, j[m]+1] = np.array(color_list)[m]
        out[i[m]-1, j[m]+1] = np.array(color_list)[m]
        out[i[m]+1, j[m]-1] = np.array(color_list)[m]
    else:
        print("no point found")


out = np.empty((h, w, 3), dtype=np.uint8)

class twoDmapper:
    # twoDmapper constructor
    def __init__(self):
        # with this variables I control when starting the computation of the transformation and it does it only once 
        self.start_computing_transformation = False
        self.start_data_acquisition_realsense =False
        self.data_acquired_from_realsense = False
        # phasepsace data
        self.DDDmarkers_phasespace = []
        # realsense data
        self.point_cloudRS = o3d.geometry.PointCloud()
        self.realsensePC = []
        self.realsensetextureCoord = [] 
        self.twoDmarkersCoord = []
        self.DDDmarkers_realsense = []
        self.DDDmarkers_realsense_color = []
        # rigid transformation
        self.registration_result = []
        self.counter = 0  # just for debugging
        # final map 
        self.map = np.zeros((1000, 1000, 3), dtype=np.uint8)
        self.pixel_centre_x = 500
        self.pixel_centre_y = 500
        self.pixel_width = 0.01

    def computeFPFH(self, point_cloud):
        # Compute FPFH features
        radius_normal = 0.5  # Radius for normal estimation
        radius_feature = 0.9  # Radius for FPFH feature computation

        # Create a KDTree parameter for efficient nearest neighbor search
        kd_tree = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)

        point_cloud.estimate_normals(kd_tree)
        # Estimate normals for the point cloud

        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            point_cloud,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return fpfh

    def execute_global_registration(self,source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        
        return result
    
    def execute_manual_global_registration(self,source, target, color_source, color_target):
        
        # pick points from two point clouds and builds correspondences
        picked_id_source = self.pick_points(source,color_source)
        picked_id_target = self.pick_points(target,color_target)
        assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
        assert (len(picked_id_source) == len(picked_id_target))
        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target

        # estimate rough transformation using correspondences
        print("Compute a rough transform using the correspondences given by user")
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(source, target,
                                                o3d.utility.Vector2iVector(corr))

        # point-to-point ICP for refinement
        print("Perform point-to-point ICP refinement")
        max_iteration = 1000
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iteration)
        threshold = 0.03  # 3cm distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),criteria=criteria)
        # Retrieve the final registration error (RMSE)
        final_rmse = reg_p2p.inlier_rmse
        print(f"Final RMSE: {final_rmse:.4f}")
        return reg_p2p
    
    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 1, 0])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def pick_points(self, pcd, color):
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        pcd_temp = copy.deepcopy(pcd)
        pcd_temp.paint_uniform_color(color)
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd_temp)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()
    
    def  rotate_point_cloud(self):
        # Rotate the point cloud to align with the phasespace coordinate system
        for i in range(len(self.realsensePC)):
            input_vec = np.append(self.realsensePC[i], 1.0)
            cur = np.matmul(self.registration_result.transformation, input_vec)
            self.realsensePC[i] = cur[:3]
        

    # in this function I read from the optitrack system and I store the markers coordinate in a list 
    def reading_marker_phasespace_and_compute_transformation(self, data):
        #debug
        #print(self.data_acquired_from_realsense,"   ",self.start_computing_transformation)
        if  self.data_acquired_from_realsense and self.start_computing_transformation:
            # when I have the data from the realsense and I have trigger the computation of the transformation I do it only once
            # when i trigger the computation of the transformation I get the measurements from phasespace
            self.start_computing_transformation = False
            self.DDDmarkers_phasespace = []

            for i in range(len(data.markers)):
                # apparently there are a lot of markers wich are not instantiated and are all in the origin 
                # we need to exclude them from the computation
                if(data.markers[i].x != 0 and data.markers[i].y != 0 and data.markers[i].z != 0):
                    # i need to rescale the coordinates of the markers from millimiters to meters
                    self.DDDmarkers_phasespace.append([data.markers[i].x/1000, data.markers[i].y/1000, data.markers[i].z/1000])
            
            # build the point clouds and the corresponding fpfh features
            source_point_cloud = o3d.geometry.PointCloud()
            source_point_cloud.points = o3d.utility.Vector3dVector(np.asarray(self.DDDmarkers_realsense))
            source_fpfh = self.computeFPFH(source_point_cloud)
            print(self.DDDmarkers_realsense)
            target_point_cloud = o3d.geometry.PointCloud()
            target_point_cloud.points = o3d.utility.Vector3dVector(np.asarray(self.DDDmarkers_phasespace))
            target_fpfh = self.computeFPFH(target_point_cloud)
            print(self.DDDmarkers_phasespace)

            # debug (testing without noise to check the icp parameters)
            #trans_init_test = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
            #                 [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            
            #print(trans_init_test)
            #source_point_cloud = copy.deepcopy(target_point_cloud)
            #source_point_cloud.transform(trans_init_test)
            # debugcd


            self.draw_registration_result(source_point_cloud, target_point_cloud, np.identity(4))

            #voxel_size = 2  
            #self.registration_result = self.execute_global_registration(source_point_cloud, target_point_cloud, source_fpfh,
            #                    target_fpfh, voxel_size)

            self.registration_result = self.execute_manual_global_registration(source_point_cloud, target_point_cloud,[1, 0, 0],[0, 1, 0])

            self.draw_registration_result(source_point_cloud, target_point_cloud,  self.registration_result.transformation)

            self.point_cloudRS.transform(self.registration_result.transformation)

            # visualize rotated point cloud
            o3d.visualization.draw_geometries([self.point_cloudRS])

            # Extract point coordinates
            cur_points = self.point_cloudRS.points

            # Extract color information
            cur_colors = self.point_cloudRS.colors

            # Convert points and colors to NumPy arrays
            cur_points_np = np.asarray(cur_points)
            cur_colors_np = np.asarray(cur_colors)

            # Print the extracted point coordinates and color information
            print("Point Coordinates:")
            print(cur_points_np)

            print("Color Information:")
            print(cur_colors_np)

            for i in range(len(cur_points_np)):
                grid_coord_x = int(np.floor((cur_points_np[i,2])/self.pixel_width + self.pixel_centre_x))
                grid_coord_y = int(np.floor((cur_points_np[i,0])/self.pixel_width + self.pixel_centre_y))

                # transform from ROS to CV coordinate:
                grid_coord_x_cv = 1000 - grid_coord_y
                grid_coord_y_cv = grid_coord_x
                self.map[grid_coord_x_cv, grid_coord_y_cv] = cur_colors_np[i,:]

            # appply transformation to the RS point cloud
            #print("apply rotations to point")
            #self.rotate_point_cloud()
            # apply point to map
            # the phasespace coordinate system is y up instead of z up.
            # hence phasespace (x,y,z) -> (z,x,y) in ROS
            #print("write the point cloud to the map")
            #cw, ch = self.current_color_frame.shape[:2][::-1]
            #v, u = (self.realsensetextureCoord * (cw, ch) + 0.5).astype(np.uint32).T
            #for i in range(len(self.realsensePC)):
            #    grid_coord_x = int(np.floor((self.realsensePC[i,2])/self.pixel_width + self.pixel_centre_x))
            #    grid_coord_y = int(np.floor((self.realsensePC[i,0])/self.pixel_width + self.pixel_centre_y))

                # transform from ROS to CV coordinate:
            #    grid_coord_x_cv = 1000 - grid_coord_y
            #    grid_coord_y_cv = grid_coord_x
            #    self.map[grid_coord_x_cv, grid_coord_y_cv] = self.current_color_frame[u[i], v[i]]

            # plot map
            print("plot the current map")
            cv2.imshow("current map", self.map)
            key = cv2.waitKey(1)  

        elif not self.data_acquired_from_realsense and self.start_computing_transformation:
            #print warning message
            self.start_computing_transformation = False
            print("Warning: I can't compute the transformation because I don't have the data from the realsense. acquire them first")


    def reading_marker_realsense(self, color_frame, depth_frame):
        if  self.start_data_acquisition_realsense:

            self.start_data_acquisition_realsense = False
            self.data_acquired_from_realsense = True
            #each time I collect the point I have to clean the data stored so far 
            #self.realsensePC, self.realsensetextureCoord = [], []
            self.DDDmarkers_realsense, self.twoDmarkersCoord = [], []
            self.DDDmarkers_realsense_color = []
            self.realsensePC, self.realsensetextureCoord = [], []
            
            #self.point_cloudRS.clear()
    
             # first I search for the red markers from the color frame
            # then I get the depth value of the pixel and I convert it in a 3d point
            # I store the 3d point in a list
            #decimated_color_frame = decimate.process(color_frame)
            #cv_image = np.array(decimated_color_frame.get_data())
            cv_image = np.array(color_frame.get_data())
             # here I get the depth value of the pixel and I convert it in a 3d point
            depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        
            # Select the ROI interactively
            roi = cv2.selectROI("Select ROI", cv_image, fromCenter=False)

            # Extract the ROI from the image
            x_roi, y_roi, w_roi, h_roi = roi
            roi_image = cv_image[y_roi:y_roi+h_roi, x_roi:x_roi+w_roi]
            #cv_image = cv2.cvtColor(np.array(decimated_color_frame.get_data()), cv2.COLOR_BGR2RGB)
            # here I search for the red markers
            # Convert the image to grayscale
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to reduce noise
            #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
              # Apply adaptive thresholding to obtain binary image
            _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Iterate through each contour
            for contour in contours:
                # get the contour center coordinates on the plane of the original
                # Get the bounding box coordinates
                moments = cv2.moments(contour)

                # Calculate the centroid of the contour
                if(moments["m00"]>0):
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])

                    self.twoDmarkersCoord.append([x_roi+cx, y_roi+cy])

                    # computing the 3d point 
                    depth = depth_frame.as_depth_frame().get_distance(x_roi+cx, y_roi+cy)
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_roi+cx, y_roi+cy], depth)
                    self.DDDmarkers_realsense.append(depth_point)

                    # plot pointson the image
                    radius = 3
                    color = (0, 255, 0)  # Green color
                    self.DDDmarkers_realsense_color.append(color)
                    thickness = -1  # Filled circle
                    # Draw a rectangle around the contour
                    cv2.circle(cv_image, (x_roi+cx, y_roi+cy), radius, color, thickness)

        
            # I show the resulting image
            cv2.imshow('color_frame', cv_image)
            cv2.waitKey(1)

            # here I compute the pointcloud
            #points = cur_pc.calculate(depth_frame)
            #cur_pc.map_to(color_frame)
            #v,t = points.get_vertices(), points.get_texture_coordinates()
            #self.realsensePC = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            #self.realsensetextureCoord = np.asanyarray(t).view(np.float32).reshape(-1, 2) 
            #self.current_color_frame = np.asanyarray(color_frame.get_data())

            #print("Saving to cur.ply...")
            #points.export_to_ply("cur.ply", color_frame)
            #print("Done")

            #self.point_cloudRS = o3d.io.read_point_cloud("cur.ply")
            #o3d.visualization.draw_geometries(self.point_cloudRS)

            # Pointcloud data to arrays
            #self.realsensePC, self.realsensetextureCoord = points.get_vertices(), points.get_texture_coordinates()
             # Convert RealSense point cloud to Open3D point cloud
           
            intrinsic = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(np.asarray(color_frame.get_data())),o3d.geometry.Image(np.asarray(depth_frame.get_data())),convert_rgb_to_intensity=False)
            pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intrinsic.width, intrinsic.height, intrinsic.fx,
                                                                      intrinsic.fy, intrinsic.ppx, intrinsic.ppy)
            self.point_cloudRS=o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
            #print(rgbd_image)
            
            # visualizing point cloud
            o3d.visualization.draw_geometries([self.point_cloudRS])






    def start_realsense_acquisition(self):
        self.start_data_acquisition_realsense = True
    
    def start_computing_rigid_transform(self):
        self.start_computing_transformation = True

           
    


def main():

     # here i create the 2d mapper object
    mapper = twoDmapper()
    rospy.init_node('2dmapper')
    rospy.Subscriber("phasespace_markers", markers_msg, mapper.reading_marker_phasespace_and_compute_transformation)
   
    # Set the publishing rate
    rate = rospy.Rate(100) #100 hz
    

    while not rospy.is_shutdown():
        # Grab camera data
        if not state.paused:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            
            # original frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # decimated and aligned frames
            decimated = decimate.process(frames).as_frameset()
            # Align the depth frame to color frame
            aligned_frames = align.process(decimated)
            depth_frame_aligned = aligned_frames.get_depth_frame()
            color_frame_aligned = aligned_frames.get_color_frame()

            depth_frame = depth_frame_aligned

            # here I apply the other filters
            depth_frame = threshold_filter.process(depth_frame)
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)
            # Convert depth to disparity
            #depth_frame = depth_to_disparity.process(depth_frame)
            # Convert disparity back to depth
            #depth_frame = disparity_to_depth.process(depth_frame)


            mapper.reading_marker_realsense(color_frame_aligned, depth_frame)

            # data for visualization 
            # Grab new intrinsics (may be changed by decimation)
            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            mapped_frame, color_source = color_frame, color_image

            points = pc_visualization.calculate(depth_frame)
            pc_visualization.map_to(mapped_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

        # Render
        now = time.time()

        # visualiztion block on the opencv window
       
        out.fill(0)

        grid(out, (0, 0.5, 1), size=1, n=10)
        frustum(out, depth_intrinsics)
        axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

        if not state.scale or out.shape[:2] == (h, w):
            pointcloud(out, verts, texcoords, color_source)
            if(mapper.data_acquired_from_realsense):
                pointTotest(out, mapper.DDDmarkers_realsense,mapper.DDDmarkers_realsense_color)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pointcloud(tmp, verts, texcoords, color_source)
            if(mapper.data_acquired_from_realsense):
                pointTotest(tmp, mapper.DDDmarkers_realsense,mapper.DDDmarkers_realsense_color)
            tmp = cv2.resize(
                tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)

        if any(state.mouse_btns):
            axes(out, view(state.pivot), state.rotation, thickness=4)

        dt = time.time() - now

        cv2.setWindowTitle(
            state.WIN_NAME, "RealSense (%dx%d) %dFPS (%.2fms) %s" %
            (w, h, 1.0/dt, dt*1000, "PAUSED" if state.paused else ""))

        cv2.imshow(state.WIN_NAME, out)
        
        key = cv2.waitKey(1)  
        # end visualization block

        # Handle key presses

        if key == ord("f"):
            print("starting calibration process, aquiring data from realsense")
            mapper.start_realsense_acquisition()

        if key == ord("s"):
            print("computing transformation matrix using phasespace mocap and apply rigid transform to realsense pointcloud")
            mapper.start_computing_rigid_transform()

        if key == ord("r"):
            state.reset()

        if key == ord("p"):
            state.paused ^= True

        if key == ord("d"):
            state.decimate = (state.decimate + 1) % 3
            decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)

        if key == ord("z"):
            state.scale ^= True

        if key == ord("c"):
            state.color ^= True

        if key == ord("k"):
            cv2.imwrite('./out.png', out)

        if key == ord("e"):
            points.export_to_ply('./out.ply', mapped_frame)

        if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
            break

        rate.sleep()

    # Stop streaming
    pipeline.stop()



if __name__ == '__main__':
    main()