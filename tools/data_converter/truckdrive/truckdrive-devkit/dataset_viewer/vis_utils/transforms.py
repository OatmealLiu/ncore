import numpy as np
import open3d as o3d
import math
import os, re
from pathlib import Path


def transform_points_4x4(points_hom, T):
    """
    points_hom: Nx4 array of homogeneous points in the 'src' frame
    T:          4x4 transform that takes src -> tgt
    returns Nx4 array of transformed points in 'tgt' frame (homogeneous)
    """
    return (T @ points_hom.T).T


def get_3d_box_corners(x_c, y_c, z_c, l, w, h, yaw_deg, pitch_deg=0, roll_deg=0):
    """
    Returns the 8 corners of an oriented 3D box in homogeneous coords (Nx4).
    The box is centered at (x_c, y_c, z_c) with length=l (front-back),
    width=w (left-right), height=h (up-down), and is rotated about Z by yaw_deg.
    """
    yaw_rad = yaw_deg
    pitch_rad = pitch_deg
    roll_rad = roll_deg  # roll_deg

    # Rotation about Z
    R_z = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0],
            [0, 0, 1],
        ]
    )

    R_y = np.array(
        [
            [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
            [0, 1, 0],
            [-np.sin(pitch_rad), 0, np.cos(pitch_rad)],
        ]
    )

    Rx_roll = np.array(
        [
            [1, 0, 0],
            [0, np.cos(roll_rad), -np.sin(roll_rad)],
            [0, np.sin(roll_rad), np.cos(roll_rad)],
        ]
    )

    # The box’s local corner offsets
    l2 = l / 2.0
    w2 = w / 2.0
    h2 = h / 2.0

    corners_local = np.array(
        [
            [l2, w2, h2],
            [l2, w2, -h2],
            [l2, -w2, h2],
            [l2, -w2, -h2],
            [-l2, w2, h2],
            [-l2, w2, -h2],
            [-l2, -w2, h2],
            [-l2, -w2, -h2],
        ]
    )  # shape (8,3)

    # Rotate
    corners_rotated = corners_local @ R_z.T

    if pitch_deg != 0:
        corners_rotated = corners_rotated @ R_y.T
    if roll_deg != 0:
        corners_rotated = corners_rotated @ Rx_roll.T

    # Translate
    corners_translated = corners_rotated + np.array([x_c, y_c, z_c])

    # Convert to homogeneous (x,y,z,1)
    ones = np.ones((8, 1))
    corners_hom = np.hstack([corners_translated, ones])  # shape (8,4)

    return corners_hom


def from_corners_to_obb(corners_4xN):
    """
    Given Nx4 corners in homogeneous coordinates (x, y, z, 1),
    fit an oriented bounding box using Open3D and return:
       {
         "x_c": center.x,
         "y_c": center.y,
         "z_c": center.z,
         "yaw": rotation about global Z (degrees),
         "l":   bounding box size along OBB's local x,
         "w":   bounding box size along OBB's local y,
         "h":   bounding box size along OBB's local z
       }

    If your box is purely upright, 'yaw' is the angle between
    the OBB local x-axis and the global X-axis in the XY-plane.
    If the box is arbitrarily oriented in 3D, this 'yaw'
    may only capture part of the orientation. See notes below.
    """

    # 1) Convert corners to Nx3 if needed.
    #    corners_4xN is shape (N,4), so skip the last column.
    corners_xyz = corners_4xN[:, :3]
    # 2) Create an Open3D point cloud from these corners.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(corners_xyz)

    # 3) Compute the oriented bounding box.
    obb = pcd.get_oriented_bounding_box()
    # - obb.center:   3D center
    # - obb.extent:   (length_x, length_y, length_z) in OBB’s local axes
    # - obb.R:        3x3 rotation matrix from OBB’s local coords -> global coords

    center = obb.center
    extent = obb.extent  # [ex, ey, ez]
    R = obb.R

    # 4) Extract an approximate 'yaw' about the global Z-axis.
    #    The local x-axis in OBB is R[:, 0]. We'll measure angle in XY plane:
    yaw_rad = np.arctan2(R[1, 0], R[0, 0])
    yaw_deg = np.degrees(yaw_rad)

    # 5) Assign length, width, height from the OBB’s extent.
    #    Note that OBB does NOT guarantee l >= w >= h. You can reorder if needed.
    l, w, h = extent

    return {
        "x": float(center[0]),
        "y": float(center[1]),
        "z": float(center[2]),
        "yaw": float(yaw_deg),
        "l": float(l),
        "w": float(w),
        "h": float(h),
    }


def lidar_timestamps_to_camera(lidar_timestamps, offset=0.0):
    """Convert lidar timestamps to camera time"""
    return [i + int(offset * 1e6) for i in lidar_timestamps]


def radar_timestamps_to_camera(radar_timestamps, offset=0.0):
    """Convert radar timestamps to camera time"""
    return [i + int(offset * 1e6) for i in radar_timestamps]


def match_timestamps_one_to_one(ts1, ts2, max_diff):
    """
    Given two sorted lists ts1 and ts2, find one-to-one matches
    where |ts1[i] - ts2[j]| <= max_diff.

    Returns a list of index pairs (i, j) that matched.
    Each i, j is used at most once.
    """
    i, j = 0, 0
    matches = []
    while i < len(ts1) and j < len(ts2):
        diff = ts1[i] - ts2[j]
        if abs(diff) <= max_diff:
            # We consider this a match:
            matches.append((i, j))
            # Move both pointers forward so each index is used once
            i += 1
            j += 1
        else:
            # Whichever timestamp is smaller, move that pointer forward
            # This tries to "catch up" to the other
            if diff < 0:
                i += 1
            else:
                j += 1
    return matches


def extract_fx_fy_cx_cy(P):
    """
    Given a 3x4 matrix P in canonical form
    [ [fx, 0,  cx, Tx],
      [0,  fy, cy, Ty],
      [0,   0,  1,  Tz] ],
    extract fx, fy, cx, cy.

    NOTE: This works if there's no skew or rotation baked into the top-left 3x3.
          If P includes rotation (R != identity) or skew, you'd need a more general factorization.
    """
    fx = P[0, 0]
    fy = P[1, 1]
    cx = P[0, 2]
    cy = P[1, 2]
    return fx, fy, cx, cy


def plane_eq_from_intrinsics(im_w, im_h, fx, fy, cx, cy):
    """
    Build near/left/right/top/bottom planes in camera coords.
    For near plane, we say z >= 0.
    For left, it's the plane that ensures u >= 0 => (fx*x + cx*z)/z >= 0 => ...
    We'll store them in a form (nx, ny, nz, d).
    """
    planes = []

    # calculating the fov
    h_fov = np.arctan(im_h / 2 / fy)
    w_fov = np.arctan(im_w / 2 / fx)

    # 1) near plane: z >= 0
    # We can represent that as normal=(0,0,1), d=0 =>  n . p >= 0
    planes.append((np.array([0, 0, 1]), 0.0))

    # 2) left plane: (fx*x + cx*z)/z >= 0 => fx*x + cx*z >= 0 => ...
    # This is an inequality in x, z. We can treat that as n=[fx, 0, cx], d=0 => n.(x,y,z) >= 0
    # planes.append((np.array([fx, 0, cx]), 0.0))
    planes.append((np.array([np.cos(w_fov), 0, np.sin(w_fov)]), 0))

    # 3) right plane: (fx*x + cx*z)/z <= W => fx*x + cx*z <= W*z => fx*x + cx*z - W*z <= 0
    # We want it as "n . p >= d", so multiply by -1:
    # -fx*x - cx*z + W*z >= 0 => n=[-fx,0,(W-cx)], d=0
    # planes.append((np.array([fx, 0, (cx - im_w)]), 0.0))
    planes.append((np.array([-np.cos(w_fov), 0, np.sin(w_fov)]), 0))

    # 4) top plane: (fy*y + cy*z)/z >= 0 => fy*y + cy*z >= 0 => n=[0,fy,cy]
    # planes.append((np.array([0, fy, cy]), 0.0))
    planes.append((np.array([0, np.cos(h_fov), np.sin(h_fov)]), 0))

    # 5) bottom plane: (fy*y + cy*z)/z <= H => fy*y + cy*z - H*z <= 0 =>
    # => -fy*y - cy*z + H*z >= 0 => n=[0,-fy, (cy - H)]
    # planes.append((np.array([0, fy, (cy - im_h)]), cy))
    planes.append((np.array([0, -np.cos(h_fov), np.sin(h_fov)]), 0))

    return planes


def clip_convex_polyhedron(points, edges, planes):
    active_points = np.array(points)

    for normal, d in planes:
        # For each plane, create a new set of points that are inside or on-plane.
        new_points = []
        # We can do an edge-based approach:
        for i1, i2 in edges:
            A = active_points[i1]
            B = active_points[i2]
            sideA = normal.dot(A) - d
            sideB = normal.dot(B) - d

            if sideA >= 0 and sideB >= 0:
                # both inside
                new_points.append(A)
                new_points.append(B)
            elif sideA * sideB < 0:
                # They cross the plane => find intersection
                t = sideA / (sideA - sideB)
                I = A + t * (B - A)
                # Keep whichever is inside
                if sideA >= 0:
                    new_points.append(A)
                if sideB >= 0:
                    new_points.append(B)
                # intersection
                new_points.append(I)
            # else both outside => discard

        if len(new_points) == 0:
            return np.empty((0, 3))

        new_points = np.unique(np.round(new_points, 6), axis=0)

        active_points = new_points

        if active_points.shape[0] == 0:
            return np.empty((0, 3))

    return active_points


def clip_polyhedron_once(points, edges, plane):
    normal, dval = plane
    new_points = []
    index_map = {}
    new_edges = []

    def get_new_index(pt):
        """Add pt to new_points if not existing, return index."""
        tup = tuple(np.round(pt, 6))  # float rounding for hashing
        if tup not in index_map:
            index_map[tup] = len(new_points)
            new_points.append(pt)
        return index_map[tup]

    for i1, i2 in edges:
        A = points[i1]
        B = points[i2]

        sideA = normal.dot(A) - dval
        sideB = normal.dot(B) - dval
        A_in = sideA >= 0
        B_in = sideB >= 0

        if A_in and B_in:
            # Both in => keep both endpoints and edge
            new_i1 = get_new_index(A)
            new_i2 = get_new_index(B)
            new_edges.append((new_i1, new_i2))

        elif A_in and not B_in:
            # A in, B out => keep A + intersection
            new_i1 = get_new_index(A)
            # find intersection
            t = sideA / (sideA - sideB)
            I = A + t * (B - A)
            new_i2 = get_new_index(I)
            new_edges.append((new_i1, new_i2))

        elif B_in and not A_in:
            # B in, A out => keep B + intersection
            new_i2 = get_new_index(B)
            t = sideA / (sideA - sideB)
            I = A + t * (B - A)
            new_i1 = get_new_index(I)
            new_edges.append((new_i1, new_i2))
        else:
            continue

    if len(new_points) == 0:
        return np.empty((0, 3)), []

    new_points = np.array(new_points, dtype=float)
    return new_points, new_edges


def clip_polyhedron(points, edges, planes):
    cur_points = points
    cur_edges = edges

    for plane in planes:
        new_points, new_edges = clip_polyhedron_once(cur_points, cur_edges, plane)
        if new_points.shape[0] == 0 or len(new_edges) == 0:
            # entire shape clipped away
            return new_points, new_edges
        cur_points = new_points
        cur_edges = new_edges

    return cur_points, cur_edges


def clip_box_against_camera_fov(corners_3d, edges, im_w, im_h, fx, fy, cx, cy):
    planes = plane_eq_from_intrinsics(im_w, im_h, fx, fy, cx, cy)
    clipped_pts = clip_polyhedron(corners_3d, edges, planes)
    return clipped_pts
