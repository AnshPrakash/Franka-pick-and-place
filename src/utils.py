import numpy as np
import pybullet as p
import open3d as o3d
from typing import Tuple
from scipy.spatial.transform import Rotation as R


def pb_image_to_numpy(rgbpx, depthpx, segpx, width, height):
    """
    Convert pybullet camera images to numpy arrays.

    Args:
        rgbpx: RGBA pixel values
        depthpx: Depth map pixel values
        segpx: Segmentation map pixel values
        width: Image width
        height: Image height

    Returns:
        Tuple of:
            rgb: RGBA image as numpy array [height, width, 4]
            depth: Depth map as numpy array [height, width]
            seg: Segmentation map as numpy array [height, width]
    """
    # RGBA - channel Range [0-255]
    rgb = np.reshape(rgbpx, [height, width, 4])
    # Depth Map Range [0.0-1.0]
    depth = np.reshape(depthpx, [height, width])
    # Segmentation Map Range {obj_ids}
    seg = np.reshape(segpx, [height, width])

    return rgb, depth, seg

def get_pcd_from_numpy(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def get_robot_view_matrix(pos, rot):
    """Copied fragment from get_ee_renders() method of simulation.py.
    """
    rot_matrix = p.getMatrixFromQuaternion(rot)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    init_camera_vector = (0, 0, 1)  # z-axis
    init_up_vector = (0, 1, 0)  # y-axis

    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix = p.computeViewMatrix(
        pos, pos + 0.1 * camera_vector, up_vector)

    return view_matrix


def visualize_point_cloud(points, sphere_radius=0.01, color=[1, 0, 0, 1], max_points=20):
    """
    For testing point cloud computation.
    Visualizes a point cloud in PyBullet by placing a small sphere at each point.

    Parameters:
      points       : (N,3) numpy array of 3D points.
      sphere_radius: Radius of the sphere used for each point.
      color        : RGBA color for the spheres.
      max_points   : Maximum number of points to visualize (for performance reasons).
                     If the cloud is larger, a random subset will be used.
    """
    N = points.shape[0]
    if N > max_points:
        indices = np.random.choice(N, max_points, replace=False)
        points = points[indices]

    # Create a visual shape for a small sphere. This shape will be re-used for all points.
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                          radius=sphere_radius,
                                          rgbaColor=color)

    # For each point, create a massless multi-body with the sphere as its visual shape.
    for pt in points:
        p.createMultiBody(baseMass=0,
                          baseVisualShapeIndex=visual_shape_id,
                          basePosition=pt.tolist())

def matrix_to_pose(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a 4x4 homogeneous transformation matrix into a translation vector and quaternion.
    
    Args:
        matrix: 4x4 homogeneous transformation matrix.
        
    Returns:
        A tuple (translation, quaternion), where:
          - translation is a (3,) np.ndarray.
          - quaternion is a (4,) np.ndarray in [x, y, z, w] order.
    """
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    quat = R.from_matrix(rotation_matrix).as_quat()  # returns [x, y, z, w]
    return translation, quat


def extract_mesh_data(object_id):
    import trimesh
    # Extract visual data
    visual_data = p.getVisualShapeData(object_id)
    # get global scaling of object
    scaling = visual_data[0][3]
    # passes the mesh file to load
    mesh_file = visual_data[0][4].decode('utf-8')
    
    # Load mesh using Open3D or trimesh
    mesh = o3d.io.read_triangle_mesh(mesh_file, enable_post_processing=True)
    if mesh.is_empty():
        tri_mesh = trimesh.load(mesh_file)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(tri_mesh.vertices),
            o3d.utility.Vector3iVector(tri_mesh.faces)
        )
    
    # Scale the mesh
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertices) * np.array(scaling))
    # possibly recenter the mesh => https://www.open3d.org/docs/latest/tutorial/Basic/transformation.html might not be necessary?
    center = mesh.get_center()
    print(f"Mesh Center Object {object_id}: ", center)
    mesh.translate(-center)
    
    # Get the object's current pose from PyBullet
    pos, orn = p.getBasePositionAndOrientation(object_id)
    # Convert quaternion to rotation matrix (3x3)
    R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    
    # Build a 4x4 homogeneous transformation matrix:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos

    # Apply the transformation to the mesh vertices
    mesh.transform(T)
    
    # Retrieve the transformed vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    return vertices, triangles

