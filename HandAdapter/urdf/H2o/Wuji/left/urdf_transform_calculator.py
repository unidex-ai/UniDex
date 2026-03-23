import numpy as np
import pybullet as p
import argparse

def get_link_transform(body_id, link_name):
    """Get 4x4 transform matrix for a specific link"""
    # Find link index (-1 for base link)
    link_index = -1
    for i in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, i)
        if info[12].decode('utf-8') == link_name:
            link_index = i
            break
    
    # Get world position and orientation
    if link_index == -1:  # Base link case
        pos, orn = p.getBasePositionAndOrientation(body_id)
    else:
        link_state = p.getLinkState(body_id, link_index, computeForwardKinematics=True)
        pos = link_state[4]  # World position
        orn = link_state[5]  # World orientation (quaternion)
    
    # Convert to 4x4 homogeneous transformation matrix
    rotation_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    tf = np.eye(4)
    tf[:3, :3] = rotation_matrix
    tf[:3, 3] = pos
    return tf

def save_transforms(source_link, target_link, urdf_file, output_prefix):
    """Calculate and save both forward and inverse transforms"""
    # Initialize PyBullet in non-graphical mode
    physics_client = p.connect(p.DIRECT)
    
    try:
        # Load URDF
        robot_id = p.loadURDF(urdf_file, [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
        
        # Get transforms for both links
        source_tf = get_link_transform(robot_id, source_link)
        target_tf = get_link_transform(robot_id, target_link)
        
        # Calculate forward transform (target in source frame)
        forward_tf = np.linalg.inv(source_tf) @ target_tf
        
        # Calculate inverse transform (source in target frame)
        inverse_tf = np.linalg.inv(forward_tf)
        
        # Save both transforms
        np.save(f"{output_prefix}_forward.npy", forward_tf)
        np.save(f"{output_prefix}_inverse.npy", inverse_tf)
        
        print(f"Saved transforms between {source_link} and {target_link}:")
        print(f"- Forward transform: {output_prefix}_forward.npy")
        print(f"- Inverse transform: {output_prefix}_inverse.npy")
        
        print("\nForward Transform ({} -> {}):".format(source_link, target_link))
        print(np.round(forward_tf, 6))
        
        print("\nInverse Transform ({} -> {}):".format(target_link, source_link))
        print(np.round(inverse_tf, 6))
        
        # Verify they're proper inverses
        identity_check = forward_tf @ inverse_tf
        if np.allclose(identity_check, np.eye(4), atol=1e-6):
            print("\n✅ Transform verification passed (forward @ inverse ≈ identity)")
        else:
            print("\n❌ Transform verification failed!")
            print("Forward @ Inverse should be identity matrix:")
            print(identity_check)
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate transforms between two URDF links')
    parser.add_argument('--urdf', required=True, help='Path to URDF file')
    parser.add_argument('--source', required=True, help='Source link name')
    parser.add_argument('--target', required=True, help='Target link name')
    parser.add_argument('--output', default='transform', help='Output filename prefix')
    
    args = parser.parse_args()
    
    save_transforms(args.source, args.target, args.urdf, args.output)