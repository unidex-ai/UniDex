import trimesh
import os

input_dir = "/home/sqz/jianhan/Dexcap-unihand/UniHand/2/meshes/visual"
output_dir = os.path.join(input_dir, "converted_objs")
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".glb"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".glb", ".obj"))
        try:
            mesh = trimesh.load(input_path, force='mesh')
            mesh.export(output_path)
            print(f"[OK] Converted: {filename} → {os.path.basename(output_path)}")
        except Exception as e:
            print(f"[FAIL] Skipped {filename}: {e}")
