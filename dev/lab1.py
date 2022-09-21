import open3d as o3d

# mesh_path = 'ImageToStl.com_pikachu.obj'
# mesh = o3d.io.read_triangle_mesh(mesh_path, True)

# dataset = o3d.data.BunnyMesh()
# mesh = o3d.io.read_triangle_mesh(dataset.path)

mesh = o3d.geometry.TriangleMesh.create_octahedron()

mesh.compute_vertex_normals()
o3d.visualization.draw(mesh, raw_mode=True)
