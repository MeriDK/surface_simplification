import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def equation_plane(v0, v1, v2):
    """
    calculate plane parameters such so:
    1. ax + by + cz + d = 0
    2. a^2 + b^2 + c^2 = 1
    """

    point_mat = np.array([v0, v1, v2])
    abc = np.matmul(np.linalg.inv(point_mat), np.array([[1], [1], [1]]))
    p = np.concatenate([abc.T, np.array(-1).reshape(1, 1)], axis=1) / (np.sum(abc ** 2) ** 0.5)
    p = p.reshape(4)

    return p


def calculate_K(vertices, triangles):
    """calculate fundamental error quadric K for each plane"""
    K_list = []

    for i in range(len(triangles)):
        t = triangles[i]
        v0, v1, v2 = vertices[t[0]], vertices[t[1]], vertices[t[2]]
        p = equation_plane(v0, v1, v2)
        K = np.outer(p, p)
        K_list.append(K)

    return K_list


def create_v_tr(triangles):
    """
    create dict {v_id: [t_id0, t_id1, ...]}
    where v_id - vertices id
    t_id0, t_id1 ... - corresponding to v_id vertex ids of triangles
    """
    v_tr = {}

    for i in range(len(triangles)):
        t = triangles[i]
        for v in t:
            if v in v_tr:
                v_tr[v].append(i)
            else:
                v_tr[v] = [i]

    return v_tr


def calculate_Q(K_list, v_tr):
    """calculate Q (4x4 matrices)"""
    v_Q = {}

    for v_id, tr_ids in v_tr.items():
        Q = sum([K_list[el] for el in tr_ids])
        v_Q[v_id] = Q

    return v_Q


def find_edges(triangles):
    """find edges of a mesh having its planes"""
    edge_1 = triangles[:, 0:2]
    edge_2 = triangles[:, 1:]
    edge_3 = np.concatenate([triangles[:, :1], triangles[:, -1:]], axis=1)
    edges = np.concatenate([edge_1, edge_2, edge_3], axis=0)

    unique_edges_trans, unique_edges_locs = np.unique(edges[:, 0] * (10 ** 10) + edges[:, 1], return_index=True)
    edges = edges[unique_edges_locs, :]

    return edges


def select_valid_pairs(vertices, triangles, t):
    """
    a pair (v1, v2) is a valid pair for contraction if either:
    1. (v1, v2 ) is an edge, or
    2. ||v1 − v2|| < t, where t is a threshold parameter
    """
    valid_pairs = np.array([])

    for i in tqdm(range(len(vertices))):
        v = vertices[i]
        distances = np.sum((vertices - v) ** 2, axis=1) ** 0.5
        valid = np.where(distances < t)[0]
        pairs = np.vstack((np.ones(len(valid)) * i, valid)).T

        if i == 0:
            valid_pairs = pairs
        else:
            valid_pairs = np.vstack((valid_pairs, pairs))

    valid_pairs = valid_pairs[valid_pairs[:, 0] != valid_pairs[:, 1]]
    valid_pairs = valid_pairs.astype(int)

    edges = find_edges(triangles)
    valid_pairs = np.vstack((valid_pairs, edges))

    valid_pairs = valid_pairs[valid_pairs[:, 0] != valid_pairs[:, 1]]

    unique_valid_pairs_trans, unique_valid_pairs_loc = np.unique(valid_pairs[:, 0] * (10 ** 10) + valid_pairs[:, 1],
                                                                 return_index=True)
    valid_pairs = valid_pairs[unique_valid_pairs_loc, :]

    return valid_pairs


def calculate_error(valid_pairs, v_Q, vertices):
    """
    Compute the optimal contraction target v¯ for each valid pair
    (v1, v2 ). The error v¯T(Q1 +Q2 )v¯ of this target vertex becomes
    the cost of contracting that pair
    """
    errors = []

    b = np.ones((vertices.shape[0], 1))
    vertices_4d = np.hstack((vertices, b))

    for i in range(len(valid_pairs)):
        pair = valid_pairs[i]
        v = (vertices_4d[pair[0]] - vertices_4d[pair[1]]) / 2

        try:
            Q1 = v_Q[pair[0]]
            Q2 = v_Q[pair[1]]
            Q = Q1 + Q2

            error = np.dot(np.dot(v, Q), v.T)
        except KeyError:
            error = 10 ** 10

        errors.append(error)

    return errors


def create_queue(valid_pairs, errors):
    """
    Place all the pairs in a heap keyed (numpy array in this implementation)
    on cost with the minimum cost pair at the top
    """
    queue = np.hstack((valid_pairs, np.reshape(errors, (-1, 1))))
    # queue = queue[queue[:, 2] != 10 ** 10]
    queue = queue[np.argsort(queue[:, 2])]

    return queue


def update_queue(queue, v0, v1, v_Q, vertices):
    """
    Update the costs of all valid pairs involving v0, v1
    """
    # drop invalid pairs from queue for v1
    queue = queue[~((queue[:, 0] == v1) | (queue[:, 1] == v1))]

    # update error value for v0
    value_pairs_to_update = queue[(queue[:, 0] == v0) | (queue[:, 1] == v0)][:, :2].astype(int)
    errors_to_update = np.array(calculate_error(value_pairs_to_update, v_Q, vertices))

    rows = np.argwhere((queue[:, 0] == v0) | (queue[:, 1] == v0))
    queue[rows, 2] = np.reshape(errors_to_update, (-1, 1))

    # sort again
    queue = queue[np.argsort(queue[:, 2])]

    return queue


def remove_one_vertex(v0, v1, vertices, triangles, v_tr):
    """
    Function to contract the pair (v1, v2) after removing it from the heap
    """
    vertices[v0] = (vertices[v0] + vertices[v1]) / 2

    # 2. change v2 to v1 in planes
    try:
        for tr in v_tr[v1]:
            triangles[tr][triangles[tr] == v1] = v0
    except KeyError:
        pass

    return vertices, triangles


def remove_vertices(vertices, triangles, ratio, t):
    """
    Iteratively remove the pair (v1, v2 ) of least cost from the heap,
    contract this pair, and update the costs of all valid pairs involving v0 and v1
    """
    valid_pairs = select_valid_pairs(vertices, triangles, t=t)
    num_steps = int(len(triangles) * (1 - ratio) / 2)
    K_list = calculate_K(vertices, triangles)
    v_tr = create_v_tr(triangles)
    v_Q = calculate_Q(K_list, v_tr)
    errors = calculate_error(valid_pairs, v_Q, vertices)
    queue = create_queue(valid_pairs, errors)

    for i in tqdm(range(num_steps)):

        v0, v1, error = queue[0]
        v0, v1 = int(v0), int(v1)
        v_Q[v0] = v_Q[v0] + v_Q[v1]
        del v_Q[v1]

        vertices, triangles = remove_one_vertex(v0, v1, vertices, triangles, v_tr)
        valid_pairs = valid_pairs[~((valid_pairs[:, 0] == v1) | (valid_pairs[:, 1] == v1))]

        triangles = np.unique(triangles, axis=0)
        triangles = triangles[(triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2]) & (
                    triangles[:, 2] != triangles[:, 0])]

        v_tr = create_v_tr(triangles)
        queue = update_queue(queue, v0, v1, v_Q, vertices)

        if len(queue) == 0:
            break

    return vertices, triangles


def mesh_simplify(mesh, ratio, t):
    print(mesh)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    vertices_new, triangles_new = remove_vertices(vertices, triangles, ratio=ratio, t=t)

    mesh_new = o3d.geometry.TriangleMesh()
    mesh_new.vertices = o3d.utility.Vector3dVector(vertices_new)
    mesh_new.triangles = o3d.utility.Vector3iVector(triangles_new)

    mesh_new.remove_unreferenced_vertices()
    print(mesh_new)

    return mesh_new


def main():
    mesh = o3d.geometry.TriangleMesh.create_torus()
    mesh_new = mesh_simplify(mesh, ratio=0.1, t=1)
    o3d.io.write_triangle_mesh('torus_ratio0.1_t1.obj', mesh_new)


if __name__ == '__main__':
    main()
