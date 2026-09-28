#!/usr/bin/env python3
"""
EMMA modified this to make it FASTER!

This script computes the point-pair features of a given
model and tries to find the model in a given scene.

source: https://github.com/whateverforever/model-globally-match-locally-python/blob/master/ppf.py 

Note: Currently, trimesh doesn't support pointcloud with normals. To combat this, you need to
      reconstruct some surface between the points (e.g. ball pivoting)
"""

import random
import time
import argparse
import itertools
from collections import defaultdict

import trimesh
import trimesh.transformations as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

#EMMA
import numpy as np
from numba import njit

@njit(fastmath=True)
def vote_numba(
    m_pairs, model_alphas_array, alpha_s, inv_alpha_step, num_angles, accumulator
):
    n_pairs = m_pairs.shape[0]

    for i in range(n_pairs):
        mA = m_pairs[i, 0]
        mB = m_pairs[i, 1]

        alpha_m = model_alphas_array[mA, mB]

        if alpha_m != -999.0:
            diff = alpha_m - alpha_s
            alpha_disc = int(diff * inv_alpha_step) % num_angles

            if alpha_disc < 0:
                alpha_disc += num_angles

            idx = mA * num_angles + alpha_disc
            accumulator[idx] += 1


def get_poses(model_path, scene_path, scene_pts_fraction=0.5, ppf_rel_dist_step=0.05):
    class Args:
        pass
    
    args = Args()
    args.model = model_path
    args.scene = scene_path
    args.fast = True
    args.scene_pts_fraction = scene_pts_fraction
    args.ppf_num_angles = 30#24 #30
    args.ppf_rel_dist_step = ppf_rel_dist_step
    args.alpha_num_angles = 30#24 #30 #EMMA!
    args.cluster_max_angle = 20 #30 #EMMA!

    _compute_ppf = compute_ppf
    _pdist_rot = pdist_rot

    if args.fast:
        import ppf_fast #this has to be builded first!

        _compute_ppf = ppf_fast.compute_ppf
        _pdist_rot = ppf_fast.pdist_rot
        print("Using fast c++ mode")
    else:
        _compute_ppf = compute_ppf
        _pdist_rot = pdist_rot
        print("Using slow python mode")

    model = trimesh.load(args.model)
    scene = trimesh.load(args.scene)

    if hasattr(model, "vertex_normals") and hasattr(scene, "vertex_normals"):
        _model_vis = model.copy()
        _scene_vis = scene.copy()
    else:
        try:
            print(
                "Your trimesh version still can't load pointclouds with normals. Trying open3d..."
            )
            import open3d as o3d

            model = o3d.io.read_point_cloud(args.model)
            scene = o3d.io.read_point_cloud(args.scene)

            model = trimesh.Trimesh(
                np.asarray(model.points), vertex_normals=np.asarray(model.normals)
            )
            scene = trimesh.Trimesh(
                np.asarray(scene.points), vertex_normals=np.asarray(scene.normals)
            )

            _model_vis = trimesh.PointCloud(vertices=model.vertices)
            _scene_vis = trimesh.PointCloud(vertices=scene.vertices)
        except ImportError as e:
            print("Fallback to open3d failed", e)
            quit()

    print("Model has", len(model.vertices), "vertices. scale=", model.scale)
    print("Scene has", len(scene.vertices), "vertices. scale=", scene.scale)

    # trimesh doesn't support .scale for point clouds
    modelscale = np.linalg.norm(np.max(model.vertices, axis=0) - np.min(model.vertices, axis=0))
    print("modelscale", modelscale)

    _model_vis.visual.vertex_colors = [[255, 0, 0, 255] for _ in _model_vis.vertices]
    _scene_vis.visual.vertex_colors = [
        [150, 200, 150, 240] for _ in _scene_vis.vertices
    ]

    # plt.figure()
    # plt.show()

    # vis = trimesh.Scene([_model_vis, _scene_vis])
    # vis.show()

    ## 1. compute ppfs of all vertex pairs in model, store in hash table
    angle_step = float(np.radians(360 / args.ppf_num_angles))
    dist_step = args.ppf_rel_dist_step * modelscale

    print("Computing model ppfs features")
    t_start = time.perf_counter()
    if args.fast:
        ppfs_model, _, model_alphas = _compute_ppf(
            to_nanobind(model.vertices),
            to_nanobind(model.vertex_normals),
            angle_step,
            dist_step
        )
    else:
        ppfs_model, _, model_alphas = _compute_ppf(
            to_nanobind(model.vertices),
            to_nanobind(model.vertex_normals),
            angle_step,
            dist_step,
            alphas=True
        )
    t_end = time.perf_counter()
    print(f"Computing ppfs for {len(model.vertices)} verts took {t_end - t_start:.2f}s")

    ## 2. choose reference points in scene, compute their ppfs
    t_start = time.perf_counter()
    if args.fast:
        _, pairs_scene, scene_alphas = _compute_ppf(
            to_nanobind(scene.vertices),
            to_nanobind(scene.vertex_normals),
            angle_step,
            dist_step,
            max_dist=modelscale,
            ref_fraction=args.scene_pts_fraction
        )
    else:
        _, pairs_scene, scene_alphas = _compute_ppf(
            to_nanobind(scene.vertices),
            to_nanobind(scene.vertex_normals),
            angle_step,
            dist_step,
            max_dist=modelscale,
            ref_fraction=args.scene_pts_fraction,
            alphas=True
        )
    t_end = time.perf_counter()
    print(f"Computing all scene ppfs took {t_end - t_start:.1f}s")

    ## 3. go through scene ppfs, look up in table if we find model ppf
    t_start = time.perf_counter()
    skipped_features = 0

    # discretization for the alpha rotation
    alpha_step = np.radians(360 / args.alpha_num_angles)

    #++++++++++++++++++++++++++++++++++++++++++++++
    if args.fast:
        print("Running fast C++ matching & voting...")
        num_s_verts = len(scene.vertices)
        scene_alphas_array = np.full((num_s_verts, num_s_verts), -999.0, dtype=np.float32)
        for (sA_idx, sB_idx), val in scene_alphas.items():
            scene_alphas_array[sA_idx, sB_idx] = val

        num_m_verts = len(model.vertices)
        model_alphas_array = np.full((num_m_verts, num_m_verts), -999.0, dtype=np.float32)
        for (mA_idx, mB_idx), val in model_alphas.items():
            model_alphas_array[mA_idx, mB_idx] = val

        alpha_step = float(np.radians(360 / args.alpha_num_angles))

        peaks = ppf_fast.match_and_vote(
            ppfs_model,
            pairs_scene,
            scene_alphas_array,
            model_alphas_array,
            num_m_verts,
            num_s_verts,
            args.alpha_num_angles,
            alpha_step
        )

        poses = []
        unit_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        # for peak in peaks:
        for sA, best_mr, best_alpha, score in peaks:
            s_r = scene.vertices[sA]
            s_normal = scene.vertex_normals[sA]
            # sA = peak.sA
            # best_mr = peak.best_mr
            # best_alpha = peak.best_alpha
            # score = peak.votes

            s_r = scene.vertices[sA]
            s_normal = scene.vertex_normals[sA]
            m_normal = model.vertex_normals[best_mr]
            
            R_scene2glob = np.eye(4)
            R_scene2glob[:3, :3] = align_vectors(s_normal, unit_x) # Reines NumPy!
            T_scene2glob = R_scene2glob @ tf.translation_matrix(-s_r)

            # Modell -> Global Transformation
            R_model2glob = np.eye(4)
            R_model2glob[:3, :3] = align_vectors(m_normal, unit_x) # Reines NumPy!
            T_model2glob = R_model2glob @ tf.translation_matrix(-model.vertices[best_mr])

            R_alpha = tf.rotation_matrix(alpha_step * best_alpha, [1, 0, 0], [0, 0, 0])
            T_model2scene = np.linalg.inv(T_scene2glob) @ R_alpha @ T_model2glob
            
            poses.append((T_model2scene, best_mr, score))

        t_end = time.perf_counter()
        print(f"Matching & voting completed in {t_end - t_start:.2f}s (found {len(poses)} peak candidates)")
    #++++++++++++++++++++++++++++++++++++++++++++++

    else:

        inv_alpha_step = 1.0 / alpha_step #Emma

        poses = []
        # accumulator we're going to reuse for each reference vert
        #accumulator = np.zeros((len(model.vertices), args.alpha_num_angles)) # Emma

        # ---- EMMA
        num_m_verts = len(model.vertices)
        num_angles = args.alpha_num_angles

        # Flaches 1D-Array im C-Speicher-Layout (float32 ist schneller als float64)
        accumulator = np.zeros(num_m_verts * num_angles, dtype=np.float32)

        # ---- EMMA
        num_s_verts = len(scene.vertices)
        scene_alphas_array = np.full((num_s_verts, num_s_verts), -999.0, dtype=np.float32)
        for (sA_idx, sB_idx), val in scene_alphas.items():
            scene_alphas_array[sA_idx, sB_idx] = val

        num_m_verts = len(model.vertices)
        model_alphas_array = np.full((num_m_verts, num_m_verts), -999.0, dtype=np.float32)

        for (mA_idx, mB_idx), val in model_alphas.items():
            model_alphas_array[mA_idx, mB_idx] = val

        t_inner_loop = 0.0      # Vorbereitung, Dict-Lookups, Alpha-Mathe, Accumulator Voting
        t_peaks = 0.0           # np.max und np.argwhere auf dem Accumulator
        t_scene_trans = 0.0     # Scene-Transformationen (s_normal, R_scene2glob, T_scene2glob)
        t_pose_calc = 0.0       # Innere Peaks-Schleife (Model-Normale, R_alpha, Matrix-Invertierung)

        t_sub_lookup_feat = 0.0   # 1. Feature Lookups & Scene-Pairs
        t_sub_lookup_scene = 0.0  # 2. scene_alphas_array Lookup
        t_sub_lookup_model = 0.0  # 3. Model-Pairs & Try-Except
        t_sub_accumulator = 0.0   # 4. Alpha-Berechnung & Accumulator += 1

        t_loop_total_start = time.perf_counter()
        # ---- EMMA

        print("Num reference verts", len(pairs_scene))
        for idx_ref, sA in enumerate(pairs_scene):
            print(
                f"{idx_ref+1}/{len(pairs_scene)}: {len(pairs_scene[sA])} paired verts for ref {sA}",
                " " * 20,
                end="\r",
            )

            # ---- Emma
            t0 = time.perf_counter()

            # one accumulator per reference vert, we set it to zero instead of re-initializing
            #accumulator[...] = 0
            accumulator.fill(0) #Emma

            for sB in pairs_scene[sA]:
                ts0 = time.perf_counter()

                if sA == sB:
                    continue

                s_feature = pairs_scene[sA][sB]
                if s_feature not in ppfs_model:
                    skipped_features += 1
                    continue

                ts1 = time.perf_counter()
                t_sub_lookup_feat += (ts1 - ts0)

                #alpha_s = scene_alphas[(sA, sB)] #EMMA
                #alpha_s = scene_alphas.get((sA, sB))
                # if alpha_s is None:
                #     print("CONTINUE ALPHA")
                #     continue
                # try:
                #     alpha_s = scene_alphas[(sA, sB)]
                # except KeyError:
                #     continue
                alpha_s = scene_alphas_array[sA, sB]
                if alpha_s == -999.0:
                    continue

                ts2 = time.perf_counter()
                t_sub_lookup_scene += (ts2 - ts1)

                m_pairs = ppfs_model[s_feature]  # Sollte ein np.ndarray mit Shape (N, 2) sein
                if len(m_pairs) > 0:
                    ts3 = time.perf_counter()
                    m_pairs_arr = np.asarray(m_pairs)
                    vote_numba(
                        m_pairs_arr,
                        model_alphas_array,
                        alpha_s,
                        inv_alpha_step,
                        num_angles,
                        accumulator,
                    )

                    ts5 = time.perf_counter()
                    t_sub_lookup_model += ts5 - ts3
                    continue #use numba EMMA
                    mAs = m_pairs_arr[:, 0]
                    mBs = m_pairs_arr[:, 1]

                    # Alle Alphas auf einmal aus dem Array lesen
                    alpha_ms = model_alphas_array[mAs, mBs]

                    # Filtern, wo alpha_m gültig ist
                    valid_mask = alpha_ms != -999.0
                    if np.any(valid_mask):
                        valid_mAs = mAs[valid_mask]
                        valid_alpha_ms = alpha_ms[valid_mask]

                        ts4 = time.perf_counter()
                        t_sub_lookup_model += (ts4 - ts3)

                        # Vektorisierte Alpha-Mathe
                        alpha_discs = (
                            (valid_alpha_ms - alpha_s) * inv_alpha_step
                        ).astype(np.int32) % args.alpha_num_angles

                        # Akkumulator auf einen Schlag in C befüllen!
                        flat_indices = valid_mAs * args.alpha_num_angles + alpha_discs
                        np.add.at(accumulator, flat_indices, 1)
                        #np.add.at(accumulator, (valid_mAs, alpha_discs), 1)

                        ts5 = time.perf_counter()
                        t_sub_accumulator += (ts5 - ts4)

                # for m_pair in ppfs_model[s_feature]:
                #     ts3 = time.perf_counter()
                #     mA, mB = m_pair
                #     #alpha_m = model_alphas[m_pair] # EMMA
                #     # alpha_m = model_alphas.get(m_pair)
                #     # if alpha_m is None:
                #     #     print("CONTINUE ALPHA_M")
                #     #     continue
                #     # try:
                #     #     alpha_m = model_alphas[m_pair]
                #     # except KeyError:
                #     #     continue
                #     alpha_m = model_alphas_array[mA, mB]
                #     if alpha_m == -999.0:
                #         continue
                #     #alpha = alpha_m - alpha_s #Emma

                #     ts4 = time.perf_counter()
                #     t_sub_lookup_model += (ts4 - ts3)

                #     # alpha_disc = int(alpha // alpha_step) Emma
                #     alpha_disc = int((alpha_m - alpha_s) * inv_alpha_step) % args.alpha_num_angles
                #     #accumulator[mA, alpha_disc] += 1
                #     accumulator[mA * num_angles + alpha_disc] += 1
                #     # accumulator[mA, (alpha_disc - 1) % args.alpha_num_angles] += 1
                #     # accumulator[mA, (alpha_disc + 1) % args.alpha_num_angles] += 1

                #     ts5 = time.perf_counter()
                #     t_sub_accumulator += (ts5 - ts4)
            
            # ---- Emma
            t1 = time.perf_counter()
            t_inner_loop += (t1 - t0)

            peak_cutoff = np.max(accumulator) * 0.9
            if peak_cutoff > 0:
                #idxs_peaks = np.argwhere(accumulator > peak_cutoff)
                idxs_1d = np.argwhere(accumulator > peak_cutoff).flatten()
        
                # 1D-Indices wieder in (best_mr, best_alpha) umrechnen:
                idxs_peaks = [(idx // num_angles, idx % num_angles) for idx in idxs_1d]
            else:
                print("SKIPPING REF VERT", sA, "NO PEAKS")
                idxs_peaks = []

            if len(idxs_peaks) == 0: #EMMA
                print("SKIPPING REF VERT", sA, "NO PEAKS")
                continue

            # ---- Emma
            t2 = time.perf_counter()
            t_peaks += (t2 - t1)

            s_r = scene.vertices[sA]
            s_normal = scene.vertex_normals[sA]

            R_scene2glob = np.eye(4)
            R_scene2glob[:3, :3] = align_vectors(s_normal, [1, 0, 0])
            T_scene2glob = R_scene2glob @ tf.translation_matrix(-s_r)

            # ---- Emma
            t3 = time.perf_counter()
            t_scene_trans += (t3 - t2)

            for best_mr, best_alpha in idxs_peaks:
                R_model2glob = np.eye(4)
                R_model2glob[:3, :3] = align_vectors(
                    model.vertex_normals[best_mr], [1, 0, 0]
                )
                T_model2glob = R_model2glob @ tf.translation_matrix(
                    -model.vertices[best_mr]
                )

                R_alpha = tf.rotation_matrix(alpha_step * best_alpha, [1, 0, 0], [0, 0, 0])
                # TODO: invert homog
                T_model2scene = np.linalg.inv(T_scene2glob) @ R_alpha @ T_model2glob
                #poses.append((T_model2scene, best_mr, accumulator[best_mr, best_alpha]))
                poses.append((T_model2scene, best_mr, accumulator[best_mr * args.alpha_num_angles + best_alpha]))

            # --- Emma
            t4 = time.perf_counter()
            t_pose_calc += (t4 - t3)

        t_loop_total = time.perf_counter() - t_loop_total_start
        print()

        # ----------------------------------------------------
        # PROFILING AUSGABE / STATISTIK
        # ----------------------------------------------------
        print("\n" + "="*50)
        print("PROFILING STATISTIK FOR MATCHING LOOP")
        print("="*50)
        print(f"Gesamtdauer der Schleife: {t_loop_total:.2f} Sekunden\n")

        print(f"1. Inner Loop (Voting/Matching) : {t_inner_loop:6.2f}s | {t_inner_loop/t_loop_total*100:5.1f}%")
        print("   ├── 1a. Feature-Lookups (s_feature) : {:6.2f}s | {:5.1f}%".format(t_sub_lookup_feat, t_sub_lookup_feat/t_loop_total*100))
        print("   ├── 1b. Scene-Alpha Array Lookup   : {:6.2f}s | {:5.1f}%".format(t_sub_lookup_scene, t_sub_lookup_scene/t_loop_total*100))
        print("   ├── 1c. Model-Alpha Lookups        : {:6.2f}s | {:5.1f}%".format(t_sub_lookup_model, t_sub_lookup_model/t_loop_total*100))
        #print("   └── 1d. Alpha-Mathe & Accumulator  : {:6.2f}s | {:5.1f}%".format(t_sub_accumulator, t_sub_accumulator/t_loop_total*100))
        print(f"2. Peak Detection (Numpy Max/Where): {t_peaks:6.2f}s | {t_peaks/t_loop_total*100:5.1f}%")
        print(f"3. Scene Transformation           : {t_scene_trans:6.2f}s | {t_scene_trans/t_loop_total*100:5.1f}%")
        print(f"4. Pose Reconstruction (Peaks)    : {t_pose_calc:6.2f}s | {t_pose_calc/t_loop_total*100:5.1f}%")
        print("="*50 + "\n")
        # ----------------------------------------------------

    print(f"Got {len(poses)} poses after matching", "+" * 20)
    print("Skipped", skipped_features, "scene pairs, not found in model")
    t_end = time.perf_counter()
    print(f"Searching ppf pairs took {t_end - t_start:.1f}s")

    if not poses or len(poses) == 0:
        print("CRITICAL: Keine Posen gefunden. Matching gescheitert.")
        return []

    t_cluster_start = time.perf_counter()
    pose_clusters = cluster_poses(
        poses,
        dist_max=modelscale * 0.5,
        rot_max_deg=args.cluster_max_angle,
        pdist_rot=_pdist_rot,
    )
    poses = pose_clusters
    t_cluster_end = time.perf_counter()
    print(f"Clustering took {t_cluster_end - t_cluster_start:.1f}s")

    # ## Visualize result
    # scene_refs = trimesh.PointCloud(
    #     [scene.vertices[idx] for idx in list(pairs_scene.keys())]
    # )
    # vis = trimesh.Scene([_scene_vis, scene_refs])
    # for T_model2scene, m_r, score in poses:
    #     model_vis = _model_vis.copy()
    #     color = (*np.random.randint(0, 255, size=3), 255)
    #     model_vis.visual.vertex_colors = [color for _ in model_vis.vertices]
    #     model_vis.apply_transform(T_model2scene)
    #     vis.add_geometry(model_vis)

    #     print("Score", score)
    #     print(np.around(T_model2scene, decimals=2))
    #     print()
    # vis.show()
    return poses


def to_nanobind(arr):
    """
    Workaround for current bug in nanobind: arrays need to be writable to be recognized
    https://github.com/wjakob/nanobind/issues/42
    """
    arr_copy = np.copy(arr)
    F_arr = np.asfortranarray(arr_copy)
    F_arr.setflags(write=True)
    return F_arr


def vector_angle_signed_x(vecA, vecB):
    assert np.isclose(np.linalg.norm(vecA), 1)
    assert np.isclose(np.linalg.norm(vecB), 1)
    return np.arctan2(np.dot(np.cross(vecA, vecB), [1, 0, 0]), np.dot(vecA, vecB))


assert np.isclose(vector_angle_signed_x([0, 1, 0], [0, 0, 1]), np.pi / 2)
assert np.isclose(vector_angle_signed_x([0, 0, 1], [0, 1, 0]), -np.pi / 2)


def align_vectors(a, b):
    """
    Computes rotation matrix that rotates a into b
    """

    v = np.cross(a, b)
    c = np.dot(a, b)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]])

    R = np.eye(3) + Vmat + (Vmat.dot(Vmat) * h)
    return R


def compute_feature(vertA, vertB, normA, normB, angle_step=None, dist_step=None):
    """
    angle_step: Angle step in radians
    """

    diffvec = vertA - vertB

    F1 = np.linalg.norm(diffvec)
    F2, F3, F4 = trimesh.geometry.vector_angle(
        [(-diffvec / F1, normA), (diffvec / F1, normB), (normA, normB)]
    )

    if dist_step and angle_step:
        prev = (F1, F2, F3, F4)
        F1 //= dist_step
        F2 //= angle_step
        F3 //= angle_step
        F4 //= angle_step
        try:
            res = tuple(int(x) for x in [F1, F2, F3, F4])
        except ValueError as e:
            print(e, "F1", F1, "F2", F2, "F3", F3, "F4", F4)
            print("prev", prev)
            return None
        return res

    return (F1, F2, F3, F4)


def homog(vec3):
    return [*vec3, 1]


def compute_ppf(
    vertices,
    normals,
    angle_step: float,
    dist_step: float,
    ref_fraction=1.0,
    ref_abs=None,
    max_dist=np.inf,
    alphas=False,
):
    table = defaultdict(list)
    ref2feature = defaultdict(dict)
    model_alphas = {}

    idxs = range(len(vertices))

    num_pts = int(ref_fraction * len(vertices))
    num_pts = min(num_pts, ref_abs or len(vertices))
    random.seed(42) #EMMA, same seed for study purpose
    idxsA = random.sample(idxs, k=num_pts)
    print(f"Going for {num_pts} reference pts ({num_pts/len(vertices) * 100:.0f}%)")

    # without KDTREE: Computing ppfs for the scene took 2134.7s
    # with KDTree:                                       814.9s
    vert_tree = KDTree(vertices)

    num = 0
    for ivertA in idxsA:
        vertA = vertices[ivertA]

        for ivertB in vert_tree.query_ball_point(vertA, max_dist):
            if ivertA == ivertB:
                continue

            normA = normals[ivertA]
            normB = normals[ivertB]
            vertB = vertices[ivertB]

            F = compute_feature(
                vertA, vertB, normA, normB, angle_step=angle_step, dist_step=dist_step
            )

            if F is None:
                continue

            num += 1
            if num < 500 or num % 10000 == 0:
                print("pair", num, f"{num/(len(vertices)**2)*100:.0f}%", end="\r")

            table[F].append((ivertA, ivertB))
            ref2feature[ivertA][ivertB] = F

            # precompute the model angles
            if alphas:
                m_r = vertA
                m_i = vertB
                m_normal = normA

                R_model2glob = np.eye(4)
                R_model2glob[:3, :3] = align_vectors(m_normal, [1, 0, 0])
                T_model2glob = R_model2glob @ tf.translation_matrix(-m_r)

                m_ig = (T_model2glob @ homog(m_i))[:3]
                m_ig /= np.linalg.norm(m_ig)
                alpha_m = vector_angle_signed_x(m_ig, [0, 0, -1])
                model_alphas[(ivertA, ivertB)] = alpha_m

    return table, ref2feature, model_alphas


def bernstein(vala, valb):
    """thanks to special sauce https://stackoverflow.com/a/34006336/10059727"""
    h = 1009
    h = h * 9176 + vala
    h = h * 9176 + valb
    return h


def rotation_between(rotmatA, rotmatB):
    """thanks to JonasVautherin https://math.stackexchange.com/q/2113634"""
    assert rotmatA.shape == (3, 3)
    assert rotmatB.shape == (3, 3)

    r_oa_t = np.transpose(rotmatA)
    r_ab = r_oa_t @ rotmatB
    #EMMA
    cos_theta = (np.trace(r_ab) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)


matA = tf.rotation_matrix(np.pi / 4, [1, 0, 0])[:3, :3]
matB = tf.rotation_matrix(np.pi / 2, [1, 0, 0])[:3, :3]
assert np.isclose(rotation_between(matA, matB), np.pi / 4), rotation_between(matA, matB)


def pdist_rot(rot_mats):
    """Returns the condensed distance matrix like pdist, but in rotation space"""
    m = len(rot_mats)
    idx = lambda i, j: m * i + j - ((i + 2) * (i + 1)) // 2
    print("Index for last pair", idx(m, m) + 1)

    # we save distance in degrees and use uint8 for smaller memory footprint
    dists = np.zeros(idx(m, m) + 1, dtype=np.uint8)
    print("dists shape", dists.shape)

    mat_idxs = np.arange(m)
    # Note: combinations() doesn't give (i,i) pairs
    # Note: combinations() keeps original ascending index order
    for idxA, idxB in itertools.combinations(mat_idxs, 2):
        dist = np.degrees(rotation_between(rot_mats[idxA], rot_mats[idxB])).astype(
            np.uint8
        )
        dists[idx(idxA, idxB)] = dist

    return dists.astype(float)


def cluster_poses(poses, dist_max=0.5, rot_max_deg=10, pdist_rot=None):
    rots = np.array([T_m2s[:3, :3] for T_m2s, _, _ in poses])
    locs = np.array([T_m2s[:3, 3] for T_m2s, _, _ in poses])
    scores = np.array([score for _, _, score in poses])

    method = "centroid"

    # 1) cluster by location
    dist_dists = pdist(locs)
    dist_dendro = linkage(dist_dists, method)
    dist_clusters = fcluster(dist_dendro, dist_max, criterion="distance")

    # 2) cluster by rotations
    # XXX optimize, we can make more smaller cluster problems, since
    # a cluster across distant poses doesn't make sense
    rot_dists = pdist_rot(rots)
    rot_dendor = linkage(rot_dists, method)
    rot_clusters = fcluster(rot_dendor, rot_max_deg, criterion="distance")

    # Combine the two clusterings, by creating new clusters
    # if two poses are in same cluster in loc and rot, they will be in new
    # common cluster (hash of both cluster ids)
    pose_clusters = bernstein(dist_clusters, rot_clusters)

    # remap the ludicrous hash values to range 0..num
    _, pose_clusters = np.unique(pose_clusters, return_inverse=True)

    cluster_scores = np.zeros(np.max(pose_clusters) + 1)
    for pose_score, pose_cluster in zip(scores, pose_clusters):
        cluster_scores[pose_cluster] += pose_score

    best_cluster_idx = np.argmax(cluster_scores)
    print(
        "Best cluster",
        best_cluster_idx,
        cluster_scores[best_cluster_idx],
        np.count_nonzero(pose_clusters == best_cluster_idx),
    )

    # plt.hist(cluster_scores, histtype="stepfilled", bins=100)
    # plt.title("Cluster Scores Histogram")
    # plt.show()

    out_ts = defaultdict(list)
    out_Rs = defaultdict(list)
    for pose_idx, cluster_idx in enumerate(pose_clusters):
        out_ts[cluster_idx].append(locs[pose_idx])
        out_Rs[cluster_idx].append(rots[pose_idx])

    sorted_clusters = np.argsort(cluster_scores)[::-1]

    for idx_cluster, top_cluster in enumerate(sorted_clusters[:10]):
        print(f"cluster {idx_cluster} contains {len(out_ts[top_cluster])} poses")

    geo = lambda x, y: np.sqrt(x * y)

    best_cluster_idx = np.argmax(cluster_scores)
    best_score = cluster_scores[best_cluster_idx]
    best_geo_score = geo(best_score, len(out_ts[best_cluster_idx]))
    best_rel_thresh = 0.5

    out_poses = []
    for cluster_idx in sorted_clusters:
        geo_score = geo(len(out_ts[cluster_idx]), cluster_scores[cluster_idx])
        if geo_score < best_rel_thresh * best_geo_score:
            continue

        print("cluster idx", cluster_idx, cluster_scores[cluster_idx], "geoscore", geo_score)
        ts = out_ts[cluster_idx]
        Rs = out_Rs[cluster_idx]

        avg_t = np.mean(ts, axis=0)
        avg_R = average_rotations(Rs)
        out_T = np.eye(4)
        out_T[:3, :3] = avg_R[:3, :3]
        out_T[:3, 3] = avg_t
        out_poses.append((out_T, 0, geo_score))

    print("Returning", len(out_poses), "clustered and averaged poses")
    return out_poses


def average_rotations(rotations):
    """thanks to jonathan https://stackoverflow.com/a/27410865/10059727"""
    Q = np.zeros((4, len(rotations)))

    for i, rot in enumerate(rotations):
        quat = tf.quaternion_from_matrix(rot)
        Q[:, i] = quat

    _, v = np.linalg.eigh(Q @ Q.T)
    quat_avg = v[:, -1]

    return tf.quaternion_matrix(quat_avg)


# trimesh todo
# - point cloud normals
# - scipy not in dependencies

# Validation

vertA = np.array([0, 0, 0])
vertB = np.array([1, 0, 0])
normA = np.array([0, 1, 0])
normB = np.array([0, 1, 0])
F1, F2, F3, F4 = compute_feature(vertA, vertB, normA, normB)
assert np.isclose(F1, 1)
assert np.isclose(F2, F3)
assert np.isclose(F2, np.radians(90)), f"F2={np.degrees(F2)}"
assert np.isclose(F4, 0)

vertA = np.array([0, 0, 0])
vertB = np.array([1, 0, 0])
normA = np.array([1, 1, 0])
normA = normA / 2 * 1.414
normB = np.array([-1, 1, 0])
normB = normB / 2 * 1.414
F1, F2, F3, F4 = compute_feature(vertA, vertB, normA, normB)
assert np.isclose(F1, 1)
assert np.isclose(F2, F3)
assert np.isclose(F2, np.radians(45), rtol=1e-3), f"F2={np.degrees(F2)}"
assert np.isclose(F4, np.radians(90), rtol=1e-3)