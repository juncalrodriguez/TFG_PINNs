#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, glob, os
import numpy as np

# ----------------------------- helpers --------------------------------
def normalise(v):
    v = np.asarray(v, float)
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v if n == 0 else v / n
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n == 0.0] = 1.0
    return v / n

def resolve_mesh_pts(pathlike: str) -> str:
    p = os.path.abspath(pathlike)
    if p.endswith(".pts") and os.path.isfile(p): return p
    if os.path.isfile(p + ".pts"): return p + ".pts"
    if os.path.isdir(p):
        cands = sorted(glob.glob(os.path.join(p, "*.pts")))
        if len(cands) == 1: return cands[0]
        raise FileNotFoundError(f"{p}: expected exactly one .pts, found {len(cands)}")
    raise FileNotFoundError(f"Cannot resolve .pts from: {pathlike}")

def read_pts_raw(pts_path: str) -> np.ndarray:
    with open(pts_path, "r") as f:
        n = int(f.readline().strip())
        arr = np.loadtxt(f)
    if arr.ndim == 1: arr = arr.reshape(1, -1)
    if arr.shape[0] != n or arr.shape[1] < 3:
        raise ValueError(f"{pts_path}: expected {n} rows & >=3 cols, got {arr.shape}")
    return arr[:, :3].astype(float)

def read_elem(elem_path: str):
    """Read CARP .elem; returns only tets (N,4) and their regions (N,)."""
    with open(elem_path, "r") as f:
        m = int(f.readline().strip())
        tets, regs = [], []
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            p = s.split()
            tag = p[0]
            if tag == "Tt":  # tetra
                n0, n1, n2, n3 = map(int, p[1:5])
                reg = int(p[5]) if len(p) >= 6 else 0
                tets.append((n0, n1, n2, n3)); regs.append(reg)
            # ignore other element types here
    tets = np.asarray(tets, dtype=np.int64)
    regs = np.asarray(regs, dtype=np.int32)
    if tets.shape[0] != m:
        print(f"[warn] {elem_path}: header={m}, read {tets.shape[0]} tetra entries")
    return tets, regs

def write_ascii_scalar(filename: str, data: np.ndarray) -> None:
    arr = np.asarray(data, float).ravel()
    with open(filename, "w") as f:
        for v in arr: f.write(f"{v:.10g}\n")
    print(f"[ASCII] wrote {filename}  (lines={arr.size})")

# --------------------------- units & sources ---------------------------
def autodetect_mm(pts: np.ndarray, tets4: np.ndarray) -> float:
    bb = pts.max(0) - pts.min(0)
    med_span = float(np.median(bb))
    if med_span > 1e6:   # looks like µm
        return 1/1000.0
    if med_span < 1.0:   # looks like mm
        return 1.0
    # quick check via first tet edge
    e = tets4[0]
    edges = [np.linalg.norm(pts[e[i]] - pts[e[j]]) for i in range(4) for j in range(i+1,4)]
    med_edge = float(np.median(edges))
    return 1/1000.0 if med_edge > 50 else 1.0

# ------------------------------ .lon I/O --------------------------------
def _first_data_line(path: str) -> str:
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"): return s
    raise ValueError(f"{path}: no data lines")

def read_lon_opencarp(lon_path: str, num_elems: int):
    """
    openCARP semantics:
      first non-comment data line is an integer nvec = 1 or 2
        1  -> only fibre per element          [fx fy fz]
        2  -> fibre and SHEET per element     [fx fy fz sx sy sz]
      (sheet is NOT a sheet-normal here)
    Returns:
      f_e: (M,3), s_e: (M,3 or None)
    """
    head = _first_data_line(lon_path)
    nvec = int(head.split()[0])
    if nvec not in (1, 2):
        raise ValueError(f"{lon_path}: first data int must be 1 or 2, got {nvec}")

    rows = []
    with open(lon_path, "r") as f:
        seen_first = False
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            if not seen_first: seen_first = True; continue  # skip the nvec line
            rows.append([float(x) for x in s.split()])
    arr = np.asarray(rows, float)
    expected_cols = 3*nvec
    if arr.shape[0] != num_elems or arr.shape[1] != expected_cols:
        raise ValueError(f"{lon_path}: expected ({num_elems},{expected_cols}), got {arr.shape}")

    f_e = normalise(arr[:, 0:3])
    s_e = None
    if nvec == 2:
        s_e = normalise(arr[:, 3:6])
        # Re-orthogonalize sheet to fibre (s ⟂ f)
        s_e = normalise(s_e - np.sum(s_e*f_e, axis=1, keepdims=True)*f_e)
    return f_e, s_e

# ---------- node frames like openCARP (no averaging; with 1D fallback) ----------
def node_to_elem_adjacency(n_nodes: int, tets4: np.ndarray):
    adj = [[] for _ in range(n_nodes)]
    for e,(a,b,c,d) in enumerate(tets4):
        adj[a].append(e); adj[b].append(e); adj[c].append(e); adj[d].append(e)
    return adj

def frames_per_node_like_opencarp(pts: np.ndarray,
                                  tets4: np.ndarray,
                                  f_e: np.ndarray,
                                  s_e: np.ndarray | None):
    """
    For each node:
      - pick the FIRST incident non-line element -> copy fibre/sheet
      - if none (degenerate case), fabricate fibre from incident edges
      - re-orthonormalize; if sheet missing, invent an orthogonal sheet
    Returns:
      f_n (N,3), s_n (N,3)
    """
    N = pts.shape[0]
    f_n = np.zeros((N,3), float)
    s_n = np.zeros((N,3), float)

    adj = node_to_elem_adjacency(N, tets4)

    for n in range(N):
        chosen = None
        for e in adj[n]:
            # tets4 are always 4-nodes; treat as non-line
            chosen = e
            break

        if chosen is not None:
            f = f_e[chosen]
            s = None if s_e is None else s_e[chosen]
        else:
            # 1D fallback: sum edge directions
            v = np.zeros(3, float)
            for e in adj[n]:
                a,b,c,d = tets4[e]
                for q in (a,b,c,d):
                    if q == n: continue
                    dvec = pts[q] - pts[n]
                    L = np.linalg.norm(dvec)
                    if L > 0: v += dvec / L
            if np.linalg.norm(v) == 0:
                f = np.array([1.0,0.0,0.0])
            else:
                f = normalise(v)
            s = None

        f = normalise(f)

        if s is None:
            # make some sheet ⟂ f
            helper = np.array([0.0,0.0,1.0]) if abs(f[2]) < 0.9 else np.array([0.0,1.0,0.0])
            s = normalise(np.cross(f, helper))
        else:
            s = normalise(s - np.dot(s, f)*f)

        # n = f × s (not stored but used later)
        f_n[n] = f
        s_n[n] = s

    return f_n, s_n

# ---------------------------- slowness metric S -------------------------
def build_S_from_fs(f: np.ndarray, s: np.ndarray, rt: float, rn: float):
    """
    Build S = f f^T + rt^2 s s^T + rn^2 n n^T  (n = f × s)
    f,s are (3,), returns (3,3)
    """
    f = normalise(f); s = normalise(s - np.dot(s,f)*f); n = normalise(np.cross(f,s))
    S = np.zeros((3,3), float)
    def outer(u): return np.outer(u,u)
    S += outer(f)
    S += (rt*rt) * outer(s)
    S += (rn*rn) * outer(n)
    return S

def build_S_nodes(f_n: np.ndarray, s_n: np.ndarray, vl: float, vt: float, vn: float):
    rt = vl / (vt if vt != 0 else vl)
    rn = vl / (vn if vn is not None and vn != 0 else vt if vt != 0 else vl)
    N = f_n.shape[0]
    S = np.empty((N,3,3), float)
    for i in range(N):
        S[i] = build_S_from_fs(f_n[i], s_n[i], rt, rn)
    return S, rt, rn

def build_S_elems(f_e: np.ndarray, s_e: np.ndarray | None, vl: float, vt: float, vn: float):
    rt = vl / (vt if vt != 0 else vl)
    rn = vl / (vn if vn is not None and vn != 0 else vt if vt != 0 else vl)
    M = f_e.shape[0]
    S = np.empty((M,3,3), float)
    for i in range(M):
        f = f_e[i]
        if s_e is None:
            helper = np.array([0.0,0.0,1.0]) if abs(f[2]) < 0.9 else np.array([0.0,1.0,0.0])
            s = normalise(np.cross(f, helper))
        else:
            s = normalise(s_e[i] - np.dot(s_e[i], f)*f)
        S[i] = build_S_from_fs(f, s, rt, rn)
    return S

# --------------------------- “edge-like” times --------------------------
def times_like_opencarp_nodes(pts_mm, srcs_mm, S_nodes, v_l):
    """
    For each node i: T_i = min_s sqrt( (x_i - s)^T S_i (x_i - s) ) / v_l
    v_l can be scalar or (N,) array.
    """
    N = pts_mm.shape[0]
    S = S_nodes
    srcs = np.asarray(srcs_mm, float)
    out = np.empty(N, float)
    v_l_arr = np.broadcast_to(np.asarray(v_l, float), (N,))

    for i in range(N):
        x = pts_mm[i]; Si = S[i]
        best = np.inf
        for s in srcs:
            d = x - s
            val = d @ Si @ d
            if val < best:
                best = val
        out[i] = np.sqrt(max(best, 0.0)) / v_l_arr[i]
    return out

def times_like_opencarp_elems(centroids_mm, srcs_mm, S_elems, v_l):
    M = centroids_mm.shape[0]
    srcs = np.asarray(srcs_mm, float)
    out = np.empty(M, float)
    for i in range(M):
        x = centroids_mm[i]; Si = S_elems[i]
        best = np.inf
        for s in srcs:
            d = x - s
            best = min(best, d @ Si @ d)
        out[i] = np.sqrt(max(best, 0.0)) / v_l
    return out

# ------------------------------- verify --------------------------------
def verify_speeds_S(S_nodes, f_n, s_n, vl, vt, vn, sample=25):
    N = S_nodes.shape[0]
    idx = np.linspace(0, N-1, min(sample, N), dtype=int)
    err_f = []; err_s = []; err_n = []
    for i in idx:
        f = normalise(f_n[i])
        s = normalise(s_n[i] - np.dot(s_n[i], f)*f)
        n = normalise(np.cross(f, s))
        S = S_nodes[i]
        v_f = vl / np.sqrt(f @ S @ f)
        v_s = vl / np.sqrt(s @ S @ s)
        v_n = vl / np.sqrt(n @ S @ n)
        err_f.append((v_f - vl)/vl)
        err_s.append((v_s - vt)/vt if vt != 0 else 0.0)
        err_n.append((v_n - vn)/vn if vn not in (None,0) else 0.0)
    ef = np.median(np.abs(err_f)); es = np.median(np.abs(err_s)); en = np.median(np.abs(err_n))
    print(f"[verify] median rel error |v_f-vl|/vl = {ef:.3e}, |v_s-vt|/vt = {es:.3e}, |v_n-vn|/vn = {en:.3e}")

# -------------------------------- main ---------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Analytical eikonal, geometry like openCARP (writes .dat).")
    p.add_argument("--meshbase", type=str, default="/home/jorge/ventricular-dt/meshes/1.0mm/block.pts", help="Path to <base> or <base>.pts")
    p.add_argument("--pts_unit", choices=["auto","mm","um"], default="auto",
                   help="Units of .pts coordinates. 'auto' tries to guess.")
    p.add_argument("--vl", type=float, default=500.0, help="v_l [mm/s]")
    p.add_argument("--ar", type=float, default=2.0, help="anisotropy ratio a (v_t = v_l/a)")
    p.add_argument("--vn_ratio", type=float, default=None, help="v_n = v_l/vn_ratio (default: ar)")
    p.add_argument("--src", type=float, nargs=3, action="append",
                   help="Source point in mm. Repeat for multiple points.")
    p.add_argument("--outdir", type=str, default="out", help="Output directory")
    p.add_argument("--no_aniso", action="store_true", help="Skip anisotropic fields")
    return p.parse_args()

def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fields = os.path.join(a.outdir, "fields"); os.makedirs(fields, exist_ok=True)

    pts_path = resolve_mesh_pts(a.meshbase)
    base = pts_path[:-4]; elem_path = base + ".elem"; lon_path = base + ".lon"
    if not os.path.isfile(elem_path): raise FileNotFoundError(elem_path)

    pts_raw = read_pts_raw(pts_path)
    tets4, _ = read_elem(elem_path)

    # units
    if a.pts_unit == "auto":
        scale = autodetect_mm(pts_raw, tets4)
    else:
        scale = 1/1000.0 if a.pts_unit == "um" else 1.0
    pts_mm = pts_raw * scale
    if scale != 1.0: print(f"[units] scaled coordinates by {scale:g} to mm")

    # speeds
    vl = a.vl/1000.0
    vt = vl/(a.ar if a.ar != 0 else 1.0)
    vn = vl/(a.vn_ratio if a.vn_ratio is not None else a.ar if a.ar != 0 else 1.0)

    # sources
    if not a.src:
        print("[src] no --src provided; using origin (0,0,0) mm")
        srcs = np.array([[-50.0, -50.0, -5.0]], float)
    else:
        srcs = np.asarray(a.src, float)

    # isotropic reference (just Euclidean / vl)
    def times_iso(X, S):
        diff = X[:,None,:] - S[None,:,:]
        d = np.linalg.norm(diff, axis=2)
        return np.min(d, axis=1) / vl

    T_iso_nodes = times_iso(pts_mm, srcs)
    write_ascii_scalar(os.path.join(fields, "T_iso_nodes.dat"), T_iso_nodes)

    # anisotropy like openCARP
    if not a.no_aniso and os.path.isfile(lon_path):
        f_e, s_e = read_lon_opencarp(lon_path, tets4.shape[0])

        # frames per node like openCARP
        f_n, s_n = frames_per_node_like_opencarp(pts_mm, tets4, f_e, s_e)

        # slowness metrics (ratios only)
        S_nodes, rt, rn = build_S_nodes(f_n, s_n, vl, vt, vn)
        print(f"[aniso] ratios: rt=vl/vt={rt:.3f}, rn=vl/vn={rn:.3f}")

        # quick verification
        verify_speeds_S(S_nodes, f_n, s_n, vl, vt, vn)

        # node times
        T_aniso_nodes = times_like_opencarp_nodes(pts_mm, srcs, S_nodes, vl)
        write_ascii_scalar(os.path.join(fields, "T_aniso_nodes.dat"), T_aniso_nodes)
        print(f"[aniso] used {os.path.basename(lon_path)} with openCARP semantics (node frames like solver)")

        # element-center fields, using element frames directly
        cent = pts_mm[tets4].mean(axis=1)
        T_iso_elems = times_iso(cent, srcs)
        write_ascii_scalar(os.path.join(fields, "T_iso_elems.dat"), T_iso_elems)

        S_elems = build_S_elems(f_e, s_e, vl, vt, vn)
        T_aniso_elems = times_like_opencarp_elems(cent, srcs, S_elems, vl)
        write_ascii_scalar(os.path.join(fields, "T_aniso_elems.dat"), T_aniso_elems)

    else:
        print(f"[aniso][warn] {'disabled' if a.no_aniso else lon_path+' not found'}; writing isotropic fields for aniso outputs")
        write_ascii_scalar(os.path.join(fields, "T_aniso_nodes.dat"), T_iso_nodes)
        cent = pts_mm[tets4].mean(axis=1)
        T_iso_elems = times_iso(cent, srcs)
        write_ascii_scalar(os.path.join(fields, "T_iso_elems.dat"), T_iso_elems)
        write_ascii_scalar(os.path.join(fields, "T_aniso_elems.dat"), T_iso_elems)

if __name__ == "__main__":
    raise SystemExit(main())
