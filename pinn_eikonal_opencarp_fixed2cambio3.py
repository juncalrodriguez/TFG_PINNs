#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PINN para la ecuación de Eikonal sobre una malla openCARP 3D.

Modos:

1) Isotrópico:
       |∇T(x)| = s0     en Ω

2) Anisotrópico (a partir de fibras/sheet del .lon):
       ∇T(x)^T G(x) ∇T(x) = 1

   donde G se construye como:
       G = Q diag([s_l^2, s_t^2, s_n^2]) Q^T
       Q = [f, s, n] (columnas), f=fibra, s=sheet, n = f × s.

Formato openCARP:
    basename.pts
    basename.elem
    basename.lon (opcional pero necesario para modo anisotrópico)

Ejemplos de uso al final del archivo.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn


# ===========================================================
# 1. Lectura de malla openCARP (.pts, .elem, .lon)
# ===========================================================

NODES_PER_TYPE: Dict[str, int] = {
    "Ln": 2,
    "Tr": 3,
    "Qd": 4,
    "Tt": 4,
    "Py": 5,
    "Pr": 6,
    "Hx": 8,
}


@dataclass
class OpenCarpMesh:
    points: np.ndarray                 # (N,3)
    elem_types: np.ndarray             # (M,)
    connectivity: List[np.ndarray]     # lista de arrays 1D
    regions: Optional[np.ndarray] = None  # (M,) o None
    fibers: Optional[np.ndarray] = None   # (M,3) o None
    sheets: Optional[np.ndarray] = None   # (M,3) o None

    def elements_of_type(
        self, etype: str
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Devuelve conectividades y datos asociados SOLO del tipo etype (p.ej. 'Tt').

        return:
            conn:   (K,n_nodes)
            reg:    (K,) o None
            fibers: (K,3) o None
            sheets: (K,3) o None
        """
        etype = etype.strip()
        mask = (self.elem_types == etype)
        idx = np.where(mask)[0]
        if idx.size == 0:
            return np.empty((0, 0), dtype=int), None, None, None

        n_nodes = NODES_PER_TYPE.get(etype)
        if n_nodes is None:
            raise ValueError(f"Tipo de elemento desconocido: {etype}")

        conn = np.zeros((idx.size, n_nodes), dtype=int)
        for k, i in enumerate(idx):
            e = self.connectivity[i]
            if len(e) != n_nodes:
                raise ValueError(
                    f"Elemento {i} de tipo {etype} tiene {len(e)} nodos, "
                    f"esperaba {n_nodes}"
                )
            conn[k, :] = e

        reg = self.regions[idx] if self.regions is not None else None
        fib = self.fibers[idx] if self.fibers is not None else None
        sht = self.sheets[idx] if self.sheets is not None else None
        return conn, reg, fib, sht


def read_pts(pts_path: str) -> np.ndarray:
    """
    Lee archivo .pts openCARP:
        n_nodes
        x0 y0 z0
        ...
    """
    coords: List[List[float]] = []
    with open(pts_path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            header = line
            break
        if header is None:
            raise ValueError(f"{pts_path}: fichero vacío o sin datos válidos")

        try:
            n_nodes = int(header.split()[0])
        except Exception as e:
            raise ValueError(f"{pts_path}: cabecera inválida '{header}'") from e

        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{pts_path}: línea inválida: '{line}'")
            x, y, z = map(float, parts[:3])
            coords.append([x, y, z])

    points = np.asarray(coords, dtype=float)
    if points.shape[0] != n_nodes:
        raise ValueError(
            f"{pts_path}: cabecera dice {n_nodes} nodos pero se leyeron {points.shape[0]}"
        )
    return points


def read_elem(
    elem_path: str, assume_zero_based: bool = True
) -> Tuple[np.ndarray, List[np.ndarray], Optional[np.ndarray]]:
    """
    Lee archivo .elem:
        n_elems
        T n0 ... n_{m-1} [region]
    """
    elem_types: List[str] = []
    connectivity: List[np.ndarray] = []
    regions: List[int] = []

    with open(elem_path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            header = line
            break
        if header is None:
            raise ValueError(f"{elem_path}: fichero vacío o sin datos válidos")

        try:
            n_elems = int(header.split()[0])
        except Exception as e:
            raise ValueError(f"{elem_path}: cabecera inválida '{header}'") from e

        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            etype = parts[0]
            if etype not in NODES_PER_TYPE:
                raise ValueError(
                    f"{elem_path}: tipo de elemento '{etype}' no soportado "
                    f"(línea: '{line}')"
                )
            n_nodes = NODES_PER_TYPE[etype]
            if len(parts) < 1 + n_nodes:
                raise ValueError(
                    f"{elem_path}: nodos insuficientes para {etype}: '{line}'"
                )

            node_tokens = parts[1:1 + n_nodes]
            node_ids = np.array([int(t) for t in node_tokens], dtype=int)
            if not assume_zero_based:
                node_ids -= 1

            if len(parts) == 1 + n_nodes:
                reg = None
            elif len(parts) == 2 + n_nodes:
                reg = int(parts[-1])
            else:
                raise ValueError(
                    f"{elem_path}: demasiados campos en línea: '{line}'"
                )

            elem_types.append(etype)
            connectivity.append(node_ids)
            if reg is not None:
                regions.append(reg)

    elem_types_arr = np.asarray(elem_types, dtype=object)
    if len(elem_types_arr) != n_elems:
        raise ValueError(
            f"{elem_path}: cabecera dice {n_elems} elementos, "
            f"pero se leyeron {len(elem_types_arr)}"
        )

    if regions and len(regions) == len(connectivity):
        regions_arr: Optional[np.ndarray] = np.asarray(regions, dtype=int)
    else:
        regions_arr = None

    return elem_types_arr, connectivity, regions_arr


def _parse_nf(header: str) -> int:
    """
    Extrae nf (1 ó 2) de la cabecera del .lon.
    """
    parts = header.split()
    nf: Optional[int] = None
    for tok in parts:
        try:
            nf = int(tok)
            break
        except ValueError:
            continue
    if nf is None or nf not in (1, 2):
        raise ValueError(f"Cabecera .lon inválida: '{header}' (nf debe ser 1 o 2)")
    return nf


def read_lon(
    lon_path: str, n_elems_expected: Optional[int] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Lee archivo .lon:
        nf
        f0x f0y f0z [s0x s0y s0z]
        ...
    """
    fibers: List[List[float]] = []
    sheets: List[List[float]] = []

    with open(lon_path, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            header = line
            break
        if header is None:
            raise ValueError(f"{lon_path}: fichero vacío o sin datos válidos")

        nf = _parse_nf(header)

        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) != 3 * nf:
                raise ValueError(
                    f"{lon_path}: se esperaban {3*nf} valores por línea, "
                    f"pero se encontraron {len(vals)} en '{line}'"
                )
            fx, fy, fz = vals[0:3]
            fibers.append([fx, fy, fz])
            if nf == 2:
                sx, sy, sz = vals[3:6]
                sheets.append([sx, sy, sz])

    fibers_arr = np.asarray(fibers, dtype=float)
    sheets_arr: Optional[np.ndarray]
    if sheets:
        sheets_arr = np.asarray(sheets, dtype=float)
    else:
        sheets_arr = None

    if n_elems_expected is not None and fibers_arr.shape[0] != n_elems_expected:
        raise ValueError(
            f"{lon_path}: se leyeron {fibers_arr.shape[0]} fibras pero el .elem "
            f"tiene {n_elems_expected} elementos"
        )

    return fibers_arr, sheets_arr


def load_opencarp_mesh(
    basename: str, assume_zero_based: bool = True, load_fibers: bool = True
) -> OpenCarpMesh:
    """
    Carga basename.pts, basename.elem y (opcional) basename.lon.
    """
    pts_path = basename + ".pts"
    elem_path = basename + ".elem"
    lon_path = basename + ".lon"

    points = read_pts(pts_path)
    elem_types, connectivity, regions = read_elem(elem_path, assume_zero_based)

    fibers = None
    sheets = None
    if load_fibers:
        try:
            fibers, sheets = read_lon(lon_path, n_elems_expected=len(elem_types))
        except FileNotFoundError:
            print(f"[WARN] No se encontró {lon_path}, continúo sin fibras.")
        except Exception as e:
            print(f"[WARN] Error leyendo {lon_path}: {e}")
            print("       Continúo sin fibras / sheets.")

    return OpenCarpMesh(
        points=points,
        elem_types=elem_types,
        connectivity=connectivity,
        regions=regions,
        fibers=fibers,
        sheets=sheets,
    )


# ===========================================================
# 2. Estructura específica para PINN sobre tetraedros
# ===========================================================

@dataclass
class TetMeshPINN:
    points: np.ndarray          # (N,3)
    tets: np.ndarray            # (M,4)
    volumes: np.ndarray         # (M,)
    regions: Optional[np.ndarray] = None  # (M,)
    fibers: Optional[np.ndarray] = None   # (M,3)
    sheets: Optional[np.ndarray] = None   # (M,3)

    @property
    def num_nodes(self) -> int:
        return self.points.shape[0]

    @property
    def num_tets(self) -> int:
        return self.tets.shape[0]


def tet_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """
    Volumen de cada tetraedro.
    """
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    v = np.abs(np.einsum("ij,ij->i", (p1 - p0), np.cross(p2 - p0, p3 - p0))) / 6.0
    return v


def build_tet_mesh_for_pinn(oc_mesh: OpenCarpMesh) -> TetMeshPINN:
    """
    Extrae solo tetraedros ('Tt') de la malla openCARP.
    """
    tets, tet_regions, tet_fibers, tet_sheets = oc_mesh.elements_of_type("Tt")
    if tets.size == 0:
        raise RuntimeError("La malla no contiene tetraedros ('Tt').")

    vols = tet_volumes(oc_mesh.points, tets)
    if np.any(vols <= 0):
        print("[WARN] Algunos tetraedros tienen volumen <= 0 (degenerados).")

    return TetMeshPINN(
        points=oc_mesh.points,
        tets=tets,
        volumes=vols,
        regions=tet_regions,
        fibers=tet_fibers,
        sheets=tet_sheets,
    )


# ===========================================================
# 3. Tensor anisotrópico G a partir de fibras/sheets
# ===========================================================

def build_anisotropic_tensors_from_fibers(
    fibers: np.ndarray,
    sheets: Optional[np.ndarray] = None,
    s_l: float = 1.0,
    s_t: float = 1.0,
    s_n: Optional[float] = None,
) -> np.ndarray:
    """
    Construye un tensor G (M,3,3) por elemento:

        G = Q diag([s_l^2, s_t^2, s_n^2]) Q^T

    con Q = [f, s, n], f=fibra, s=sheet ortogonal a f, n=f×s.
    Si sheets es None, se genera una base ortogonal arbitraria.
    Si s_n es None, se toma s_n = s_t.

    Ecuación anisotrópica:
        grad T^T G grad T = 1
    """
    M = fibers.shape[0]
    f = np.array(fibers, dtype=float)
    # normaliza fibra
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-15

    if sheets is not None:
        s = np.array(sheets, dtype=float)
        # ortogonaliza y normaliza sheet respecto a f
        s = s - np.sum(s * f, axis=1, keepdims=True) * f
        s /= np.linalg.norm(s, axis=1, keepdims=True) + 1e-15
    else:
        # genera una sheet arbitraria ortogonal
        tmp = np.tile(np.array([1.0, 0.0, 0.0]), (M, 1))
        # evita casi paralelos
        mask = np.abs(np.sum(f * tmp, axis=1)) > 0.9
        tmp[mask] = np.array([0.0, 1.0, 0.0])
        s = tmp - np.sum(tmp * f, axis=1, keepdims=True) * f
        s /= np.linalg.norm(s, axis=1, keepdims=True) + 1e-15

    # normal = f × s
    n = np.cross(f, s)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-15

    if s_n is None:
        s_n = s_t

    diag_vals = np.array([s_l**2, s_t**2, s_n**2], dtype=float)
    G_all = np.zeros((M, 3, 3), dtype=float)
    for i in range(M):
        Q = np.column_stack([f[i], s[i], n[i]])   # (3,3)
        G_all[i] = Q @ np.diag(diag_vals) @ Q.T

    return G_all


# ===========================================================
# 4. PINN para eikonal (iso/anisotrópico)
# ===========================================================

def device_auto():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=1, width=128, depth=5, act="tanh", nonneg=True):
        super().__init__()
        acts = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        activation = acts.get(act, nn.Tanh)

        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(activation())
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(activation())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

        self.nonneg = nonneg
        if nonneg:
            self.softplus = nn.Softplus()

    def forward(self, x):
        y = self.net(x)
        if self.nonneg:
            y = self.softplus(y)
        return y


def grad_t(T, x):
    """
    ∇T con autograd.
    """
    grad_outputs = torch.ones_like(T)
    g = torch.autograd.grad(
        outputs=T,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return g


def sample_interior_tets(
    mesh: TetMeshPINN, batch_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Muestreo interior:
      - elige tetraedros con probabilidad proporcional a su volumen
      - muestrea barycentros ~ Dirichlet(1,1,1,1)
      - devuelve coords físicas x
    """
    vols = mesh.volumes
    p = vols / vols.sum()
    p = np.maximum(p, 1e-16)
    p /= p.sum()

    elem_ids = np.random.choice(mesh.num_tets, size=batch_size, p=p)

    # Dirichlet(1,1,1,1) por exponenciales
    alphas = np.random.exponential(scale=1.0, size=(batch_size, 4))
    alphas /= alphas.sum(axis=1, keepdims=True)

    tet_nodes = mesh.tets[elem_ids]                 # (B,4)
    tet_points = mesh.points[tet_nodes]             # (B,4,3)
    x = np.einsum("bij,bi->bj", tet_points, alphas) # (B,3)

    return elem_ids, x, alphas


def sample_sources(mesh: TetMeshPINN, source_nodes: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Muestreo de nodos fuente (T=0).
    """
    if source_nodes.size == 0:
        raise ValueError("No se han proporcionado nodos fuente.")
    ids = np.random.choice(source_nodes, size=batch_size, replace=True)
    x = mesh.points[ids]
    return x


def train_pinn_eikonal(
    mesh: TetMeshPINN,
    source_nodes: np.ndarray,
    s0: float = 1.0,
    epochs: int = 20000,
    batch_interior: int = 4096,
    batch_source: int = 512,
    w_pde: float = 1.0,
    w_bc: float = 100.0,
    w_mono: float = 0.1,
    lr: float = 1e-3,
    act: str = "tanh",
    anisotropic: bool = False,
    G_elems: Optional[np.ndarray] = None,
) -> nn.Module:

    import time  # para medir tiempo por epoch

    device = device_auto()
    print("Entrenando en dispositivo:", device)

    model = MLP(in_dim=3, out_dim=1, width=128, depth=5, act=act, nonneg=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    source_nodes = np.asarray(source_nodes, dtype=int)

    G_elems_torch = None
    if anisotropic:
        if G_elems is None:
            raise ValueError("Modo anisotrópico requiere G_elems (tensor por elemento).")
        G_elems_torch = torch.as_tensor(G_elems, dtype=torch.float32, device=device)
      
    #inicio del cronómetro total
    t_total_ini = time.time()
    loss_hist = []
    

    for ep in range(1, epochs + 1):

        t0 = time.time()  # inicio de epoch

        opt.zero_grad()

        # -------- Interior --------
        elem_ids_np, x_int_np, _ = sample_interior_tets(mesh, batch_interior)
        x_int = torch.tensor(x_int_np, dtype=torch.float32, device=device, requires_grad=True)
        T_int = model(x_int)
        gT = grad_t(T_int, x_int)

        if anisotropic:
            elem_ids_torch = torch.as_tensor(elem_ids_np, dtype=torch.long, device=device)
            G_batch = G_elems_torch[elem_ids_torch]
            quad = torch.einsum("bi,bij,bj->b", gT, G_batch, gT).unsqueeze(1)
            res_pde = quad - 1.0
            loss_pde = torch.mean(res_pde**2)
        else:
            grad_norm = torch.linalg.norm(gT, dim=1, keepdim=True)
            s_val = s0 * torch.ones_like(T_int)
            res_pde = grad_norm - s_val
            loss_pde = torch.mean(res_pde**2)

        # -------- Fuente T = 0 --------
        if source_nodes.size > 0 and batch_source > 0:
            x_src_np = sample_sources(mesh, source_nodes, batch_source)
            x_src = torch.as_tensor(x_src_np, dtype=torch.float32, device=device)
            T_src = model(x_src)
            loss_bc = torch.mean(T_src**2)
        else:
            loss_bc = torch.zeros((), dtype=torch.float32, device=device)

        loss_mono = torch.mean(torch.relu(-T_int))

        loss = w_pde * loss_pde + w_bc * loss_bc + w_mono * loss_mono
        loss_hist.append(loss.item())
        

        loss.backward()
        opt.step()

        # tiempo por iteración
        dur = time.time() - t0

        # -------- IMPRIMIR SOLO CADA 100 EPOCHS --------
        if ep % 100 == 0 or ep == 1 or ep == epochs:
            print(
                f"Epoch {ep}/{epochs} - tiempo {dur:.4f}s - "
                f"loss={loss.item():.4e}  PDE={loss_pde.item():.4e}  BC={loss_bc.item():.4e}"
            )
    t_total = time.time() - t_total_ini
    print(f"\n[INFO] Entrenamiento completado en {t_total:.1f} s (~{t_total/60:.2f} min)")

    np.savetxt(
        "C:/tfgn/meshalyzer_win64/loss_history.dat",
        np.array(loss_hist),
        fmt="%.8e"
    )
    print("[INFO] loss_history.dat guardado")

    return model, loss_hist


# ===========================================================
# 5. Cálculo de T nodal + export a VTU / DAT
# ===========================================================

def compute_T_at_nodes(mesh: TetMeshPINN, model: nn.Module) -> np.ndarray:
    """
    Evalúa la PINN en todos los nodos de la malla y devuelve un vector (N,)
    con los tiempos T de cada nodo en el orden del .pts.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(mesh.points, dtype=torch.float32, device=device)
        T = model(x).cpu().numpy().reshape(-1)
    return T


def export_T_to_vtu(mesh: TetMeshPINN, T: np.ndarray, out_path: str):
    """
    Exporta el campo T en nodos a un VTU para ver en ParaView.
    Requiere meshio: pip install meshio
    """
    try:
        import meshio
    except ImportError:
        print("[WARN] meshio no está instalado, no se exporta VTU.")
        return

    T = np.asarray(T).reshape(-1)
    if T.shape[0] != mesh.num_nodes:
        raise ValueError(
            f"export_T_to_vtu: T tiene longitud {T.shape[0]} pero la malla tiene "
            f"{mesh.num_nodes} nodos."
        )

    cells = [("tetra", mesh.tets)]
    mesh_out = meshio.Mesh(points=mesh.points, cells=cells, point_data={"T_pinn": T})
    meshio.write(out_path, mesh_out)
    print(f"Guardado VTU con T_pinn en: {out_path}")


def export_T_to_dat(mesh: TetMeshPINN, T: np.ndarray, out_path: str):
    """
    Exporta el campo T en nodos a un fichero de texto .dat con una sola columna:
        T_0
        T_1
        ...
        T_{N-1}

    El orden es exactamente el de los nodos en el .pts (índices 0...(N-1)).
    """
    T = np.asarray(T).reshape(-1)
    if T.shape[0] != mesh.num_nodes:
        raise ValueError(
            f"export_T_to_dat: T tiene longitud {T.shape[0]} pero la malla tiene "
            f"{mesh.num_nodes} nodos."
        )

    np.savetxt(out_path, T, fmt="%.8e")
    print(f"Guardado DAT con T_pinn (una columna) en: {out_path}")


# ===========================================================
# 6. CLI
# ===========================================================

def parse_args():
    p = argparse.ArgumentParser(description="PINN eikonal (iso/anisotrópico) sobre malla openCARP.")
    p.add_argument("--basename", required=True,
                   help="Basename de la malla (basename.pts, basename.elem, [basename.lon])")
    p.add_argument("--sources", type=int, nargs="*", default=[],
                   help="Índices de nodos fuente (T=0), separados por espacios.")
    p.add_argument("--s0", type=float, default=1.0,
                   help="Slowness isotrópica s0 (|∇T| = s0). Ignorado en modo anisotrópico.")
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--batch-interior", type=int, default=4096)
    p.add_argument("--batch-source", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--act", type=str, default="tanh", choices=["tanh", "relu", "silu", "gelu"])
    p.add_argument("--out-vtu", type=str, default="T_pinn.vtu",
                   help="Fichero de salida VTU (opcional, vacío para desactivar).")

    # Modo anisotrópico
    p.add_argument("--anisotropic", action="store_true",
                   help="Usar formulación anisotrópica ∇T^T G ∇T = 1 con fibras del .lon.")
    p.add_argument("--sl", type=float, default=1.0,
                   help="Slowness longitudinal (dirección fibra) en modo anisotrópico.")
    p.add_argument("--st", type=float, default=1.0,
                   help="Slowness transversal (sheet) en modo anisotrópico.")
    p.add_argument("--sn", type=float, default=None,
                   help="Slowness normal en modo anisotrópico (si no se da, sn=st).")

    # Export DAT
    p.add_argument("--out-dat", type=str, default="",
                   help="Fichero de salida DAT (una columna con T por nodo, vacío para desactivar).")

    return p.parse_args()


def main():
    args = parse_args()

    print("Cargando malla openCARP desde basename:", args.basename)
    oc_mesh = load_opencarp_mesh(args.basename, assume_zero_based=True, load_fibers=True)
    tet_mesh = build_tet_mesh_for_pinn(oc_mesh)

    print(f"N nodos totales:      {tet_mesh.num_nodes}")
    print(f"N tetraedros usados:  {tet_mesh.num_tets}")
    if tet_mesh.fibers is not None:
        print(f"Fibras por tetraedro: {tet_mesh.fibers.shape}")
    else:
        print("No se han cargado fibras (lon no encontrado o error).")

    # Construir G para modo anisotrópico si se pide
    G_elems = None
    if args.anisotropic:
        if tet_mesh.fibers is None:
            raise RuntimeError("Modo anisotrópico solicitado pero no hay fibras (.lon).")
        G_elems = build_anisotropic_tensors_from_fibers(
            tet_mesh.fibers,
            sheets=tet_mesh.sheets,
            s_l=args.sl,
            s_t=args.st,
            s_n=args.sn,
        )
        print("Modo anisotrópico: G_elems shape =", G_elems.shape)

    # Entrenar PINN
    model, loss_hist = train_pinn_eikonal(
        mesh=tet_mesh,
        source_nodes=np.array(args.sources, dtype=int),
        s0=args.s0,
        epochs=args.epochs,
        batch_interior=args.batch_interior,
        batch_source=args.batch_source,
        lr=args.lr,
        act=args.act,
        anisotropic=args.anisotropic,
        G_elems=G_elems,
    )

    
    # Calcular T en nodos una sola vez
    T_nodes = compute_T_at_nodes(tet_mesh, model)

    # Exportar VTU si se pide
    if args.out_vtu:
        export_T_to_vtu(tet_mesh, T_nodes, args.out_vtu)

    # Exportar DAT si se pide
    if args.out_dat:
        export_T_to_dat(tet_mesh, T_nodes, args.out_dat)


if __name__ == "__main__":
    main()
