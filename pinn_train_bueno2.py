import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ======================================================================================
# PROBLEM OVERVIEW:
# This script solves the Eikonal Equation: |grad T| = 1/v
#   - T: Activation time (what we want to find)
#   - v: Conduction velocity (given constant, e.g., 1.0 mm/ms)
#
# IMPLEMENTATION STRATEGY: "Distance Factorized PINN"
# Standard PINNs struggle with point sources because T(source)=0 is a hard constraint
# and the gradient at the source is singular (undefined).
#
# Solution: We define the output as T(x) = distance(x) * Network(x)
#   - distance(x): Euclidean distance from source. Enforces T=0 at source automatically.
#   - Network(x): Learns the "slowness" correction factor.
# ======================================================================================

# ==========================================
# 1. MESH LOADING (openCARP format)
# ==========================================
def load_opencarp_mesh(basename, scale_to_mm=True):
    """
    Loads cardiac mesh files (.pts for nodes, .elem for elements).
    Commonly used in cardiac electrophysiology simulations (openCARP/Carpentry).
    """
    pts_file = basename + ".pts"
    elem_file = basename + ".elem"
    
    if not os.path.exists(pts_file):
        sys.exit(f"Error: Could not find mesh file {pts_file}")

    print(f"Loading {pts_file}...")
    # Load points (nodes). Handle potential header lines in different file versions.
    try: raw_pts = np.loadtxt(pts_file, skiprows=1)
    except: raw_pts = np.loadtxt(pts_file)
    
    # Unit conversion: Meshes are often in microns, but physics is easier in mm.
    if scale_to_mm:
        print("Scaling mesh coordinates by 0.001 (microns -> mm)")
        raw_pts = raw_pts * 0.001

    # Load connectivity (tetrahedra). We only need indices, typically columns 1-4.
    try: raw_elems = np.loadtxt(elem_file, dtype=int, skiprows=1, usecols=(1, 2, 3, 4))
    except: 
        # Fallback for mixed-type headers
        raw_elems = np.loadtxt(elem_file, dtype=str, skiprows=1)
        raw_elems = raw_elems[:, 1:5].astype(int)

    return raw_pts, raw_elems

def write_vtu(filename, pts, elems, data_array):
    """
    Writes the result to a VTU file for visualization in ParaView.
    """
    try:
        import meshio
        cells = [("tetra", elems)]
        mesh = meshio.Mesh(pts, cells, point_data={"T": data_array})
        mesh.write(filename)
        print(f"Saved output to {filename}")
    except ImportError:
        print("Warning: 'meshio' not found. Saving .dat instead.")
        np.savetxt(filename.replace('.vtu', '.dat'), np.hstack((pts, data_array)))


def write_dat(filename, pts, data_array):
    """
    Writes the mesh points and solution data to a space-separated .dat file.
    Format: x y z value
    Useful for plotting in Gnuplot, Matlab, or Python scripts without mesh libraries.
    """
    # Ensure data_array is the right shape (N, 1)
    if data_array.ndim == 1:
        data_array = data_array[:, None]
        
    output_data = np.hstack((pts, data_array))
    
    # fmt='%.6f' ensures precision. Header describes columns.
    np.savetxt(filename, data_array, fmt='%.6f')
    print(f"Saved solution to {filename}")


# ==========================================
# 2. SYMBOLIC DISTANCE PINN
# ==========================================
class DistancePINN(nn.Module):
    def __init__(self, layers=[3, 64, 64, 64, 64, 1], init_slowness=1.0):
        super(DistancePINN, self).__init__()
        self.net = nn.Sequential()
        
        # Standard Fully Connected Neural Network (MLP)
        self.net.add_module("layer_in", nn.Linear(layers[0], layers[1]))
        self.net.add_module("act_in", nn.Tanh()) # Tanh activation is smooth (good for derivatives)
        
        for i in range(1, len(layers)-1):
            self.net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.net.add_module(f"act_{i}", nn.Tanh())
        
        # CRITICAL INITIALIZATION TRICK:
        # We initialize the network bias to the 'init_slowness' (1/v).
        # Since T = dist * Network, initially T ~= dist * (1/v) = dist / v.
        # This is the exact solution for a homogeneous medium! 
        # The training starts extremely close to the solution, drastically speeding up convergence.
        self.net[-1].bias.data.fill_(init_slowness)

    def forward(self, x_norm, x_phys, src_phys):
        """
        The Forward Pass defines the ansatz (the shape of the solution).
        
        Args:
           x_norm: Normalized coordinates ([-1, 1]). Neural networks learn best with these.
           x_phys: Physical coordinates (mm). We need these to calculate real distances.
           src_phys: Source location in physical space.
        """
        # 1. Calculate Euclidean Distance SYMBOLICALLY inside the graph
        # Why? If we pre-calculated distance as a number, PyTorch would see it as a constant.
        # By calculating it here using tensors, PyTorch knows d(dist)/dx.
        # We add 1e-8 to avoid sqrt(0) error at the exact source location.
        dist = torch.sqrt(torch.sum((x_phys - src_phys)**2, dim=1, keepdim=True) + 1e-8)
        
        # 2. Get Scaling Factor from Network
        # The NN takes normalized coordinates to predict 'N', the local slowness factor.
        N = self.net(x_norm)
        
        # 3. Factorization Formula
        # T(x) = distance(x) * N(x)
        # This guarantees that when distance is 0 (at source), T is 0.
        # It forces the wave to be spherical-ish, preventing flat/planar trivial solutions.
        T = dist * N
        return T

# ==========================================
# 3. MAIN SCRIPT
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--basename', type=str, required=True, help='Mesh filename prefix')
    parser.add_argument('--out-vtu', type=str, default='T_pinn.vtu')
    parser.add_argument('--cv', type=float, default=1.0, help='Conduction Velocity (mm/ms)')
    parser.add_argument('--source', nargs=3, type=float, default=[0.0, 0.0, 0.0], help='Source coords x y z')
    
    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--w-pde', type=float, default=1.0)
    parser.add_argument('--scale-to-mm', action='store_true', default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")

    # 1. Load Data
    pts, elems = load_opencarp_mesh(args.basename, args.scale_to_mm)
    
    # 2. Coordinate Setup (CRITICAL STEP)
    # We maintain TWO sets of coordinates:
    
    # Set A: Physical Coordinates (mm)
    # Used for the physics equation (Distance calculation, gradients in real units)
    X_phys = torch.tensor(pts, dtype=torch.float32, device=device, requires_grad=True)
    
    # Set B: Normalized Coordinates (approx -1 to 1)
    # Used as INPUT to the neural network.
    # NNs struggle to converge if inputs are very small (0.001) or very large (100).
    lb = pts.min(0)
    ub = pts.max(0)
    dims = ub - lb
    pts_norm = 2.0 * (pts - lb) / dims - 1.0
    X_norm = torch.tensor(pts_norm, dtype=torch.float32, device=device, requires_grad=True)
    
    # Source Setup
    src_loc = torch.tensor([args.source], dtype=torch.float32, device=device)
    
    # Calculate Target Slowness (s = 1/v)
    # Eikonal equation: |grad T| = s
    target_slowness = 1.0 / args.cv
    print(f"Target Slowness: {target_slowness:.4f} (CV={args.cv})")

    # 3. Model Initialization
    # Pass target_slowness so the model starts "almost correct"
    model = DistancePINN(init_slowness=target_slowness).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("Starting training with Symbolic Distance Factorization...")

    # 4. Training Loop
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        
        # Forward Pass
        # We pass X_norm for the NN weights, but X_phys for the distance physics
        T_pred = model(X_norm, X_phys, src_loc)
        
        # Gradient Calculation (Automatic Differentiation)
        # Compute dT/dx, dT/dy, dT/dz
        # Note: We differentiate w.r.t X_phys. This gives us the PHYSICAL gradient directly.
        # If we diff'd w.r.t X_norm, we'd have to manually multiply by the scaling factor.
        grads = torch.autograd.grad(T_pred, X_phys, torch.ones_like(T_pred), create_graph=True)[0]
        
        # Calculate Magnitude of Gradient: |grad T|
        norm_grad = torch.norm(grads, dim=1, keepdim=True)
        
        # --- SINGULARITY MASKING ---
        # The gradient of distance (1/2*sqrt(r)) explodes at r=0 (the source).
        # This numerical explosion will cause NaN loss if not handled.
        # We calculate squared distance and mask out points inside a small radius (0.5mm).
        dist_sq = torch.sum((X_phys - src_loc)**2, dim=1, keepdim=True)
        mask = (dist_sq > 0.5**2).float() # 1.0 if outside safe zone, 0.0 if inside
        
        # Physics Loss (Eikonal Residual)
        # Loss = mean( (|grad T| - 1/v)^2 )
        # We only sum the loss where mask == 1.0
        loss_pde = torch.sum(mask * (norm_grad - target_slowness)**2) / (torch.sum(mask) + 1e-8)
        
        loss = args.w_pde * loss_pde
        
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0 or epoch == 1:
            print(f"Ep {epoch:6d} | Loss: {loss.item():.6f}")

    # --- SAVE MODEL ---
    # Saving weights allows us to reload this later for multi-source evaluation
    model_save_path = args.basename + "_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

    # 5. Output Results
    with torch.no_grad():
        T_final = model(X_norm, X_phys, src_loc).cpu().numpy()
        
    write_vtu(args.out_vtu, pts, elems, T_final)
    write_dat("T_pinn.dat", pts, T_final)

if __name__ == "__main__":
    main()