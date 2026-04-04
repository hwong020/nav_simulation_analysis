import sys
import numpy as np
import mdtraj as md

# ===== INPUT =====
# Command-line arguments: topology file and trajectory file
top = sys.argv[1]      # confout.gro
traj = sys.argv[2]     # traj.xtc
# Contact cutoff in nanometers (0.4 nm = 4 Å)
cutoff = 0.4

# ===== LOAD =====
# Load the trajectory and access its topology
t = md.load(traj, top=top)
topology = t.topology

# Select heavy atoms (non-hydrogen) for protein and ligand
protein_atoms = topology.select("protein and not element H")
ligand_atoms = topology.select("resname LIG and not element H")

if len(ligand_atoms) == 0:
    raise ValueError("No ligand with resname LIG found.")

# Collect all protein residues in the topology
protein_residues = [res for res in topology.residues if res.is_protein]

# Map atom index → residue index in the full topology
atom_to_res = np.array([atom.residue.index for atom in topology.atoms])

# Track per-residue contact counts (number of frames with any contact)
contact_counts = np.zeros(len(protein_residues), dtype=int)
n_frames = t.n_frames

# Progress setup
progress_step = max(1, n_frames // 100)  # update every 1%

# ===== MAIN LOOP =====
# For each frame, compute distances from all protein atoms to all ligand atoms.
# A residue is considered "in contact" for that frame if ANY of its atoms
# are within the cutoff distance to ANY ligand atom.
for frame in range(n_frames):

    # Progress print
    if frame % progress_step == 0:
        percent = (frame / n_frames) * 100
        print(f"Progress: {percent:.1f}% ({frame}/{n_frames})", end="\r")

    # Coordinates for current frame (shape: n_atoms x 3)
    prot_xyz = t.xyz[frame, protein_atoms, :]
    lig_xyz = t.xyz[frame, ligand_atoms, :]

    # Pairwise distances between protein and ligand atoms
    dists = np.linalg.norm(
        prot_xyz[:, None, :] - lig_xyz[None, :, :],
        axis=2
    )

    # Find indices of atom pairs within the cutoff
    contacts = np.where(dists < cutoff)

    if len(contacts[0]) == 0:
        continue

    # Map contacting protein atoms to unique residue indices
    contacting_atoms = protein_atoms[contacts[0]]
    contacting_residues = np.unique(atom_to_res[contacting_atoms])

    # Increment once per frame for each residue that has any contact
    contact_counts[contacting_residues] += 1

# Final progress line
print(f"Progress: 100.0% ({n_frames}/{n_frames})")

# ===== OUTPUT CSV =====
# Probability = fraction of frames with at least one contact for each residue
with open("contact_probability.csv", "w") as f:
    f.write("ResID,Resname,Probability\n")
    for i, res in enumerate(protein_residues):
        prob = contact_counts[i] / n_frames
        f.write(f"{res.resSeq},{res.name},{prob:.4f}\n")

print("Done. Output written to contact_probability.csv")