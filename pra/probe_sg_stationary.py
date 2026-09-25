"""Compare two parameterizations of the same diagonal SG correction fit."""

from pathlib import Path
import numpy as np
from scipy.linalg import expm
from sine_gordon import sglib as sg
from duffing import duffing_padded_hybrid_validation as df
from pra.records import write_json, provenance, ROOT


def main():
    rows=[]
    for n,d,k in ((16,12,8),(48,24,16)):
        basis=sg.build_mode_basis(n,np.sqrt(3));h=5.2/np.sqrt(3)/10
        cosine=(basis.U_phi*np.cos(2*basis.lam_phi))@basis.U_phi.conj().T
        target=expm(-1j*h*.3*(np.eye(n)-cosine))
        inputs=np.eye(n,dtype=complex)[:,:k]
        layers=4;theta=np.zeros((layers,d))
        theta[0,:k]=np.angle(np.diag(target)[:k])
        zero_alpha=np.zeros((layers,2))
        paired_alpha=np.array([[.5,0],[-.5,0],[0,.5],[0,-.5]])
        compiler=df.SnapDisplacementCompiler(n,d,layers,inputs,target@inputs,0.)
        zero=df.pack_parameters(theta,zero_alpha)
        paired=df.pack_parameters(theta,paired_alpha)
        f0,g0=compiler.value_and_grad(zero);f1,g1=compiler.value_and_grad(paired)
        u0=df.dense_ansatz_unitary(n,d,layers,zero)
        u1=df.dense_ansatz_unitary(n,d,layers,paired)
        rows.append(dict(N=n,d=d,columns=k,zero_displacement_loss=f0,paired_displacement_loss=f1,
            same_initial_unitary_difference=float(np.linalg.norm(u0-u1)),zero_displacement_gradient_norm=float(np.linalg.norm(g0)),
            paired_displacement_gradient_norm=float(np.linalg.norm(g1)),paired_snap_gradient_norm=float(np.linalg.norm(g1[:layers*d]))))
    write_json(ROOT/"pra/records/sg_parity_stationary_probe.json",dict(rows=rows,regularization=0.,
        scope="One objective and gradient evaluation per parameterization. Opposite adjacent displacements cancel at zero intervening SNAP phases, leaving the same diagonal unitary. No optimizer or trajectory run. This isolates the parameterization geometry, not achievable final accuracy.",
        diagonal_phases="arg of the exact local target diagonal on the input columns",provenance=provenance()))


if __name__=="__main__":main()
