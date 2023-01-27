import torch
import numpy as np
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from prostat.utils.geometry import get_dihedrals


def plot_bonds(pos, bond_idcs, c='red'):
    _bond_idcs = bond_idcs.T
    x_bonds = []
    y_bonds = []
    z_bonds = []

    for i in range(_bond_idcs.shape[1]):
        x_bonds.extend([pos[_bond_idcs[0][i], 0].item(), pos[_bond_idcs[1][i], 0].item(), None])
        y_bonds.extend([pos[_bond_idcs[0][i], 1].item(), pos[_bond_idcs[1][i], 1].item(), None])
        z_bonds.extend([pos[_bond_idcs[0][i], 2].item(), pos[_bond_idcs[1][i], 2].item(), None])

    return go.Scatter3d(
            x=x_bonds,
            y=y_bonds,
            z=z_bonds,
            name='bonds',
            mode='lines',
            line=dict(color=c, width=2),
            hoverinfo='none')


def plot(nth, pos1, pos2, P, p1, p2, dataset, bond_idcs=None):
    subplots = 1
    fig = make_subplots(rows=1, cols=subplots, specs=[[{'type': 'scene'}]*subplots], shared_xaxes=True, horizontal_spacing=0)

    trace_1 = go.Scatter3d(
        x=pos1[:, 0],
        y=pos1[:, 1],
        z=pos1[:, 2],
        name='atoms',
        text=dataset['bead_types'],
        mode='markers',
        marker=dict(symbol='circle', color=dataset['z'], opacity=0.5, size=5)
    )

    trace_2 = go.Scatter3d(
        x=pos2[:, 0],
        y=pos2[:, 1],
        z=pos2[:, 2],
        name='atoms',
        text=dataset['bead_types'],
        mode='markers',
        marker=dict(symbol='circle', color='green', opacity=0.5, size=5)
    )

    data = [trace_1, trace_2]

    if bond_idcs is not None:
        data.append(plot_bonds(pos1, bond_idcs, c='red'))
        data.append(plot_bonds(pos2, bond_idcs, c='blue'))

    c = P[nth:nth+1].cpu().detach().numpy()

    for h in range(0, c.shape[-1], 3):
        data.append(go.Scatter3d(
            x=c[:, h],
            y=c[:, h+1],
            z=c[:, h+2],
            name='atoms',
            mode='markers',
            marker=dict(symbol='cross', color='blue', opacity=0.5, size=5)
        ))

    p1_ = p1[nth:nth+1].cpu().detach().numpy()
    for h in range(0, p1_.shape[-1], 3):
        data.append(go.Scatter3d(
            x=c[:, h] + p1_[:, h],
            y=c[:, h+1] + p1_[:, h+1],
            z=c[:, h+2] + p1_[:, h+2],
            name='atoms',
            mode='markers',
            marker=dict(symbol='diamond-open', color='yellow', opacity=0.5, size=5)
        ))

    p2_ = p2[nth:nth+1].cpu().detach().numpy()
    for h in range(0, p2_.shape[-1], 3):
        data.append(go.Scatter3d(
        x=c[:, h] + p2_[:, h],
        y=c[:, h+1] + p2_[:, h+1],
        z=c[:, h+2] + p2_[:, h+2],
        name='atoms',
        mode='markers',
        marker=dict(symbol='diamond', color='orange', opacity=0.5, size=5)
    ))

    for d in data:
        fig.add_trace(d, row=1, col=1)

    layout = go.Layout(
        scene=dict(
        xaxis = dict(
            nticks=3,
            range=[pos1.min() - 0.2, pos1.max() + 0.2],
            backgroundcolor="rgba(0,0,0,0.2)",
            gridcolor="whitesmoke",
            showbackground=True,
            showgrid=True,
            ),
        yaxis = dict(
            nticks=3,
            range=[pos1.min() - 0.2, pos1.max() + 0.2],
            backgroundcolor="rgba(0,0,0,0.1)",
            gridcolor="whitesmoke",
            showbackground=True,
            showgrid=True,
            ),
        zaxis = dict(
            nticks=3,
            range=[pos1.min() - 0.2, pos1.max() + 0.2],
            backgroundcolor="rgba(0,0,0,0.4)",
            gridcolor="whitesmoke",
            showbackground=True,
            showgrid=True,
            ),
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        scene_aspectmode='cube',
        autosize=False,
        width=1000,
        height=1000,
    )

    fig.update_layout(layout)
    fig.show()

def plot_dihedral_distribution(dataset, pos_recon, minimized_pos_recon, n=100):
    dih = get_dihedrals(torch.from_numpy(dataset['pos_test']), dataset["dihedral_idcs"][:n]).detach().cpu()
    dih_pred = get_dihedrals(pos_recon, dataset["dihedral_idcs"][:n]).detach().cpu()
    dih_pred_recon = get_dihedrals(minimized_pos_recon, dataset["dihedral_idcs"][:n]).detach().cpu()
    plot_hist(dih, dih_pred, dih_pred_recon)

    ca_idcs = np.array([i for i, bt in enumerate(dataset['bead_types']) if bt.split("_")[1] == "CA"])
    c_idcs = np.array([i for i, bt in enumerate(dataset['bead_types']) if bt.split("_")[1] == "C"])
    n_idcs = np.array([i for i, bt in enumerate(dataset['bead_types']) if bt.split("_")[1] == "N"])
    backboone_idcs = np.sort(np.concatenate([ca_idcs, c_idcs, n_idcs]))
    phi_psi_dihedrals = dataset["dihedral_idcs"][np.isin(dataset["dihedral_idcs"], backboone_idcs).all(axis=1)]

def plot_hist(dih, dih_pred, dih_pred_recon):
    n_dih = dih.shape[1]
    rows = int(np.sqrt(n_dih))
    cols = int(np.ceil(n_dih / rows))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    for i, (dih_row, dih_pred_row, dih_pred_recon) in enumerate(zip(dih.T, dih_pred.T, dih_pred_recon.T)):
        _, _, _ = axs[i//cols, i%cols].hist(dih_row, bins=20, histtype='step', linewidth=2, facecolor='c', hatch='/', edgecolor='k', fill=False)
        _, _, _ = axs[i//cols, i%cols].hist(dih_pred_row, bins=20, histtype='step', linewidth=2, facecolor='cyan', hatch='/', edgecolor='cyan', fill=False)
        _, _, _ = axs[i//cols, i%cols].hist(dih_pred_recon, bins=20, histtype='step', linewidth=2, facecolor='orange', hatch='/', edgecolor='orange', fill=False)
        axs[i//cols, i%cols].set_xlim([-np.pi, np.pi])
    plt.show()