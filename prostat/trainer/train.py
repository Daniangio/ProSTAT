import torch
from tqdm.auto import tqdm
from numpy import log

from prostat.utils.geometry import get_dihedrals


def criterion(pred, target, aggr_f='sum'):
    if aggr_f == 'sum':
        return -torch.exp(-torch.norm(pred - target, dim=-1)).sum(dim=-1).mean()
    assert aggr_f == 'mean'
    return -torch.exp(-torch.norm(pred - target, dim=-1)).mean()


def criterion_ramachandran(pred, target, dihedral_idcs):
    pred_dih = get_dihedrals(pred, dihedral_idcs)
    target_dih = get_dihedrals(target, dihedral_idcs)
    loss_cos = torch.abs(torch.cos(pred_dih) - torch.cos(target_dih)).sum(axis=1)
    loss_sin = torch.abs(torch.sin(pred_dih) - torch.sin(target_dih)).sum(axis=1)
    return loss_cos.mean() + loss_sin.mean()


def train_autoencoder(model, dataset, conf):
    criterion_rmse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf['lr'], betas=(0.5, 0.999))
    pos_train = torch.from_numpy(dataset['pos_train']).double().to(conf['device'])

    _, _, _, _, _, _, _, contribution_beads = model(pos_train)
    contribution_beads_loss = torch.zeros_like(contribution_beads, dtype=torch.float32)
    contribution_beads_loss[:, 0] = 1.
    contribution_beads_loss = torch.log(contribution_beads_loss + 1e-10).mean()
    loss = None
    print(f'Loss:  log_recon |    rmse    | bead_independence | orthogonality | ramachandran')
    for i in tqdm(range(conf['epochs'])):
        out, P, p1, p2, st, rot_axes, normalized_weights, contribution_beads = model(pos_train)
        loss_recon = criterion(out, pos_train, aggr_f='mean')
        loss_rmse = criterion_rmse(out, pos_train) * 10
        # loss_sparsity = torch.log(model.encoder_lin.weights + 1e-10).sum() / -(np.log(1e-10) * (model.locality - 1) * (model.hidden + model.hidden_remainder) * 3)
        loss_bead_independence = torch.log(normalized_weights[contribution_beads.bool()] + 1e-10).mean() / -contribution_beads_loss
        loss_orthogonality = torch.log(torch.abs(torch.einsum('bij,bij->bi', p1, p2)) + 1e-10).sum() / (-log(1e-10) * p1.shape[0] * p2.shape[1]) * 10
        loss_ramachandran = criterion_ramachandran(out, pos_train, dataset['dihedral_idcs'])

        loss = loss_recon + loss_rmse + loss_bead_independence + loss_orthogonality + loss_ramachandran # + loss_sparsity

        if not i % conf['log_every']:
            print(f'Loss: {loss_recon.item():10.4f} | {loss_rmse.item():10.4f} | {loss_bead_independence.item():10.4f} | {loss_orthogonality.item():10.4f} | {loss_ramachandran.item():10.4f}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            while not torch.allclose(model.encoder_lin.weights.sum(dim=0), torch.ones_like(model.encoder_lin.weights.sum(dim=0), dtype=torch.double), 1e-10):
                weight_norm = torch.abs(model.encoder_lin.weights)/torch.abs(model.encoder_lin.weights.sum(dim=0, keepdim=True))
                
                steepness = 100
                weight_norm[..., 1:] = weight_norm[..., 1:] * torch.sigmoid(weight_norm[..., :1]*steepness - 1/(2*model.locality)*steepness)
                weight_norm = weight_norm/weight_norm.sum(dim=0, keepdim=True)

                model.encoder_lin.weights = model.encoder_lin.weights.copy_(weight_norm)
            while not torch.allclose(model.decoder_lin.weights.sum(dim=2), torch.ones_like(model.decoder_lin.weights.sum(dim=2), dtype=torch.double), 1e-10):
                weight_norm = torch.abs(model.decoder_lin.weights)/torch.abs(model.decoder_lin.weights.sum(dim=2, keepdim=True))
                model.decoder_lin.weights = model.decoder_lin.weights.copy_(weight_norm)

    print(f'Loss: {loss}')