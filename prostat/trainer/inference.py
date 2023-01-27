import torch

from prostat.trainer.minimization import minimization


def test_autoencoder(model, dataset, conf):
    criterion_rmse = torch.nn.MSELoss()

    pos_test = torch.from_numpy(dataset['pos_test']).double().to(conf['device'])
    # rot = get_R(phi=np.pi, theta=np.pi/2, psi=np.pi/3., device=device).double()
    # pos_test = torch.einsum("ij,bkj->bki", rot, pos_test)

    print('Inferencing...')
    pos_recon, pos_beads, v1, v2, st, rot_axes, normalized_weights, contribution_beads = model(pos_test)
    v12 = model.cross_prod(v1.reshape(v1.size(0), -1), v2.reshape(v2.size(0), -1)).reshape(v1.size(0), v1.shape[1], -1)
    print("Prediction RMSE:", criterion_rmse(pos_recon, pos_test) * 10)

    print("Minimizing structure...")
    minimized_pos_recon = minimization(pos_recon.detach(), dataset, conf)
    print("Minimized Prediction RMSE:", criterion_rmse(minimized_pos_recon, pos_test.detach().cpu()) * 10)
    return pos_recon.detach().cpu(), minimized_pos_recon.detach().cpu(), pos_beads.detach().cpu(), v1.detach().cpu(), v2.detach().cpu(), v12.detach().cpu()