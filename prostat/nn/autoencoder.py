import torch
from e3nn import o3


def q_conjugate(q: torch.DoubleTensor):
    scaling = torch.tensor([1, -1, -1, -1], device=q.device)
    return scaling * q

def q_mult(q1, q2):
    w1, x1, y1, z1 = torch.unbind(q1, -1)
    w2, x2, y2, z2 = torch.unbind(q2, -1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return torch.stack((w, x, y, z), -1)

def qv_mult(q1, v2):
    if v2.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {v2.shape}.")
    real_parts = v2.new_zeros(v2.shape[:-1] + (1,))
    q2 = torch.cat((real_parts, v2), -1)
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[..., 1:]


class AELinear(torch.nn.Module):
    def __init__(
        self,
        locality,
        hidden,
        num_out_vectors,
        **kwargs
    ):
        super().__init__()
        self.num_out_vectors = num_out_vectors
        self.locality = locality
        self.hidden = hidden
        weights = torch.nn.Parameter(torch.randn(self.locality, self.hidden, self.num_out_vectors))
        self.register_parameter("weights", weights)        
    
    def forward(self, x):
        return torch.einsum("bijk,ijl->bjlk", x, self.weights)


class AutoEncoder(torch.nn.Module):

    def __init__(
        self,
        num_atoms,
        locality = 20,
        desired_stride = 10,
        **kwargs
    ):
        super().__init__()
        self.locality = locality
        self.desired_stride = desired_stride
        assert desired_stride <= locality, "Cannot stride more than locality, or you are loosing some points"

        self.n = num_atoms
        if self.locality > self.n:
            self.locality = self.n
        self.hidden_remainder = int((self.n - self.locality) % self.desired_stride > 0)
        self.hidden = (self.n - self.locality) // self.desired_stride + self.hidden_remainder + 1
        
        self.partial_n = self.locality + self.desired_stride * (self.hidden - self.hidden_remainder - 1)
        assert self.locality * self.hidden >= self.partial_n, "Cannot cover all points"

        p_p1_p2 = 3
        self.encoder_lin = AELinear(self.locality, self.hidden, p_p1_p2)

        self.cross_prod = o3.TensorProduct(
            f"{self.hidden}x1o",
            f"{self.hidden}x1o",
            f"{self.hidden}x1e",
            [(0, 0, 0, "uuu", False)],
            irrep_normalization='none'
        )

        self.decoder_lin = AELinear(p_p1_p2, self.hidden, self.locality)

        angles_polar = torch.nn.Parameter(torch.randn(self.hidden * self.locality, 1))
        self.register_parameter("angles_polar", angles_polar)
        angles_azimutal = torch.nn.Parameter(torch.randn(self.hidden * self.locality, 1))
        self.register_parameter("angles_azimutal", angles_azimutal)
        scaling = torch.nn.Parameter(torch.randn(self.hidden * self.locality, 1))
        self.register_parameter("scaling", scaling)
    
    def encode(self, x, batch):
        partial_input = x[:, :self.partial_n]
        b, i, j = partial_input.stride()
        stride = b, i, self.desired_stride*i, j

        st = torch.as_strided(partial_input, (batch, self.locality, self.hidden - self.hidden_remainder, 3), stride)
        if self.hidden_remainder:
            remainder = x[:, -self.locality:, None, :]
            st = torch.cat([st, remainder], dim=2)
        embedding = self.encoder_lin(st)

        P, I1, I2 = embedding.split([1, 1, 1], dim=2)
        P, I1, I2 = P.squeeze(2), I1.squeeze(2), I2.squeeze(2)
        p1 = I1 - P
        p2 = I2 - P
        
        p1_norm = torch.norm(p1, dim=-1, keepdim=True)
        p2_norm = torch.norm(p2, dim=-1, keepdim=True)
        p1_mask = (p1_norm == 0)
        p2_mask = (p2_norm == 0)
        p1_norm = p1_norm.masked_fill(p1_mask, value=1.)
        p2_norm = p2_norm.masked_fill(p2_mask, value=1.)
        
        p1 = p1 / p1_norm
        p2 = p2 / p2_norm
        return P, p1, p2, st
    
    def decode(self, P, p1, p2, batch):
        device = P.device

        p12 = self.cross_prod(p1.reshape(p1.size(0), -1), p2.reshape(p2.size(0), -1)).reshape(p1.size(0), p1.shape[1], -1)
        
        stacked = torch.stack([p1, p2, p12], dim=1) # (batch, p1_p2_p12, hidden, xyz)
        rot_axes = self.decoder_lin(stacked)
        #rot_axes = rot_axes / torch.norm(rot_axes, dim=-1, keepdim=True)
        rot_axes = rot_axes.reshape(-1, 3)

        q_polar = self.get_queterions(batch=batch, rot_axes=rot_axes, angles=self.angles_polar)

        v1 = p1.reshape(-1, 3).repeat_interleave(self.locality, 0)
        v3 = qv_mult(q_polar, v1).reshape(batch, self.hidden, self.locality, -1)
        
        # q_polar = self.get_queterions(batch=batch, rot_axes=p12.reshape(-1, 3), angles=self.angles_polar)
        # q_azimutal = self.get_queterions(batch=batch, rot_axes=p1.reshape(-1, 3), angles=self.angles_azimutal)

        # v1 = p1.reshape(-1, 3).repeat_interleave(self.locality, 0)
        # v2 = qv_mult(q_polar, v1)
        # v3 = qv_mult(q_azimutal, v2).reshape(batch, self.hidden, self.locality, -1)

        reconstructed = torch.zeros((batch, self.n, 3)).to(device)
        contribution_beads = torch.zeros((self.n, self.hidden)).to(device)
        idcs = torch.arange(1, self.n + 1, dtype=int)
        started_beads = torch.ceil(idcs/self.desired_stride).int()
        started_beads[self.partial_n - self.locality:] = started_beads[self.partial_n - self.locality]
        ended_beads = torch.max(torch.ceil((idcs - self.locality)/self.desired_stride), torch.zeros_like(idcs)).int()
        if self.hidden_remainder:
            ended_beads[self.partial_n:] = ended_beads[self.partial_n]
            started_beads[-self.locality:] += 1

        for i, (h_from, h_to) in enumerate(zip(ended_beads, started_beads)):
            contribution_beads[i, h_from:h_to] = 1.

        normalized_weights = torch.zeros((self.n, self.hidden)).to(device)
        for shift, h_weights in enumerate(self.encoder_lin.weights[..., 0]): # (P)
            matrix = torch.eye(h_weights.size(0)).to(device) * h_weights
            matrix = matrix[:self.hidden - self.hidden_remainder, :]
            normalized_weights[shift:self.partial_n + 1 + shift - self.locality:self.desired_stride] += contribution_beads[shift:self.partial_n + 1 + shift - self.locality:self.desired_stride] * matrix
        if self.hidden_remainder:
            normalized_weights[-self.locality:, -1] = self.encoder_lin.weights[:, -1, 0]
        normalized_weights = normalized_weights / normalized_weights.sum(dim=1, keepdim=True)

        for h, shift in enumerate(range(0, self.partial_n - self.locality + 1, self.desired_stride)):
            reconstruction = self.scaling[shift:shift + self.locality] * v3[:, h] + P[:, h:h+1]
            reconstructed[:, shift:shift + self.locality] += reconstruction * normalized_weights[shift:shift + self.locality, h][None, :, None]
        if self.hidden_remainder:
            reconstruction = self.scaling[-self.locality:] * v3[:, -1] + P[:, -1:]
            reconstructed[:, -self.locality:] += reconstruction * normalized_weights[-self.locality:, -1][None, :, None]
        return reconstructed, rot_axes, normalized_weights, contribution_beads

    def get_queterions(self, batch, rot_axes, angles):
        real_parts = torch.cos(angles)
        real_parts = real_parts.repeat(batch, 1)
        imaginary_parts_multiplier = torch.sin(angles)
        imaginary_parts_multiplier = imaginary_parts_multiplier.repeat(batch, 1)
        imaginary_parts = imaginary_parts_multiplier * rot_axes
        q = torch.cat((real_parts, imaginary_parts), -1)
        return q
    
    def forward(self, x):
        batch = x.size(0)

        P, p1, p2, st = self.encode(x, batch)
        reconstructed, p1p2, normalized_weights, contribution_beads = self.decode(P, p1, p2, batch)
        
        return reconstructed, P, p1, p2, st, p1p2, normalized_weights, contribution_beads