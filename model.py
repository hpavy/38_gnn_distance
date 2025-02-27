import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, hyper_param):
        super().__init__()
        self.nb_hidden = hyper_param['nb_hidden']
        self.nb_hidden_encode = hyper_param['nb_hidden_encode']
        self.dim_latent = hyper_param['dim_latent']
        self.nb_gn = hyper_param['nb_gn']
        self.gnn = nn.ModuleList(
            [MLP(dim_in=3, dim_out=self.dim_latent, nb_hidden=self.nb_hidden_encode)]
            + [GN(self.nb_hidden, self.dim_latent, hyper_param['nb_neighbours']) for _ in range(hyper_param['nb_gn'])]
            + [MLP(dim_in=self.dim_latent, dim_out=3, nb_hidden=self.nb_hidden_encode)]
        )

    def forward(self, x, edge_neighbours):
        for block in self.gnn:
            x = block(x, edge_neighbours)
        return x


class GN(nn.Module):
    def __init__(self, nb_hidden, dim_latent, nb_neighbours):
        super().__init__()
        self.mlp_neigh = nn.ModuleList(
            [MLP(dim_latent, dim_latent, nb_hidden)] +  # pour lui même
            [MLP(dim_latent, dim_latent, nb_hidden) for _ in range(nb_neighbours)]  # pour les voisins
            )
        self.LayerNorm = torch.nn.LayerNorm(dim_latent)

    def forward(self, x, edge_neighbours):
        message = self.mlp_neigh[0](x, edge_neighbours)  # batch_size * nb_noeuds * dim_latent
        for k in range(1, edge_neighbours.shape[1]+1):
            message = message + self.mlp_neigh[k](x[:, edge_neighbours[:, k-1]], edge_neighbours)
        return self.LayerNorm(x + F.relu(message))


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, nb_hidden):
        # self.dim_in = dim_in
        # self.dim_out = dim_out
        super().__init__()
        self.linear_first = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        self.hidden = nn.ModuleList([
            nn.Linear(dim_out, dim_out) for _ in range(nb_hidden)
        ])
        self.mlp = self.linear_first + self.hidden
        self.initial_param()

    def forward(self, x, _):
        for layer in self.mlp:
            x = F.relu(layer(x))
        return x

    def initial_param(self):
        for layer in self.mlp:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
