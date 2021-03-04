"""
Collaborative Filtering

author: taikannakada
"""

"""
Collaborative filtering
-------------------------

Learning Latent Factors

1. randomly initialize latent paramters
2. calculate prediction --> some similarity measure
3. calculate loss

Input:

From the prospective of link prediction,
users --> layer 1, say, gene nework
movies --> layer 2, say, disease network

"""

n_gene = len("size of gene network")
n_disease = len("size of disease network")
n_n_factors = 5

gene_factors = torch.randn(n_gene, n_factors)
disease_factors = torch.randn(n_disease, n_factors)


def create_params(size):
    return nn.Paraameter(torch.zeros(*size).normal_(0, 0.01))

class DotProductBias(Module):
    """
    Probabilistic Matrix Factorization
    -----------------------------------

    a dotproduct approach to collaborative Filtering
    """

    def __init__(self, n_gene, n_disease, n_factors, y_range=(0,5.5)):
        self.gene_factors = create_params([n_gene, n_disease])
        self.gene_bias = create_params([n_gene])
        self.disease_factors = create_params([n_disease, n_factors])
        self.disease_bias = create_params([n_disease])
        self.y_range = y_range

    def forward(self, x):
        """
        forward pass
        """
        gene = self.gene_factors(x[:,0])
        disease = self.disease_factors(x[:,1])
        res = (gene * disease).sum(dim=1, keepdim=True)
        res += self.gene_bias(x[:,0]) + self.disease_bias(x[:,1])
        return sigmoid_range(res, *self.y_range)



class NN_Collab(nn.Module):
    """
    Deep Learning approach to collaborative Filtering
    """
    def __init__(self, gene_size, disease_size, y_range=(0,5.5), n_act=100):
        self.gene_factors = create_params([n_gene, n_disease])
        self.disease_factors = create_params([n_disease, n_factors])
        self.layers = nn.Sequential(nn.Linear(gene_size[1] + disease_size[1], n_act),
                                    nn.Relu(),
                                    nn.Linear(n_act, 1))
        self.y_range = y_range

    def forward(self, x):
        embedding = self.gene_factors(x[:,0]), self.disease_factors(x[:,1])




model = DotProduct(n_gene, n_disease, 50)
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)
