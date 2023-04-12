import torch.nn as nn
import torch


class DistMult(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(DistMult, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.init_weights()
        self.loss = nn.BCELoss()

    def init_weights(self):
        nn.init.xavier_normal_(self.entity_embeddings.weight.data)
        nn.init.xavier_normal_(self.relation_embeddings.weight.data)

    def forward(self, e1, r):
        e1_emb = self.entity_embeddings(e1)
        r_emb = self.relation_embeddings(r)

        resDistMult = torch.mm(e1_emb * r_emb, self.entity_embeddings.weight.transpose(1, 0))
        return torch.sigmoid(resDistMult)

    def predict(self, e1, r, e2):
        e1_emb = self.entity_embeddings(e1)
        r_emb = self.relation_embeddings(r)
        e2_emb = self.entity_embeddings(e2)

        resDistMult = torch.mm(e1_emb * r_emb, e2_emb.transpose(1, 0))
        return torch.sigmoid(resDistMult) > 0.5


#
# -------------- LiteralE DistMult --------------
#
# LitEmbedding implements the function g from the paper
class LitEmbedding(nn.Module):

    def __init__(self, num_numerical_literals, num_text_literals, embedding_dim):
        super(LitEmbedding, self).__init__()
        self.W_ze = nn.Linear(embedding_dim, embedding_dim)
        self.W_zl_num = nn.Linear(num_numerical_literals, embedding_dim)
        self.W_zl_text = nn.Linear(num_text_literals, embedding_dim)
        self.bias = nn.Parameter(torch.zeros(embedding_dim))
        self.W_h = nn.Linear(embedding_dim + num_numerical_literals + num_text_literals, embedding_dim)

    def forward(self, e1_emb, e_num_lit, e_text_lit):
        z_e = self.W_ze(e1_emb)
        z_l_num = self.W_zl_num(e_num_lit)
        z_l_text = self.W_zl_text(e_text_lit)
        z = torch.sigmoid(z_e + z_l_num + z_l_text + self.bias)
        h = torch.tanh(self.W_h(torch.cat((e1_emb, e_num_lit, e_text_lit), dim=1)))

        return z * h + (1 - z) * e1_emb


# DistMultLit implements the function f_DistMult(g(e_i, l_i), g(e_j, l_j), r_k) from the paper
class DistMultLit(nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals, text_literals, embedding_dim):
        super(DistMultLit, self).__init__()
        self.embedding_dim = embedding_dim
        self.numerical_literals = numerical_literals
        self.text_literals = text_literals
        self.num_numerical_literals = numerical_literals.shape[1]
        self.num_text_literals = text_literals.shape[1]
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.literal_embeddings = LitEmbedding(self.num_numerical_literals, self.num_text_literals, embedding_dim)
        self.loss = nn.BCELoss()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.entity_embeddings.weight.data)
        nn.init.xavier_normal_(self.relation_embeddings.weight.data)

    def forward(self, e1, r):
        e1_emb = self.entity_embeddings(e1)
        r_emb = self.relation_embeddings(r)

        e1_num_lit = self.numerical_literals[e1]
        e1_text_lit = self.text_literals[e1]
        e1_emb = self.literal_embeddings(e1_emb, e1_num_lit, e1_text_lit)
        e2_all_emb = self.literal_embeddings(self.entity_embeddings.weight, self.numerical_literals, self.text_literals)

        resDistMult = torch.mm(e1_emb * r_emb, e2_all_emb.transpose(1, 0))
        return torch.sigmoid(resDistMult)
