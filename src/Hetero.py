import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block, InnerProductInteraction
from fuxictr.pytorch.torch_utils import get_activation


class Hetero(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="Hetero",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 num_cross_layers=4,
                 pnn_hidden_units=[400, 400],
                 pnn_batchnorm=True,
                 fn_hidden_units=[400, 400],
                 gate_hidden_units=[400, 400],
                 gate_dropout_rate=0.1,
                 aux_loss_alpha=0.1,
                 net_dropout=0.1,
                 batch_norm=False,
                 hidden_activations="ReLU",
                 decorr_strength=0.1,
                 decorr_type='feature',
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(Hetero, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        if feature_map.dataset_id == "taobaoad_x1":
            self.userid_index = feature_map.get_column_index("userid")
        else:
            self.userid_index = feature_map.get_column_index("user_id")
        input_dim = feature_map.sum_emb_out_dim()
        num_fields = feature_map.num_fields
        self.embedding_dim = embedding_dim
        self.expert_centroids = nn.Parameter(torch.empty(3, embedding_dim))
        torch.nn.init.orthogonal_(self.expert_centroids, gain=0.1)
        self.expert_centroids_mapping = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(embedding_dim, embedding_dim)
            ) for _ in range(3)
        ])
        self.decorr_strength = decorr_strength
        self.decorr_type = decorr_type
        self.topk = 2
        self.ECN = ExponentialCrossNetwork(input_dim=input_dim,
                                           num_cross_layers=num_cross_layers,
                                           net_dropout=net_dropout,)
        self.PNN = ProductNetwork(num_fields=num_fields,
                                  embedding_dim=embedding_dim,
                                  hidden_units=pnn_hidden_units,
                                  batch_norm=pnn_batchnorm,
                                  net_dropout=net_dropout,
                                  hidden_activations=hidden_activations)
        self.FN = FinalNet(input_dim=input_dim * 2,
                           hidden_units=fn_hidden_units,
                           hidden_activations=hidden_activations,
                           dropout_rates=net_dropout,
                           batch_norm=batch_norm,
                           num_fields=num_fields)
        self.user_gate = MLP_Block(input_dim=embedding_dim,
                                   output_dim=3,
                                   hidden_units=gate_hidden_units,
                                   hidden_activations=hidden_activations,
                                   output_activation=None,
                                   dropout_rates=net_dropout,
                                   batch_norm=batch_norm)
        self.experts = [self.ECN, self.PNN, self.FN]
        self.gate_dropout_rate = gate_dropout_rate
        self.aux_loss_alpha = aux_loss_alpha
        self.dropout = nn.Dropout(0.5)
        self.softmax = get_activation("softmax")
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def calculate_decorrelation_loss(self):
        if self.decorr_strength <= 0:
            return 0.0
        if self.decorr_type == 'feature':
            if not hasattr(self, 'last_feature_emb') or self.last_feature_emb is None:
                return 0.0
            weights = []
            for expert in enumerate(self.experts):
                if hasattr(expert, 'mlp'):
                    first_layer = list(expert.mlp.mlp)[0]
                    feature_rep = first_layer(self.last_feature_emb)
                    weights.append(feature_rep)
                elif hasattr(expert, 'dnn') and hasattr(expert.dnn, 'mlp'):
                    first_layer = list(expert.dnn.mlp)[0]
                    feature_rep = first_layer(self.last_feature_emb)
                    weights.append(feature_rep)
                elif hasattr(expert, 'layer') and len(expert.layer) > 0:
                    first_layer = expert.layer[0]
                    if hasattr(first_layer, 'linear'):
                        feature_rep = first_layer.linear(self.last_feature_emb)
                    else:
                        feature_rep = first_layer(self.last_feature_emb)
                    weights.append(feature_rep)
                elif hasattr(expert, 'w') and len(expert.w) > 0:
                    feature_rep = expert.w[0](self.last_feature_emb)
                    weights.append(feature_rep)
        else:
            return 0.0

        total_loss = 0.0
        pair_count = 0
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                cos_sim = F.cosine_similarity(
                    weights[i].flatten(),
                    weights[j].flatten(),
                    dim=0,
                    eps=1e-6)
                total_loss += torch.abs(cos_sim)
                pair_count += 1
        return total_loss / pair_count if pair_count > 0 else 0.0

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb = self.embedding_layer(X)  # B × F × D
        if self.decorr_type == 'feature':
            self.last_feature_emb = feature_emb.detach().flatten(start_dim=1)
        userid_emb = feature_emb[:, self.userid_index, :]  # B × D
        user_gates = self.user_gate(userid_emb)
        if self.training:
            self.update_expert_centroids()
        logit1, x1 = self.ECN(feature_emb)
        logit2, x2 = self.PNN(feature_emb)
        logit3, x3 = self.FN(feature_emb)
        user_gates, _, topk_indices = self.calculate_topk_scores(user_gates, userid_emb)
        expert_outputs = torch.stack([logit1, logit2, logit3], dim=1)
        user_logit = torch.sum(user_gates.unsqueeze(-1) * expert_outputs, dim=1)
        y_pred = self.output_activation(user_logit)

        if self.training and self.aux_loss_alpha > 0:
            scores_aux = user_gates
            topk_idx = topk_indices
            mask_ce = F.one_hot(topk_idx, num_classes=3).sum(dim=1)
            ce = mask_ce.float().mean(dim=0)
            Pi = scores_aux.mean(dim=0)
            fi = ce * 3
            aux_loss = (Pi * fi).sum() * self.aux_loss_alpha
        else:
            aux_loss = torch.tensor(0.0).to(user_gates.device)
        return_dict = {"y_pred": y_pred,
                       "y1": torch.sigmoid(logit1),
                       "y2": torch.sigmoid(logit2),
                       "y3": torch.sigmoid(logit3),
                       "aux_loss": aux_loss,}
        return return_dict

    def update_expert_centroids(self):
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                if hasattr(expert, 'w') and len(expert.w) > 0:
                    first_layer_weight = expert.w[0].weight.data
                    if first_layer_weight.dim() == 2:
                        pooled_weight = torch.mean(first_layer_weight, dim=0, keepdim=False)
                        if pooled_weight.shape[0] != self.embedding_dim:
                            pooled_weight = torch.zeros(self.embedding_dim).to(first_layer_weight.device)
                        pooled_weight = pooled_weight.unsqueeze(0).unsqueeze(-1)
                        centroid = self.expert_centroids_mapping[i](pooled_weight)
                    else:
                        zero_tensor = torch.zeros(1, self.embedding_dim, 1).to(first_layer_weight.device)
                        centroid = self.expert_centroids_mapping[i](zero_tensor)
                elif hasattr(expert, 'dnn') and hasattr(expert.dnn, 'mlp'):
                    first_layer_weight = list(expert.dnn.mlp)[0].weight.data
                    if first_layer_weight.dim() == 2:
                        pooled_weight = torch.mean(first_layer_weight, dim=0, keepdim=False)
                        if pooled_weight.shape[0] != self.embedding_dim:
                            pooled_weight = torch.zeros(self.embedding_dim).to(first_layer_weight.device)
                        pooled_weight = pooled_weight.unsqueeze(0).unsqueeze(-1)
                        centroid = self.expert_centroids_mapping[i](pooled_weight)
                    else:
                        zero_tensor = torch.zeros(1, self.embedding_dim, 1).to(first_layer_weight.device)
                        centroid = self.expert_centroids_mapping[i](zero_tensor)
                elif hasattr(expert, 'layer') and len(expert.layer) > 0:
                    first_layer = expert.layer[0]
                    if hasattr(first_layer, 'linear'):
                        first_layer_weight = first_layer.linear.weight.data
                    else:
                        first_layer_weight = first_layer.weight.data
                    if first_layer_weight.dim() == 2:
                        pooled_weight = torch.mean(first_layer_weight, dim=0, keepdim=False)
                        if pooled_weight.shape[0] != self.embedding_dim:
                            pooled_weight = torch.zeros(self.embedding_dim).to(first_layer_weight.device)
                        pooled_weight = pooled_weight.unsqueeze(0).unsqueeze(-1)
                        centroid = self.expert_centroids_mapping[i](pooled_weight)
                    else:
                        zero_tensor = torch.zeros(1, self.embedding_dim, 1).to(first_layer_weight.device)
                        centroid = self.expert_centroids_mapping[i](zero_tensor)
                else:
                    zero_tensor = torch.zeros(1, self.embedding_dim, 1)
                    centroid = self.expert_centroids_mapping[i](zero_tensor)
                
                self.expert_centroids.data[i] = centroid

    def calculate_topk_scores(self, gates, userid_emb):
        topk_values, topk_indices = torch.topk(gates, k=self.topk, dim=1)
        all_gates = torch.softmax(gates, dim=1)
        masked_gates = torch.full_like(gates, -float('inf'))
        masked_gates.scatter_(1, topk_indices, topk_values)
        normalized_gates = F.softmax(masked_gates, dim=1)
        with torch.no_grad():
            token_expert_affinities = userid_emb.matmul(self.expert_centroids.transpose(0, 1))
            affinity_gates = F.softmax(token_expert_affinities, dim=1)
        all_gates = (all_gates + affinity_gates) / 2
        if self.training and self.gate_dropout_rate > 0:
            normalized_gates = F.dropout(normalized_gates,
                                         p=self.gate_dropout_rate,
                                         training=True)
        return all_gates, normalized_gates, topk_indices

    def add_loss(self, inputs):
        return_dict = self.forward(inputs)
        y_true = self.get_labels(inputs)
        y_pred = torch.clamp(return_dict["y_pred"], min=1e-6, max=1 - 1e-6)
        total_loss = self.loss_fn(y_pred, y_true, reduction='mean')
        if self.training:
            expert_outputs = []
            for i in range(3):
                expert_outputs.append(return_dict[f"y{i + 1}"])
            expert_losses = []
            for i, y_expert in enumerate(expert_outputs):
                y_expert = torch.clamp(y_expert, min=1e-6, max=1 - 1e-6)
                expert_loss = self.loss_fn(y_expert, y_true, reduction='mean')
                expert_losses.append(expert_loss)
            epsilon = 1e-8
            valid_expert_losses = []
            valid_expert_indices = []
            for i, expert_loss in enumerate(expert_losses):
                if expert_loss.detach() >= total_loss.detach():
                    valid_expert_losses.append(expert_loss)
                    valid_expert_indices.append(i)
            if len(valid_expert_losses) > 0:
                valid_losses_tensor = torch.stack(valid_expert_losses)
                inv_sqrt_losses = torch.pow(valid_losses_tensor.detach() + epsilon, -1)
                sum_inv_sqrt = torch.sum(inv_sqrt_losses)
                weights = inv_sqrt_losses / sum_inv_sqrt
                for i, (weight, expert_loss) in enumerate(zip(weights, valid_expert_losses)):
                    total_loss += weight * expert_loss
            if self.aux_loss_alpha > 0:
                total_loss += return_dict["aux_loss"]
            if self.decorr_strength > 0:
                decorr_loss = self.calculate_decorrelation_loss()
                total_loss += self.decorr_strength * decorr_loss
        return total_loss

class ExponentialCrossNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 num_cross_layers=3,
                 net_dropout=0.1,):
        super(ExponentialCrossNetwork, self).__init__()
        self.num_cross_layers = num_cross_layers
        self.batch_norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        for i in range(num_cross_layers):
            self.w.append(nn.Linear(input_dim, input_dim, bias=False))
            self.b.append(nn.Parameter(torch.zeros((input_dim,))))
            self.batch_norm.append(nn.BatchNorm1d(input_dim))
            if net_dropout > 0:
                self.dropout.append(nn.Dropout(net_dropout))
            nn.init.uniform_(self.b[i].data)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for i in range(self.num_cross_layers):
            H = self.w[i](x)
            if len(self.batch_norm) > i:
                H = self.batch_norm[i](H)
            x = x * (H + self.b[i]) + x
            if len(self.dropout) > i:
                x = self.dropout[i](x)
        return self.fc(x), x


class ProductNetwork(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim=16,
                 hidden_units=[400, 400, 400],
                 batch_norm=False,
                 net_dropout=0.1,
                 hidden_activations="ReLU"):
        super(ProductNetwork, self).__init__()
        self.inner_product_layer = InnerProductInteraction(num_fields, output="inner_product")
        input_dim = int(num_fields * (num_fields - 1) / 2) + num_fields * embedding_dim
        self.dnn = MLP_Block(input_dim=input_dim,
                             output_dim=None,
                             hidden_units=hidden_units,
                             hidden_activations=hidden_activations,
                             output_activation=None,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)
        self.fc = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        inner_products = self.inner_product_layer(x)
        dense_input = torch.cat([x.flatten(start_dim=1), inner_products], dim=1)
        x = self.dnn(dense_input)
        return self.fc(x), x


class FinalNet(nn.Module):
    def __init__(self, num_fields, input_dim, hidden_units=[], hidden_activations=None,
                 dropout_rates=0.1, batch_norm=True):
        # Replacement of MLP_Block, identical when order=1
        super(FinalNet, self).__init__()
        if type(dropout_rates) != list:
            dropout_rates = [dropout_rates] * len(hidden_units)
        if type(hidden_activations) != list:
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.layer = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.activation = nn.ModuleList()
        hidden_units = [input_dim] + hidden_units
        self.field_gate = FinalGate(num_fields)
        for idx in range(len(hidden_units) - 1):
            self.layer.append(FinalLinear(hidden_units[idx], hidden_units[idx + 1]))
            if batch_norm:
                self.norm.append(nn.BatchNorm1d(hidden_units[idx + 1]))
            if dropout_rates[idx] > 0:
                self.dropout.append(nn.Dropout(dropout_rates[idx]))
            self.activation.append(get_activation(hidden_activations[idx]))
        self.fc = nn.Linear(hidden_units[-1], 1)

    def forward(self, X):
        X = self.field_gate(X)
        X = X.flatten(start_dim=1)
        X_i = X
        for i in range(len(self.layer)):
            X_i = self.layer[i](X_i)
            if len(self.norm) > i:
                X_i = self.norm[i](X_i)
            if self.activation[i] is not None:
                X_i = self.activation[i](X_i)
            if len(self.dropout) > i:
                X_i = self.dropout[i](X_i)
        return self.fc(X_i), X_i


class FinalLinear(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        """ A replacement of nn.Linear to enhance multiplicative feature interactions.
            `residual_type="concat"` uses the same number of parameters as nn.Linear
            while `residual_type="sum"` doubles the number of parameters.
        """
        super(FinalLinear, self).__init__()
        assert output_dim % 2 == 0, "output_dim should be divisible by 2."
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x):
        h = self.linear(x)
        h2, h1 = torch.chunk(h, chunks=2, dim=-1)
        h = torch.cat([h2, h1 * h2], dim=-1)
        return h


class FinalGate(nn.Module):
    def __init__(self, num_fields):
        super(FinalGate, self).__init__()
        self.linear = nn.Linear(num_fields, num_fields)

    def reset_custom_params(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.ones_(self.linear.bias)

    def forward(self, feature_emb):
        gates = self.linear(feature_emb.transpose(1, 2)).transpose(1, 2)
        out = torch.cat([feature_emb, feature_emb * gates], dim=1)  # b x 2f x d

        return out
