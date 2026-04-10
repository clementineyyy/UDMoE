import torch
from torch import nn
import torch.nn.functional as F
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbedding, MLP_Block
from fuxictr.pytorch.torch_utils import get_activation


class Homo(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="Homo",
                 gpu=-1,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 topk=2,
                 dnn_hidden_units=[400, 400],
                 dnn_netdropout=0.1,
                 dnn_batchnorm=False,
                 gate_hidden_units=[400, 400],
                 gate_dropout_rate=0.1,
                 aux_loss_alpha=0.1,
                 num_specific_experts=4,
                 net_dropout=0.1,
                 batch_norm=False,
                 hidden_activations="ReLU",
                 decorr_strength=0.1,
                 decorr_type='feature',
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(Homo, self).__init__(feature_map,
                                     model_id=model_id,
                                     gpu=gpu,
                                     embedding_regularizer=embedding_regularizer,
                                     net_regularizer=net_regularizer,
                                     **kwargs)
        self.topk = topk
        self.embedding_layer = FeatureEmbedding(feature_map, embedding_dim)
        if feature_map.dataset_id == "taobaoad_x1":
            self.userid_index = feature_map.get_column_index("userid")
        else:
            self.userid_index = feature_map.get_column_index("user_id")
        input_dim = feature_map.sum_emb_out_dim()
        self.num_specific_experts = num_specific_experts
        self.embedding_dim = embedding_dim
        self.expert_centroids = nn.Parameter(torch.empty(self.num_specific_experts, embedding_dim))
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
        self.specific_experts = nn.ModuleList([MLP_Block(input_dim=input_dim,
                                              output_dim=1,
                                              hidden_units=dnn_hidden_units,
                                              hidden_activations=hidden_activations,
                                              output_activation=None,
                                              dropout_rates=dnn_netdropout,
                                              batch_norm=dnn_batchnorm, ) for _ in range(self.num_specific_experts)])
        self.user_gate = MLP_Block(input_dim=embedding_dim,
                                   output_dim=self.num_specific_experts,
                                   hidden_units=gate_hidden_units,
                                   hidden_activations=hidden_activations,
                                   output_activation=None,
                                   dropout_rates=net_dropout,
                                   batch_norm=batch_norm)
        self.gate_dropout_rate = gate_dropout_rate
        self.aux_loss_alpha = aux_loss_alpha
        self.dropout = nn.Dropout(0.5)
        self.softmax = get_activation("softmax")
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def calculate_decorrelation_loss(self):
        if self.decorr_strength <= 0 or self.num_specific_experts < 2:
            return 0.0
        if self.decorr_type == 'feature':
            if not hasattr(self, 'last_feature_emb') or self.last_feature_emb is None:
                return 0.0
            weights = []
            for expert in self.specific_experts:
                first_layer = list(expert.mlp)[0]
                feature_rep = first_layer(self.last_feature_emb)
                weights.append(feature_rep)  # [batch_size, hidden_dim]
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
        user_gates, normalized_gates, topk_indices = self.calculate_topk_scores(user_gates, userid_emb)
        specific_logits_tensor = None
        for i in range(self.num_specific_experts):
            logit = self.specific_experts[i](feature_emb.flatten(start_dim=1))
            if specific_logits_tensor is None:
                specific_logits_tensor = logit  # B × 1
            else:
                specific_logits_tensor = torch.cat([specific_logits_tensor, logit], dim=1)  # B × num_experts
        specific_logits = [specific_logits_tensor[:, i] for i in range(self.num_specific_experts)]
        user_logit = torch.sum(user_gates * specific_logits_tensor, dim=1)
        y_pred = self.output_activation(user_logit)

        if self.training and self.aux_loss_alpha > 0:
            scores_aux = user_gates
            topk_idx = topk_indices
            mask_ce = F.one_hot(topk_idx, num_classes=self.num_specific_experts).sum(dim=1)
            ce = mask_ce.float().mean(dim=0)
            Pi = scores_aux.mean(dim=0)
            fi = ce * self.num_specific_experts
            aux_loss = (Pi * fi).sum() * self.aux_loss_alpha
        else:
            aux_loss = torch.tensor(0.0).to(user_gates.device)
        return_dict = {"y_pred": y_pred, "aux_loss": aux_loss, "gate_weights": user_gates}
        for i in range(self.num_specific_experts):
            return_dict[f"y{i + 1}"] = torch.sigmoid(specific_logits[i])
        return return_dict

    def update_expert_centroids(self):
        with torch.no_grad():
            for i, expert in enumerate(self.specific_experts):
                if hasattr(expert, 'dnn') and hasattr(expert.dnn, 'mlp'):
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
                else:
                    zero_tensor = torch.zeros(1, self.embedding_dim, 1).to(self.expert_centroids.device)
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
        total_loss = self.loss_fn(y_pred.unsqueeze(-1), y_true, reduction='mean')
        if self.training:
            expert_outputs = []
            for i in range(self.num_specific_experts):
                expert_outputs.append(return_dict[f"y{i + 1}"])
            expert_losses = []
            for i, y_expert in enumerate(expert_outputs):
                y_expert = torch.clamp(y_expert, min=1e-6, max=1 - 1e-6)
                expert_loss = self.loss_fn(y_expert.unsqueeze(-1), y_true, reduction='mean')
                expert_losses.append(expert_loss)
            epsilon = 1e-8
            losses = torch.stack(expert_losses)
            inv_sqrt_losses = torch.pow(losses.detach() + epsilon, -1)
            sum_inv_sqrt = torch.sum(inv_sqrt_losses)
            weights = inv_sqrt_losses / sum_inv_sqrt
            for i, (weight, expert_loss) in enumerate(zip(weights, expert_losses)):
                total_loss += weight * expert_loss
            if self.aux_loss_alpha > 0:
                total_loss += return_dict["aux_loss"]
            if self.decorr_strength > 0:
                decorr_loss = self.calculate_decorrelation_loss()
                total_loss += self.decorr_strength * decorr_loss

        return total_loss

