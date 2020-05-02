import torch
import torch.nn as nn

from typing import Sequence, Tuple


class MLP(nn.Module):
    def __init__(self, shapes: Sequence[int]):
        super().__init__()
        blocks = []

        for i in range(len(shapes) - 1):
            blocks.append(nn.Linear(shapes[i], shapes[i + 1]))
            blocks.append(nn.ReLU())  # type: ignore

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        n_batch, input_dim, n_relation = x.size()
        return self.blocks(x.view(-1, input_dim)).view(n_batch, -1, n_relation)


class AttributePredictor(nn.Module):
    def __init__(self, shapes: Sequence[int]):
        super().__init__()
        blocks = []

        for i in range(len(shapes) - 1):
            blocks.append(nn.Linear(shapes[i], shapes[i + 1]))
            blocks.append(nn.ReLU())  # type: ignore

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


def reform_relation_triplet(
        objects: torch.Tensor,
        triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    assert objects.size(2) == triplet[0].size(1)
    assert objects.size(2) == triplet[1].size(1)

    return torch.cat([
        torch.matmul(objects, triplet[0]),
        torch.matmul(objects, triplet[1]), triplet[2]
    ],
                     dim=1)


def aggregation(objects: torch.Tensor, externals: torch.Tensor,
                effects: torch.Tensor,
                triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ) -> torch.Tensor:
    return torch.cat([
        objects, externals,
        torch.matmul(effects, triplet[0].transpose(dim0=1, dim1=2))
    ],
                     dim=1)


def aggregate_prediction(prediction: torch.Tensor) -> torch.Tensor:
    return prediction.sum(dim=2, keepdim=False)


class InteractionNetwork(nn.Module):
    def __init__(self, config: dict):
        super(InteractionNetwork, self).__init__()
        relation_module_config = config["relation"]
        object_module_config = config["object"]

        self.relation_module = MLP(relation_module_config["shapes"])
        self.object_module = MLP(object_module_config["shapes"])

        self.predict_attribute = False

        if config.get("attribute") is not None:
            attribute_module_config = config["attribute"]
            self.attribute_module = AttributePredictor(
                attribute_module_config["shapes"])
            self.predict_attribute = True

    def forward(self, objects, externals, triplet):
        relation_input = reform_relation_triplet(objects, triplet)

        effects = self.relation_module(relation_input)
        aggregated_input = aggregation(objects, externals, effects, triplet)

        prediction = self.object_module(aggregated_input)

        if self.predict_attribute:
            aggregated_prediction = aggregate_prediction(prediction)
            attribute = self.attribute_module(aggregated_prediction)
            return prediction, attribute
        else:
            return prediction


def get_model(config: dict):
    model_config = config["model"]
    model_name = model_config["name"]

    if model_name == "IN":
        return InteractionNetwork(model_config["params"])
    else:
        raise NotImplementedError
