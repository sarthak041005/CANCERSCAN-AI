import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


class CancerCNN(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        n = self.backbone.fc.in_features
        # Linear layer matching the trained lung checkpoint
        self.backbone.fc = nn.Linear(n, num_classes)

    def forward(self, x):
        return self.backbone(x)

    @property
    def feature_layer(self):
        return self.backbone.layer4[-1]

    def load_state_dict(self, state_dict, strict=True):
        # Allow loading state dict whether keys have 'backbone.' prefix or not
        current_keys = set(self.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        if not ckpt_keys.issubset(current_keys):
            # Check if checkpoint keys are missing 'backbone.' prefix
            prefixed = {}
            for k, v in state_dict.items():
                if k.startswith("fc."):
                    prefixed[f"backbone.{k}"] = v
                elif not k.startswith("backbone."):
                    prefixed[f"backbone.{k}"] = v
                else:
                    prefixed[k] = v
            try:
                return super().load_state_dict(prefixed, strict=strict)
            except Exception:
                pass
        return super().load_state_dict(state_dict, strict=False)
