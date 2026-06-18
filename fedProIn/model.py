from torchvision.models import ResNet18_Weights
import torch
from bhaang.Medical_imaging.model import Model_master
from configs import num_prototypes_per_class, num_classes

class model_with_trainable_prototypes(torch.nn.Module):
    def __init__(self, model, num_classes=num_classes, num_prototypes_per_class=5, feature_dim=512):
        super(model_with_trainable_prototypes, self).__init__()
        self.model = model
        self.model.fc = torch.nn.Identity()
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.feature_dim = feature_dim
        
        # Initialize prototypes
        self.prototypes = torch.nn.Parameter(torch.randn(num_classes, num_prototypes_per_class, feature_dim))

    def forward(self, x):
        features = self.model(x)
        return features, self.prototypes

Resnet18_obj = Model_master('torch')
Resnet18_model = Resnet18_obj.get_model('pytorch/vision', 'resnet18', weights=ResNet18_Weights.DEFAULT)

sample_input = torch.randn(1, 3, 128, 128)

Resnet18_model = model_with_trainable_prototypes(Resnet18_model, num_classes=num_classes, num_prototypes_per_class= num_prototypes_per_class, feature_dim=512)
Resnet18_obj.model = Resnet18_model

Resnet18_obj.display_model_layers(sample_input)