from torch import nn
import torchvision.models


class CombineBaseModelWithClassifier(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)

        return x


class MergeMultiTaskModel(nn.Module):
    def __init__(self, models):
        """ All model used to merge need to have 2 attribute .backbone and .classifier
            Where .backbone of all models are similar.
        """
        super().__init__()
        self.num_models = len(models)
        self.backbone = models[0].backbone
        self.classifier = []
        for model_idx in range(self.num_models):
            self.classifier.append(models[model_idx].classifier)

    def forward(self, x):
        x = self.backbone(x)
        outputs = self.classifier[0](x)
        for model_idx in range(1, self.num_models):
            outputs = torch.cat((outputs, self.classifier[model_idx](x)), dim=1)

        return outputs


class ModelsGenerator:
    def __init__(self, base_model):
        self.base_model = base_model

        self.classification_layer_name = None
        if hasattr(self.base_model, 'classifier'):
            self.classification_layer_name = 'classifier'
        elif hasattr(self.base_model, 'fc'):
            self.classification_layer_name = 'fc'
        else:
            raise ValueError(type(self).__name__ + ": This model has not contain fc and classifier layer")

    def create_new_classifier(self, num_classes):
        return nn.Sequential

    def create_single_model(self, num_classes):
        setattr(self.base_model, self.classification_layer_name, self.create_new_classifier(num_classes))
        return self.base_model

    def create_multitask_model(self, num_models, num_cls_per_model):
        if type(num_cls_per_model) is int:
            num_cls_per_model = [num_cls_per_model] * num_models

        # Delete old classification layer
        setattr(self.base_model, self.classification_layer_name, nn.Dropout(p=0))

        classifier, models = [0] * num_models, [0] * num_models
        for model_idx in range(num_models):
            classifier[model_idx] = self.create_new_classifier(num_cls_per_model[model_idx])
            models[model_idx] = CombineBaseModelWithClassifier(self.base_model, classifier[model_idx])

        return models


class EfficientNet(ModelsGenerator):
    def __init__(self, version):
        super().__init__(getattr(torchvision.models, 'efficientnet_b' + str(version))(pretrained=True))
        self.dropout_p = self.base_model.classifier[0].p
        self.in_features = self.base_model.classifier[1].in_features

    def create_new_classifier(self, num_classes):
        return nn.Sequential(nn.Dropout(p=self.dropout_p, inplace=True),
                             nn.Linear(self.in_features, num_classes),
                             nn.Sigmoid())


class ResNext101_32x8d(ModelsGenerator):
    def __init__(self):
        resnext101 = torchvision.models.resnext101_32x8d(pretrained=True)
        super().__init__(resnext101)
        self.in_features = self.base_model.fc.in_features

    def create_new_classifier(self, num_classes):
        return nn.Sequential(nn.Linear(self.in_features, num_classes),
                             nn.Sigmoid())


class RegNet_x_8gf(ModelsGenerator):
    def __init__(self):
        regnetx8gf = torchvision.models.regnet_x_8gf(pretrained=True)
        super().__init__(regnetx8gf)
        self.in_features = self.base_model.fc.in_features

    def create_new_classifier(self, num_classes):
        return nn.Sequential(nn.Linear(self.in_features, num_classes),
                             nn.Sigmoid())


class VGG_bn(ModelsGenerator):
    def __init__(self, version):
        vgg_bn = getattr(torchvision.models, 'vgg' + str(version) + '_bn')(pretrained=True)
        super().__init__(vgg_bn)
        self.dropout_p = self.base_model.classifier[2].p

    def create_new_classifier(self, num_classes):
        return nn.Sequential(nn.Linear(512 * 7 * 7, 4096),
                             nn.ReLU(True),
                             nn.Dropout(p=self.dropout_p),
                             nn.Linear(4096, 4096),
                             nn.ReLU(True),
                             nn.Dropout(p=self.dropout_p),
                             nn.Linear(4096, num_classes),
                             nn.Sigmoid())


if __name__ == '__main__':
    from torchvision.io import read_image
    from torchvision.transforms import Resize
    import torch

    image1 = read_image(r'D:\Machine Learning Project\5kCompliance\dataset\train\images\2.jpg').float()
    image1 = Resize((300,300))(image1)
    image2 = read_image(r'D:\Machine Learning Project\5kCompliance\dataset\train\images\21.jpg').float()
    image2 = Resize((300, 300))(image1)
    image = torch.stack((image1, image2), dim=0)
    print(image.size())

    models = VGG_bn(version=16).create_multitask_model(2, 1)
    model = MergeMultiTaskModel(models)

    print(model(image))
