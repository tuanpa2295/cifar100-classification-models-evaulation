from models.resnet import ResNet50, ResNet101, ResNet152
from models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from models.googlenet import google_net
from models.efficientnet import efficient_net

models_mappings = {'ResNet50': ResNet50(), 'ResNet101': ResNet101(),
                     'ResNet152': ResNet152(), 
                     'VGG11': vgg11(), 
                     'VGG11_Batch_Normalization': vgg11_bn(), 'VGG13': vgg13(), 
                     'VGG13_Batch_Normalization': vgg13_bn(), 'VGG16': vgg16(), 
                     'VGG16_Batch_Normalization': vgg16_bn(), 
                     'VGG19': vgg19(), 'VGG19_Batch_Normalization': vgg19_bn(),
                     'GoogleNet': google_net(),
                     'EfficientNet': efficient_net()}

def use_model(model_name='ResNet50'):
    return models_mappings.get(model_name); 