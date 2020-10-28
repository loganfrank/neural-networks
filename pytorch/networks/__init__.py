from .modified_resnet import ModifiedResNet18
from .modified_resnet import ModifiedResNet34
from .modified_resnet import ModifiedResNet50
from .modified_resnet import ModifiedResNet101
from .modified_resnet import ModifiedResNet152
from .resnet import ResNet18
from .resnet import ResNet34
from .resnet import ResNet50
from .resnet import ResNet101
from .efficientnet import EfficientNet

NETWORKS = ['resnet18_modified', 'resnet34_modified', 'resnet50_modified', 'resnet101_modified', 'resnet152_modified', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 
            'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'efficientnetb8']

CIFAR10_NETWORKS = ['resnet18_modified', 'resnet34_modified', 'resnet50_modified', 'resnet101_modified', 'resnet152_modified', 'efficientnetb0']

IMAGENET_NETWORKS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnetb0''efficientnetb1', 'efficientnetb2', 
                     'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'efficientnetb8']

def load_network(network):
    if network == 'resnet18_modified':
        network = ModifiedResNet18
    elif network == 'resnet34_modified':
        network = ModifiedResNet34
    elif network == 'resnet50_modified':
        network = ModifiedResNet50
    elif network == 'resnet101_modified':
        network = ModifiedResNet101
    elif network == 'resnet152_modified':
        network = ModifiedResNet152
    elif network == 'resnet18':
        network = ResNet18
    elif network == 'resnet34':
        network = ResNet34
    elif network == 'resnet50':
        network = ResNet50
    elif network == 'resnet101':
        network = ResNet101
    elif 'efficientnet' in network:
        ## WORK IN PROGRESS
        version = network[-2:]
        network = f'efficientnet-{version}'
        network = EfficientNet.from_name(network)
    else:
        raise('Unknown network')
    return network

def retrieve_network(dataset, network):
    if dataset == 'cifar10':
        assert network in CIFAR10_NETWORKS
    elif dataset == 'furniture':
        assert network in IMAGENET_NETWORKS
    else:
        raise('Not yet implemented')

    network = load_network(network)
    return network