from .cifar import ModifiedResNet18
from .resnet import ResNet18
from .resnet import ResNet34
from .resnet import ResNet50
from .resnet import ResNet101
from .efficientnet import EfficientNet

NETWORKS = ['resnet18_modified', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 
            'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'efficientnetb8']

CIFAR10_NETWORKS = ['resnet18_modified', 'efficientnetb0']

IMAGENET_NETWORKS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnetb0''efficientnetb1', 'efficientnetb2', 
                     'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7', 'efficientnetb8']

def load_network(network, num_classes, batch_norms=None):
    if network == 'resnet18_modified':
        network = ModifiedResNet18(num_classes, batch_norms)
    elif network == 'resnet18':
        network = ResNet18(num_classes, batch_norms)
    elif network == 'resnet34':
        network = ResNet34(num_classes, batch_norms)
    elif network == 'resnet50':
        network = ResNet50(num_classes, batch_norms)
    elif network == 'resnet101':
        network = ResNet101(num_classes, batch_norms)
    elif 'efficientnet' in network:
        version = network[-2:]
        network = f'efficientnet-{version}'
        network = EfficientNet.from_name(network)
    else:
        raise('Unknown network')
    return network

def retrieve_network(dataset, network, num_classes, batch_norms=None):
    if dataset == 'cifar10':
        assert network in CIFAR10_NETWORKS
    elif dataset == 'furniture':
        assert network in IMAGENET_NETWORKS
    else:
        raise('Not yet implemented')

    network = load_network(network, num_classes, batch_norms)
    return network