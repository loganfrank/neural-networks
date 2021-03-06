## PyTorch imports
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageNet(Dataset):

    subsets = {
        'imagenet' : [],

        'spv' : ['n02701002', 'n02814533', 'n02930766', 'n03100240', 'n03594945', 'n03670208', 'n03770679', 'n03777568', 
                 'n04037443', 'n04285008', 'n02704792', 'n03345487', 'n03417042', 'n03796401', 'n03977966', 'n03930630', 
                 'n04461696', 'n04467665', 'n03444034', 'n03785016', 'n04252225', 'n03272562', 'n04310018', 'n03384352', 
                 'n03478589', 'n04252077', 'n04389033', 'n04065272', 'n04335435', 'n04465501'],

        'furniture' : ['n02791124', 'n03376595', 'n04099969', 'n04429376', 'n03891251', 'n04344873', 'n04447861', 'n02804414', 
                       'n03125729', 'n03131574', 'n02870880', 'n03016953', 'n03018349', 'n03742115', 'n03179701', 'n03201208', 
                       'n03290653', 'n03337140', 'n03388549', 'n04380533', 'n04550184'],

        'animal' : ['n01440764', 'n01443537', 'n02526121', 'n02606052', 'n02607072', 'n02643566', 'n02655020', 'n02640242', 
                    'n02641379', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n02514041', 'n02536864', 
                    'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 
                    'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 
                    'n01616318', 'n01622779', 'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 
                    'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672', 'n02018795', 
                    'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849', 
                    'n02013706', 'n02018207', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n01860187', 
                    'n02017213', 'n02051845', 'n02056570', 'n02058221', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 
                    'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 
                    'n01669191', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333', 
                    'n01693334', 'n01694178', 'n01695060', 'n01675722', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 
                    'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01739381', 'n01740131', 'n01737021', 
                    'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 
                    'n02119789', 'n02119022', 'n02120079', 'n02120505', 'n02114367', 'n02114548', 'n02114712', 'n02114855', 
                    'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02137549', 'n02138441', 'n02125311', 'n02127052', 
                    'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136', 'n02133161', 
                    'n02134084', 'n02134418', 'n02441942', 'n02442845', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 
                    'n02443114', 'n02509815', 'n02510455', 'n02364673', 'n02342885', 'n02346627', 'n02356798', 'n02361337', 
                    'n02363005', 'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02325366', 'n02328150', 'n02326432', 
                    'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521', 'n02403003', 'n02408429', 
                    'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 
                    'n02437616', 'n02454379', 'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 
                    'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219', 
                    'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02492035', 'n02497673', 'n02500267', 'n02504013', 
                    'n02504458', 'n01871265', 'n01872401', 'n01873310', 'n01877812', 'n01882714', 'n01883070', 'n01768244', 
                    'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 
                    'n01776313', 'n01784675', 'n01978287', 'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 
                    'n01985128', 'n01986214', 'n01990800', 'n02264363', 'n02165105', 'n02165456', 'n02167151', 'n02168699', 
                    'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 
                    'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02268443', 'n02268853', 
                    'n02276258', 'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n01910747', 'n01914609', 
                    'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 
                    'n01968897', 'n02317335', 'n02319095', 'n02321529', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 
                    'n01806143', 'n01806567', 'n01807496', 'n02087046', 'n02085620', 'n02085782', 'n02085936', 'n02086079', 
                    'n02086240', 'n02086646', 'n02086910', 'n02092339', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 
                    'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091244', 
                    'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02097209', 'n02097047', 'n02097130', 'n02093256', 
                    'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 
                    'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 
                    'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02102177', 'n02102040', 
                    'n02101388', 'n02101556', 'n02102318', 'n02102480', 'n02102973', 'n02099267', 'n02099429', 'n02099601', 
                    'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02087394', 
                    'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02104029', 'n02104365', 'n02107142', 'n02107312', 
                    'n02110627', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 
                    'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02108089', 'n02108422', 'n02108551', 
                    'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02111889', 'n02112018', 
                    'n02112137', 'n02112350', 'n02110341', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 
                    'n02112706', 'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02123045', 
                    'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02091032', 'n02091134']
    }

    normalization_values = {
        'imagenet' : {
            -1   : {'mean' : [], 'std' : []}
        },
        'furniture' : {
            -1   : {'mean' : [], 'std' : []}
        },
        'animal' : {
            -1   : {'mean' : [], 'std' : []}
        }
    }

    def __init__(self, subset, dataframe, image_directory, phase, transform=None, visualize=False):
        """
        Still under construction.
        """
        self.classes_index = classes_index
        self.classes = torch.unique(self.classes_index)
        self.classes_names = classes_names
        self.classes_to_index = classes_to_index
        self.instances = instances
        self.phase = phase
        self.transform = transform
        self.visualize = visualize

    def num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, index):
        image = self.instances[index]
        image = Image.fromarray(image)
        label = self.classes_index[index].numpy().astype(np.int64)

        if self.transform:
            image = self.transform(image)

        if self.visualize:
            image = np.asarray(image)
            fig = plt.figure()
            plt.imshow(image)
            plt.title(self.classes_names[label])
            plt.show()

        return (image, label)

    @staticmethod
    def create_train_and_test(subset, dataframe, image_directory, train_transform, test_transform, visualize=False):
        """
        Still under construction.
        """
        # For some reason train.targets and test.targets are torch tensors in MNIST, but python lists for CIFAR10
        train.targets = torch.tensor(train.targets)
        test.targets = torch.tensor(test.targets)

        ordered_labels, indexes = torch.sort(train.targets)
        ordered_data = train.data[indexes]

        uniques, counts = torch.unique(test.targets, return_counts=True)

        train_targets = torch.empty(0, dtype=torch.int64)
        train_instances = np.empty([0, 32, 32, 3], dtype=np.uint8)
        val_targets = torch.empty(0, dtype=torch.int64)
        val_instances = np.empty([0, 32, 32, 3], dtype=np.uint8)

        for num in uniques:
            num = num.item()
            data = ordered_data[ordered_labels == num]
            labels = ordered_labels[ordered_labels == num]

            train_data, val_data = np.split(data, [len(data) - counts[num]])
            train_labels, val_labels = torch.split(labels, [len(data) - counts[num], counts[num]])

            train_targets = torch.cat([train_targets, train_labels])
            train_instances = np.concatenate([train_instances, train_data])
            val_targets = torch.cat([val_targets, val_labels])
            val_instances = np.concatenate([val_instances, val_data])

        del train_data, train_labels, val_data, val_labels

        train_dataset = CIFAR10(train_targets, train.classes, train.class_to_idx, train_instances, phase='train', transform=train.transform, visualize=visualize)
        val_dataset = CIFAR10(val_targets, train.classes, train.class_to_idx, val_instances, phase='val', transform=test.transform, visualize=visualize)

        return train_dataset, val_dataset
    
    @staticmethod
    def create_validation(subset, dataframe, image_directory, test_transform, visualize=False):
        """
        Still under construction.
        """
        test_dataset = CIFAR10(test.targets, test.classes, test.class_to_idx, test.data, phase='test', transform=test.transform, visualize=visualize)
        return test_dataset