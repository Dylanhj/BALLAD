
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from randaug import *
import os

def calculate_momentum_weight(momentum_loss, epoch):

    momentum_weight = ((momentum_loss[epoch-1]-torch.mean(momentum_loss[epoch-1,:]))/torch.std(momentum_loss[epoch-1,:]))
    momentum_weight = ((momentum_weight/torch.max(torch.abs(momentum_weight[:])))/2+1/2).detach().cpu().numpy()

    return momentum_weight

class memoboosted_Imagenet(Dataset):
    def __init__(self, root, txt, rank_k, class_num = 1000):
        # super().__init__(**kwds)
        self.img_path = []
        self.labels = []
        self.class_num = class_num
        self.idxsNumPerClass = [0] * class_num
        self.rank_k = rank_k

        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                label = int(line.split()[1])
                self.labels.append(label)
                # self.targets.append(label)
                self.idxsNumPerClass[label] += 1
                idx_num += 1

        # self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(100)]
        # self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]

        # self.momentum_weight=np.empty(len(sublist))
        self.momentum_weight[:]=0

    def update_momentum_weight(self, momentum_loss, epoch):
        momentum_weight_norm = calculate_momentum_weight(momentum_loss, epoch)
        self.momentum_weight = momentum_weight_norm

    def __getitem__(self, idx):
        # img = self.data[idx]
        # img = Image.fromarray(img).convert('RGB')
        path = self.img_path[idx]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        label = self.labels[idx]

        if self.rand_k == 1:
            # We remove the rand operation when adopt small aug 
            min_strength = 10 # training stability
            memo_boosted_aug = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=BICUBIC),
                        transforms.RandomHorizontalFlip(p=0.5),
                        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        # transforms.RandomGrayscale(p=0.2),
                        RandAugment_prob(self.rand_k, min_strength + (self.args.rand_strength - min_strength)*self.momentum_weight[idx], 1.0*self.momentum_weight[idx]),
                        transforms.ToTensor(),
                    ])
        else:
            min_strength = 5 # training stability
            memo_boosted_aug = transforms.Compose([
                        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=BICUBIC),
                        transforms.RandomHorizontalFlip(p=0.5),
                        # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                        # transforms.RandomGrayscale(p=0.2),
                        RandAugment_prob(self.rand_k, min_strength + (self.args.rand_strength - min_strength)*self.momentum_weight[idx]*np.random.rand(1), 1.0*self.momentum_weight[idx]),
                        transforms.ToTensor(),
                    ])

        # imgs = [memo_boosted_aug(img), memo_boosted_aug(img)]
        img = memo_boosted_aug(img)

        return img, label, idx
