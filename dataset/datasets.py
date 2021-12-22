from torch.utils.data import Dataset
from utils.util import *
import torch,random,bisect
import numpy as np

class OneDataset(Dataset):
    def __init__(self, pth2label, transform, target=False):
        self.pth2label = pth2label
        self.transform = transform
        self.target = target

    def __getitem__(self, index):
        item = self.pth2label[index]
        img = load_img(item[0])
        img = self.transform(img)

        if self.target:
            return img,int(item[1]),item[0]  # data,label,pth

        return img,int(item[1])

    def __len__(self):
        return len(self.pth2label)

class AlignDataset(Dataset):
    def __init__(self,pth2labels,transforms,config):
        self.config = config
        self.p2ls = [self.split_class(pth2labels[0]),self.split_class(pth2labels[1])]
        self.tfs = [transforms[0],transforms[1]]
        self.bts = self.limit_bts()

    def __len__(self):
        return self.config["class_num"]

    def __getitem__(self, index):
        data = [None]*5
        for i,d in enumerate(["s","t"]):
            cur_pths = self.p2ls[i]
            inds = random.choices(range(len(cur_pths[index])),k=max(self.bts[0][index],self.bts[1][index]))
            pths = np.asarray([cur_pths[index][ind][0] for ind in inds],dtype=np.str)
            label = torch.tensor([int(cur_pths[index][ind][1]) for ind in inds])
            imgs = torch.stack([self.tfs[i](load_img(p)) for p in pths],0)

            if d == "s":
                data[0] = imgs
                data[1] = label
            elif d == "t":
                data[2] = imgs
                data[3] = pths
                data[4] = label
        return data[0],data[1],data[2],data[3],data[4]

    def split_class(self,pth2label):

        results = {}
        for c in range(self.config["class_num"]):
            results[c] = pth2label[np.where(pth2label[:,1]==str(c))]
        return results

    def limit_bts(self):
        results = [None]*2
        bts = [self.config["s_bs"],self.config["t_bs"]]
        for i,d in enumerate(["s","t"]):
            results[i] = {}
            for c in range(self.config["class_num"]):
                results[i][c] = min(bts[i],len(self.p2ls[i][c]))
        return results
    def consturct(self,dict):
        results={}
        keys = np.asarray(list(dict.keys()),dtype=str)
        values = np.asarray(list(dict.values()),dtype=str)
        for c in range(self.config["class_num"]):
            index_c = np.where(values==str(c))
            keys_c = np.asarray(keys[index_c]).reshape(-1,1)
            values_c = np.asarray(values[index_c]).reshape(-1,1)
            results[c] = np.concatenate((keys_c,values_c),1)
        self.p2ls[1] = results
        # print(results)

class ClassDataset(Dataset):
    def __init__(self,pth2label,transform,config,align_bs):
        self.config = config
        self.pth2label = self.split_class(pth2label)
        self.transform = transform
        self.align_bs = align_bs
        # self.bs = self.limit_bts()

    def split_class(self,pth2lable):
        result = {}
        for c in range(self.config["class_num"]):
            result[c] = pth2lable[np.where(pth2lable[:,1]==str(c))]
        return result
    def limit_bts(self):
        result = {}
        bs = self.align_bs
        for c in range(self.config["class_num"]):
            result[c] = min(bs,len(self.pth2label[c]))
        return result

    def __len__(self):
        return self.config["class_num"]

    def __getitem__(self, index):
        inds = random.choices(range(len(self.pth2label[index])),k=self.align_bs)
        pths = np.asarray([self.pth2label[index][ind][0] for ind in inds],dtype=np.str)
        labels = torch.tensor([int(self.pth2label[index][ind][1]) for ind in inds])
        imgs = torch.stack([self.transform(load_img(p)) for p in pths],0)

        return imgs,labels

    def construct(self, pth2label):
        results = {}
        keys = np.asarray(list(pth2label.keys()), dtype=str)
        values = np.asarray(list(pth2label.values()), dtype=str)
        for c in range(self.config["class_num"]):
            index_c = np.where(values==str(c))
            keys_c = np.asarray(keys[index_c]).reshape(-1,1)
            values_c = np.asarray(values[index_c]).reshape(-1,1)
            results[c] = np.concatenate((keys_c,values_c),1)
        self.pth2label = results

class DgAlighDataset(Dataset):
    def __init__(self,*datasets,choose_num=3):
        self.datasets = datasets
        self.all = list(range(len(self.datasets)))
        self.choose_num = choose_num

    def __len__(self):
        return self.datasets[0].__len__()

    def __getitem__(self, index):
        datas,labels = [],[]
        for i in range(self.choose_num):
            data,label = self.datasets[self.all[i]].__getitem__(index)
            datas.append(data)
            labels.append(label)
        return datas,labels

    def construct(self,pths2labels):
        length = len(pths2labels)
        for i in range(length):
            self.datasets[i+1].construct(pths2labels[i])


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)              

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx