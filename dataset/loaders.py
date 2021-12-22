from torch.utils.data import DataLoader,BatchSampler,RandomSampler
import torch
import numpy as np

class ClassAlignLoader(object):
    def __init__(self,dataset,num_selected_classes,num_workers,collate_fn):
        self.dataset = dataset
        self.num_selected_classes = num_selected_classes
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        batch_sampler = BatchSampler(RandomSampler(self.dataset), self.num_selected_classes, drop_last=False)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,batch_sampler=batch_sampler,
                                                      collate_fn=self.collate_fn,num_workers=self.num_workers)

    def construct(self,dict):
        self.dataset.consturct(dict)
        batch_sampler = BatchSampler(RandomSampler(self.dataset), self.num_selected_classes, drop_last=False)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_sampler=batch_sampler,
                                                      collate_fn=self.collate_fn,
                                                      num_workers=self.num_workers)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        dataset_len = 0.0
        for c in range(self.dataset.config["class_num"]):
            c_len = max([len(self.dataset.p2ls[i][c])//self.dataset.bts[i][c] for i in range(2)])
            dataset_len += c_len

        dataset_len = np.ceil(1.0*dataset_len/self.num_selected_classes)
        return dataset_len