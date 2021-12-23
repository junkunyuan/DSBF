import numpy as np
import os.path as osp

from torch.utils.data import DataLoader
from dataset.datasets import *
from dataset.config import *
from utils.util import write_log


class Factory(object):
    def ConfigFactory(self, dataset_name):
        dataset_name = dataset_name.lower()
        if dataset_name in ["office-31", "office", "office_31", "office31"]:
            return office_31
        elif dataset_name in ["office-home", "office_home", "officehome"]:
            return office_home
        elif dataset_name in ["pacs"]:
            return pacs
        elif dataset_name in ["dac"]:
            return dac
        elif dataset_name in ["face_gender"]:
            return face_gender
        elif dataset_name in ["office-8"]:
            return office_8
        elif dataset_name in ["all"]:
            return pacs, office_31, office_home, dac, face_gender
        return None

    def UnLabelDgDataFac(self, args):
        target = args.config["domains"][args.t_da_i]
        source_label = args.config["domains"][args.label]
        source_unlabels = [args.config["domains"][i] for i in args.unlabel]
        class_dsets = []
        write_log(args.out_file, "Labeled source dataset: {}".format(source_label))

        """Get target dataset."""
        pth2label_target = np.loadtxt(osp.join(args.config["data_list_dir"], target + "_test.txt"), dtype=str)
        target_dset = OneDataset(pth2label_target, args.config["transforms"][1])

        """Labeled source dataset."""
        pth2label_train = np.loadtxt(osp.join(args.config["data_list_dir"], source_label + "_train.txt"), dtype=str)
        if args.d_name == "pacs":
            pth2label_val = np.loadtxt(osp.join(args.config["data_list_dir"], source_label + "_crossval.txt"), dtype=str)
            source_label_dsets = [OneDataset(pth2label_train, args.config["transforms"][0]),
                              OneDataset(pth2label_val, args.config["transforms"][1])]   # train and test datasets
            pth2label = np.concatenate([pth2label_train, pth2label_val],0)
        else:
            source_label_dsets = [OneDataset(pth2label_train, args.config["transforms"][0]), target_dset]  # train and test datasets
            pth2label = pth2label_train
        class_dsets.append(ClassDataset(pth2label, args.config["transforms"][0], args.config, args.align_bs))

        """Get unlabeled source datasets."""
        source_unlabel_dsets = []
        for source_unlabel in source_unlabels:
            write_log(args.out_file, "Unlabeled source dataset: {}".format(source_unlabel))
            pth2label_train = np.loadtxt(osp.join(args.config["data_list_dir"], source_unlabel + "_train.txt"), dtype=str)
            if args.d_name == "pacs":
                pth2label_val = np.loadtxt(osp.join(args.config["data_list_dir"], source_unlabel+"_crossval.txt"), dtype=str)
                pth2label = np.concatenate([pth2label_train,pth2label_val], 0)
            else:
                pth2label = pth2label_train
            source_unlabel_dsets.append([
                OneDataset(pth2label, args.config["transforms"][0], True),
                OneDataset(pth2label, args.config["transforms"][1], True)
            ])

            class_dsets.append(ClassDataset(pth2label, args.config["transforms"][0], args.config, args.align_bs))

        """Align dataset."""
        align_dset = DgAlighDataset(*class_dsets, choose_num=args.choose_num)

        return source_label_dsets, source_unlabel_dsets, target_dset, align_dset

    def UnLabelDgFac(self, args):
        source_label_dsets, source_unlabel_dsets, target_dset, align_dset = self.UnLabelDgDataFac(args)
        loaders = {}
        
        """Labeled source dataloader. """
        loaders["source_label_train"] = DataLoader(source_label_dsets[0], batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
        loaders["source_label_val"] = DataLoader(source_label_dsets[1], batch_size=args.bs*3, shuffle=False, num_workers=args.workers, pin_memory=True)

        """unlabel source dataloader"""
        loaders["source_unlabel_trains"] = []
        loaders["source_unlabel_vals"] = []
        for dsets in source_unlabel_dsets:
            loaders["source_unlabel_trains"].append(DataLoader(dsets[0], batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True))
            loaders["source_unlabel_vals"].append(DataLoader(dsets[1], batch_size=args.bs*3, shuffle=False, num_workers=args.workers, pin_memory=True))

        """Target dataloader."""
        loaders["target"] = DataLoader(target_dset, batch_size=args.bs*3, shuffle = False, num_workers=args.workers, pin_memory=True)

        """Align dataloader."""
        loaders["align"] = DataLoader(align_dset, batch_size=args.num_selected_classes, shuffle=True, num_workers=args.workers, pin_memory=True)

        return loaders



