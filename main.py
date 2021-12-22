from scipy.spatial.distance import cdist
from operator import itemgetter
import os.path as osp
import torch,random,os
from torch import nn
import numpy as np
from utils.util import print_args
from models.dg_network import resnet50, resnet18, FeatBootleneck, Classifier
from models.attentions_new import DGAttention as NEWDGAttention
from factory import Factory
from utils.run_util import *
from utils.util import ForeverDataIterator,write_log
from loss import Lim
from tensorboardX import SummaryWriter
import argparse

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.iter_step = 20
        self.outf = args.out_file
        self.output_dir = args.output_dir
        self.writer = SummaryWriter(args.output_dir + "/target_{}/logs".format(args.t_da_i))
        self.accs = []

        """Load model."""
        if args.net == 'resnet18':
            self.netF = resnet18(pretrained=True, classes=args.config["class_num"]).cuda()
            bottleneck_dim = 256
        elif args.net == 'resnet50':
            self.netF = resnet50(pretrained=True, classes=args.config["class_num"]).cuda()
            bottleneck_dim = 256
        else:
            self.netF = resnet18(pretrained=True, classes=args.config["class_num"]).cuda()
            bottleneck_dim = 256
        self.netB = FeatBootleneck(self.netF.in_features,type="wn",bottleneck_dim=bottleneck_dim).to(device)
        self.netC = Classifier(args.config["class_num"],bottleneck_dim=bottleneck_dim).to(device)
        self.netIV1 = FeatBootleneck(self.netF.in_features, type="wn",bottleneck_dim=bottleneck_dim).to(device)
        self.netIV2 = FeatBootleneck(self.netF.in_features, type="wn",bottleneck_dim=bottleneck_dim).to(device)
        self.netA = NEWDGAttention(len(args.loaders["source_unlabel_trains"]),in_dim=bottleneck_dim).to(device)

        """Initalize optimizers."""
        params_groups = self.get_params(self.netIV1, self.netIV2)
        self.optimizer_FBC, self.scheduler_FBC = get_optim_and_scheduler(params_groups[2], args.max_epoch, args.lr)
        self.optimizer_FB, self.scheduler_FB = get_optim_and_scheduler(params_groups[0], args.max_epoch, args.lr)
        self.optimizer_C, self.scheduler_C = get_optim_and_scheduler(params_groups[1], args.max_epoch, args.lr)
        self.optimizer_IV1, self.scheduler_IV1 = get_optim_and_scheduler(params_groups[3][0], args.max_epoch, args.lr)
        self.optimizer_IV2, self.scheduler_IV2 = get_optim_and_scheduler(params_groups[3][1], args.max_epoch, args.lr)
        self.optimizer_A, self.scheduler_A = get_optim_and_scheduler(self.netA.parameters(), args.max_epoch, args.lr)

        """Prepare criterion function."""
        self.criterion_iv = nn.MSELoss()
        self.criterion_im = Lim(1e-5)
        self.criterion_ce = nn.CrossEntropyLoss()

    def pse_label(self):
        """Calculate the label before every epoch."""
        self.netF.eval(),self.netB.eval(),self.netC.eval()
        pseudo_labels = []
        for i, loader in enumerate(self.args.loaders["source_unlabel_vals"]):
            write_log(self.outf,"unlabel dataset {} get pseudo labels:".format(i))
            pseudo_labels.append(self.obtain_label(loader,self.netF,self.netB,self.netC))
        self.args.loaders["align"].dataset.construct(pseudo_labels)
        return pseudo_labels


    def train_epoch_IV(self, epoch):
        """Train the models."""
        self.netF.train(), self.netB.train(), self.netC.train(), self.netIV1.train(),self.netIV2.train()
        length = int(len(self.args.loaders["source_unlabel_trains"][0]))
        unlabel_trains_loader = []
        for loader in self.args.loaders["source_unlabel_trains"]:
            unlabel_trains_loader.append(ForeverDataIterator(loader))
        align_loader = ForeverDataIterator(self.args.loaders["align"])
        self.max_acc = 0

        if self.args.pseudo:
            pseudo_labels = self.pse_label()

        for it in range(length):
            """Use unlabeled source dataset to train netF and netB."""
            for i,loader in enumerate(unlabel_trains_loader):
                datas,pred,pths = next(loader)
                pred = pred.to(self.device)
                datas = datas.to(self.device)
                features = self.netB(self.netF(datas))
                outputs = self.netC(features)
                if self.args.pseudo:
                    pred = torch.tensor(itemgetter(*pths)(pseudo_labels[i]),dtype=torch.long).cuda()
                ce_loss = self.criterion_ce(outputs,pred)                         
                im_loss = self.criterion_im(outputs)
                loss_ce_im = (ce_loss + im_loss) * self.args.lambda_
                self.optimizer_FB.zero_grad()
                loss_ce_im.backward()
                self.optimizer_FB.step()
                self.writer.add_scalar("{}/loss_ce".format(i), ce_loss, epoch * length + it)
                self.writer.add_scalar("{}/loss_im".format(i), im_loss, epoch * length + it)
                self.writer.add_scalar("{}/loss_ce_im".format(i), loss_ce_im, epoch * length + it)


            """Train IV."""
            datas,labels = next(align_loader)
            (data1, data2, data3), label1, label2, label3 = shape_data(*datas), labels[0].reshape(-1), labels[
                1].reshape(-1), labels[2].reshape(-1)
            data1, data2, data3 = data1.to(self.device), data2.to(self.device), data3.to(self.device)
            label1, label2, label3 = label1.to(self.device), label2.to(self.device), label3.to(self.device)

            self.netF.eval(),self.netB.eval()
            out1 = self.netB(self.netF(data1))
            out2 = self.netIV1(self.netF(data2))
            out3 = self.netIV2(self.netF(data3))
            loss12_iv = 0.5 * self.criterion_iv(out1,out2)
            loss13_iv = 0.5 * self.criterion_iv(out1,out3)
            loss_iv = (loss12_iv + loss13_iv) * self.args.gamma
            self.optimizer_IV1.zero_grad(), self.optimizer_IV2.zero_grad()
            loss_iv.backward()
            self.optimizer_IV1.step(), self.optimizer_IV2.step()
            self.netF.train(), self.netB.train()

            self.writer.add_scalar("loss_iv", loss_iv, epoch * length + it)
            self.writer.add_scalar("loss12_iv", loss12_iv, epoch * length + it)
            self.writer.add_scalar("loss13_iv", loss13_iv, epoch * length + it)

            """Train C (classifier calibrating)."""
            out2 = self.netIV1(self.netF(data2))
            out3 = self.netIV2(self.netF(data3))
            if self.args.att > 0:
                (out2,out3), _ = self.netA(out2, out3)
            out2 = self.netC(out2)
            out3 = self.netC(out3)

            self.optimizer_C.zero_grad(), self.optimizer_A.zero_grad()
            loss2 = self.criterion_ce(out2, label2) * 0.5 * self.args.gamma       
            loss2.backward(retain_graph=True)
            loss3 = self.criterion_ce(out3,label3) * 0.5 * self.args.gamma
            loss3.backward()
            self.optimizer_A.step(), self.optimizer_C.step()

            self.writer.add_scalar("loss_ce2", loss2, epoch * length + it)
            self.writer.add_scalar("loss_ce3", loss3, epoch * length + it)

            if it % self.args.step == 0 and it !=0:
                self.netF.eval(), self.netB.eval(), self.netC.eval()
                acc = self.test(self.args.loaders["target"])
                acc_source = self.test(self.args.loaders["source_label_train"])
                self.accs.append("{:.2f}".format(acc))
                write_log(self.outf,"test; Epoch: {}, acc: {:.2f}/{:.2f}".format(epoch, acc,acc_source))
                if acc > self.max_acc:
                    self.max_acc = acc
                    self.max_netF = self.netF.state_dict()
                    self.max_netB = self.netB.state_dict()
                    self.max_netC = self.netC.state_dict()
                self.netF.train(),self.netB.train(),self.netC.train()


        self.netF.eval(), self.netB.eval(), self.netC.eval()
        acc = self.test(self.args.loaders["target"])
        write_log(self.outf,"test; Epoch: {}, acc: {:.2f}\n".format(epoch, acc))
        self.writer.add_scalar("acc", acc, epoch)

    def pre_train(self):
        best_netF,best_netB,best_netC = None,None,None
        self.netF.train(),self.netB.train(),self.netC.train()
        loader_len = len(self.args.loaders["source_label_train"])
        max_iter = int(1.5*self.args.max_epoch) * loader_len
        max_acc = 0

        for it in range(max_iter):
            try:
                datas,labels = iter_source_label.next()
            except:
                iter_source_label = iter(self.args.loaders["source_label_train"])
                datas,labels = iter_source_label.next()
            datas,labels = datas.to(self.device),labels.to(self.device)
            """use data1 to train model FBC"""
            out = self.netC(self.netB(self.netF(datas)))
            loss_ce = self.criterion_ce(out,labels)
            self.optimizer_FBC.zero_grad()
            loss_ce.backward()
            self.optimizer_FBC.step()
            self.writer.add_scalar("loss_ce",loss_ce,it)

            if ((it % self.args.step) == 0 and it !=0) or it == max_iter:
                self.netF.eval(),self.netB.eval(),self.netC.eval()
                acc = self.test(self.args.loaders["source_label_val"])
                write_log(self.outf,"val: iter_num: {}/{}, acc: {:.2f}".format(it,max_iter,acc))
                if acc > max_acc:
                    max_acc = acc
                    best_netF = self.netF.state_dict()
                    best_netB = self.netB.state_dict()
                    best_netC = self.netC.state_dict()
                self.netF.train(),self.netB.train(),self.netC.eval()
        if self.args.d_name == "pacs":
            torch.save(best_netF,osp.join(self.output_dir,"source_F.pt"))
            torch.save(best_netB,osp.join(self.output_dir,"source_B.pt"))
            torch.save(best_netC,osp.join(self.output_dir,"source_C.pt"))
        else:
            torch.save(self.netF.state_dict(), osp.join(self.output_dir, "source_F.pt"))
            torch.save(self.netB.state_dict(), osp.join(self.output_dir, "source_B.pt"))
            torch.save(self.netC.state_dict(), osp.join(self.output_dir, "source_C.pt"))

        self.netF.eval(), self.netB.eval(), self.netC.eval()
        acc = self.test(self.args.loaders["target"])
        write_log(self.outf,"target: acc: {:.2f}".format(acc))

    def test(self, loader):
        total = 0
        with torch.no_grad():
            correct = 0
            for it, (data, labels) in enumerate(loader):
                data, labels = data.to(self.device), labels.to(self.device)
                class_logit = self.netC(self.netB(self.netF(data)))
                _, cls_pred = class_logit.max(dim=1)
                correct += torch.sum(cls_pred == labels.data)
                total += data.size(0)
        return (float(correct)/total) * 100

    def get_params(self,*netIVs):
        params_group_FBC = []
        params_group_FB = []
        params_group_C = []
        for k, v in self.netF.named_parameters():
            params_group_FBC += [{"params": v, "lr": self.args.lr*0.1}]
            params_group_FB += [{"params": v, "lr": self.args.lr*0.1}]
        for k, v in self.netB.named_parameters():
            params_group_FBC += [{"params": v, "lr": self.args.lr}]
            params_group_FB += [{"params": v, "lr": self.args.lr}]
        for k, v in self.netC.named_parameters():
            params_group_FBC += [{"params": v, "lr": self.args.lr}]
            params_group_C += [{'params': v, 'lr': self.args.lr}]

        params_group_IVs = []
        for netIV in netIVs:
            params_group_IV = []
            for k, v in netIV.named_parameters():
                params_group_IV += [{"params": v, "lr": self.args.lr}]
            params_group_IVs.append(params_group_IV)

        return params_group_FB, params_group_C, params_group_FBC, params_group_IVs

    def train(self):
        """Pre-train (model initialization)."""
        write_log(self.outf,"**********  pre_train  ***********")
        if (not osp.exists(osp.join(self.output_dir,"source_F.pt"))) or (not self.args.model_reuse):
            self.pre_train()

        """train"""
        self.netF.load_state_dict(torch.load(osp.join(self.output_dir, "source_F.pt")))
        self.netB.load_state_dict(torch.load(osp.join(self.output_dir, "source_B.pt")))
        self.netC.load_state_dict(torch.load(osp.join(self.output_dir, "source_C.pt")))
        write_log(self.outf,"*********  train  ***********")
        for self.current_epoch in range(self.args.max_epoch):
            self.scheduler_FB.step()
            self.scheduler_C.step()
            self.scheduler_FBC.step()
            self.scheduler_IV1.step()
            self.scheduler_IV2.step()
            self.scheduler_A.step()
            self.train_epoch_IV(self.current_epoch)
        self.accs.sort()
        write_log(self.outf, 'Final acc:{}\n\n'.format(self.accs[-1]))

        torch.save(self.max_netF, osp.join(self.output_dir, "source_F2.pt"))
        torch.save(self.max_netB, osp.join(self.output_dir, "source_B2.pt"))
        torch.save(self.max_netC, osp.join(self.output_dir, "source_C2.pt"))

    def obtain_label(self,loader, netF, netB, netC):
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                pths = np.asarray(data[2],dtype=str)
                inputs = inputs.cuda()
                feas = netB(netF(inputs))
                outputs = netC(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_pth = pths
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_pth = np.concatenate((all_pth,pths),0)
                    all_label = torch.cat((all_label, labels.float()), 0)

        all_output = nn.Softmax(dim=1)(all_output)

        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

        dd = cdist(all_fea, initc, "cosine")
        pred_label = dd.argmin(axis=1)

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, "cosine")
            pred_label = dd.argmin(axis=1)

        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
        log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        write_log(self.outf,log_str)

        pred_label = list(pred_label.astype("int"))
        all_pth = list(all_pth)

        return dict(zip(all_pth,pred_label))


def analytic_para():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='DG')
    
    """Object dataset setting."""
    parser.add_argument("--outdir",type=str,default="DG")
    parser.add_argument("--pseudo",type=int,default=1, help="1: SLDG; 0: CDG")
    parser.add_argument("--d_name", type=str, default="pacs", help="dataset name")
    parser.add_argument("--t_da_i", type=int, default=0, help="the index of target domain in dataset config")
    parser.add_argument("--bs", type=int, default=64, help="batch_size")
    
    """Alignment setting."""
    parser.add_argument("--align_bs", type=int, default=4, help="align source batch_size")
    parser.add_argument("--num_selected_classes", type=int, default=4, help="classes to choose in every step")
    parser.add_argument("--choose_num", type=int, default=3)
    
    """General setting."""
    parser.add_argument('--model_reuse', type=int, default=0, help="1: reuse the trained model; 0: train a new model")
    parser.add_argument('--max_epoch', type=int, default=20, help="maximum epoch")
    parser.add_argument('--step', type=int, default=10, help="pre step")
    parser.add_argument('--workers', type=int, default=4, help="number of workers")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--lambda_', type=float, default=1,help='hyper-parameter lambda')
    parser.add_argument('--gamma', type=float, default=1, help='hyper-parameter gamma')
    parser.add_argument("--label",type=int, default=1, help="labeled source dataset")
    parser.add_argument("--att", type=int, default=1, help="whether use attention")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="gpu id")
    
    parser.add_argument("--net", type=str, default="resnet18")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    """Set random seed."""
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    return args

def set_output(args):
    args.output_dir = osp.join(args.outdir + str(args.seed),
                               args.d_name,
                               args.config["domains"][args.t_da_i],
                               "label_source_" + str(args.label))
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.out_file = open(osp.join(args.output_dir, 'log.txt'), 'w')
    args.out_file.write(print_args(args) + '\n')
    args.out_file.flush()

def main():
    args = analytic_para()
    for label in range(4):
        if label == args.t_da_i: continue
        args.label = label  # Set labeled source domain.
        factory = Factory()
        args.config = factory.ConfigFactory(args.d_name)
        args.unlabel = [i for i in range(len(args.config["domains"])) if i not in [args.label, args.t_da_i]]  # Set unlabeled source domain.

        set_output(args)
        write_log(args.out_file, "Target: {}".format(args.config["domains"][args.t_da_i]))
        args.loaders = factory.UnLabelDgFac(args)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer = Trainer(args, device)
        trainer.train()

if __name__ == "__main__":
    main()


# You may run experiments with the following commands.

# PACS dataset & SLDG task & target_data: t_da_i = 0(Ar), 1(Ca), 2(Ph), 3(Sk).
# nohup python main.py --gpu_id 0 --step 10 --d_name pacs --t_da_i 0 > pacs-SLDG-data0.txt 2>&1 &

# Office-Home dataset & SLDG task & target_data: t_da_i = 0(Ar), 1(Cl), 2(Pr), 3(Rw).
# nohup python main.py --gpu_id 0 --step 15 --max_epoch 30 --d_name office-home --t_da_i 0 > office-home-SLDG-data0.txt 2>&1 &

# PACS dataset & CDG task & target_data: t_da_i = 0(Ar), 1(Ca), 2(Ph), 3(Sk).
# nohup python main.py --gpu_id 0 --step 10 --d_name pacs --t_da_i 0 --pseudo 0 > pacs-SLDG-data0.txt 2>&1 &

# Office-Home dataset & CDG task & target_data: t_da_i = 0(Ar), 1(Cl), 2(Pr), 3(Rw).
# nohup python main.py --gpu_id 0 --step 15 --max_epoch 30 --d_name office-home --t_da_i 0 --pseudo 0 > office-home-SLDG-data0.txt 2>&1 &