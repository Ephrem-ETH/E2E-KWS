import os
import time
import random
import argparse
import logging
import numpy as np
import shutil
import torch
from torch import nn, autograd
import  torchaudio
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from model import RNNModel
import tensorboard_logger as tb
from torch.utils.tensorboard import SummaryWriter
#from DataLoader import SequentialLoader, TokenAcc
from DataLoader import data_processing, TokenAcc
from torchaudio.datasets import SPEECHCOMMANDS
from prettytable import PrettyTable
import warnings
warnings.simplefilter("ignore", UserWarning)


parser = argparse.ArgumentParser(description='PyTorch LSTM CTC Acoustic Model on TIMIT.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--start_epoch', type=int, default=1,
                    help='start epoch')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--bi', default=False, action='store_true', 
                    help='whether use bidirectional lstm')
parser.add_argument('--noise', default=False, action='store_true',
                    help='add Gaussian weigth noise')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--stdout', default=False, action='store_true', help='log in terminal')
parser.add_argument('--out', type=str, default='exp/ctc_lr1e-3',
                    help='path to save the final model')
parser.add_argument('--cuda', default=True, action='store_false')
parser.add_argument('--init', type=str, default='',
                    help='Initial am parameters')
parser.add_argument('--gradclip', default=False, action='store_true')
parser.add_argument('--schedule', default=False, action='store_true')
parser.add_argument('--resume', type=str, default='', help="resume the model training")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, 'args'), 'w') as f:
    f.write(str(args))
if args.stdout: logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
else: logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', filename=os.path.join(args.out, 'train.log'), level=logging.INFO)
tb.configure(args.out)
# Writer will output to ./runs/ directory by default
writer = SummaryWriter()
random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)
# Hyperparameters

start_epoch = 1

model = RNNModel(80, 29, 250, 3, args.dropout, bidirectional=args.bi)
if args.init: model.load_state_dict(torch.load(args.init))
else: 
    for param in model.parameters(): torch.nn.init.uniform(param, -0.1, 0.1)
if args.cuda: 
    model.cuda()
    device = torch.device('cuda')

#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


#if args.resume: load_checkpoint(args.resume)
if args.batch_size > 0 :
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity =True).to(device)
    # criterion = CTCLoss(size_average=True).to(device)
else:
    criterion = nn.CTCLoss(blank=0, reduction='mean').to(device)


# path for GSC training data 
train_path = "./data/train"
dev_path = "./data/dev"
if not os.path.isdir(train_path):
        os.makedirs(train_path)

# Path for GSC dev't set
if not os.path.isdir(dev_path):
        os.makedirs(dev_path)

# train_set = SubsetSC("training", train_path )
# dev_set = SubsetSC("validation", dev_path)

# Datasets

train_set = torchaudio.datasets.SPEECHCOMMANDS(root= train_path , subset="training", download=True)
dev_set = torchaudio.datasets.SPEECHCOMMANDS(root= dev_path , subset="validation", download=True)

train_loader = data.DataLoader(dataset=train_set, pin_memory=True,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=lambda x: data_processing(x, 'train'))

dev_loader = data.DataLoader(dataset=dev_set, pin_memory=True,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=lambda x: data_processing(x, 'valid'))
# print(len(train_loader))
# print(len(dev_loader))
# exit()
                        
# Checkpoint
def save_checkpoint(state, best_model, is_best=False, best_model_dir= args.out + '/best_model_dir'):
    
        torch.save(state, best_model)
 

def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    return model, optimizer, start_epoch



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
# count_parameters(model)
# print("Done!")
# exit()

if args.start_epoch:
    start_epoch = args.start_epoch
    
if args.resume: 
	model,optimizer,start_epoch = load_checkpoint(args.resume)


tri = cvi = 0
patience = 20
def eval(epoch):
    global cvi
    losses = []
    tacc = TokenAcc()
    for i, (xs, ys, xlen, ylen) in enumerate(dev_loader):
        x = Variable(torch.FloatTensor(xs), volatile=True)
        x = torch.squeeze(x,1).transpose(1,2).cuda()
        ys = np.hstack([ys[i, :j] for i, j in enumerate(ylen)])
        y = Variable(torch.IntTensor(ys))
        xl = Variable(torch.IntTensor(xlen)); yl = Variable(torch.IntTensor(ylen))
        model.eval()
        out = model(x)[0]
        out = F.log_softmax(out, dim=2)
        #print(out.shape)
        loss = criterion(out.transpose(0,1).contiguous(), y, xl, yl)
        # loss = F.ctc_loss(out.transpose(0,1).contiguous(), y, xl, yl)
        loss = float(loss.data) * len(xlen) # batch size
        losses.append(loss)
        tacc.update(out.data.cpu().numpy(), xlen, ys)
        tb.log_value('cv_loss', loss/len(xlen), epoch * len(dev_loader) + i)
        writer.add_scalar('cv_loss', loss/len(xlen), epoch * len(dev_loader) + i)
        writer.add_scalar('Loss/validation', loss/len(xlen), epoch)
        cvi += 1
    return sum(losses) / len(dev_loader), tacc.getAll()

def train():
    def adjust_learning_rate(optimizer, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    def add_noise(x):
        dim = x.shape[-1]
        noise = torch.normal(torch.zeros(dim), 0.075)
        if x.is_cuda: noise = noise.cuda()
        x.data += noise

    global tri
    global patience
    
    prev_loss = 10000
    best_model = None
    lr = args.lr
    for epoch in range(start_epoch, args.epochs):
        totloss = 0; losses = []
        start_time = time.time()
        tacc = TokenAcc()
        # if epoch == 10:
        #     break
        for i, (xs, ys, xlen, ylen) in enumerate(train_loader):
            # breaking point
            if i == 5000:
                break
            x = Variable(torch.FloatTensor(xs))
            x = torch.squeeze(x,1).transpose(1,2)
                            
            if args.cuda: x = x.cuda()
            if args.noise: add_noise(x)
            ys = np.hstack([ys[i, :j] for i, j in enumerate(ylen)])
            y = Variable(torch.IntTensor(ys)) 
            xl = Variable(torch.IntTensor(xlen)); yl = Variable(torch.IntTensor(ylen))
            model.train()
            optimizer.zero_grad()
            out = model(x)[0]
            out = F.log_softmax(out, dim=2)
            # print(f" log_probs:{out.shape} , target: {y.shape}, xlen: {xl.shape}, ylen: {yl.shape}")
            loss = criterion(out.transpose(0,1).contiguous(), y, xl, yl)
            # loss = F.ctc_loss(out.transpose(0,1).contiguous(), y, xl, yl)
            loss.backward()
            loss = float(loss.data) * len(xlen) # batch size
            totloss += loss; losses.append(loss)
            tacc.update(out.data.cpu().numpy(), xlen, ys)
            if args.gradclip: grad_norm = nn.utils.clip_grad_norm(model.parameters(), 200)
            optimizer.step()

            tb.log_value('train_loss', loss/len(xlen), epoch * len(train_loader) + i)
            writer.add_scalar('train_loss', loss/len(xlen), epoch * len(train_loader) + i )
            writer.add_scalar('Loss/train', loss/len(xlen), epoch)
            if args.gradclip: tb.log_value('train_grad_norm', grad_norm, tri)
            tri += 1
            # print(f"epoch {epoch} loss {loss}")
            if i % args.log_interval == 0 and i > 0:
                loss = totloss /args.batch_size/ args.log_interval
                #print(loss)
                logging.info('[Epoch %d Batch %d] loss %.2f, CER %.2f'%(epoch, i, loss, tacc.get()))
                totloss = 0

        losses = sum(losses) /len(train_loader)
       
        val_l, cer = eval(epoch)
        logging.info('[Epoch %d] time cost %.2fs, train loss %.2f, CER %.2f; cv loss %.2f, CER %.2f; lr %.3e'%(
            epoch, time.time()-start_time, losses, tacc.getAll(), val_l, cer, lr
        ))
        # Save checkpoint
        checkpoint = {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
      
        if val_l < prev_loss:
            is_best=True
            prev_loss = val_l
            best_model = '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}'.format(args.out, epoch, losses, val_l)
            # torch.save(model.state_dict(), best_model)          
        else:
            patience = patience - 1
            torch.save(model.state_dict(), '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}_rejected'.format(args.out, epoch, losses, val_l))
            #model.load_state_dict(torch.load(best_model))
            #model,optimizer,start_epoch = load_checkpoint(best_model)
            checkpoint = torch.load(best_model)
            model.load_state_dict(checkpoint['state_dict'])
            if args.cuda: model.cuda()
            if args.schedule:
                lr /= 2
                adjust_learning_rate(optimizer, lr)
        save_checkpoint(checkpoint, best_model, is_best)
        if patience ==0:
            break
            

if __name__ == '__main__':
    train()
