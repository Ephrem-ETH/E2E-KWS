import argparse
import logging
import math
import os
import time

import editdistance
import torch
from torch.utils import data
import torchaudio
from torch.autograd import Variable
import torch.nn.functional as F
import torchaudio.transforms as T
import numpy as np
from model import Transducer, RNNModel
from DataLoader import TokenAcc, index_map
from torchaudio.datasets import SPEECHCOMMANDS


# Hyperparameters
sample_rate = 16000
n_fft = 1024
win_length = 320 #20ms
hop_length = 160 #10ms 
n_mels = 80
n_mfcc = 80  #23
mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc,
 melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'win_length': win_length, 'hop_length': hop_length})

def data_processing(data, data_type="test"):
    mfccs = []
    labels = []
    input_lengths = []
    label_lengths = []
    fileid_audio = " "
    for (waveform, _, utterance, speaker_id, utterance_id) in data:
        if data_type == 'test':
            mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1)
            label = utterance.lower().split()
            fileid_audio = str(speaker_id) + "-" + str(utterance_id) 

    return mfcc, label, fileid_audio



parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Acoustic Model on TIMIT.')
parser.add_argument('model', help='trained model filename')
parser.add_argument('--beam', type=int, default=0, help='apply beam search, beam width')
parser.add_argument('--ctc', default=False, action='store_true', help='decode CTC acoustic model')
parser.add_argument('--bi', default=False, action='store_true', help='bidirectional LSTM')
parser.add_argument('--dataset', default='test', help='decoding data set')
parser.add_argument('--out', type=str, default='', help='decoded result output dir')
args = parser.parse_args()

logdir = args.out if args.out else os.path.dirname(args.model) + '/decode.log'
# if args.out: os.makedirs(args.out, exist_ok=True)
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt="%H:%M:%S", filename=logdir, level=logging.INFO)

# Load model
Model = RNNModel if args.ctc else Transducer
model = Model(80, 29, 250, 3, bidirectional=args.bi)
state = torch.load(args.model, map_location='cpu')
# model.load_state_dict(torch.load(args.model, map_location='cpu'))
model.load_state_dict(state['state_dict'])

#use_gpu = torch.cuda.is_available()
use_gpu = True
if use_gpu:
    model.cuda()

# data set
test_path = "./data/test"
dev_path = "./data/dev"
if not os.path.isdir("./data/test"):
    os.makedirs("./data/test")

if not os.path.isdir("./data/dev"):
    os.makedirs("./data/dev")


test_set = torchaudio.datasets.SPEECHCOMMANDS(root= test_path , subset="testing", download=True)
dev_set = torchaudio.datasets.SPEECHCOMMANDS(root= dev_path , subset="validation", download=True)



test_loader = data.DataLoader(dataset=test_set,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'test'))
# print(len(test_set))
# exit()
# Word map
with open('conf/keywords', 'r') as f:
    keymap = {}
    for line in f:
        line = line.lower().split() 
        keymap[line[0]] = line[0]



# Calculate sentence level character error rate(CER)
def calculate_cer(y,t):
    y = ' '.join(y)
    t = ' '.join(t)
    char_t_len = len(t)
    return char_t_len, editdistance.eval(list(y), list(t))

  
# Class labels for 12-lables classification task

CLASS_LABELS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

model.eval()
def decode():
    logging.info('Decoding transduction model:')
    total_word, total_char, total_cer, total_wer, correct_pred_35, total_pred_35, correct_pred_12, total_pred_12 = 0,0,0,0,0,0,0,0
    for i, (xs, label,k) in enumerate(test_loader):
       
        xs = Variable(torch.FloatTensor(xs[None, ...]), volatile=True)
        if use_gpu:
            xs = xs.cuda()
        if args.beam > 0:
            y,nll = model.beam_search(xs, args.beam)
            print("beam {0} , label{1} ".format(y,label))
        else:
            y, nll = model.greedy_decode(xs)
        # y = [keymap.get(i) for i in y if keymap.get(i)]
        # t = [keymap.get(i) for i in label if keymap.get(i)]

        # Computing 12-labels classification accuracy
        pred = ''.join(y)
        ref = ''.join(label)
        if ref in CLASS_LABELS:
            if pred==ref:
                correct_pred_12 +=1
            total_pred_12 +=1
        elif not ref:          # To check silence 
            if ref == pred:
                correct_pred_12 +=1
            total_pred_12 +=1
        else:
            correct_pred_12 +=1
            total_pred_12 +=1

       # Computing 35-labels classification accuracy
        if pred==ref:
            correct_pred_35 +=1
        total_pred_35 +=1
        #Compute CER
        sen_len, cer = calculate_cer(y,label)
        #print(f"cer : {cer} correctly classified : {correct_pred}, total prediction: {total_pred}")
        total_cer += cer; total_char += sen_len
        logging.info('[{}]: {}'.format(k, ' '.join(label)))
        logging.info('[{}]: {}\nlog-likelihood: {:.2f}\n'.format(k, ' '.join(y),nll))
    accuracy_12 = 100*correct_pred_12 / total_pred_12
    accuracy_35 = 100*correct_pred_35 /total_pred_35
    logging.info('12-labels Accuracy {:.2f}% and 35-labels Accuracy {:.2f}%'.format(accuracy_12, accuracy_35))
    logging.info('{} set {} CER {:.2f}%'.format(
        args.dataset.capitalize(), 'CTC' if args.ctc else 'Transducer', 100*total_cer/total_char))

decode()


