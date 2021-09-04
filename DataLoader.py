import torch
import torchaudio
from torch import nn, autograd, utils
import torchaudio.transforms as T
import numpy as np
import warnings
warnings.simplefilter("ignore", UserWarning)
index_map = {}
class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        with open('char_map.txt','r') as char_map_str:
            char_map_str = char_map_str
            self.char_map = {}
            self.index_map = {}
            for line in char_map_str:
                ch, index = line.split()
                self.char_map[ch] = int(index)
                self.index_map[int(index)] = ch
            #self.index_map[1] = ' '
        print(self.char_map)
        index_map = self.index_map
        print(index_map)
    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:  
            if c == ' ':
                ch = self.char_map['>']
            else:
                 ch = self.char_map.get(c)
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('>', ' ')


# mfcc_transform = nn.Sequential(
#     torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=60),
#     torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
#     torchaudio.transforms.TimeMasking(time_mask_param=35)
# )
sample_rate = 16000
n_fft = 1024
win_length = 320 #20ms
hop_length = 160 #10ms 
n_mels = 80
n_mfcc = 80  #23
mfcc_transform = T.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc,
 melkwargs={'n_fft': n_fft, 'n_mels': n_mels, 'win_length': win_length, 'hop_length': hop_length})


       
#print("mfcc ok!")
#mfcc = mfcc_transform(waveform)

#valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

text_transform = TextTransform()


def data_processing(data, data_type="train"):
    mfccs = []
    labels = []
    input_lengths = []
    label_lengths = []
    fileid_audio = " "
    for (waveform, _, utterance, speaker_id, utterance_id) in data:
        if data_type == 'train':
            mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1)        
       
        else:
            mfcc = mfcc_transform(waveform).squeeze(0).transpose(0, 1)
            
            
        fileid_audio = str(speaker_id) + "-" + str(utterance_id)   
        mfccs.append(mfcc)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(mfcc.shape[0])
        label_lengths.append(len(label))

    mfccs = nn.utils.rnn.pad_sequence(mfccs, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return mfccs, labels, input_lengths, label_lengths

import editdistance
class TokenAcc():
    def __init__(self, blank=0):
        self.err = 0
        self.cnt = 0
        self.tmp_err = 0
        self.tmp_cnt = 0
        self.blank = 0
    
    def update(self, pred, xlen, label):
        ''' label is one dimensinal '''
        pred = np.vstack([pred[i, :j] for i, j in enumerate(xlen)])
        e = self._distance(pred, label)
        c = label.shape[0]
        self.tmp_err += e; self.err += e
        self.tmp_cnt += c; self.cnt += c
        return 100 * e / c

    def get(self, err=True):
        # get interval
        if err: res = 100 * self.tmp_err / self.tmp_cnt
        else: res = 100 - 100 * self.tmp_err / self.tmp_cnt
        self.tmp_err = self.tmp_cnt = 0
        return res

    def getAll(self, err=True):
        if err: return 100 * self.err / self.cnt
        else: return 100 - 100 * self.err / self.cnt

    def _distance(self, y, t):
        if len(y.shape) > 1: 
            y = np.argmax(y, axis=1)
        prev = self.blank
        hyp = []
        for i in y:
            if i != self.blank and i != prev: hyp.append(i)
            prev = i
        return editdistance.eval(hyp, t)
