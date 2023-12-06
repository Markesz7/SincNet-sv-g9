import torch
from torch.autograd import Variable
import soundfile as sf 

from data_io import ReadList,read_conf,str_to_bool

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
import numpy as np
from dnn_models import MLP,flip
from dnn_models import SincNet as CNN 
from data_io import ReadList,read_conf,str_to_bool

threshold = 0.8 # Change this for different results

# Copy your network definitions here
# Reading cfg file
options=read_conf()

#[data]
tr_lst=options.tr_lst
te_lst=options.te_lst
test_model_path = options.test_model_path
test_files = options.test_files
pt_file=options.pt_file
class_dict_file=options.lab_dict
data_folder=options.data_folder+'/'
output_folder=options.output_folder

#[windowing]
fs=int(options.fs)
cw_len=int(options.cw_len)
cw_shift=int(options.cw_shift)

#[cnn]
cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act=list(map(str, options.cnn_act.split(',')))
cnn_drop=list(map(float, options.cnn_drop.split(',')))


#[dnn]
fc_lay=list(map(int, options.fc_lay.split(',')))
fc_drop=list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act=list(map(str, options.fc_act.split(',')))

#[class]
class_lay=list(map(int, options.class_lay.split(',')))
class_drop=list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act=list(map(str, options.class_act.split(',')))


#[optimization]
lr=float(options.lr)
batch_size=int(options.batch_size)
N_epochs=int(options.N_epochs)
N_batches=int(options.N_batches)
N_eval_epoch=int(options.N_eval_epoch)
seed=int(options.seed)

# Converting context and shift in samples
wlen=int(fs*cw_len/1000.00)
wshift=int(fs*cw_shift/1000.00)

# validation list
wav_lst_te=ReadList(test_files)
snt_te=len(wav_lst_te)

# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
          'fs': fs,
          'cnn_N_filt': cnn_N_filt,
          'cnn_len_filt': cnn_len_filt,
          'cnn_max_pool_len':cnn_max_pool_len,
          'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
          'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
          'cnn_use_laynorm':cnn_use_laynorm,
          'cnn_use_batchnorm':cnn_use_batchnorm,
          'cnn_act': cnn_act,
          'cnn_drop':cnn_drop,          
          }

CNN_net=CNN(CNN_arch)
CNN_net.cuda()

# Loading label dictionary
lab_dict=np.load(class_dict_file, allow_pickle=True).item()

DNN1_arch = {'input_dim': CNN_net.out_dim,
          'fc_lay': fc_lay,
          'fc_drop': fc_drop, 
          'fc_use_batchnorm': fc_use_batchnorm,
          'fc_use_laynorm': fc_use_laynorm,
          'fc_use_laynorm_inp': fc_use_laynorm_inp,
          'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
          'fc_act': fc_act,
          }

DNN1_net=MLP(DNN1_arch)
DNN1_net.cuda()


DNN2_arch = {'input_dim':fc_lay[-1] ,
          'fc_lay': class_lay,
          'fc_drop': class_drop, 
          'fc_use_batchnorm': class_use_batchnorm,
          'fc_use_laynorm': class_use_laynorm,
          'fc_use_laynorm_inp': class_use_laynorm_inp,
          'fc_use_batchnorm_inp':class_use_batchnorm_inp,
          'fc_act': class_act,
          }

print(DNN2_arch)
DNN2_net=MLP(DNN2_arch)
DNN2_net.cuda()

cost = nn.NLLLoss()

# Batch_dev
Batch_dev=128


CNN_net.eval()
DNN1_net.eval()
DNN2_net.eval()

# Load trained model
checkpoint = torch.load(test_model_path)
CNN_net.load_state_dict(checkpoint['CNN_model_par'])
DNN1_net.load_state_dict(checkpoint['DNN1_model_par'])
DNN2_net.load_state_dict(checkpoint['DNN2_model_par'])

loss_sum=0
err_sum=0
err_sum_snt=0

with torch.no_grad():  
    for i in range(snt_te):

        [signal, fs] = sf.read(data_folder+wav_lst_te[i])

        signal=torch.from_numpy(signal).float().cuda().contiguous()
        lab_batch=lab_dict[wav_lst_te[i]]

        # split signals into chunks
        beg_samp=0
        end_samp=wlen
        
        N_fr=int((signal.shape[0]-wlen)/(wshift))
        #print(N_fr)

        sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()
        lab= Variable((torch.zeros(N_fr+1)+int(lab_batch)).cuda().contiguous().long())
        pout=Variable(torch.zeros(N_fr+1,class_lay[-1]).float().cuda().contiguous())
        count_fr=0
        count_fr_tot=0

        while end_samp<signal.shape[0]:
            sig_arr[count_fr,:]=signal[beg_samp:end_samp]
            beg_samp=beg_samp+wshift
            end_samp=beg_samp+wlen
            count_fr=count_fr+1
            count_fr_tot=count_fr_tot+1
            if count_fr==Batch_dev:
                inp=Variable(sig_arr)
                pout[count_fr_tot-Batch_dev:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))
                count_fr=0
                sig_arr=torch.zeros([Batch_dev,wlen]).float().cuda().contiguous()

        if count_fr>0:
            inp=Variable(sig_arr[0:count_fr])
            pout[count_fr_tot-count_fr:count_fr_tot,:]=DNN2_net(DNN1_net(CNN_net(inp)))


        predictions = torch.nn.functional.softmax(pout, dim=-1)[:,lab_batch]
        #print(predictions)

        count_rejected= len(predictions[predictions < threshold])
        
        err = count_rejected / N_fr

        sentence_pred = torch.mean(predictions)

        err_sum_snt=err_sum_snt+(sentence_pred < threshold).float()
        err_sum=err_sum+err
    
    err_tot_dev_snt=err_sum_snt/snt_te
    err_tot_dev=err_sum/snt_te
    print("err_te=%f err_te_snt=%f" % (err_tot_dev,err_tot_dev_snt))