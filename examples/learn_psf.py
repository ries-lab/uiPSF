
import h5py as h5

import matplotlib.pyplot as plt
from tqdm import tqdm
# append the path of the parent directory as long as it's not a real package
import sys

from tkinter import EXCEPTION, messagebox as mbox
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib


#%% load parameters
paramfile = sys.argv[1]
L = psflearninglib()
L.getparam(paramfile)

#%%
try:
    L.getpsfclass()
    toc = 0
    pbar = tqdm(total=100,desc='1/6: loading data',bar_format = "{desc}: [{elapsed}s] {postfix[0]}{postfix[1][time]:>4.2f}s",postfix=["total time: ", dict(time=0)])
    images = L.load_data()
    
    pbar.postfix[1]['time'] = pbar._time()-pbar.start_t   
    toc = pbar._time()-pbar.start_t   
    pbar.close()
    pbar = tqdm(total=100,desc='2/6: cutting rois',bar_format = "{desc}: [{elapsed}s] {postfix[0]}{postfix[1][time]:>4.2f}s",postfix=["total time: ", dict(time=toc)])
    
    dataobj = L.prep_data(images)
    
    pbar.postfix[1]['time'] = toc+ pbar._time()-pbar.start_t    
    pbar.update()
    toc = pbar.postfix[1]['time'] 
    pbar.close()
    psfobj,fitter = L.learn_psf(dataobj,time=toc)
    
    resfile = L.save_result(psfobj,dataobj,fitter)

    #mbox.showinfo('learning status','learning is done.')
except Exception as e:
    print("\n{}:".format(type(e).__name__),"{}".format(e))
    #print(e)
    #print('learning is not sucessful')
    mbox.showerror("{}:".format(type(e).__name__),"{}".format(e))
#%%
#f = h5.File(resfile, 'r')
#d1 = f['res']
#d1.keys()
#f.close()
# %%
