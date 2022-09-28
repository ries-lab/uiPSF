
#%% imports


# append the path of the parent directory as long as it's not a real package
import sys

from tqdm import tqdm
from time import sleep

from tkinter import EXCEPTION, messagebox as mbox
sys.path.append("..")
from psflearning.psflearninglib import psflearninglib


#%% load parameters
#paramfile = sys.argv[1]
#paramfile = r'D:\Sheng\data\04-28-2022 bead\bead_600_001\bead1_600__560_00000_00024_mode000_amp000_par.json'
paramfile = r'Z:\projects\PSFLearning\exampledata\40nm_M2\Pos0_40nm_bead_50nm_Z_1\Pos0_40nm_bead_50nm_Z_1_MMStack_Default.ome_par.json'
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

except Exception as e:
    print("\n{}:".format(type(e).__name__),"{}".format(e))
    #print(e)
    #print('learning is not sucessful')
    #mbox.showerror("{}:".format(type(e).__name__),"{}".format(e))
#%%
import h5py as h5
resfile = r'Z:\2021\21_12\211207_SL_bead_3D_M2\40nm_bead_50nm\psfmodel_LL_zernike_single.h5'
f = h5.File(resfile, 'r')
d1 = f['res']
#d1.keys()
#f.close()
# %%
psfobj,fitter = L.learn_psf(dataobj,time=toc)

#%%


pbar = tqdm(total=100,desc='learning',bar_format = "{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt} {postfix[0]}{postfix[1][value]}",postfix=["current loss: ", dict(value=0)])

for i in range(0,100):
    pbar.postfix[1]['value'] = i
    
    pbar.update(1)
    sleep(0.01)

#pbar.close()
toc = pbar.last_print_t-pbar.start_t
print(toc)
# %%
pbar = tqdm(total=1000, desc='learning',bar_format = "{desc}: {n_fmt}/{total_fmt} [{elapsed}s] {rate_fmt}, {postfix[0]}{postfix[2][loss]:>4.2f}, {postfix[1]}{postfix[2][time]:>4.2f}",postfix=["current loss: ","total time: ", dict(loss=0,time=0)])
for i in range(0,100):
    pbar.postfix[-1]['loss'] = i
    pbar.postfix[-1]['time'] = pbar._time()-pbar.start_t
    pbar.update()
    sleep(0.01)
    #pbar.clear()
pbar.close()

# %%
