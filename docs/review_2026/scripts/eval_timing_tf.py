# Timing of one loss+gradient evaluation of the uiPSF zernike_vector bead model: eager (as in uiPSF) vs tf.function vs XLA
import os
import sys, time, numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'bead_fit_scaling.py')).read().split("def run(")[0])   # builds gt model + data with N beads
import tensorflow as tf
p = PSFZernikeBased(opts()); p.psftype='vector'; d = Data(data); p.data=d
init,_ = p.calc_initials(d); p.ind=[0,N]
V = [tf.Variable(x) for x in init]
rois = tf.constant(data)
w = [1,1,0,0.01,1,1,1,0,0.1]
def lossgrad():
    with tf.GradientTape() as t:
        l = mse_real_zernike(p.calc_forward_images(V), rois, V, 1.0, w)
    return l, t.gradient(l, V)
def loss_xlogy(model, data):  # Poisson deviance without boolean masking (static shapes)
    m = tf.maximum(model, 1e-6)
    return tf.reduce_mean(m - data - tf.math.xlogy(data, m) + tf.math.xlogy(data, tf.maximum(data,1e-6)))
def lossgrad2():
    with tf.GradientTape() as t:
        l = loss_xlogy(p.calc_forward_images(V), rois)
    return l, t.gradient(l, V)
f_graph = tf.function(lossgrad2)
f_xla = tf.function(lossgrad2, jit_compile=True)
def bench(f, n=10):
    f(); t=time.time()
    for _ in range(n): r=f(); _=r[0].numpy()
    return (time.time()-t)/n
print(f"N={N} beads, Nz={Nz}, roi 21x21, bin 2, pupil 64, vector")
print("eager (uiPSF):        %.3f s/eval" % bench(lossgrad))
print("eager, xlogy loss:    %.3f s/eval" % bench(lossgrad2))
print("tf.function:          %.3f s/eval" % bench(f_graph))
try: print("tf.function+XLA:      %.3f s/eval" % bench(f_xla))
except Exception as e: print("XLA failed:", type(e).__name__, str(e)[:200])
