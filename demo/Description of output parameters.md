# Description of output parameters from PSF learning
In the end of each demo notebook, it listed all the output parameters obtained from PSF learning. Here we provide the description of those parameters from different imaging modalities. 
## List of output parameters
### Single channel    
|**Parameters**| Description|
|:----------------------------|:---------------------------|
|**locres**|localization results of the data used for learning | 
|&nbsp;&nbsp;&nbsp;**CRLB**|CRLB, theoretical localization variance of each variable|
|&nbsp;&nbsp;&nbsp;**LL**|Loglikelihood ratio of each emitter|
|&nbsp;&nbsp;&nbsp;**loc**|Estimated positions, unit: pixel|
|&nbsp;&nbsp;&nbsp;**coeff**|Spline coefficients used for spline based localization algorithm|
|&nbsp;&nbsp;&nbsp;**coeff_reverse**|Same as `coeff` but with z dimension reversed|
|&nbsp;&nbsp;&nbsp;**coeff_bead**|Same as `coeff` but only for localizing bead data|
|**res**|PSF learning results| 
|&nbsp;&nbsp;&nbsp;**I_model**|Learned PSF model for modelling single molecules, a 3D matrix|
|&nbsp;&nbsp;&nbsp;**I_model_reverse**|Same as `I_model` but with z dimension reversed|
|&nbsp;&nbsp;&nbsp;**I_model_bead**|Learned PSF model for modelling bead data|
|&nbsp;&nbsp;&nbsp;**bg**|Learned background values of each emitter|
|&nbsp;&nbsp;&nbsp;**intensity**|Learned total photon count of each emitter|
|&nbsp;&nbsp;&nbsp;**pos**|Learned x,y,z positions of each emitter, unit: pixel|
|&nbsp;&nbsp;&nbsp;**pupil**|Learned pupil function, a 2D complex matrix|
|&nbsp;&nbsp;&nbsp;**zernike_coeff**|Learned Zernike coefficients of the pupil function, including both the coefficients for pupil magnitude and pupil phase|
|&nbsp;&nbsp;&nbsp;**sigma**|Learned widths in x,y of the Gaussian blurring kernel, unit: pixel|
|&nbsp;&nbsp;&nbsp;**drift_rate**|Learned x,y drift for each bead stack, unit: pixel per z slice|
|&nbsp;&nbsp;&nbsp;**cor**|Pixel coordinates of final emitters|
|&nbsp;&nbsp;&nbsp;**cor_all**|Pixel coordinates of all candidate emitters|
|&nbsp;&nbsp;&nbsp;**apodization**|The apodization term of the pupil, a 2D matrix|
|&nbsp;&nbsp;&nbsp;**zernike_polynomials**|The matrix representation of each Zernike polynomials used in learning, a set of 2D matrices|
|&nbsp;&nbsp;&nbsp;**offset**|The minimum value of `I_model`, ideally it should be greater than zero|
|**rois**| | 
|&nbsp;&nbsp;&nbsp;**cor**|Pixel coordinates of final emitters|
|&nbsp;&nbsp;&nbsp;**fileID**|Data file No. of final emitters|
|&nbsp;&nbsp;&nbsp;**image_size**|The image size of the raw data, unit: pixel|
|&nbsp;&nbsp;&nbsp;**psf_data**|The selected rois of final emitters|
|&nbsp;&nbsp;&nbsp;**psf_fit**|The PSF models of final emitters, same size as `psf_data`|
### Multi-channel 
Below list parameters that are different from [single channel](#Single%20channel)
|**Parameters**| Description|
|:----------------------------|:---------------------------|
|**res**|PSF learning results| 
|&nbsp;&nbsp;&nbsp;**T**|Affine transformation matrix between each target channel to the reference channel, a stack of 3x3 matrices|
|&nbsp;&nbsp;&nbsp;**channelN**|Learned results from Nth channel, see `res` in [single channel](#Single%20channel), `N` counts from 0.|
|&nbsp;&nbsp;&nbsp;**imgcenter**|The pixel coordinate of the image center from the raw data, it defines the rotation center of `T`|
|&nbsp;&nbsp;&nbsp;**xyshift**|The initial estimation of the lateral shift between the target channel to the reference channel, unit: pixel|
### 4Pi 
The first level output parameters are the same as the ones in [multi-channel](#Multi-channel), however the parameters in `channelN` are different from the ones in [single channel](#Single%20channel), below list the difference.
|**Parameters**| Description|
|:----------------------------|:---------------------------|
|**channelN**|Learned results from Nth channel |
|&nbsp;&nbsp;&nbsp;**I_model**|Learned model for matrix I in the IAB model, a 3D matrix|
|&nbsp;&nbsp;&nbsp;**A_model**|Learned model for matrix A and B in the IAB model, a complex 3D matrix|
|&nbsp;&nbsp;&nbsp;**I_model_reverse**|Same as `I_model` but with z dimension reversed|
|&nbsp;&nbsp;&nbsp;**A_model_reverse**|Same as `A_model` but with z dimension reversed|
|&nbsp;&nbsp;&nbsp;**intensity**|Learned total photon (`real(intensity)`) and interference phase (`angle(intensity)`) of each emitter, a complex vector|
|&nbsp;&nbsp;&nbsp;**phase_dm**|Learned relative phases of the three axial scans in one dataset, a vector of three values|
|&nbsp;&nbsp;&nbsp;**pupil1**|Learned pupil function of the top emission path, a 2D complex matrix|
|&nbsp;&nbsp;&nbsp;**pupil2**|Learned pupil function of the bottom emission path, a 2D complex matrix|
|&nbsp;&nbsp;&nbsp;**zernike_coeff_mag**|Learned Zernike coefficients of the magnitude parts of `pupil1` and `pupil2`|
|&nbsp;&nbsp;&nbsp;**zernike_coeff_phase**|Learned Zernike coefficients of the phase parts of `pupil1` and `pupil2`|
|&nbsp;&nbsp;&nbsp;**modulation_depth**|Learned modulation depth, defines the weight factor of the coherent part of the PSF model|
|&nbsp;&nbsp;&nbsp;**offset**|The minimum value of the PSF model, ideally it should be greater than zero. In IAB model, the PSF model at interference phase equal to zero is $PSF_{model}=I_{model}-2\left\|A_{model}\right\|$|
|&nbsp;&nbsp;&nbsp;**Zphase**|The stage position (in pixels) multiplied by $2\pi$|
### Field dependent
Below list parameters that are different from [single channel](#Single%20channel)
|**Parameters**| Description|
|:----------------------------|:---------------------------|
|**locres**|localization results of the data used for learning | 
|&nbsp;&nbsp;&nbsp;*others*|Corresponding values from averaged PSF model|
|&nbsp;&nbsp;&nbsp;**loc_FD**|Estimated positions from the PSF model for each emitter|
|**res**|PSF learning results| 
|&nbsp;&nbsp;&nbsp;**I_model_all**|Learned PSF model for each emitter, a set of 3D matrices|
|&nbsp;&nbsp;&nbsp;**I_model_bead**|Learned averaged PSF model for modelling bead data, a 3D matrix|
|&nbsp;&nbsp;&nbsp;**I_model**|Learned averaged PSF model for modelling single molecules, a 3D matrix|
|&nbsp;&nbsp;&nbsp;**pupil**|Learned pupil function of each emitter, a set of 2D complex matrix|
|&nbsp;&nbsp;&nbsp;**zernike_coeff**|Learned Zernike coefficients of the pupil function of each emitter, including both the coefficients for pupil magnitude and pupil phase. A set of 2D arrays|
|&nbsp;&nbsp;&nbsp;**zernike_map**|Learned aberration maps of both the pupil magnitude and pupil phase for each Zernike polynomial|
### *In situ* PSF
Below list additional parameters from *in situ* learning 
|**Parameters**| Description|
|:----------------------------|:---------------------------|
|**res** or **res/channelN**|PSF learning results| 
|&nbsp;&nbsp;&nbsp;**stagepos**|Learned stage position, a positive scalar, unit: micron|
|&nbsp;&nbsp;&nbsp;**zoffset**|z position of the first slice in the learned PSF model, unit: pixel|
|&nbsp;&nbsp;&nbsp;**sampleheight**|Learned thickness of the sample chamber in a 4Pi system, unit: micron|



