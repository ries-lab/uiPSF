# Description of user defined parameters 
List of all parameters defined in [config_base.yaml](config_base.yaml). Some parameters are system specific or for advanced settings. Users are not required to work with all parameters. We divided those parameters into [system specific](systemtype), [channel specific](channeltype) and [PSF specific](psftype) parameters. For most application, users only need to edit or add system type config file [e.g. 4pi.yaml](systemtype/4pi.yaml) and update the parameters in the demo notebook. 
|**Parameters**| Description|
|:----------------------------|:---------------------------|
|**datapath** | *string*, full path to the data|  
|**keyword** | *string*, keyword for the data file, used for automatic finding files with the same keyword|
|**savename** | *string*, full path to the result folder/filename (no extension)|  
|**subfolder** | *string*, keyword for the subfolder name if each bead data is saved in a individual folder|  
|**format** | *string*, options are `{'.mat', '.tif', '.h5', '.czi'}`, data format|  
|**stage_mov_dir** | *string*, options are `{'normal', 'reverse'}`, normal direction is where stage move towards the objective when taking a z-stack bead data| 
|**gain**| *float*, camera gain that converts the raw pixel value to photon by multiplication   
|**ccd_offset**| *float*, camera offset, the average pixel value at no light
|**roi**| | 
|&nbsp;&nbsp;&nbsp;**roi_size**|*vector[int]*, crop size of each emitter in `[y,x]` or `[z,y,x]`|
|&nbsp;&nbsp;&nbsp;**gauss_sigma**|*vector[int]*, smooth kernel size of a Gaussian filter in `[y,x]` or `[z,y,x]`|
|&nbsp;&nbsp;&nbsp;**max_kernel**|*vector[int]*, kernel size of a maximum filter in `[y,x]` or `[z,y,x]`|
|&nbsp;&nbsp;&nbsp;**peak_height**|*float*, relative intensity above which the emitters are selected|
|&nbsp;&nbsp;&nbsp;**max_bead_number**|*int*, maximum number of beads to be selected|
|&nbsp;&nbsp;&nbsp;**bead_radius**|*float*, `unit: micron`, radius of the bead|
|**pixel_size**| | 
|&nbsp;&nbsp;&nbsp;**x**|*float*, `unit: micron`, pixel size in x at the sample plane|
|&nbsp;&nbsp;&nbsp;**y**|*float*, `unit: micron`, pixel size in y at the sample plane|
|&nbsp;&nbsp;&nbsp;**z**|*float*, `unit: micron`, pixel size in z at the sample plane|
|**FOV**| | 
|&nbsp;&nbsp;&nbsp;**y_center**|*int*, y coordinate of defined FOV, within which emitters are selected|
|&nbsp;&nbsp;&nbsp;**x_center**|*int*, x coordinate of defined FOV, within which emitters are selected|
|&nbsp;&nbsp;&nbsp;**radius**|*int*, radius of defined FOV, within which emitters are selected|
|&nbsp;&nbsp;&nbsp;**z_start**|*+int*, start slice in z dimension, e.g. `1` means ignore the first slice|
|&nbsp;&nbsp;&nbsp;**z_end**|*-int*, end slice in z dimension, e.g. `-1` means ignore the last slice|
|&nbsp;&nbsp;&nbsp;**z_step**|*int*, sampling step in z, e.g. `2` means sample at every 2 slices from the original data|
|**dual**| | 
|&nbsp;&nbsp;&nbsp;**mirrortype** | *string*, options are `{'up-down','left-right'}`, mirror arrangement between two channels|
|&nbsp;&nbsp;&nbsp;**channel_arrange** | *string*, options are `{'up-down','left-right'}`, channel arrangement between two channels|
|**multi**| | 
|&nbsp;&nbsp;&nbsp;**channel_size** | *vector[int]*, size of each channel in `[y,x]`, for MFM system|
|**fpi**| | 
|&nbsp;&nbsp;&nbsp;**modulation_period** | *float*, `unit: micron`, modulation period of a 4Pi-SMLM system|
|**LLS**| | 
|&nbsp;&nbsp;&nbsp;**skew_const** | *float*, `unit: pixel`, translation in `[y,x]` per z slice, relative to the detection objective in a LLS system|
|**option**| | 
|&nbsp;&nbsp;&nbsp;**imaging**| | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**emission_wavelength** | *float*, `unit: micron`, central wavelength of the emission filter|
|&nbsp;&nbsp;&nbsp;**RI**| | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**imm** | *float*, refractive index of the immersion medium|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**med** | *float*, refractive index of the sample medium|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**cov** | *float*, refractive index of the coverglass|
|&nbsp;&nbsp;&nbsp;**NA**| *float*, numerical aperture of the objective| 
|&nbsp;&nbsp;&nbsp;**insitu**| | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**stage_pos** | *float*, `unit: micron`, position of the sample stage, equal to 0 at the coverslip and positive when move the objective towards the coverslip|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**min_photon** | *float*, quantile of the photon below which the emitters are rejected|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**partition_data** | *bool*, options are `{true, false}`, `true` means partition the emitters|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**partition_size** | *vector[int]*, define partition size, `[NO. z segments, NO. emitters per segment]` or `[NO. z segments, NO. y segments, NO. x segments, NO. emitters per segment]` for insitu-FD|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**zernike_index** | *vector[int]*, indices of non-zero Zernike coefficients for an initial pupil, if `[]`, search from lower or higher order Zernike polynomials|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**zernike_coeff** | *vector[float]*, values of Zernike coefficients defined in `zernike_index`|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**z_range** | *float*, `unit: micron`, z range of the insitu PSF model|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**zkorder_rank** | *string*, options are `{'L', 'H'}`, searching range of zernike coefficient, `'L'` means searching from 5 to 21 Zernike polynomials, `'H'` means searching from 22 to 45 Zernike polynomials|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**var_stagepos** | *bool*, options are `{true, false}`, `true` means estimate the stage position|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**repeat** | *int*, repeat number for insitu PSF modelling, the previous PSF model will be used for the next iteration|
|&nbsp;&nbsp;&nbsp;**fpi**| | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**link_zernikecoeff** | *bool*, options are `{true, false}`, `true` means link the Zernike coefficients between the four channels of a 4Pi-SMLM system|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**phase_dm** | *vector[float]*, `unit: radian`, a vector of three phase positions of a bead stack|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**sampleheight** | *float*, `unit: micron`, height of the sample chamber between the two coverslips|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**var_sampleheight** | *bool*, options are `{true, false}`, `true` means estimate the sample height|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**phase_delay_dir** | *string*, options are `{'descend', 'ascend'}`, direction of the phase increment between the four channels|
|&nbsp;&nbsp;&nbsp;**multi**| | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**defocus_offset** | *float*, `unit: micron`, defocus of the first channel in a MFM system|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**defocus_delay** | *float*, `unit: micron`, defocus increment between the channels in a MFM system|
|&nbsp;&nbsp;&nbsp;**model**| | 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**pupilsize** | *int*, pixel size of the pupil image|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**n_max** | *int*, maximum radial order of the Zernike polynomials used in modelling| 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**zernike_nl** | *vector([int,int])*, define the Zernike terms for PSF modelling in terms of `[n, l]` index, e.g. `[[2,2],[2,-2]]`, if `[]`, all zernike terms defined by `n_max` will be used| 
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**blur_sigma** | *float*, `unit: pixel`, the standard deviation of a 2D Gaussian kernel used to account for the extra blur of the measured PSF|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**var_blur** | *bool*, options are `{true, false}`, `true` means estimate the `blur_sigma`|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**with_apoid** | *bool*, options are `{true, false}`, `true` means include the apodization term in the pupil|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**const_pupilmag** | *bool*, options are `{true, false}`, `true` means set the pupil magnitude to be a unit circle
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**symmetric_mag** | *bool*, options are `{true, false}`, `true` means use only radial symmetric Zernike polynomials for the pupil magnitude|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**with_IMM** | *bool*, options are `{true, false}`, `true` means include index mismatch aberration, only used for agarose bead sample|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**init_pupil_file** | *string*, full path to the initial PSF file, the output file from uiPSF|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**estimate_drift** | *bool*, options are `{true, false}`, `true` means estimate the lateral drifts in z from bead data|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**var_photon** | *bool*, options are `{true, false}`, `true` means estimate the intensity fluctuation in z from bead data|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**bin** | *int*, upsampling rate of the camera pixel size, e.g. `2` means the pixel size of the upsampled PSF model is half of the camera pixel size|
|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**division** | *int*, number of divisions per lateral dimension of the FOV, used for modelling field-dependent aberration
|**PSFtype**| *string*, type of PSF model, options are `{'voxel', 'zernike', 'pupil', 'zernike_vector', 'pupil_vector', 'zernike_FD', 'zernike_vector_FD', 'insitu_zernike', 'insitu_pupil', 'insitu_FD'}`| 
|**channeltype**| *string*, type of system channel, options are `{'single', 'multi', '4pi'}`|
|**datatype**| *string*, type of data, options are `{'bead', 'smlm'}`|
|**loss_weight**| |
|&nbsp;&nbsp;&nbsp;**mse1**| *float*, weight for MSE in the loss function| 
|&nbsp;&nbsp;&nbsp;**mse2**| *float*, weight for modified MSE in the loss function| 
|&nbsp;&nbsp;&nbsp;**smooth**| *float*, weight for smooth regularization in the loss function|
|&nbsp;&nbsp;&nbsp;**edge**| *float*, weight for reducing edge effect in the loss function| 
|&nbsp;&nbsp;&nbsp;**psf_min**| *float*, weight for ensuring positive PSF model in the loss function| 
|&nbsp;&nbsp;&nbsp;**bg_min**| *float*, weight for ensuring positive background in the loss function| 
|&nbsp;&nbsp;&nbsp;**photon_min**| *float*, weight for ensuring positive photon in the loss function| 
|&nbsp;&nbsp;&nbsp;**Inorm**| *float*, weight for ensuring sum of the PSF model in lateral dimension being constant in the loss function| 
|&nbsp;&nbsp;&nbsp;**gxy_min**| *float*, weight for ensuring small lateral drifts in the loss function| 
|**rej_threshold**| |
|&nbsp;&nbsp;&nbsp;**bias_z**| *float*, relative z localization bias, above which the emitters will be rejected for re-learning step|
|&nbsp;&nbsp;&nbsp;**mse**| *float*, relative MSE error, above which the emitters will be rejected for re-learning step|
|&nbsp;&nbsp;&nbsp;**photon**| *float*, relative photon count, above which the emitters will be rejected for re-learning step|
|**usecuda**| *bool*, options are `{true, false}`, `true` means use GPU for localization|
|**plotall**| *bool*, options are `{true, false}`, `true` means show plots during the modelling process|
|**ref_channel**| *int*, index of the reference channel, for multi-channel or 4Pi system|
|**batch_size**| *int*, maximum number of emitters to be modeled at one time|
|**iteration**| *int*, number of iterations for the optimization process|
|**varname**| *string*, the variable name for the data, used for simulated matlab data|
|**filelist**| *vector[string]*, a list of data files, if `[]`, the data files will be identified based on the `keyword`|
|**swapxy**| *bool*, options are `{true, false}`, `true` means permute the x, y dimensions of the raw images|








