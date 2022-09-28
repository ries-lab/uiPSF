clearvars
%% enviroment path
if ispc
    envpath = 'C:\Users\Ries Lab\anaconda3\envs\myenv';
    runpath = 'C:\Users\Ries Lab\git\PSFlearningTF3\psfmodelling\examples';
else
    envpath = '/Applications/anaconda3/envs/myenv';
    runpath = '/Users/shengliu/Documents/Python/psflearningTF3/psfmodelling/examples';
end
[p1,env]=fileparts(envpath);
condapath=fileparts(p1);

%% setup parameters
default_paramfile = 'params_default.json';

fid = fopen(default_paramfile); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
f = jsondecode(str);

datapath = 'Z:\projects\PSFLearning\exampledata\LLS_100\'; % Z: t2ries 
datapath = strrep(strrep(datapath,'\',filesep),'/',filesep);

data_keyword = 'bead';
subfolder = ''; % keyword for subfolder, leave it empty, if there is no subfolder
savename = 'psfmodel';
data_format = '.czi'; % '.mat', '.tif'
PSFtype = 'voxel'; % 'voxel', 'pupil','zernike','zernike_vector'
chtype = 'single'; % 'single','multi','4pi'

gain = 1;
ccd_offset = 100;
roi_size = [27,30,30];
gaus_sigma = [6,2,2];
max_kernel = [9,3,3];
peak_height = 0.5;

pixelsize_x = 0.145; %um
pixelsize_y = 0.145; %um
pixelsize_z = 0.1;%um
bead_radius = 0.0;%um
loss_Inorm = 0; % it is better to set to zero for most cases
loss_smooth = 1; % for pupil based learning, it will smooth the pupil, better set to 0
iteration = 100;
FOV = struct('y_center',250,'x_center',1150,'radius',300,'z_start',0,'z_end',0,'z_step',1); % if radius is zero, use the full FOV, 'z_start' is counting as 0,1,2..., 'z_end' is counting as 0,-1,-2...
rej_threshold = struct('bias_z',5,'mse',3,'photon',1.5);

% 0: false, 1:true
estdrift = 1;
varphoton = 1; % set to 1, for all PSF types
usecuda = 1; % use GPU or CPU for localization, will be set to zero automatically for mac or if error for loading GPU function
plotall = 1;

% for dual channel (multi) only
mirrortype = 'up-down';
channel_arrange = 'up-down';
ref_channel = 0;

% for 4pi only
zT = 0.26; %um

% for LLS PSF only
skew_const = [-1.194,0];

% for zernike or pupil based learning only
optparam = struct('emission_wavelength',0.6,'RI',1.516,'RI_med',1.33,'RI_cov',1.516,'NA',1.43,'pupilsize',64,'n_max',9,'gauss_filter_sigma',2);


f.datapath = datapath;
f.keyword = data_keyword;
f.subfolder = subfolder;
f.savename = savename;
f.format = data_format;
f.PSFtype = PSFtype;
f.channeltype = chtype;
f.mirrortype = mirrortype;
f.channel_arrange = channel_arrange;
f.ref_channel = ref_channel;
f.gain = gain;
f.ccd_offset = ccd_offset;
f.roi_size = roi_size;
f.gaus_sigma = gaus_sigma;
f.max_kernel = max_kernel;
f.peak_height = peak_height;
f.pixelsize_x = pixelsize_x;
f.pixelsize_y = pixelsize_y;
f.pixelsize_z = pixelsize_z;
f.bead_radius = bead_radius;
f.loss_weight.Inorm = loss_Inorm;
f.loss_weight.smooth = loss_smooth;
f.iteration = iteration;
f.rej_threshold = rej_threshold;
f.option_params = optparam;
f.FOV = FOV;
f.modulation_period = zT;
f.estimate_drift = estdrift==1;
f.vary_photon = varphoton==1;
f.usecuda = usecuda==1;
f.plotall = plotall==1;
f.skew_const = skew_const;

paramfile = 'params.json';
encode_str = jsonencode(f,'PrettyPrint',true);
fid = fopen(paramfile,'w'); 
fwrite(fid,encode_str); 
fclose(fid);
%% run python script
pythonfile = 'learn_psf.py';
command = ['python ' pythonfile ' ' paramfile];

if ispc
    pcall=['call "' condapath '\Scripts\activate.bat" ' env ' & cd "' runpath '" & ' command ' & exit &'];
    
else
    cd(runpath)
    pcall=[envpath '/bin/'  command ];
end
[status, results]=system(pcall,'-echo');
%% load h5
addpath('easyh5')
filename = [datapath,savename,'_',PSFtype,'_',chtype,'.h5'];
Fm = loadh5(filename);
val = h5readatt(filename,'/','params');
params = jsondecode(val);

