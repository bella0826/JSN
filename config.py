# Super parameters
clamp = 2.0
channels_in = 1
log10_lr = -5.2
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 4
lamda_guide = 2         
lamda_low_frequency = 1
device_ids = [0]

# Train:
batch_size = 2
cropsize = 256
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 256
batchsize_val = 2
shuffle_val = False
val_freq = 100


# Dataset
TRAIN_PATH = '/home/han/DIV2K_train_HR/DIV2K_train_HR/'
VAL_PATH = '/home/han/DIV2K_valid_HR/DIV2K_valid_HR/'
BACKWARD_PATH = '/home/han/HiNET/image/steg_0607/'
format_train = 'png'
format_val = 'png'
format_backward = 'jpg'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = '/home/han/model/'
checkpoint_on_error = True
SAVE_freq = 100

IMAGE_PATH = '/home/han/HiNET/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover1/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret1/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg1/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev1/'

# Load:
suffix = 'model_checkpoint_01000_5.pt'
tain_next = False
trained_epoch = 0

# channel after DCT
channel_dct = 4
blocksize_dct = 128