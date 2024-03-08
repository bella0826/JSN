# Super parameters
clamp = 2.0
channels_in = 1
log10_lr = -5.0
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 3
lamda_guide = 1         
lamda_low_frequency = 0
device_ids = [0]

# Train:
batch_size = 16
cropsize = 224
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 512
batchsize_val = 2
shuffle_val = False
val_freq = 50


# Dataset
TRAIN_PATH = '/home/han/DIV2K_train_HR/DIV2K_train_HR/'
VAL_PATH = '/home/han/DIV2K_valid_HR/DIV2K_valid_HR/'
BACKWARD_PATH = '/home/han/HiNET/image/YCbCr/'
format_train = 'png'
format_val = 'png'
format_backward = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

MODEL_PATH = '/home/han/model/'
checkpoint_on_error = True
SAVE_freq = 50

IMAGE_PATH = '/home/han/HiNET/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover1/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret1/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg1/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev1/'
IMAGE_PATH_backward = IMAGE_PATH + 'YCbCr/'

# Load:
suffix = 'model_with8x8_3.pt'
tain_next = False
trained_epoch = 0

# LoadCb
suffix_cb = 'model_cb.pt'

# channel after DCT
channel_dct = 64
blocksize_dct = 8