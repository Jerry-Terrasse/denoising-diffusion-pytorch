import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

import torchvision
from torchvision.transforms.functional import resize

from matplotlib import pyplot as plt

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 100,    # number of steps
    sampling_timesteps=25
)

# training_images = torch.rand(8, 3, 128, 128) # images are normalized from 0 to 1
# loss = diffusion(training_images)
# loss.backward()

# after a lot of training

# sampled_images = diffusion.sample(batch_size = 4)
# sampled_images.shape # (4, 3, 128, 128)


trainer = Trainer(
    diffusion,
    './cifar10_dog',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

trainer.train()

# fname = "lena.jpg"
# img = torchvision.io.read_image(fname)
# # to 128x128
# img = resize(img, [128, 128], antialias=True)
# img = img.float() / 255.0

# for t in range(0, 1000, 100):
#     img = diffusion.q_sample(img, torch.tensor([t]))
#     plt.imshow(img.permute(1, 2, 0))
#     plt.show()

# # show
# plt.imshow(img.permute(1, 2, 0))
# plt.show()