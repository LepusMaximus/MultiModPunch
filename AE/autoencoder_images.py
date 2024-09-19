import os
import numpy as np
import urllib.request
from urllib.error import HTTPError
import tensorboard as tf
import pytorch_lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.notebook import tqdm
import mlflow.pytorch
from torchsummary import summary
from PIL import Image


# Tensorboard extension (for visualization purposes later)
#%load_ext tensorboard

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = os.environ.get("PATH_DATASETS", "data")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/tutorial9")

# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()
sns.set()

from lib.utils.dataloader import DSet, get_loader, MonochromeDataset as TrainDataset, MonochromeTestDataset as TestDataset, img_by_name
DATASET_PATH = "/media/maxi/T7 Shield/Arbeit/V32/punch_dataset"
#DATASET_PATH = "D:\\Arbeit\\V32\\punch_dataset"

#creates train set
train_set = TrainDataset(DATASET_PATH, img_size=256, augmentations=False, class_channels=2)
train_set.add_data("Images")

#creates test set
train_set, test_set = train_set.split_dset(0.2)
train_set, val_set = train_set.split_dset(0.1)


train_loader = get_loader(train_set, batch_size=24)


def compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True, range=(-1, 1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4, 2))
    plt.title(f"{title_prefix} Loss: {loss.item():4.2f}")
    plt.imshow(grid)
    plt.axis("off")
    plt.show()

# We define a set of data loaders that we can use for various purposes later.
train_loader = data.DataLoader(train_set, batch_size=512, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)

lr_monitor = LearningRateMonitor(logging_interval='epoch')
checkpoint_callback = ModelCheckpoint(
    dirpath='/media/maxi/T7 Shield/Arbeit/AutoEncoder/runs', every_n_epochs=10,
    filename='l6_{epoch}-{val_loss}', monitor="val_loss", save_top_k=5)


#train_loader = get_loader(train_set, batch_size=24)
#test_loader = get_loader(test_set, batch_size=24)
#val_loader = get_loader(val_set, batch_size=24)

#def create image
def create_log_image(img):
    img = img.detach().cpu().numpy()[0, 0]
    p = Image.fromarray((img* 255.9999).astype(np.uint8))
    p.save("Image_test/test.png")
    log_image("Image_test/test.png")

# Log an image file as an artifact
def log_image(file_path, artifact_subdir='images'):
    mlflow.log_artifact(file_path, artifact_subdir)

def get_train_images(num):
    return torch.stack([train_set[i][0] for i in range(num)], dim=0)

class Autoencoder(L.LightningModule):
    def __init__(self, depth, bottleneck_size):
        super(Autoencoder, self).__init__()
        self.save_hyperparameters()

        self.encoder = self.build_encoder(depth, bottleneck_size)
        self.decoder = self.build_decoder(depth, bottleneck_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def build_encoder(self, depth, bottleneck_size):
        layers = []
        in_channels = 1  # Grayscale image

        for _ in range(depth):
            layers += [
                nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels * 2),
                nn.ReLU(inplace=True),
            ]
            in_channels *= 2

        layers += [
            nn.Flatten(),
            nn.Linear(in_channels * 8 * 8, bottleneck_size),
            nn.ReLU(inplace=True),
        ]

        return nn.Sequential(*layers)

    def build_decoder(self, depth, bottleneck_size):
        layers = []
        in_channels = 1 * 32 # Initial input channels in decoder

        layers += [
            nn.Linear(bottleneck_size, in_channels * 8 * 8),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, torch.Size([32,8,8]))
        ]

        for _ in range(depth-1):
            layers += [
                nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
            ]
            in_channels //= 2

        layers += [
            nn.ConvTranspose2d(in_channels, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        ]

        return nn.Sequential(*layers)

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="mean")
        #loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=10, min_lr=5e-7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)

class GenerateCallback(Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                create_log_image(reconst_imgs[0])
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))


if __name__ == "__main__":
    depth = 5
    bottleneck_size = 2048
    model = Autoencoder(depth=depth, bottleneck_size=bottleneck_size)
    summary(model, (1, 256,256), device="cpu")

    # for i in range(2):
    #     # Load example image
    #     img, _ = train_set[i]
    #     img_mean = img.mean(dim=[1, 2], keepdims=True)
    #
    #     compare_imgs(img, img, "Original -")
    #
    #     # Shift image by one pixel
    #     SHIFT = 1
    #     img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
    #     img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
    #     img_shifted[:, :1, :] = img_mean
    #     img_shifted[:, :, :1] = img_mean
    #     compare_imgs(img, img_shifted, "Shifted -")
    #
    #     # Set half of the image to zero
    #     img_masked = img.clone()
    #     img_masked[:, : img_masked.shape[1] // 2, :] = img_mean
    #     compare_imgs(img, img_masked, "Masked -")

    checkpoint_callback = ModelCheckpoint(
        dirpath='/media/maxi/T7 Shield/Arbeit/AutoEncoder/saved_models/autoencoder_images_batchnorm', every_n_epochs=1,
        filename='Images2048_{epoch}-{val_loss}', monitor="val_loss", save_top_k=1)
    #early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode='min', stopping_threshold=0.00005)

    mlflow.pytorch.autolog()
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=500,
        callbacks=[TQDMProgressBar(refresh_rate=10), lr_monitor, checkpoint_callback]
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    #pretrained_filename = os.path.join(CHECKPOINT_PATH, "cifar10_%i.ckpt" % latent_dim)
    mlflow.set_experiment('Image_Autoencoder_batch')
    #
    with mlflow.start_run(run_name="2048_train_ep200" ) as run:
            trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}

    # latent_dims = sorted(k for k in model_dict)
    # val_scores = [model_dict[k]["result"]["val"][0]["test_loss"] for k in latent_dims]
    #
    # fig = plt.figure(figsize=(6, 4))
    # plt.plot(
    #     latent_dims, val_scores, "--", color="#000", marker="*", markeredgecolor="#000", markerfacecolor="y", markersize=16
    # )
    # plt.xscale("log")
    # plt.xticks(latent_dims, labels=latent_dims)
    # plt.title("Reconstruction error over latent dimensionality", fontsize=14)
    # plt.xlabel("Latent dimensionality")
    # plt.ylabel("Reconstruction error")
    # plt.minorticks_off()
    # plt.ylim(0, 100)
# plt.show()