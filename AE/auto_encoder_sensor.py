import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl
import polars as po
import mlflow
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchsummary import summary
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import LearningRateMonitor


class FileCollector:
    """
    base class for listing files inside a directory
    """
    def __init__(self, base_dir):
        """
        Params:
            basedir:
                root folder for data listing
        """
        self.base_dir = base_dir
        self.images = []

    def add_data(self, img_dir, subdirs = True):
        """
        adds the data within the given directories to the file list

        img_dir and mask_dir must have same internal structure and files must be named the same

        Params:
            img_dir:
                folder with the images

                path is evaluated relative to given base dir
            mask_dir:
                folder with the masks

                path is evaluated relative to given base path
            subdirs:
                flag, if subdirectories of img_dir and mask_dir should be evaluated
        """
        self.__collect_image_names(img_dir, subdirs = subdirs)
        self.size = len(self.images)

    def __collect_image_names(self, img_path, current = "", subdirs = True):
        """
        collects the data at the given location

        collection is handeled recursive

        the collected data are tuples with structure:
            (img_path, mask_path, path_to_image_inside_img_path)

        Params:
            img_path:
                path with images, relative to base dir
            mask_path:
                path with masks, relative to base dir
            current:
                current prefix of subdirectory
            subdirs:
                flag, if subdirectories of img_dir and mask_dir should be evaluated
        """
        #get data about current folder
        path, folders, files=next(os.walk(os.path.join(self.base_dir, img_path, current)))

        #collect images in style (img_path, mask_path, file_path)
        for image in files:
            self.images.append((img_path, os.path.join(current, image)))

        #traverse subfolders if requested
        if subdirs:
            for folder in folders:
                self.__collect_image_names(img_path, os.path.join(current, folder), subdirs)

class CustomCSVDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        # Load and preprocess your data point
        # For example, assuming each CSV file contains a different data point
        data = po.read_csv(file_path)
        if self.transform:
            data = self.transform(data)
        return data

    def next(self):
        """
        handles generator iteration
        """
        #check current index
        if self.index >= self.size:
            raise StopIteration("out of elements")

        #load data at given image
        data = self.getitem(self.index)

        #increase index
        self.index += 1

        return data

    def __next__(self):
        return self.next()

class CustomPolarsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, train_size=0.8, val_size=0.1, test_size=0.1, feature=None):
        super(CustomPolarsDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.feature=feature

    def setup(self, stage=None):
        # Create a list of file paths for your CSV files
        all_files = [os.path.join(self.data_dir, filename) for filename in os.listdir(self.data_dir) if filename.endswith('.csv')]
        all_files = all_files[654:]
        # Split the data into training, validation, and test sets
        train_files, test_files = train_test_split(all_files, test_size=self.test_size, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=self.val_size/(self.train_size + self.val_size), random_state=42)
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomCSVDataset(train_files, transform=self.preprocess_data)
            self.val_dataset = CustomCSVDataset(val_files, transform=self.preprocess_data)
        if stage == 'test' or stage is None:
            self.test_dataset = CustomCSVDataset(test_files, transform=self.preprocess_data)

    def preprocess_data(self, data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(data[self.feature].to_numpy().reshape(-1,1))
        X_tensor = torch.FloatTensor(normalized_data.flatten())
        return X_tensor

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class FlexibleTimeSeriesAutoencoder(pl.LightningModule):
    def __init__(self, input_size, hidden_sizes, bottleneck_size, learning_rate=0.001):
        super(FlexibleTimeSeriesAutoencoder, self).__init__()
        self.save_hyperparameters()

        # Encoder
        encoder_layers = []
        for i in range(len(hidden_sizes)):
            encoder_layers.append(nn.Linear(input_size if i == 0 else hidden_sizes[i - 1], hidden_sizes[i]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(hidden_sizes[i]))
        encoder_layers.append(nn.Linear(hidden_sizes[-1], bottleneck_size))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for i in range(len(hidden_sizes) - 1, -1, -1):
            decoder_layers.append(nn.Linear(bottleneck_size if i == 2 else hidden_sizes[i+1], hidden_sizes[i]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(hidden_sizes[i]))
        decoder_layers.append(nn.Linear(hidden_sizes[0], input_size))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        # Loss function
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, inputs)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="mean")
        #loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

if __name__ == "__main__":
    # Load time series data from a CSV file
    data_dir = "/media/maxi/T7 Shield/Arbeit/V32/punch_dataset/Sensors"

    custom_polars_data_module = CustomPolarsDataModule(data_dir, batch_size=64, feature="Force")
    # Assuming the time series data is in columns 'feature1', 'feature2', ..., 'featureN'
    features = 5300

    # Instantiate the PyTorch Lightning model with multiple layers and a flexible bottleneck size
    input_size = 5300
    hidden_sizes = [4096, 2048, 1024]  # Adjust the number and size of hidden layers as needed
    bottleneck_size = 1024  # You can change this size based on your requirements
    autoencoder = FlexibleTimeSeriesAutoencoder(input_size, hidden_sizes, bottleneck_size)
    #summary(autoencoder,[5300])

    checkpoint_callback = ModelCheckpoint(
        dirpath='/media/maxi/T7 Shield/Arbeit/AutoEncoder/saved_models/autoencoder_Sensors', every_n_epochs=1,
        filename='Force1024_{epoch}-{val_loss}', monitor="val_loss", save_top_k=1)


    # Set up MLflow experiment tracking
    mlflow.set_experiment('Sensor_bottleneck_lr')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Set up a PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=30,
                         accelerator="auto",
                         callbacks=[TQDMProgressBar(refresh_rate=10), lr_monitor, checkpoint_callback])

    # Start MLflow run
    with mlflow.start_run(run_name="Force_512"):
        # Log model hyperparameters
        mlflow.log_params(autoencoder.hparams)

        # Train the autoencoder
        trainer.fit(autoencoder, custom_polars_data_module)

        # Log the final metrics and model
        mlflow.log_metrics({'final_train_loss': autoencoder.trainer.callback_metrics['train_loss'].item(),
                            'final_val_loss': autoencoder.trainer.callback_metrics['val_loss'].item()})

        #mlflow.pytorch.log_model(autoencoder, "mlruns/models")

    # Load the model from MLflow for inference
    #loaded_model = mlflow.pytorch.load_model("models")

    # Perform inference on the test data
#with torch.no_grad():
#        test_outputs = loaded_model(test_data)


# from scipy.stats import shapiro
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
#
#data = po.read_csv("/media/maxi/T7 Shield/Arbeit/V32/punch_dataset/Sensors/10612.csv")
#
# # Assuming 'data' is your dataset (replace it with your actual data)
# stat, p_value = shapiro(data)
#
# # Check the p-value
# alpha = 0.05
# if p_value > alpha:
#     print("The data looks normally distributed (fail to reject the null hypothesis)")
# else:
#     print("The data does not look normally distributed (reject the null hypothesis)")
#
# # Specify the window size for the rolling mean
# window_size = 3
#
# datatmp=data.with_columns(rolling_mean=po.col('Force').rolling_mean(window_size))["rolling_mean"]
#
# plt.plot(data.with_columns(rolling_mean=po.col('Force').rolling_mean(window_size))["rolling_mean"])
# plt.show()
#
# plt.hist(data["Force"], bins='auto', alpha=0.7, color='blue', edgecolor='black')
# plt.title('Histogram of Data')
# plt.xlabel('Values')
# plt.ylabel('Frequency')
# plt.show()
#
# sm.qqplot(datatmp, line='s')
# plt.title('Q-Q Plot')
# plt.show()

