import matplotlib.pyplot as plt

from auto_encoder_sensor import FlexibleTimeSeriesAutoencoder
import polars as po
import torch
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(data,feature):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data[feature].to_numpy().reshape(-1, 1))
    X_tensor = torch.FloatTensor(normalized_data.flatten())
    return X_tensor, scaler

def Force_1024():
    feature='Force'
    input_size = 5300
    hidden_sizes = [4096, 2048, 1024]   # Adjust the number and size of hidden layers as needed
    bottleneck_size = 1024  # You can change this size based on your requirements
    autoencoder = FlexibleTimeSeriesAutoencoder(input_size, hidden_sizes, bottleneck_size)
    autoencoder = autoencoder.load_from_checkpoint(
        "/media/maxi/T7 Shield/Arbeit/AutoEncoder/saved_models/autoencoder_Sensors/Force1024_epoch=22-val_loss=0.0013920576311647892.ckpt")

    autoencoder.eval()
    autoencoder.cuda()
    return autoencoder, feature

def AE_1024():
    feature='AE'
    input_size = 5300
    hidden_sizes = [4096, 2048, 1024]   # Adjust the number and size of hidden layers as needed
    bottleneck_size = 1024  # You can change this size based on your requirements
    autoencoder = FlexibleTimeSeriesAutoencoder(input_size, hidden_sizes, bottleneck_size)
    autoencoder = autoencoder.load_from_checkpoint(
        "/media/maxi/T7 Shield/Arbeit/AutoEncoder/saved_models/autoencoder_Sensors/AE1024_epoch=21-val_loss=0.0014418830396607518.ckpt")

    autoencoder.eval()
    autoencoder.cuda()
    return autoencoder, feature

def inference_encoder(model,file_path, feature):
    data = po.read_csv(file_path)
    input, scaler = preprocess_data(data, feature)
    pred = model.encoder(input.cuda().unsqueeze(0))
    np_pred = pred.detach().cpu().numpy()[0]
    return pred, np_pred


if __name__ == "__main__":
    # feature='AE'
    # input_size = 5300
    # hidden_sizes = [4096, 2048, 1024]   # Adjust the number and size of hidden layers as needed
    # bottleneck_size = 1024  # You can change this size based on your requirements
    # autoencoder = FlexibleTimeSeriesAutoencoder(input_size, hidden_sizes, bottleneck_size)
    # autoencoder = autoencoder.load_from_checkpoint(
    #     "/media/maxi/T7 Shield/Arbeit/AutoEncoder/saved_models/autoencoder_Sensors/AE1024_epoch=21-val_loss=0.0014418830396607518.ckpt")
    #
    # autoencoder.eval()
    # autoencoder.cuda()
    autoencoder, feature=AE_1024()

    file_path='/media/maxi/T7 Shield/Arbeit/V32/punch_dataset/Sensors/38517.csv'
    pred,np_pred=inference_encoder(autoencoder,file_path, feature)

    data = po.read_csv(file_path)
    input, scaler = preprocess_data(data, feature)

    pred = autoencoder(input.cuda().unsqueeze(0))
    pred = pred.detach().cpu().numpy()[0]
    pred = scaler.inverse_transform(pred.reshape(-1, 1))

    plt.plot(data[feature], label='org')
    plt.plot(pred,label='re')
    plt.legend()
    plt.show()