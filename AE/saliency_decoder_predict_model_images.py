import random
import numpy as np
import os
import json
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time
import torch
from autoencoder_images import Autoencoder


def preprocess_data(imgpath):
    with open(imgpath, 'rb') as f:
        img = Image.open(f)
        image = img.convert('L')
    img_size = 256
    creator_tensor = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    image = creator_tensor(image)
    return image

def show_data(imgpath):
    with open(imgpath, 'rb') as f:
        img = Image.open(f)
        image = img.convert('L')
    img_size = 256
    creator_tensor = transforms.Compose([transforms.Resize((img_size, img_size))])
    image = creator_tensor(image)
    return image


def Image_2048():
    depth = 5
    bottleneck_size = 2048
    model = Autoencoder(depth=depth, bottleneck_size=bottleneck_size)
    model = Autoencoder.load_from_checkpoint("/media/maxi/T7 Shield/Arbeit/AutoEncoder/saved_models/autoencoder_images_batchnorm/Images2048_epoch=499-val_loss=0.0026328936219215393.ckpt")

    model.eval()
    #model.cuda()
    return model

def Decoder_Image_2048():
    depth = 5
    bottleneck_size = 2048
    model = Autoencoder(depth=depth, bottleneck_size=bottleneck_size)
    dec_model = Autoencoder.build_decoder(model, depth=depth, bottleneck_size=bottleneck_size)
    model = Autoencoder.load_from_checkpoint("/media/maxi/T7 Shield/Arbeit/AutoEncoder/saved_models/autoencoder_images_batchnorm/Images2048_epoch=499-val_loss=0.0026328936219215393.ckpt")
    dec_model.load_state_dict(model.decoder.state_dict())
    #model.eval()
    #model.cuda()
    return dec_model

def inference_im_encoder(model, impath):
    image=preprocess_data(impath)
    pred = model.encoder(image.cuda().unsqueeze(0))
    np_pred = pred.detach().cpu().numpy()[0]
    return pred, np_pred

def inference(model,impath):
    image=preprocess_data(impath)
    pred = model(image.cuda().unsqueeze(0))
    np_pred = pred.detach().cpu().numpy()[0,0]
    return pred, np_pred

def inference_sal(model,impath):
    image=preprocess_data(impath)
    image.requires_grad = True
    pred = model(image.cuda().unsqueeze(0))
    np_pred = pred.detach().cpu().numpy()[0,0]
    return pred, np_pred

def decode_output(output):
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    list = ["Eve", "Alice", "Bob"]

    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 3)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
    return top5_catid[0]

if __name__ == "__main__":
    #
    data_dir = "/media/maxi/T7 Shield/Arbeit/V32/punch_dataset/Images"
    all_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if
                 filename.endswith('.jpg')]
    bottle=Image_2048()
    model=Decoder_Image_2048()
    test = torch.randn(1,2048)
    test = model(test)
    np_pred = test.detach().cpu().numpy()[0][0]

    plt.imshow(np_pred, cmap=plt.cm.gray)
    plt.show()
    meani = []

    ranlist=random.choices(all_files,k=32)[:50]
    for i in ranlist:
        pred,np_pred=inference_im_encoder(bottle,i)
        meani.append(np_pred)



        #p = Image.fromarray((np_pred*255.9999).astype(np.uint8))
        #p.show()
        #pathsave='/media/maxi/T7 Shield/Arbeit/AutoEncoder/Image_test'
        #file=i.split('/')[-1][:-4]
        #print(os.path.join(pathsave,file+'_'+str(bottleneck_size)+'ep200_ergebnis.png'))
        #p.save(os.path.join(pathsave,file+'_'+str(bottleneck_size)+'ep200_ergebnis.png'))

    meani=np.array(meani)
    meani_one=np.mean(meani, axis=0)

    # test = model(torch.from_numpy(meani))
    # np_pred = test.detach().cpu().numpy()[0][0]

    test = model(torch.from_numpy(meani_one).unsqueeze(0))
    np_pred = test.detach().cpu().numpy()[0][0]
    p = Image.fromarray((np_pred * 255.9999).astype(np.uint8))
    pathsave='/media/maxi/T7 Shield/Arbeit/AutoEncoder/attention'
    print(os.path.join(pathsave,'001_mean.png'))
    p.save(os.path.join(pathsave, '001_mean.png'))

    for i in tqdm(range(0,2048)):
        mani= meani_one.copy()
        mani[i] = 100
        z = np.zeros_like(mani)
        z[i] = 100


        test = model(torch.from_numpy(mani).unsqueeze(0))
        np_mani = test.detach().cpu().numpy()[0][0]

        plt.imsave(os.path.join(pathsave, '00'+str(i) + '_manipuliert_mean.png'), np_mani, cmap=plt.cm.gray)
        tt=Image.fromarray((np_pred - np_mani * 255.9999).astype(np.uint8))
        im_invert = ImageOps.invert(tt)
        tt.save(os.path.join(pathsave, '00'+str(i) + 'diff_mean_manipuliert.png'))
        im_invert.save(os.path.join(pathsave, '00'+str(i) + 'diff_mean_manipuliert_invert.png'))

        test = model(torch.from_numpy(z).unsqueeze(0))
        z = test.detach().cpu().numpy()[0][0]

        plt.imsave(os.path.join(pathsave, '00'+str(i)+'_plain.png'), z , cmap=plt.cm.hot)

    #plt.imshow(np_pred, cmap=plt.cm.gray)
    #plt.show()

    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 2, 1)
    # plt.imshow(np_pred, cmap=plt.cm.gray)
    # plt.xticks([])
    # plt.yticks([])
    # plt.subplot(1, 2, 2)
    # plt.imshow(np_pred-np_mani, cmap=plt.cm.gray)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()





    #p.show()

    #print(model.encoder(image.cuda().unsqueeze(0)))

