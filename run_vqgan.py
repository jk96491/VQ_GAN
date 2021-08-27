import torch
from VQ_GAN.Parser_args import parse_Arg
from torchvision.utils import save_image
from Utils import CIFARLoadData
from Utils import SkinDataLoad
from Utils import get_device
from Utils import saveImages
from  VQ_GAN.model import vq_gan
import os

args = parse_Arg()

batch_size = 32

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 4

embedding_dim = 64
num_embeddings = 256

commitment_cost = 0.25

decay = 0.99

just_gen = True

#mode = "dry"
#mode = "normal"
mode = "wet"

train_loader = SkinDataLoad(mode, args.batch_size, True)

device1 = 'cuda:0'
device2 = 'cuda:1'

device = [device2, device1]

target_epoch = 9950


def save_model(model, mode, epoch):
    path = "Trained_Models/{0}".format(mode)
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), "{0}/{1}.th".format(path, epoch))


def load_model(model, mode, target_epoch):
    path = "Trained_Models/{0}".format(mode)
    os.makedirs(path, exist_ok=True)
    model.load_state_dict(torch.load("{0}/{1}.th".format(path, target_epoch), map_location=lambda storage, loc: storage))


def generate_image(model, mode, target_epoch):
    load_model(model, mode, target_epoch)
    for count in range(10):
        for i, data in enumerate(train_loader):
            real_images, label, fileNames = data
            inputs = real_images.clone().to(device1)

            fake = model.generate_image(inputs)

            for i in range(len(fake)):
                if label[i] == 0:
                    path = "generate_images/dry"
                elif label[i] == 1:
                    path = "generate_images/normal"
                else:
                    path = "generate_images/wet"

                os.makedirs(path, exist_ok=True)
                save_image(fake[i], "{0}/{1}_{2}.png".format(path, fileNames[i], count), nrow=1, normalize=False)

            print(count + 1)

def learn_model():
    for epoch in range(args.n_epochs):
        for i, data in enumerate(train_loader):
            real_images, label, fileNames = data

            current_batch_size = real_images.size(0)

            inputs = real_images.clone().to(device1)

            real_labels = torch.ones(current_batch_size, 1).detach().to(device1)
            fake_labels = torch.zeros(current_batch_size, 1).detach().to(device1)

            discriminator_loss, generator_image = model.learn_discriminator(inputs, real_labels, fake_labels)
            generator_loss, perplexity = model.learn_generator(inputs, real_labels)

            if epoch % 50 == 0:
                log_data = "[Epoch %d/%d] [Batch %d/%d] [Discriminator_loss: %f] [Generator_loss: %f] [perplexity : %f]" % \
                           (epoch + 1, args.n_epochs, i + 1, len(train_loader), discriminator_loss, generator_loss,
                            perplexity)
                print(log_data)

                f = open("log.txt", 'a')
                f.write("{0}\n".format(log_data))
                f.close()

                for i in range(len(fileNames)):
                    if label[i] == 0:
                        path = "images/dry/{0}".format(fileNames[i])
                    elif label[i] == 1:
                        path = "images/normal/{0}".format(fileNames[i])
                    else:
                        path = "images/wet/{0}".format(fileNames[i])

                    os.makedirs(path, exist_ok=True)
                    save_image(generator_image[i], "{0}/Epoch_{1}.png".format(path, epoch), nrow=1, normalize=False)

                save_model(model, mode, epoch)


if __name__ == '__main__':
    model = vq_gan(args, num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim,
                  commitment_cost, decay, device)

    if just_gen is False:

        learn_model()
    else:
        generate_image(model, mode, target_epoch)



