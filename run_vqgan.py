import torch
from VQ_GAN.Parser_args import parse_Arg
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from VQ_GAN.model import vq_gan
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os

training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                     # transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                    #  transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))
args = parse_Arg()

batch_size = 256
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 4

embedding_dim = 64
num_embeddings = 256
commitment_cost = 0.25
decay = 0.99

train_loader = DataLoader(training_data,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True)

device1 = 'cuda:0'
device2 = 'cuda:0'

device = [device1, device1]

final_index = len(training_data.targets) // batch_size

target_epoch = 9950


def save_model(model, epoch):
    path = "Trained_Models"
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), "{0}/{1}.th".format(path, epoch))


def load_model(model, target_epoch):
    path = "Trained_Models"
    os.makedirs(path, exist_ok=True)
    model.load_state_dict(torch.load("{0}/{1}.th".format(path, target_epoch), map_location=lambda storage, loc: storage))


def learn_model():
    for epoch in range(args.n_epochs):
        for i, data in enumerate(train_loader):
            real_images, label = data

            current_batch_size = real_images.size(0)

            inputs = real_images.clone().to(device1)

            real_labels = torch.ones(current_batch_size, 1).detach().to(device1)
            fake_labels = torch.zeros(current_batch_size, 1).detach().to(device1)

            discriminator_loss, generator_image = model.learn_discriminator(inputs, real_labels, fake_labels)
            generator_loss, perplexity = model.learn_generator(inputs, real_labels)


            log_data = "[Epoch %d/%d] [Batch %d/%d] [Discriminator_loss: %f] [Generator_loss: %f] [perplexity : %f]" % \
                       (epoch + 1, args.n_epochs, i + 1, len(train_loader), discriminator_loss, generator_loss,
                        perplexity)
            print(log_data)

            if i == final_index:
                save_image(generator_image[:16], "images/Epoch_{0}.png".format(epoch), nrow=4, normalize=False)



if __name__ == '__main__':
    model = vq_gan(args, num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim,
                  commitment_cost, decay, device)

    learn_model()



