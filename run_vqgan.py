import torch
from VQ_GAN.Parser_args import parse_Arg
from torchvision.utils import save_image
from Utils import CIFARLoadData
from Utils import SkinDataLoad
from Utils import get_device
from Utils import saveImages
from  VQ_GAN.model import vq_gan


args = parse_Arg()

batch_size = 32
num_training_updates = 55000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

train_loader = SkinDataLoad(args.batch_size, True)

device1 = 'cuda:0'
device2 = 'cuda:1'

device = [device2, device1]

if __name__ == '__main__':
    model = vq_gan(args, num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim,
                  commitment_cost, decay, device)

    for epoch in range(args.n_epochs):
        for i, data in enumerate(train_loader):
            real_images, label = data
            current_batch_size = real_images.size(0)

            inputs = real_images.clone().to(device1)
            noise = torch.zeros(current_batch_size, args.noise_dim, 1, 1).normal_(0, 1)

            real_labels = torch.ones(current_batch_size, 1).detach().to(device1)
            fake_labels = torch.zeros(current_batch_size, 1).detach().to(device1)

            discriminator_loss, generator_image = model.learn_discriminator(inputs, real_labels, fake_labels)
            generator_loss, perplexity = model.learn_generator(inputs, real_labels)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [Discriminator_loss: %f] [Generator_loss: %f] [perplexity : %f]"
                % (epoch + 1, args.n_epochs, i + 1, len(train_loader), discriminator_loss, generator_loss, perplexity)
            )

            batches_done = epoch * len(train_loader) + i

            if batches_done % len(train_loader) == 0 and epoch % 10 == 0:
                real_data = inputs[0:16].cpu()
                test_data = generator_image[0:16].cpu()
                label = label[0:16]

                dry = test_data[label == 0]
                normal = test_data[label == 1]
                wet = test_data[label == 2]

                if len(dry) != 0:
                    saveImages(real_data[label == 0], dry, epoch, "dry")
                if len(normal) != 0:
                    saveImages(real_data[label == 1], normal, epoch, "normal")
                if len(wet) != 0:
                    saveImages(real_data[label == 2], wet, epoch, "wet")


