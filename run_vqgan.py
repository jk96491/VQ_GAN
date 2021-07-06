import torch
from VQ_GAN.Parser_args import parse_Arg
from torchvision.utils import save_image
from Utils import CIFARLoadData
from Utils import get_device
from  VQ_GAN.model import vq_gan


args = parse_Arg()

batch_size = 256
num_training_updates = 55000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

train_loader = CIFARLoadData(args.batch_size, True, True)

device = get_device()

model = vq_gan(args, num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay, device).to(device)

for epoch in range(args.n_epochs):
    for i, data in enumerate(train_loader):
        real_images, _ = data
        current_batch_size = real_images.size(0)

        inputs = real_images.clone().to(device)
        noise = torch.zeros(current_batch_size, args.noise_dim, 1, 1).normal_(0, 1)

        real_labels = torch.ones(current_batch_size, 1).detach().to(device)
        fake_labels = torch.zeros(current_batch_size, 1).detach().to(device)

        discriminator_loss, generator_image = model.learn_discriminator(inputs, real_labels, fake_labels)
        generator_loss, perplexity = model.learn_generator(inputs, real_labels)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [Discriminator_loss: %f] [Generator_loss: %f] [perplexity : %f]"
            % (epoch + 1, args.n_epochs, i + 1, len(train_loader), discriminator_loss, generator_loss, perplexity)
        )

        batches_done = epoch * len(train_loader) + i
        if batches_done % args.sample_interval == 0:
            temp = generator_image[0:64]
            save_image(generator_image[0:64], "images/%d.png" % batches_done, nrow=args.nrow, normalize=True)

