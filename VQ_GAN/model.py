import torch.nn as nn
from VQ_VAE.VQ_VAE import vq_vae
from VQ_GAN.Discriminator_layer import Discriminator
from torch import optim


class vq_gan(nn.Module):
    def __init__(self, args, num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay):
        super(vq_gan, self).__init__()
        self.args = args

        self.Generator = vq_vae(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim,
              commitment_cost, decay)

        self.Discriminator = Discriminator(self.args)

        self.optimizer_generator = optim.Adam(self.Generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        self.optimizer_discriminator = optim.Adam(self.Discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

        self.adversarial_loss = nn.BCELoss()

        self.recon_loss = nn.MSELoss()

        self.valid = None
        self.fake = None
        self.real_loss = None

    def learn_discriminator(self, inputs, real_labels, fake_labels):
        output = self.Discriminator(inputs)
        real_loss = self.adversarial_loss(output, real_labels)

        _, fake, _ = self.Generator(inputs)
        discriminator_result = self.Discriminator(fake.detach())
        fake_loss = self.adversarial_loss(discriminator_result, fake_labels)

        loss = real_loss + fake_loss

        self.optimizer_discriminator.zero_grad()
        loss.backward()
        self.optimizer_discriminator.step()

        return loss.item(), fake

    def learn_generator(self, inputs, label):
        vq_loss, fake, perplexity = self.Generator(inputs)
        discriminator_result = self.Discriminator(fake)
        recon_loss = self.recon_loss(inputs, fake)
        loss = self.adversarial_loss(discriminator_result, label) + vq_loss + recon_loss

        self.optimizer_generator.zero_grad()
        loss.backward()
        self.optimizer_generator.step()

        return loss.item(), perplexity