from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
      random noise in the range (-1, 1).
    """
    noise = None
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    # Replace "pass" statement with your code
    noise=2*torch.rand([batch_size,noise_dim],dtype=dtype,device=device)-1
    #torch.rand生成的是[0,1)上的均匀分布

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    # Replace "pass" statement with your code
    model=nn.Sequential(
        nn.Linear(784,256),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(256,256),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(256,1)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement generator.                                               #
    ############################################################################
    # Replace "pass" statement with your code
    model=nn.Sequential(
        nn.Linear(noise_dim,1024),
        nn.ReLU(),
        nn.Linear(1024,1024),
        nn.ReLU(),
        nn.Linear(1024,784),
        nn.Tanh()
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    # Replace "pass" statement with your code
    N=logits_real.shape[0]
    real_labels=torch.ones(N,device=logits_real.device)
    fake_labels=torch.zeros(N,device=logits_real.device)
    real_loss=nn.functional.binary_cross_entropy_with_logits(logits_real.reshape(N),real_labels,reduction="mean")  #训练的时候发现logits_real的形状可能是(N,1)
    fake_loss=nn.functional.binary_cross_entropy_with_logits(logits_fake.reshape(N),fake_labels,reduction="mean")
    loss=real_loss+fake_loss

    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    # Replace "pass" statement with your code
    N=logits_fake.shape[0]
    true_labels=torch.ones(N,device=logits_fake.device)
    loss=nn.functional.binary_cross_entropy_with_logits(logits_fake.reshape(N),true_labels,reduction="mean")
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    # Replace "pass" statement with your code
    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        betas=[0.5,0.999]
    )
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_discriminator_loss.                                     #
    ##############################################################################
    # Replace "pass" statement with your code
    
    N=scores_real.shape[0]
    loss=(0.5*(scores_real-1)*(scores_real-1)+0.5*scores_fake*scores_fake).sum()/N

    
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_generator_loss.                                         #
    ##############################################################################
    # Replace "pass" statement with your code
    N=scores_fake.shape[0]
    loss=(0.5*(scores_fake-1)*(scores_fake-1)).sum()/N
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    # Replace "pass" statement with your code
    model=nn.Sequential(
        nn.Unflatten(dim=1,unflattened_size=(1,28,28)),  #(N,1,28,28)
        nn.Conv2d(1,32,kernel_size=5,stride=1),  #(N,32,24,24)
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d(kernel_size=2,stride=2),  #(N,32,12,12)
        nn.Conv2d(32,64,kernel_size=5,stride=1),  #(N,64,8,8)
        nn.LeakyReLU(negative_slope=0.01),
        nn.MaxPool2d(kernel_size=2,stride=2),  #(N,64,4,4)
        nn.Flatten(),  #(N,1024)
        nn.Linear(1024,1024),
        nn.LeakyReLU(negative_slope=0.01),
        nn.Linear(1024,1)   #(N,1)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    # Replace "pass" statement with your code
    model=nn.Sequential(
        nn.Linear(noise_dim,1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Linear(1024,7*7*128),  #(N,7*7*128)
        nn.ReLU(),
        nn.BatchNorm1d(7*7*128),
        nn.Unflatten(dim=1,unflattened_size=(128,7,7)),  #(N,128,7,7)
        nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4,stride=2,padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),  #对于卷积中间量的批归一化，num_features是(N, C, H, W)里面的C
        nn.ConvTranspose2d(in_channels=64,out_channels=1,kernel_size=4,stride=2,padding=1),
        nn.Tanh(),
        nn.Flatten()

    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
