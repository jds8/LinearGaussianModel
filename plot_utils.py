#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import torch
import torch.distributions as dist
import imageio


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def plot_normal(mu, sigma, label, clr):
    xs = torch.arange(mu-10, mu+10, 0.1)
    pxs = dist.Normal(mu, sigma).log_prob(xs).exp()
    plt.plot(xs, pxs, label=label, color=clr)
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.title('Normal Distribution')
    plt.legend()

def compare_normals(mu1, sigma1, label1, mu2, sigma2, label2, save_dir, plot_name):
    plot_normal(mu1, sigma1, label1, 'b')
    plot_normal(mu2, sigma2, label2, 'r')
    plt.savefig('{}/{}'.format(save_dir, plot_name))
    plt.close()

def generate_gif(filenames, save_dir, gif_name):
    if not filenames:
        print('No files from which to make a gif.')
    else:
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('{}/{}.gif'.format(save_dir, gif_name), images)


if __name__ == "__main__":
    file_name_base = 'NormalComparison{}.png'
    filenames = []
    for i in range(1, 10):
        filename = file_name_base.format(i)
        compare_normals(0, 1, 'posterior', 0+1/i, 1+1/i, 'proposal', '.', filename)
        filenames.append(filename)
    generate_gif(filenames, '.', 'MovingNormal')
