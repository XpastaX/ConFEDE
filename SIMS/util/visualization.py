print(__doc__)
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time
from dataloader.SIMS import SIMSDataloader
import config
from model.net.constrastive.TVA_fusion import TVA_fusion
import torch
from tqdm import tqdm




device = config.DEVICE
test = SIMSDataloader('test', shuffle=False, num_workers=0,
                      batch_size=config.SIMS.downStream.TVAExp_fusion.batch_size)
config.seed=1234
model = TVA_fusion(config=config).to(device)

with torch.no_grad():
    model.eval()

    x_t_simi1 = []
    x_v_simi1 = []
    x_a_simi1 = []
    x_t_dissimi1 = []
    x_v_dissimi1 = []
    x_a_dissimi1 = []
    bar = tqdm(test, disable=True)
    label = []
    for index, sample in enumerate(bar):
        _label = sample['regression_labels'].clone().detach().to(device).float()
        Ts, Vs, As, Td, Vd, Ad = model(sample, None, return_loss=False, return_all_fea=True)
        x_t_simi1.append(Ts)
        x_v_simi1.append(Vs)
        x_a_simi1.append(As)
        x_t_dissimi1.append(Td)
        x_v_dissimi1.append(Vd)
        x_a_dissimi1.append(Ad)
        label.append(_label)

    x_t_simi1 = torch.cat(x_t_simi1, dim=0)
    x_v_simi1 = torch.cat(x_v_simi1, dim=0)
    x_a_simi1 = torch.cat(x_a_simi1, dim=0)
    x_t_dissimi1 = torch.cat(x_t_dissimi1, dim=0)
    x_v_dissimi1 = torch.cat(x_v_dissimi1, dim=0)
    x_a_dissimi1 = torch.cat(x_a_dissimi1, dim=0)

n_samples = 457 * 2
n_components = 2
(fig, subplots) = plt.subplots(2, figsize=(8, 8))

X = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1), dim=0)
y = torch.cat((torch.zeros(457), torch.ones(457), torch.ones(457) + 1,torch.ones(457) + 2,torch.ones(457) + 3,torch.ones(457) + 4),)
X = X.cpu()
red = y == 0
green = y == 1
blue = y == 2
cyan = y == 3
yellow = y == 4
magenta = y == 5

tsne = manifold.TSNE(n_components=3, init='pca',
                     random_state=0, perplexity=100)
Y = tsne.fit_transform(X)


ax = subplots[0]
ax.scatter(X[red, 0], X[red, 1], c="r",s=1)
ax.scatter(X[green, 0], X[green, 1], c="g",s=1)
ax.scatter(X[blue, 0], X[blue, 1], c="b",s=1)
ax.scatter(X[cyan, 0], X[cyan, 1], c="c",s=1)
ax.scatter(X[yellow, 0], X[yellow, 1], c="y",s=1)
ax.scatter(X[magenta, 0], X[magenta, 1], c="m",s=1)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


model.load_model(name='TVA_fusion' + '_best_' + "Mult_acc_2")

with torch.no_grad():
    model.eval()

    x_t_simi1 = []
    x_v_simi1 = []
    x_a_simi1 = []
    x_t_dissimi1 = []
    x_v_dissimi1 = []
    x_a_dissimi1 = []
    bar = tqdm(test, disable=True)
    label = []
    for index, sample in enumerate(bar):
        _label = sample['regression_labels'].clone().detach().to(device).float()
        Ts, Vs, As, Td, Vd, Ad = model(sample, None, return_loss=False, return_all_fea=True)
        x_t_simi1.append(Ts)
        x_v_simi1.append(Vs)
        x_a_simi1.append(As)
        x_t_dissimi1.append(Td)
        x_v_dissimi1.append(Vd)
        x_a_dissimi1.append(Ad)
        label.append(_label)

    x_t_simi1 = torch.cat(x_t_simi1, dim=0)
    x_v_simi1 = torch.cat(x_v_simi1, dim=0)
    x_a_simi1 = torch.cat(x_a_simi1, dim=0)
    x_t_dissimi1 = torch.cat(x_t_dissimi1, dim=0)
    x_v_dissimi1 = torch.cat(x_v_dissimi1, dim=0)
    x_a_dissimi1 = torch.cat(x_a_dissimi1, dim=0)


# X = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1), dim=0)
X = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1), dim=0)
y = torch.cat((torch.zeros(457), torch.ones(457), torch.ones(457) + 1,torch.ones(457) + 2,torch.ones(457) + 3,torch.ones(457) + 4),)
X = X.cpu()
red = y == 0
green = y == 1
blue = y == 2
cyan = y == 3
yellow = y == 4
magenta = y == 5

tsne = manifold.TSNE(n_components=3, init='pca',
                     random_state=0, perplexity=100)
Y = tsne.fit_transform(X)


ax = subplots[1]
ax.scatter(X[red, 0], X[red, 1], c="r",s=1)
ax.scatter(X[green, 0], X[green, 1], c="g",s=1)
ax.scatter(X[blue, 0], X[blue, 1], c="b",s=1)
ax.scatter(X[cyan, 0], X[cyan, 1], c="c",s=1)
ax.scatter(X[yellow, 0], X[yellow, 1], c="y",s=1)
ax.scatter(X[magenta, 0], X[magenta, 1], c="m",s=1)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')


# for i, perplexity in enumerate(perplexities):
#     ax = subplots[0][i + 1]
#
#     t0 = time()
#     tsne = manifold.TSNE(n_components=n_components, init='random',
#                          random_state=0, perplexity=perplexity)
#     Y = tsne.fit_transform(X)
#     t1 = time()
#     print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
#     ax.set_title("Perplexity=%d" % perplexity)
#     ax.scatter(Y[red, 0], Y[red, 1], c="r")
#     ax.scatter(Y[green, 0], Y[green, 1], c="g")
#     ax.xaxis.set_major_formatter(NullFormatter())
#     ax.yaxis.set_major_formatter(NullFormatter())
#     ax.axis('tight')

# Another example using s-curve
# X, color = datasets.make_s_curve(n_samples, random_state=0)


plt.show()
