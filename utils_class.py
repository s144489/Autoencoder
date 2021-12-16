import torch
import torch.nn as nn
import torch.nn.functional as F

import numba
from numba import jit
#from umap import UMAP
import numba_scipy
import scipy


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from copy import deepcopy
import matplotlib.gridspec as grd
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
import seaborn as sns
from scipy import stats
#from ax.plot.contour import plot_contour
import plotly.io as pio
import matplotlib.pyplot as plt
from plotly import graph_objs as go
#from ax.plot.trace import optimization_trace_single_method
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.cm as cm
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import plot_confusion_matrix



# ------------------------------------------------------------

import logging
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')


class GuidedSaliency(object):
    def __init__(self, model):
        self.model = deepcopy(model)
        self.model.eval()

    def guided_relu_hook(self, module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0),)

    def generate_saliency(self, input, target):
        # https://medium.com/@zhang_yang/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29
        # freezes the input
        input.requires_grad = True

        self.model.zero_grad()

        for module in self.model.modules():
            if type(module) == nn.Softplus:
                module.register_backward_hook(self.guided_relu_hook)

            output = self.model(input)
            # creating tensor with zeroes
            grad_outputs = torch.zeros_like(output)

            # puts ones in the target column
            grad_outputs[:, target] = 1

            output.backward(gradient=grad_outputs)
            input.requires_grad = False
        # returns the gradients[cells,genes] for specific hidden unit
        return input.grad.clone()


def sns_distplot(args, train_loader):
    '''
    max = 8
    min = 4

    x = torch.tensor([[1, 2, 3, 3, 5, 6, 7, 8, 9, 0, 1, 2, 3]], dtype=torch.float64)
    fig = sns.distplot(x, hist=False, rug=True)
    print(x.shape)
    print(x)
    print("1")
    plt.show()


    x = (max-min)*torch.rand(1, 80, dtype=torch.float64)
    print(x)
    print(x.shape)
    fig = sns.distplot(x, hist=False, rug=True,**kwargs)
    print("2")

    plt.show()


    x = (max-min)*torch.rand(1, 100, dtype=torch.float64)
    print(x.shape)
    print("3")
    fig = sns.distplot(x, hist=False, rug=True)
    plt.show()


    x= (max-min)*torch.rand(1, 1000, dtype=torch.float64)
    print(x.shape)
    fig = sns.distplot(x, hist=False, rug=True)
    print("4")
    plt.show()

    x= (max-min)*torch.rand(1, 4000, dtype=torch.float64)
    print(x.shape)
    fig = sns.distplot(x, hist=False, rug=True)
    print("5")
    plt.show()


    x=torch.tensor([[0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14,15,16,17,18,19,20]], dtype=torch.float64)
    fig = sns.distplot(x, hist=False, rug=True, color="w")
    plt.show()
    print("6")
    '''

    for batch_idx, (data, target) in enumerate(train_loader):
        # print(data)
        # print(data.dtype)
        # data = data.to(device=args.device, dtype=args.dtype)
        # print(len(data))
        # print(data.dtype)
        # print(data.shape)
        # print("max",torch.max(data))
        # print("min", torch.min(data))
        # print("Sum",torch.sum(data))
        # print(data)

        data = data.cpu().numpy()

        fig = sns.kdeplot(data[0], bw=2)
        # fig=sns.distplot(data, hist=False, rug=True)

    if args.path != None:
        plt.legend("")
        plt.title("Distribution" + " " + args.organ)
        # plt.ylabel("Density")
        if args.norm_type != None:
            plt.xlabel(args.norm_type)
            plt.savefig(args.path + args.norm_type + "_" + args.dataset + "_" + args.organ + '.png')
        else:
            plt.savefig("Order" + "_" + args.dataset + "_" + args.organ + '.png')
    else:
        plt.legend("")
        plt.title("Distribution" + " " + args.organ)
        if args.norm_type == None:
            plt.xlabel("Counts")
        else:
            plt.xlabel(args.norm_type)
        plt.ylabel("Density")
        plt.show()
    print("Done making sns plot")


def train(args, self, model, train_loader, optimizer, epoch, Optimization_loop=False, **kwargs):
    model.train()
    loss_func = F.poisson_nll_loss

    support = np.linspace(-4, 4, 200)
    kernels = []

    # This runs trough all data
    for batch_idx, (data, target) in enumerate(train_loader):
        # print(batch_idx,data,target)
        data = data.to(device=args.device, dtype=args.dtype)
        # target = target.to(device=args.device, dtype=args.dtype)

        optimizer.zero_grad()
        representation, scores = model(data)

        # sns.distplot(data, hist=False, rug=True);
        # if args.plot==True and epoch==1 and Optimization_loop==False:
        # print("Max",torch.max(data))
        # print("scores", scores)
        if args.norm_type == "Order":
            # print("order loss function")
            loss = loss_function_nll_poisson2(scores, data, loss_func, eps=1e-8, log_input=False, full=True)
            # loss_function_nll_poisson2(inputs, targets, loss_func, eps=1e-8, log_input=False, full=True)

        else:

            # print("scores", scores)
            loss = loss_func(scores, data, **kwargs)

        # print("scores", scores)
        # loss = loss_func(scores, data, **kwargs)
        # print("loss returned",loss)
        # loss2=loss_func(scores,data,**kwargs)
        # print("own loss", loss.item())
        # print("pytorch loss", loss.item())

        # print("diagnol",torch.eye(args.hidden_size).to(device=args.device, dtype=args.dtype).shape)
        # print("shape loss",torch.mm(model.fc2.weight.data.transpose(0,1),model.fc2.weight.data).shape)
        # print("shape loss2",torch.mm(model.fc2.weight.data, model.fc2.weight.data.transpose(0,1)).shape)
        # Add L1 regularization to avoid over fitting the data L1: sum of abselute values in vector
        if args.l1_reg != 0:
            all_params = torch.cat([x[1].view(-1) for x in model.state_dict().items()])
            l1_regularization = args.l1_reg * torch.norm(all_params, 1)
            loss += l1_regularization
        if args.l_orthog != 0:  # sum of sqrt(values^2) W|T*W-I
            loss += torch.norm(
                args.l_orthog * (
                        torch.eye(args.hidden_size).to(device=args.device, dtype=args.dtype) -
                        torch.mm(model.fc2.weight.data.transpose(0, 1), model.fc2.weight.data)
                ),
                2
            )

        # print("loss 128 dim",loss)
        # print("loss 64 dim",loss2)
        # print("shape", model.fc2.weight.data.shape)
        # print("shape transposed mm",torch.mm(model.fc2.weight.data, model.fc2.weight.data.transpose(1,0).shape)
        # Calculate the gradients by calling the backward function on the final loss
        loss.backward()
        # updates the weights/model
        optimizer.step()
        # print(data)

        if Optimization_loop == False:
            if batch_idx % args.log_interval == 0:
                # ("bactch idx",str(batch_idx),"len data",str(len(data)))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item() / len(train_loader)))


def train_classifier(args, model, train_loader, optimizer, epoch, **kwargs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=args.device, dtype=args.dtype)
        target = target.to(device=args.device, dtype=args.dtype).view(target.shape[0], 1)
        optimizer.zero_grad()
        scores = model(data)
        loss = loss_function_nll_poisson(scores, data, log_input=False, full=False)
        # Add L1 regularization
        if args.l1_reg != 0.0:
            all_params = torch.cat([x[1].view(-1) for x in model.state_dict().items()])
            l1_regularization = args.l1_reg * torch.norm(all_params, 1)
            loss += l1_regularization
        loss.backward()
        optimizer.step()
        if batch_idx == 0:
            print_result = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / target.shape[0])
    return print_result


def train_regression(args, model, train_loader, optimizer, epoch, loss_func=F.mse_loss, **kwargs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device=args.device, dtype=args.dtype)
        target = target.to(device=args.device, dtype=args.dtype).view(target.shape[0], 1)
        optimizer.zero_grad()
        scores = model(data)
        loss = loss_function_nll_poisson(scores, target, full=True)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / target.shape[0]))


def check_accuracy(self, args, model, test_loader, Optimization_loop=False, **kwargs):
    loss_func = F.poisson_nll_loss

    model.eval()
    test_loss = 0
    num = 0
    # test_loss_check=0

    with torch.no_grad():
        # for data, target in test_loader:
        count = 0
        for data, target in test_loader:

            data = data.to(device=args.device, dtype=args.dtype)
            representation, scores = model(data)
            count += 1
            if args.norm_type == "Order":

                # print("order loss function")
                test_loss += loss_function_nll_poisson2(scores, data, loss_func, eps=1e-8, log_input=False, full=True)
                # loss_function_nll_poisson2(inputs, targets, loss_func, eps=1e-8, log_input=False, full=True)

            else:

                print("scores", scores)
                test_loss += loss_func(scores, data, **kwargs)

            # test_loss += loss_func(scores, data, **kwargs)
            # print("pytorch loss",loss_func(scores, data,**kwargs).item())
            # print("new loss",loss_function_nll_poisson(scores,data,log_input=False,full=False).item())
            num += 1
        # Average loss is computed
        test_loss /= num
        # test_loss_check/=num
        print(count)
        # print("Final pytorch loss",test_loss_check.item())
        # print("final new loss",test_loss.item())

        '''
        if Optimization_loop==True:
            #print("Test loss",test_loss.item())

            if self.Best_optimized_loss==None:
                self.Best_optimized_loss=test_loss.item()

                if args.path!=None and "/" in args.path:
                    print("Saves initial checkpoint")
                    torch.save({
                            'epoch': args.epochs,
                            'args': args,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': test_loss,
                            }, args.path.rsplit('/',1)[0]+"/"+"Best_model.tar")
                else:
                    torch.save({
                        'epoch': args.epochs,
                        'args': args,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': test_loss,
                    }, "Best_model.tar")
            elif self.Best_optimized_loss> test_loss.item():
                self.Best_optimized_loss=test_loss.item()
                print("Enters 2. saving point")
                if "/" in args.path and args.path != None:
                    torch.save({
                    'args': args,
                    'epoch': args.epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': test_loss,
                    }, args.path.rsplit('/',1)[0]+"/"+"Best_model.tar")
                else:
                    torch.save({
                    'args': self.args,
                    'epoch': args.epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': test_loss,
                    }, "Best_model.tar")

        '''

    return test_loss.item()


def check_accuracy_classifier(args, model, test_loader):
    model.eval()
    test_loss = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device=args.device, dtype=args.dtype)
            target = target.to(device=args.device, dtype=args.dtype).view(target.shape[0], 1)
            scores = model(data)
            test_loss += ((scores - target).abs_() < 0.25).sum().item()
            num += target.shape[0]
        test_loss /= num
    return test_loss


def check_accuracy_regression(args, model, test_loader, loss_func=F.mse_loss, **kwargs):
    model.eval()
    test_loss = 0
    num = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device=args.device, dtype=args.dtype)
            target = target.to(device=args.device, dtype=args.dtype).view(target.shape[0], 1)
            scores = model(data)
            test_loss += loss_func(scores, data, **kwargs)
            num += target.shape[0]
        test_loss /= num
    return test_loss


def plot_embedding(method, args, dataset, model=None):
    x = torch.tensor(dataset.expressions.values, device=args.device, dtype=args.dtype)
    if model:
        model.eval()
        x, scores = model(x)
    if method == 'tsne':
        embedding_repres = TSNE(init='random').fit_transform(x.cpu().numpy())
    elif method == 'umap':
        reducer = umap.UMAP()
        embedding_repres = reducer().fit_transform(x.cpu().numpy())
    fig = plt.scatter(embedding_repres[:, 0], embedding_repres[:, 1], c=dataset.subtypes, alpha=1, marker='.', s=5)
    return fig


def get_gene_set_dict(dataset, signatures='msigdb.v6.2.symbols.gmt.txt'):
    from collections import defaultdict
    gene_set_dict = defaultdict(list)
    with open(signatures) as go_file:
        for line in go_file.readlines():
            if line.startswith('HALLMARK_'):
                gene_set = line.strip().split('\t')
                gene_set_dict[gene_set[0]] = list(np.where(dataset.expressions.columns.str.upper().isin(gene_set[1:])))

    # print(gene_set_dict)
    return gene_set_dict


def get_common_model_part(model, model_encoder):
    # the optimal or current state of the model
    pretrained_dict = model.state_dict()
    # print("pretrained_dict",len(pretrained_dict))

    # The weights of the representation layer but not used.
    model_dict = model_encoder.state_dict()
    # print("model_dict",len(model_dict))

    # 1. filter out unnecessary keys So only the keys used for the encoding
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model_encoder.load_state_dict(pretrained_dict)
    # print("model_encoder")

    return model_encoder


def get_gene_set_scores(args, model, dataset, gene_set_dict, condition):
    # con
    gene_set_score = np.zeros((args.more_hidden_size, len(gene_set_dict)))
    gene_set_std = np.zeros((args.more_hidden_size, len(gene_set_dict)))
    grads = GuidedSaliency(model)
    # args.more_hidden_size
    for i in range(args.more_hidden_size):
        if condition is None:
            x = torch.tensor(dataset.expressions.values, device=args.device, dtype=args.dtype)
        else:
            # extrating data from a certain cell type
            x = torch.tensor(dataset.__getitem__([ix for (ix, x) in enumerate(condition) if x])[0],
                             device=args.device, dtype=args.dtype)
        # gradients for specific more_hidden_size unit.
        saliency_input = grads.generate_saliency(x, i).abs()

        for j, gene_set in enumerate(gene_set_dict.values()):
            # print(saliency_input[:, gene_set[0]].median(1)[0])
            # print(saliency_input[:, gene_set[0]].median(1)[0].shape)
            # print(saliency_input[:, gene_set[0]].median(1)[0].mean())
            # computes the median value of the gradient for the Hallmark_Pathways off all cells, and then takes the mean()/std() of that.
            gene_set_score[i, j] = saliency_input[:, gene_set[0]].median(1)[0].mean()
            gene_set_std[i, j] = saliency_input[:, gene_set[0]].median(1)[0].std()

    gene_set_score = pd.DataFrame(gene_set_score, columns=[x[9:] for x in gene_set_dict.keys()])

    gene_set_std = pd.DataFrame(gene_set_std, columns=[x[9:] for x in gene_set_dict.keys()])

    return gene_set_score, gene_set_std


def get_gene_set_scores2(args, model, dataset, gene_set_dict, condition):
    colnames = dataset.expressions.columns
    gene_set_score = np.zeros((args.more_hidden_size, len(dataset.expressions.columns)))
    gene_set_score_genes = np.zeros((args.more_hidden_size, len(dataset.expressions.columns)))
    gene_set_std = np.zeros((args.more_hidden_size, len(dataset.expressions.columns)))
    grads = GuidedSaliency(model)
    # args.more_hidden_size
    for i in range(args.more_hidden_size):
        if condition is None:
            x = torch.tensor(dataset.expressions.values, device=args.device, dtype=args.dtype)
        else:
            # extrating data from a certain cell type
            x = torch.tensor(dataset.__getitem__([ix for (ix, x) in enumerate(condition) if x])[0],
                             device=args.device, dtype=args.dtype)

        # gradients for specific more_hidden_size unit.
        saliency_input = grads.generate_saliency(x, i).abs()
        # gene_set_score[i,:] = saliency_input[:, :].median(1)[0].mean()
        # if i == 0:
        #   print("saliency head", saliency_input[0:5, 0:5])

        for j in range(len(dataset.expressions.columns)):
            # computes the median value of the gradient for the Hallmark_Pathways off all cells, and then takes the mean()/std() of that.
            # median(of each columns indicated by (1))
            # median(1)[0], only extracts the median vaues and not the indices of the pathway
            # if i == 0 and j == 0:
            # print(saliency_input[:, j])
            # print("shape", saliency_input[:, j].shape)
            # print(saliency_input[:, j].mean(axis=0))
            # print("shape", saliency_input[:, j].mean())
            gene_set_score[i, j] = saliency_input[:, j].mean()
            gene_set_std[i, j] = saliency_input[:, j].std()

    gene_set_score = pd.DataFrame(gene_set_score, columns=dataset.expressions.columns)
    gene_set_std = pd.DataFrame(gene_set_std, columns=dataset.expressions.columns)

    return gene_set_score, gene_set_std


def score_gene_set(dataset, input_dim, args, model, signatures, type="Genes", condition=None):
    ## Gene sets
    gene_set_dict = get_gene_set_dict(dataset, signatures)
    # print(gene_set_dict)

    ## Get the encoding part of the model
    import models
    model_encoder = models.AutoencoderTwoLayers_encoder(input_dim, args.hidden_size, args.more_hidden_size).to(
        args.device)
    model_encoder = get_common_model_part(model, model_encoder)

    ## Evaluate the gene set contribution using saliency maps
    if type == "Pathways":
        gene_set_score, gene_set_std = get_gene_set_scores(args, model_encoder, dataset, gene_set_dict, condition)

    else:
        gene_set_score, gene_set_std = get_gene_set_scores2(args, model_encoder, dataset, gene_set_dict, condition)

    return gene_set_score, score_gene_set


def get_embedding(method, args, dataset, model=None, dataset2=None):
    if dataset2 != None:


        try:

            joined_data = np.concatenate((dataset.expressions.values, dataset2.expressions.values), axis=0)
            x = torch.tensor(joined_data, device=args.device, dtype=args.dtype)
        except:

            joined_data = np.concatenate((dataset.expressions, dataset2.expressions), axis=0)
            x = torch.tensor(joined_data, device=args.device, dtype=args.dtype)

        if model:
            model.eval()
            x, scores = model(x)

    else:

        try:

            x = torch.tensor(dataset.expressions.values, device=args.device, dtype=args.dtype)


        except:

            x = torch.tensor(dataset.expressions, device=args.device, dtype=args.dtype)


        if model:
            model.eval()
            x, scores = model(x)


    if method == 'tsne':
        embedding_repres = TSNE(init='random').fit_transform(x.cpu().numpy())
    elif method == 'umap':
        reducer = umap.UMAP()
        embedding_repres = reducer().fit_transform(x.cpu().numpy())
    elif method == 'pca':
        pca = PCA(n_components=2)
        embedding_repres = pca.fit_transform(x.cpu().numpy())

    try:
        return embedding_repres, pca

    except:
        return embedding_repres


def get_singular_values_norm(args, dataset, model):
    # based on the representation layer
    x = torch.tensor(dataset.expressions.values, device=args.device, dtype=args.dtype)
    model.eval()  # evaluation mode
    # x is the bottleneck layer
    x, scores = model(x)
    # print((x.cpu().numpy()).shape)

    s = np.linalg.svd(x.cpu().numpy(), compute_uv=False)

    s = np.linalg.norm(s, 2)  # 2-D matrix as input, is nomalized

    return s


def plot_clusters_full_color_multiple(subtitle, clusters, embedding, title, fig):
    # cluster = subtypes



    colors_use=[ 'purple', 'green', 'blue', 'orange', 'red',"teal","black","peru", "maroon", "magenta","navy","lawngreen", "Grey",
                 "darkorchid","darkorange","gold","brown","darkred",
                "springgreen", "royalblue", "crimson", "violet", "khaki","lightpink", "olivedrab", "tan", "cornflowerblue", "lightcoral"]

    # print(clusters)
    cluster_list = list(set(clusters))
    cluster_list.sort()
    num_clusters = len(cluster_list)
    gs = grd.GridSpecFromSubplotSpec(1, 2, width_ratios=[20, 1], subplot_spec=fig)
    ax = plt.subplot(gs[0])
    for i in range(num_clusters):
        cluster_i = cluster_list[i]
        if cluster_i == 'anone':
            ax.scatter(embedding[clusters == cluster_i, 0], embedding[clusters == cluster_i, 1], label=cluster_i, marker='.', s=2, alpha=1) #c=colors_use[i]
        else:
            ax.scatter(embedding[clusters == cluster_i, 0], embedding[clusters == cluster_i, 1], label=cluster_i,marker='.', s=2,) #c=colors_use[i]
    if title:
        ax.set_title(title + " " + subtitle, fontsize=14)

    # legend = ax.legend(frameon=True)

    # for legend_handle in legend.legendHandles:
    # legend_handle._legmarker.set_markersize(9)

    #ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), fontsize=10)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), markerscale=6, fontsize=10)

    # ax.set_xlim([0, 250000])
    # ax.set_ylim([0, 250000])


def plot_gene_set_score(ax, gene_set_score, gene_set_score_comp=None, diverging=True, plot_nonlin_factor=5,
                        show_percent=0.85, cbar_draw=True, cbar_ax=None, hide_black=False):
    grays = cm.get_cmap('gray', 100)
    cmap = colors.ListedColormap(grays(np.power(np.linspace(0, 1, 101), plot_nonlin_factor)))
    sns.set(font_scale=.6)
    names = gene_set_score.columns
    scaler = MinMaxScaler()
    rowindex = gene_set_score.index
    gene_set_score = scaler.fit_transform(gene_set_score)
    gene_set_score = pd.DataFrame(gene_set_score, columns=names)
    gene_set_score.index = rowindex
    if hide_black:
        sns.heatmap(gene_set_score.T.loc[:, (gene_set_score.T > 0.7).any()], cmap=cmap, yticklabels=True,
                    xticklabels=True, ax=ax, cbar=cbar_draw, cbar_ax=cbar_ax, vmin=0, vmax=1)
    else:
        sns.heatmap(gene_set_score.T, cmap=cmap, yticklabels=True, xticklabels=True, ax=ax, cbar=cbar_draw,
                    cbar_ax=cbar_ax, vmin=0, vmax=1)

        # https://seaborn.pydata.org/tutorial/distributions.html
    if gene_set_score_comp is not None:
        if not diverging:
            scaler = MinMaxScaler()
            rowindex = gene_set_score_comp.index
            gene_set_score_comp = scaler.fit_transform(abs(gene_set_score_comp))
            gene_set_score_comp = pd.DataFrame(gene_set_score_comp, columns=names)
            gene_set_score_comp.index = rowindex
            gene_set_score_comp.where(gene_set_score > show_percent, np.nan, inplace=True)
            reds = cm.get_cmap('Reds', 100)
            cmap = colors.ListedColormap(reds(np.linspace(0, .7, 101)))
            if hide_black:
                sns.heatmap(gene_set_score_comp.T.loc[:, (gene_set_score.T > 0.7).any()], cmap=cmap, yticklabels=True,
                            xticklabels=True, ax=ax, vmin=0, vmax=1, cbar=cbar_draw, cbar_ax=cbar_ax)
            else:
                sns.heatmap(gene_set_score_comp.T, cmap=cmap, yticklabels=True, xticklabels=True, ax=ax,
                            cbar=cbar_draw, cbar_ax=cbar_ax, vmin=0, vmax=1)
        else:
            scale_const = gene_set_score_comp.abs().max(axis=0)
            rowindex = gene_set_score_comp.index
            gene_set_score_comp = gene_set_score_comp / scale_const
            gene_set_score_comp = pd.DataFrame(gene_set_score_comp, columns=names)
            gene_set_score_comp.index = rowindex
            gene_set_score_comp.where(gene_set_score > show_percent, np.nan, inplace=True)
            reds = cm.get_cmap('bwr', 100)
            cmap = colors.ListedColormap(reds(np.linspace(0.1, 0.9, 101)))
            if hide_black:
                sns.heatmap(gene_set_score_comp.T.loc[:, (gene_set_score.T > 0.7).any()], cmap=cmap, yticklabels=True,
                            xticklabels=True, ax=ax, vmin=-1, vmax=1, cbar=cbar_draw, cbar_ax=cbar_ax)
            else:
                sns.heatmap(gene_set_score_comp.T, cmap=cmap, yticklabels=True, xticklabels=True, ax=ax,
                            cbar=cbar_draw, cbar_ax=cbar_ax, vmin=-1, vmax=1)
    return ax


def make_plot(subtitle, args, dataset, title, dest, model=None, file=None, plot='umap', dataset2=None, Channel=False,
              Label=False, dim=(9, 5)):
    # Channel = sequncing channel

    # Label = add label to test and train dataset.


    fig = plt.figure(figsize=dim)
    outer_grid = grd.GridSpec(1, 1)
    panel = outer_grid[0]

    if dataset2 != None:

        if plot == "pca":

            embedding, pca = get_embedding(plot, args, dataset, model, dataset2)


        else:

            embedding = get_embedding(plot, args, dataset, model, dataset2)

            # print("subtypes, before", dataset.subtypes.shape)
            # print(dataset2.subtypes.iloc[:,1])
            # subtypes= dataset.subtypes + dataset2.subtypes

        if Label == True:
            for i in range(dataset2.subtypes.shape[0]):
                dataset2.subtypes[i] = dataset2.subtypes[i] + ' test'

        # subtypes = subtypes.reset_index(drop=True)
        # subtypes = subtypes.drop(columns=['index'])
        # print(subtypes)

        if Channel == True:
            s1 = dataset.subtypes.shape[0]
            s2 = dataset2.subtypes.shape[0]
            substypes_test = ny = s1 * ['Train'] + s2 * ['Test']
            substypes_test = pd.Series(substypes_test)

            plot_clusters_full_color_multiple(subtitle, substypes_test, embedding, title, panel)

        else:
            subtypes = pd.concat([dataset.subtypes, dataset2.subtypes])
            plot_clusters_full_color_multiple(subtitle, subtypes, embedding, title, panel)



    else:

        if plot == "pca":
            embedding, pca = get_embedding(plot, args, dataset, model)
        else:
            embedding = get_embedding(plot, args, dataset, model)

        plot_clusters_full_color_multiple(subtitle, dataset.subtypes, embedding, title, panel)

    if plot == "pca":
        plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0] * 100))
        plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1] * 100))

    plt.legend(fontsize=10, markerscale=10, bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.xticks([])
    plt.yticks([])
    fig.tight_layout()

    if file != None:
        file.savefig()

    else:
        plt.show()
        plt.close('all')


def make_heatmap(args, dataset, model, file=None,cell_type=None):
    print("make heatmap")

    try:
        input_dim = dataset.expressions.shape[1]
    except:
        input_dim = dataset.shape[1]

    for element in ["Genes", "Pathways"]:
        gene_set_score, gene_set_std = score_gene_set(dataset, input_dim, args, model, signatures=args.signatures,
                                                      type=element)
        print("args.path",args.path)
        print("args.test_dataset: ",args.test_dataset)
        if  "/" in args.path:
            print("saving to XL file")
            print("Heatmap complex path")

            # print("path",args.path.rsplit('/', 1)[0] + "/" + 'Matrices_'+element+"_"+args.organ+'.xlsx')
            # print("split path",args.path.rsplit('/', 1)[0])
            # print("element",element)
            # print("organ", args.organ)
            # print(args.test_dataset)
            if args.one_cell ==True:

                print("in the right place shoul write new path")

                print("cell type",cell_type)
                cell_type=cell_type.replace(" ", "_")
                print(args.path.rsplit('/', 1)[0] + "/" +
                                        "Onecell_"+cell_type +"_"+ args.test_dataset + "_trained_" + args.organ + "_" + args.dataset + "_" + element + ".xlsx")
                writer = pd.ExcelWriter(args.path.rsplit('/', 1)[0] + "/" +
                                        "One_"+cell_type + args.test_dataset + "_trained_" + args.organ + "_" + args.dataset + "_" + element + ".xlsx")
                gene_set_score.to_excel(writer, '{}'.format(cell_type))


            elif args.test_dataset != None:

                print("NOO also in here")
                print("in the right place shoul write new path")
                writer = pd.ExcelWriter(args.path.rsplit('/', 1)[0] + "/" +
                                        "modelling_" + args.test_dataset + "_trained_" + args.organ + "_" + args.dataset + "_" + element + ".xlsx")
                gene_set_score.to_excel(writer, '{}'.format("Whole dataset"))
            


            else:
                print("in else in utils make heatmap ")
                writer = pd.ExcelWriter(
                    args.path.rsplit('/', 1)[0] + "/" + 'Matrices_' + element + "_" + args.organ + '.xlsx')
                gene_set_score.to_excel(writer, '{}'.format("Whole dataset"))




        elif args.path != None:

            if args.test_dataset != None:

                writer = pd.ExcelWriter(
                    "modelling_" + args.test_dataset + "_trained_" + args.organ + "_" + args.dataset + "_" + element + '.xlsx')
                gene_set_score.to_excel(writer, '{}'.format("Whole dataset"))

            elif args.one_cell == True:
                print("in the right place shoul write new path 2 ")

                print("cell type", cell_type)
                cell_type = cell_type.replace(" ", "_")
                print(args.path.rsplit('/', 1)[0] + "/" +
                      "Onecell_" + cell_type + args.test_dataset + "_trained_" + args.organ + "_" + args.dataset + "_" + element + ".xlsx")
                writer = pd.ExcelWriter("One_" + cell_type + "_"+args.test_dataset + "_trained_" + args.organ + "_" + args.dataset + "_" + element + ".xlsx")
                gene_set_score.to_excel(writer, '{}'.format("Whole dataset"))



            else:
                writer = pd.ExcelWriter('Matrices_' + element + "_" + args.organ + '.xlsx')
                gene_set_score.to_excel(writer, '{}'.format("Whole dataset"))
            print("Heatmap with simple  path")


        else:
            print("gene_set_score", gene_set_score)

        if args.one_cell==False:

            i = 0
            fig = plt.figure(figsize=(7, 7))
            outer_grid = grd.GridSpec(int(math.ceil((len(dataset.subtypes.unique()) + 1) / 2)), 2)
            ax = fig.add_subplot(outer_grid[i])
            plot_gene_set_score(ax, gene_set_score, cbar_draw=True, hide_black=True)
            ax.set_title('Whole dataset', fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=4)
            # print(sorted(dataset.subtypes.unique()))

            for subtype in sorted(dataset.subtypes.unique()):
                print("Entering subtype loop")
                print(subtype)
                i += 1
                gene_set_score_a, gene_set_std_a = score_gene_set(dataset, input_dim, args, model,
                                                                  signatures=args.signatures, type=element,
                                                                  condition=(dataset.subtypes == subtype))

                if args.path != None:

                    if len(subtype.strip()) > 30:
                        subtype_short = subtype.replace(" ", "")
                        if len(subtype_short.strip()) > 30:
                            gene_set_score_a.to_excel(writer, '{}'.format(subtype_short[1:30]))


                        else:
                            gene_set_score_a.to_excel(writer, '{}'.format(subtype_short))

                    else:
                        gene_set_score_a.to_excel(writer, '{}'.format(subtype.strip()))


                else:
                    print("gene_set_score", gene_set_score_a)

                gene_set_score_a = gene_set_score_a.loc[(gene_set_score != 0).any(axis=1), :]
                gene_set_score_b, gene_set_std_b = score_gene_set(dataset, input_dim, args, model,
                                                                  signatures=args.signatures, type=element,
                                                                  condition=(dataset.subtypes != subtype))
                gene_set_score_b = gene_set_score_b.loc[(gene_set_score != 0).any(axis=1), :]

                ax = fig.add_subplot(outer_grid[i])
                plot_gene_set_score(ax, gene_set_score.loc[(gene_set_score != 0).any(axis=1), :],
                                    gene_set_score_b - gene_set_score_a, diverging=True, cbar_draw=False, hide_black=True)
                ax.set_title(subtype, fontsize=12)
                ax.set_xticklabels(ax.get_xticklabels(), fontsize=4)

            fig.tight_layout()
            if args.path == None:
                plt.show()
                plt.close('all')
            else:
                # file.savefig('Heatmap_{}.eps'.format(args.dataset), bbox_inches='tight', format='eps')
                file.savefig()

        if args.path != None:
            print("close file")
            writer.save()


def plot_test_loss(test_losses, train_losses, title, file=None):
    fig = plt.figure()
    # test_losses = torch.stack(test_losses)
    # train_losses = torch.stack(train_losses)

    # fig, ax = plt.subplots()
    plt.plot(np.arange(1, len(test_losses) + 1) * 10, test_losses, label="Test loss")
    plt.plot(np.arange(1, len(train_losses) + 1) * 10, train_losses, label="Training loss")

    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # to continue the code
    # draw()
    if file != None:
        file.savefig()
    else:
        plt.show()
        plt.close('all')


def plot_test_accuracy(accuracy_tests, accuracy_trains, title, dest=None):
    accuracy_tests = torch.stack(accuracy_tests)
    accuracy_trains = torch.stack(accuracy_trains)

    # fig, ax = plt.subplots()
    plt.plot(np.arange(1, len(accuracy_tests)) * 10, test_losses.cpu().numpy()[1:], label="Accuracy Testing")
    plt.plot(np.arange(1, len(taccuracy_trains)) * 10, train_losses.cpu().numpy()[1:], label="Accuracy Training")

    # plt.set(xlabel='epochs', ylabel='loss', title=title)
    plt.legend()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    # to continue the code
    # draw()

    if dest:
        fig.savefig(dest)
    plt.show(block=False)
    plt.close('all')


def velten_markers(dataset):
    annot = pd.read_csv('data/Velten_PhenoData.csv', sep=' ')
    clusters = pd.DataFrame(1, dataset.expressions.index, ['HSC', 'MPP', 'CMP', 'MLP', 'MEP', 'GMP'])

    # For all the same
    index = annot.loc[clusters['MEP'] == 1, 'FACS_SSC.Area'].nlargest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, :] = 0

    index1 = annot.loc[clusters['MEP'] == 1, 'FACS_FSC.Area'].nsmallest(int(clusters['MEP'].sum() / 4)).index
    index2 = annot.loc[clusters['MEP'] == 1, 'FACS_SSC.Area'].nlargest(int(clusters['MEP'].sum() / 4)).index
    clusters.loc[index1, :] = 0
    clusters.loc[index2, :] = 0

    index = annot.loc[clusters['MEP'] == 1, 'FACS_Lin'].nlargest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, :] = 0

    # MEP
    index = annot.loc[clusters['MEP'] == 1, 'FACS_cd38'].nsmallest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, 'MEP'] = 0

    index = annot.loc[clusters['MEP'] == 1, 'FACS_cd34'].nsmallest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, 'MEP'] = 0

    index = annot.loc[clusters['MEP'] == 1, 'FACS_cd10'].nlargest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, 'MEP'] = 0

    index = annot.loc[clusters['MEP'] == 1, 'FACS_cd135'].nlargest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, 'MEP'] = 0

    index = annot.loc[clusters['MEP'] == 1, 'FACS_cd45RA'].nlargest(int(clusters['MEP'].sum() / 2)).index
    clusters.loc[index, 'MEP'] = 0

    # CMP
    index = annot.loc[clusters['CMP'] == 1, 'FACS_cd38'].nsmallest(int(clusters['CMP'].sum() / 2)).index
    clusters.loc[index, 'CMP'] = 0

    index = annot.loc[clusters['CMP'] == 1, 'FACS_cd34'].nsmallest(int(clusters['CMP'].sum() / 2)).index
    clusters.loc[index, 'CMP'] = 0

    index = annot.loc[clusters['CMP'] == 1, 'FACS_cd10'].nlargest(int(clusters['CMP'].sum() / 2)).index
    clusters.loc[index, 'CMP'] = 0

    index = annot.loc[clusters['CMP'] == 1, 'FACS_cd135'].nsmallest(int(clusters['CMP'].sum() / 2)).index
    clusters.loc[index, 'CMP'] = 0

    # GMP
    index = annot.loc[clusters['GMP'] == 1, 'FACS_cd38'].nsmallest(int(clusters['GMP'].sum() / 2)).index
    clusters.loc[index, 'GMP'] = 0

    index = annot.loc[clusters['GMP'] == 1, 'FACS_cd34'].nsmallest(int(clusters['GMP'].sum() / 2)).index
    clusters.loc[index, 'GMP'] = 0

    index = annot.loc[clusters['GMP'] == 1, 'FACS_cd10'].nlargest(int(clusters['GMP'].sum() / 2)).index
    clusters.loc[index, 'GMP'] = 0

    index = annot.loc[clusters['GMP'] == 1, 'FACS_cd135'].nsmallest(int(clusters['GMP'].sum() / 2)).index
    clusters.loc[index, 'GMP'] = 0

    index = annot.loc[clusters['GMP'] == 1, 'FACS_cd45RA'].nsmallest(int(clusters['GMP'].sum() / 2)).index
    clusters.loc[index, 'GMP'] = 0

    # HSC
    index = annot.loc[clusters['HSC'] == 1, 'FACS_cd34'].nsmallest(int(clusters['HSC'].sum() / 2)).index
    clusters.loc[index, 'HSC'] = 0

    index = annot.loc[clusters['HSC'] == 1, 'FACS_cd38'].nlargest(int(clusters['HSC'].sum() / 2)).index
    clusters.loc[index, 'HSC'] = 0

    index = annot.loc[clusters['HSC'] == 1, 'FACS_cd90'].nsmallest(int(clusters['HSC'].sum() / 2)).index
    clusters.loc[index, 'HSC'] = 0

    index = annot.loc[clusters['HSC'] == 1, 'FACS_cd45RA'].nlargest(int(clusters['HSC'].sum() / 2)).index
    clusters.loc[index, 'HSC'] = 0

    # MPP
    index = annot.loc[clusters['MPP'] == 1, 'FACS_cd34'].nsmallest(int(clusters['MPP'].sum() / 2)).index
    clusters.loc[index, 'MPP'] = 0

    index = annot.loc[clusters['MPP'] == 1, 'FACS_cd38'].nlargest(int(clusters['MPP'].sum() / 2)).index
    clusters.loc[index, 'MPP'] = 0

    index = annot.loc[clusters['MPP'] == 1, 'FACS_cd90'].nlargest(int(clusters['MPP'].sum() / 2)).index
    clusters.loc[index, 'MPP'] = 0

    index = annot.loc[clusters['MPP'] == 1, 'FACS_cd45RA'].nlargest(int(clusters['MPP'].sum() / 2)).index
    clusters.loc[index, 'MPP'] = 0

    # MLP
    index = annot.loc[clusters['MLP'] == 1, 'FACS_cd34'].nsmallest(int(clusters['MLP'].sum() / 2)).index
    clusters.loc[index, 'MLP'] = 0

    index = annot.loc[clusters['MLP'] == 1, 'FACS_cd38'].nlargest(int(clusters['MLP'].sum() / 2)).index
    clusters.loc[index, 'MLP'] = 0

    index = annot.loc[clusters['MLP'] == 1, 'FACS_cd45RA'].nsmallest(int(clusters['MLP'].sum() / 2)).index
    clusters.loc[index, 'MLP'] = 0

    index = annot.loc[clusters['MLP'] == 1, 'FACS_cd10'].nsmallest(int(clusters['MLP'].sum() / 2)).index
    clusters.loc[index, 'MLP'] = 0

    return clusters


# Function only used for ToyXORDataset
def find_marker_genes(args, dataset, model, topk):
    grads = GuidedSaliency(model)
    x = torch.tensor(dataset.__getitem__([ix for (ix, x) in enumerate(dataset.subtypes == 1) if x])[0],
                     device=args.device, dtype=args.dtype)
    saliency_input_1 = grads.generate_saliency(x, 0).abs().sum(0)

    # print(topk)
    indices = saliency_input_1.topk(topk, largest=True)
    # returns the indices of the largest elements
    marker_genes_detected = pd.DataFrame({'marker_gene': list(dataset.expressions.columns[indices[1].cpu().numpy()]),
                                          'saliency_score': indices[0].cpu().numpy()})
    # print(marker_genes_detected.head())
    # print( marker_genes_detected['correct'].head())

    marker_genes_detected['correct'] = marker_genes_detected['marker_gene'].map(
        lambda x: x in list(dataset.marker_genes.loc[dataset.marker_genes['module_id'].isin([3, 4, 6]), 'gene_id']))
    marker_percent = [x in list(marker_genes_detected['marker_gene']) for x in list(
        dataset.marker_genes.loc[dataset.marker_genes['module_id'].isin([3, 4, 6]), 'gene_id']
    )]
    marker_percent = sum(marker_percent) / len(marker_percent)
    # print(marker_percent)

    return marker_percent, marker_genes_detected


def loss_function_nll_poisson2(inputs, targets, loss_func, eps=1e-8, log_input=False, full=False):
    # print(inputs)
    # print(targets)
    # print("went in loop")
    loss_final = 0
    m = 0
    for input, target in zip(inputs, targets):
        # print("input", input)

        n = 0
        loss = 0
        if log_input == True:
            '''
            k = torch.exp(input) - target * input
            loss += k.sum()
            n = len(target)

            loss=loss/n
            '''
            # print("computes log_input")

            loss += loss_func(input, target, log_input=True)
            # print("loss in function",loss)


        elif full == True:

            idx = torch.where(target > 1)
            bool_array = target > 1
            # print("input",input)
            # print("target",target)
            array_False = np.where(bool_array == False)
            # print("false",array_False)
            array_True = np.where(bool_array == True)
            # print("True",array_True)

            # print("input[array_True]",input[array_True])
            # print("target[array_True]", target[array_True])

            if len(target[array_False]) >= 1:
                # print("full simple")

                k = torch.sub(input[array_False], target[array_False] * torch.log(input[array_False] + eps))
                # print("initial loss", k)
                # print("len",len(target[array_False]))
                loss += k.sum()
                # print("loss initial",loss)
                n += len(target[array_False])
                print("n1", n)

            if len(target[array_True]) >= 1:
                # print("full complicated")
                # loss+=input - target * math.log(input+eps) + target* math.log(target) - target + 0.5 * math.log(2 * math.pi * target)
                k = torch.sub(input[array_True], target[array_True] * torch.log(input[array_True] + eps)) + target[
                    array_True] * torch.log(target[array_True]) - target[array_True] + 0.5 * torch.log(
                    2 * math.pi * target[array_True])

                loss += k.sum()
                n += len(target[array_True])

            loss = loss / n
            # print(loss)

        else:
            # rint("else path")
            k = torch.sub(input, target * torch.log(input + eps))
            loss += k.sum()
            n = len(target)
            loss = loss / n
            # print("inside",loss)

        m += 1
        # print("inside/outside loss",loss)
        loss_final += loss

    return loss_final / m


def plot_optimization(model, experiment):
    # go.FigureWidget()
    # from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

    figure = plot_contour(model=model, param_x="lr", param_y='l_orthog', metric_name='loss')
    figure = go.Figure(figure[0])

    # best_objectives = np.array([[trial.objective_mean * 100 for trial in experiment.trials.values()]])
    # best_objective_plot = optimization_trace_single_method(
    #    y=np.maximum.accumulate(best_objectives, axis=1),
    #    title="Model performance vs. # of iterations",
    #    ylabel="Classification loss, %",
    # )

    # figures.append(go.Figure(figure))
    figure1 = go.Figure(best_objective_plot[0])

    if "/" in args.path:
        pio.write_image(figure, file=args.path.rsplit('/', 1)[0] + "/" + 'Optimization_1.pdf')
        # pio.write_image(figure1, file=args.path.rsplit('/', 1)[0] + "/" + 'Optimization_2.pdf')

    elif args.path != None:
        pio.write_image('Optimization_1.pdf')
        # pio.write_image('Optimization_2.pdf')
    else:
        fig.show()
    '''


        merger = PdfFileMerger(strict=False)
        PDF.close()

        arr = os.listdir()
        print(arr)

        # Merge plots
        for PDF_file in [args.path + '.pdf', args.path + '_1.pdf', args.path + '_2.pdf']:
            # input1 = open(PDF_file, "rb")
            print(PDF_file)
            merger.append(fileobj=PDF_file)

        output = open("merged.pdf", "wb")
        merger.write(output)
        merger.close()


    else
    #fig.show()
    '''


def define_Kwargs(norm_type):
    if norm_type == None or norm_type == "CMP":
        kwargs = {'log_input': False, 'full': True}
        # print("NONE or CMP")
    else:
        kwargs = {'log_input': True, 'full': False}

    return kwargs


def K_NeighborsClassifier(label_name,name_test_data, train_dataset, train_subtypes, test_dataset, test_subtypes, args, model=None,
                          In_both_datasets=True, file=None, title="", path=None):
    print("K_NeighborsClassifier")
    print("train_dataset", train_dataset.expressions.shape)
    print("test_dataset", test_dataset.expressions.shape)

    embedding_train = Run_data_trough_model(train_dataset, args, model)

    embedding_test = Run_data_trough_model(test_dataset, args, model)



    print("embedding_train", embedding_train.shape)
    print("train_subtypes", train_subtypes.shape)
    print("embedding_test", embedding_test.shape)
    print("test_subtypes", test_subtypes.shape)

    fig = plt.figure()
    neighbors_settings = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]

    training_accuracy = []
    test_accuracy = []
    Prediction_check = []

    print("path",path)

    if path!=None:
        writer = pd.ExcelWriter(path.rsplit('/', 1)[0]+ "/"+name_test_data+"_"+label_name+'.xlsx', engine='xlsxwriter')

    for n_neighbors in neighbors_settings:

        common_cell_types = set(test_subtypes).intersection(set(train_subtypes))
        print(common_cell_types)

        if len(common_cell_types) != 0:

            clf = KNeighborsClassifier(n_neighbors=n_neighbors)

            # Train the model using the training sets
            clf.fit(embedding_train, train_subtypes)

            # print("test_subtypes",set(test_subtypes))
            # print("train_subtypes", set(train_subtypes))

            if In_both_datasets == True:
                bool_array = []
                for subtype in test_subtypes:

                    if subtype in set(train_subtypes):
                        bool_array.append(True)
                    else:
                        bool_array.append(False)

                    # 0:Overcast, 2:Mild
                    # print("Test set predictions: {}".format(predicted))
                embedding_test = embedding_test[bool_array, :]

                test_subtypes = test_subtypes[bool_array]

                predicted = clf.predict(embedding_test)

                prediction_dict = {}
                count_True = 0
                count_False = 0

                print("Confusion matrix")


                unique_label = np.unique([test_subtypes, predicted])
                cmtx = pd.DataFrame(
                    confusion_matrix(test_subtypes, predicted, labels=unique_label),
                    index=['true:{:}'.format(x) for x in unique_label],
                    columns=['pred:{:}'.format(x) for x in unique_label])
                print(cmtx)

                cmtx.to_excel(writer, sheet_name=str(n_neighbors))


                #plot_confusion_matrix(clf, test_subtypes, predicted)

                #plt.savefig('Confusion_matrix'+name_test_data+'.pdf')

                '''
                test_subtypes = np.array(test_subtypes)
                for i in range(len(test_subtypes)):

                    if predicted[i] == test_subtypes[i]:
                        count_True += 1
                        # print(predicted[i]==test_subtypes[i])
                    else:
                        count_False += 1
                        if "True: " + test_subtypes[i] + ", Predicted: " + predicted[i] in prediction_dict.keys():
                            # print("True: "+ test_subtypes[i]+", Predicted: "+predicted[i])
                            prediction_dict["True: " + test_subtypes[i] + ", Predicted: " + predicted[i]] += 1
                        else:
                            prediction_dict["True: " + test_subtypes[i] + ", Predicted: " + predicted[i]] = 1
                            
                '''

                #print(prediction_dict)

                #Prediction_check.append(count_True / (count_True + count_False))

                #training_accuracy.append(clf.score(embedding_train, train_subtypes))

                #test_accuracy.append(clf.score(embedding_test, test_subtypes))


        else:
            print("No common cell types. K NeighborsClassifier not performed")

    writer.save()
    '''
    if len(common_cell_types) != 0:
        print("training_accuracy", training_accuracy)
        print("test_accuracy", test_accuracy)
        print("print(Prediction_check)", Prediction_check)

        plt.plot(neighbors_settings, training_accuracy, label='Training')
        plt.plot(neighbors_settings, test_accuracy, label='Test')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of Neighbors')
        plt.legend()
        plt.title("K Nearest Neighbor plot" + title)

        if file != None:
            file.savefig()

        else:
            plt.show()
            plt.close('all')
    '''


def Run_data_trough_model(dataset, args, model=None):
    try:
        x = torch.tensor(dataset.expressions.values, device=args.device, dtype=args.dtype)
    except:
        x = torch.tensor(dataset.expressions, device=args.device, dtype=args.dtype)

    if model:
        model.eval()
        x, scores = model(x)

    return x


def make_plot2(subtitle, args, dataset, title, model=None, file=None, plot='umap', dataset2=None, dim=(9, 5)):
    if dataset2!=None:
        print("Having 2 datasets")

        if plot == "pca":

            embedding, pca = get_embedding(plot, args, dataset, model, dataset2)


        else:

            embedding = get_embedding(plot, args, dataset, model, dataset2)


        Topics=["Both","Channel", "Label"]

        #print("embedding",embedding)
        #print(set(dataset2.subtypes))
        #print(set(dataset.subtypes))


        for element in Topics:
            print("Element",element)

            fig = plt.figure(figsize=dim)
            #fig=plt.figure(num=1, figsize=(13, 13), dpi=80, facecolor='w', edgecolor='k')
            outer_grid = grd.GridSpec(1, 1)
            panel = outer_grid[0]




            if element=="Label":

                print("length",dataset2.subtypes.shape[0])
                print(dataset2.subtypes)
                subtypes_label = [""] * dataset2.subtypes.shape[0]
                print(len(subtypes_label))


                for i in range(dataset2.subtypes.shape[0]):
                    dataset2.subtypes[i] = dataset2.subtypes[i] + ' *'

                #for i in range(dataset2.subtypes.shape[0]):
                  #  dataset2.subtypes[i] = dataset2.subtypes[i] + ' *'


                #for i in range(len(subtypes_label)):
                 #   print("new label", subtypes_label[i])
                  #  print("old label", dataset2.subtypes[i])

                #subtypes_label = pd.Series(subtypes_label)
                new_subtypes = pd.concat([dataset.subtypes, dataset2.subtypes])



                plot_clusters_full_color_multiple(subtitle, new_subtypes, embedding, title, panel)



            elif element=="Channel":
                s1 = dataset.subtypes.shape[0]
                s2 = dataset2.subtypes.shape[0]
                substypes_test = s1 * ['Training                          '] + s2 * ['Testing']
                substypes_test = pd.Series(substypes_test)

                print("in channel")
                print("shape embeddings", embedding.shape)
                print("subtypes", substypes_test.shape)

                plot_clusters_full_color_multiple(subtitle, substypes_test, embedding, title, panel)

            elif element == "Both":
                subtypes = pd.concat([dataset.subtypes, dataset2.subtypes])

                plot_clusters_full_color_multiple(subtitle, subtypes, embedding, title, panel)


            if plot == "pca":
                plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0] * 100))
                plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1] * 100))

            #plt.legend(fontsize=10, markerscale=10, bbox_to_anchor=(1.01, 1,0.1, 1.04), loc="upper left")
            plt.legend(fontsize=10, markerscale=10, bbox_to_anchor=(1.01, 1, 10, 1.04), loc="upper left", )
            plt.xticks([])
            plt.yticks([])
            fig.tight_layout()

            if file != None:
                file.savefig()

            else:
                plt.show()
                plt.close('all')

            if element== "Label":
                for i in range(dataset2.subtypes.shape[0]):
                    dataset2.subtypes[i] = dataset2.subtypes[i][:-2]

    else:
        fig = plt.figure(figsize=dim)
        outer_grid = grd.GridSpec(1, 1)
        panel = outer_grid[0]

        if plot == "pca":
            embedding, pca = get_embedding(plot, args, dataset, model)
        else:
            embedding = get_embedding(plot, args, dataset, model)



        plot_clusters_full_color_multiple(subtitle, dataset.subtypes, embedding, title, panel)

        if plot == "pca":
            plt.xlabel('PC 1 (%.2f%%)' % (pca.explained_variance_ratio_[0] * 100))
            plt.ylabel('PC 2 (%.2f%%)' % (pca.explained_variance_ratio_[1] * 100))

        #(x, y, width, height)
        #plt.legend(fontsize=10, markerscale=10, bbox_to_anchor=(1.01, 1,0.1, -0.05), loc="upper left")
        plt.xticks([])
        plt.yticks([])
        fig.tight_layout()

        if file != None:
            file.savefig()

        else:
            plt.show()
            plt.close('all')



    # Channel = sequncing channel

    # Label = add label to test and train dataset.





