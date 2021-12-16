import matplotlib.pyplot as plt
import seaborn as sns
import platform
import sys
import torch.nn as nn
import _pickle as cPickle



#python main_new_optimized.py Tabula_Muris --organ Muscle  --path Output/Tabula_Muris/Muscle/Test --norm_type logCPM --optimize Yes --epochs 100

print("python version", platform.python_version())
import os

print("system version", sys.version)
# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn.init as weight_init
from pathlib import Path
import datasets
import models
import utils_class
import argparse
import config
import numpy as np
import os.path
import torch
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import plotly.io as pio
from plotly import graph_objs as go

# optimization
#from ax.service.managed_loop import optimize
#from ax.plot.contour import plot_contour
#from ax.plot.trace import optimization_trace_single_method
#from ax.service.managed_loop import optimize


import pickle


#python3 main_new_optimized.py Tabula_Muris --organ "Marrow"  --path_model "Output/Tabula_muris/Marrow/25_03_20

class Network(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch saliency maps for single cell RNA-seq expression matrix.')
        parser.add_argument('dataset', type=str, help='Select dataset among PBMC, Paul, Velten, Tabula_Muris, Merged_10X_FACS.')
        parser.add_argument('--epochs', type=int, default=600,
                            help='Define the number of epochs that you want to run default 300')
        parser.add_argument('--path', type=str, default=None, help='Path to safe plots')
        parser.add_argument('--plot', type=bool, default=False, help='Make plots')
        parser.add_argument('--organ', type=str, default='Muscle',
                            help='Which organ do you want to run the data on fx Bladder Brain_Microglia-counts, Brain_Neurons, Colon, Fat, Heart, Kidney, Liver, Lung, Mammary, Marrow, Muscle, Pancreas, Skin, Spleen, Thymus, Tongue, Trachea')
        parser.add_argument('--norm_type', type=str, default="None",
                            help='Data preprocessing: e.g  logCPM, logcounts else None')
        parser.add_argument('--path_model', type=str, default=None,
                            help='Data preprocessing: e.g Output/Tabula_muris/Heart/25_03_20 ')
        parser.add_argument('--optimize', type=bool, default=False,
                            help='Data preprocessing: Yes if desired  ')
        parser.add_argument('--test_dataset', type=str, default=None,
                            help='Give full path to data set ')
        parser.add_argument('--dataset_test', type=str, default=None,
                            help='Give full path to data set ')
        parser.add_argument('--one_cell', type=bool, default=False, help='Give full path to data set ')


        args = parser.parse_args()
        print("start", args)

        if args.dataset == 'PBMC':
            self.dataset = datasets.PBMCDataset(args.norm_type)
            self.args = config.pbmc_config(args)
        elif args.dataset == 'Paul':
            self.dataset = datasets.PaulDataset(args.norm_type)
            self.args = config.paul_config(args)
        elif args.dataset == 'Velten':
            self.dataset = datasets.VeltenDataset(args.norm_type)
            self.args = config.velten_config(args)
        elif args.dataset == 'Tabula_Muris':
            self.dataset = datasets.Tabula_MurisDataset(args)
            self.args = config.Tabula_Muris_config(args)

        elif args.dataset == 'Simulation':
            self.dataset = datasets.Simulation_data(args.file)
            self.args = config.Simulation_config(args)
        elif args.dataset == '10X':
            self.dataset = datasets.X_10(args)
            self.args = config.X_10_config(args)
        elif args.dataset == 'Harmony':
            self.dataset = datasets.Harmony(args)
            self.args = config.Harmony_config(args)
        elif args.dataset == 'Seurat':
            self.dataset = datasets.Seurat(args)
            self.args = config.Seurat_config(args)
        elif args.dataset == 'Human':
            self.dataset = datasets.Human(args)
            self.args = config.Human_config(args)

        else:
            print('Select dataset among PBMC, Paul, Velten,Tabula_Muris.')
            print(args)
            quit()


        

        if self.args.path != None and self.args.test_dataset :
            if '/' in self.args.path:
                self.pdf = PdfPages( self.args.path.rsplit('/', 1)[0] +"/" +"modelling_"+self.args.test_dataset + "_trained_"+ self.args.organ+"_"+self.args.dataset+".pdf")
            else:
                self.pdf = PdfPages("modelling_" + self.args.test_dataset + "_trained_" + self.args.organ + "_" + self.args.dataset + ".pdf")

        elif self.args.path != None:
            self.pdf = PdfPages(self.args.path + ".pdf")

        else:
            self.pdf = None

        torch.manual_seed(self.args.seed)

        self.input_dim = self.dataset.expressions.shape[1]
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.train_loader = DataLoader(
            self.dataset, batch_size=self.args.batch_size,
            sampler=datasets.ChunkSampler(int(self.args.train_percent * len(self.dataset))), **kwargs)
        self.test_loader = DataLoader(
            self.dataset, batch_size=self.args.test_batch_size,
            sampler=datasets.ChunkSampler(len(self.dataset), int(self.args.train_percent * len(self.dataset))),
            **kwargs)

        self.full_loader = DataLoader(
            self.dataset, batch_size=self.args.test_batch_size,
            sampler=datasets.ChunkSampler(len(self.dataset), int(0.99999*len(self.dataset))),
            **kwargs)

        # Additional parameters needed
        self.Best_optimized_loss = None

        if self.args.plot == True:
            try:
                utils_class.make_plot("", self.args, self.dataset,
                                      'PCA of original dataset {} from {}'.format(self.args.dataset,self.args.organ),
                                      'figures/UMAP_original_{}.eps'.format(self.args.dataset), plot="pca",
                                      file=self.pdf)

                utils_class.make_plot("", self.args, self.dataset,
                                      'UMAP of original dataset {} from {}'.format(self.args.dataset,self.args.organ),
                                      'figures/UMAP_original_{}.eps'.format(self.args.dataset), plot="umap",
                                      file=self.pdf)

            except Exception as e:
                print(e)
                print("Could not plot original pca or UMAP")

        utils_class.make_plot("", self.args, self.dataset,
                              'UMAP of original dataset {} from {}'.format(self.args.dataset, self.args.organ),
                              'figures/UMAP_original_{}.eps'.format(self.args.dataset), plot="tsne",
                              file=self.pdf)



       # if self.args.plot == True:
        #    utils_class.make_plot2(
        #        '', self.args, self.dataset,
        #        '{}, original dataset {}'.format(self.args.dataset, self.args.organ),
        #        file=self.pdf, plot='pca')

        #    utils_class.make_plot2(
        #        '', self.args, self.dataset,
        #        '{}, original dataset {}'.format(self.args.dataset, self.args.organ),
        #        file=self.pdf, plot='umap')





        # utils_class.paga_plot(self.dataset)



    def Define_Network(self):

        self.model = models.AutoencoderTwoLayers(self.input_dim, self.args.hidden_size, self.args.more_hidden_size).to(
            self.args.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, nesterov=True,
                                   weight_decay=self.args.l2_reg)



    def Optimize_hyperparameters_using_Ax_platform(self, parameters):

        print(parameters)
        # try:
        for key, value in parameters.items():
            # print(key,value)
            self.args[key] = value

        kwargs = utils_class.define_Kwargs(self.args.norm_type)
        print("Kwargs", kwargs)

        # print("args used in optimization", self.args)
        self.Define_Network()

        test_losses = []
        train_losses = []

        for epoch in range(1, self.args.epochs + 1):
            # start_time = time.time()

            model = utils_class.train(self.args, self, model=self.model, train_loader=self.train_loader,
                                      optimizer=self.optimizer, epoch=epoch, Optimization_loop=True, **kwargs)
            # print("Time to run loss function", time.time() - start_time)
            if epoch % 10 == 0:
                test_loss = utils_class.check_accuracy(self, self.args, self.model, self.test_loader,
                                                       Optimization_loop=True,
                                                       **kwargs)

                train_loss = utils_class.check_accuracy(self, self.args, self.model, self.train_loader,
                                                        Optimization_loop=True, **kwargs)
                test_losses.append(test_loss)
                train_losses.append(train_loss)

        try:
            print("test losses", test_losses)
            print("train losses", train_losses)
            print("Test loss", test_loss)

        except Exception as e:
            print(e)
            print("Could not print test- or train-losses")

        try:
            if self.args.plot == True:
                utils_class.make_plot('lr: {:.4e}, l_orthog: {:.4e}'.format(self.args.lr, self.args.l_orthog, 5),
                                      self.args,
                                      self.dataset,
                                      'PCA of model dataset optimized {}'.format(self.args.dataset),
                                      'figures/UMAP_original_{}.eps'.format(self.args.dataset), model=self.model,
                                      file=self.pdf, plot='pca')

                utils_class.make_plot('lr: {:.4e}, l_orthog: {:.4e}'.format(self.args.lr, self.args.l_orthog, 5),
                                      self.args,
                                      self.dataset,
                                      'UMAP of model dataset optimized {}'.format(self.args.dataset),
                                      'figures/UMAP_original_{}.eps'.format(self.args.dataset), model=self.model,
                                      file=self.pdf, plot='umap')




        except Exception as e:
            print(e)
            print("Could not make model UMAP or pca")



        try:
            if self.args.plot == True:
                utils_class.plot_test_loss(test_losses, train_losses, title="Test loss Optimization", file=self.pdf)

        except Exception as e:
            print(e)
            print("could not plot test loss")

        try:
            print("Singular Value", utils_class.get_singular_values_norm(self.args, self.dataset, self.model))


        except Exception as e:
            print(e)
            print("Could not calculate singular valules")

        # except Exception as e:
        # print(e)
        # print("Could not train")
        # try:

        try:
            test_loss
        except:
            print("test loss not defined, due to lack of epochs")
            test_loss = 100000

        # except:
        #   print("Could not save model")
        if self.Best_optimized_loss == None:

            self.Best_optimized_loss = test_loss

            print("Saves initial checkpoint")

            if self.args.path != None and "/" in self.args.path:
                # models.AutoencoderTwoLayers(self.input_dim, self.args.hidden_size, self.args.more_hidden_size).to(self.args.device)
                torch.save({
                    'epoch': self.args.epochs,
                    'args': self.args,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': test_loss,
                    'test_losses': test_losses,
                    'train_losses': train_losses,
                }, self.args.path.rsplit('/', 1)[0] + "/" + "Best_model.tar")
            elif self.args.path != None:
                torch.save({
                    'epoch': self.args.epochs,
                    'args': self.args,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': test_loss,
                    'test_losses': test_losses,
                    'train_losses': train_losses,
                }, "Best_model.tar")
            else:
                torch.save({
                    'epoch': self.args.epochs,
                    'args': self.args,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': test_loss,
                    'test_losses': test_losses,
                    'train_losses': train_losses,
                }, "Best_model.tar")
        elif self.Best_optimized_loss > test_loss:
            self.Best_optimized_loss = test_loss
            print("Enters 2. saving point")
            if self.args.path != None and "/" in self.args.path:
                torch.save({
                    'args': self.args,
                    'epoch': self.args.epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': test_loss,
                    'test_losses': test_losses,
                    'train_losses': train_losses,
                }, self.args.path.rsplit('/', 1)[0] + "/" + "Best_model.tar")
            elif self.args.path != None:
                torch.save({
                    'args': self.args,
                    'epoch': self.args.epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': test_loss,
                    'test_losses': test_losses,
                    'train_losses': train_losses,
                }, "Best_model.tar")
            else:
                torch.save({
                    'epoch': self.args.epochs,
                    'args': self.args,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': test_loss,
                    'test_losses': test_losses,
                    'train_losses': train_losses,
                }, "Best_model.tar")

        return utils_class.check_accuracy(self, self.args, self.model, self.test_loader, Optimization_loop=True,
                                          **kwargs)

    def Train_Network(self, best_hyperparameters=None):
        minimum_test_loss = 10000000

        # kwargs = {'bw': 2}
        #utils_class.sns_distplot(self.args, self.train_loader)

        kwargs = utils_class.define_Kwargs(self.args.norm_type)
        print("Kwargs", kwargs)

        if best_hyperparameters != None:

            if self.args.path != None and "/" in self.args.path:
                checkpoint = torch.load(self.args.path.rsplit('/', 1)[0] + "/" + "Best_model.tar")
                self.args.epochs = checkpoint['epoch']
                self.args = checkpoint['args']
                test_losses_optimization = checkpoint['test_losses']
                train_losses_optimization = checkpoint['train_losses']
                loss = checkpoint['loss']
                self.model = models.AutoencoderTwoLayers(self.input_dim, self.args.hidden_size,
                                                         self.args.more_hidden_size).to(self.args.device)
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                           nesterov=True, weight_decay=self.args.l2_reg)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.model.load_state_dict(checkpoint['model_state_dict'])


            else:
                checkpoint = torch.load("Best_model.tar")
                self.args.epochs = checkpoint['epoch']
                self.args = checkpoint['args']
                loss = checkpoint['loss']
                test_losses_optimization = checkpoint['test_losses']
                train_losses_optimization = checkpoint['train_losses']
                self.model = models.AutoencoderTwoLayers(self.input_dim, self.args.hidden_size,
                                                         self.args.more_hidden_size).to(self.args.device)
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                           nesterov=True, weight_decay=self.args.l2_reg)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'args' in checkpoint.keys():
                print("Loading arguments from saved network")
                arguments = checkpoint['args']

                for key, value in arguments.items():
                    # print(key,value)
                    self.args[key] = value
                #print("checkpoint", arguments)

            print("Epochs during optimzation " + str(self.args.epochs) + " epochs after optimization " + str(
                int(self.args.epochs * 5)))
            self.args.epochs = int(self.args.epochs * 5)

            loss_check = utils_class.check_accuracy(self, self.args, self.model, self.test_loader,
                                                    Optimization_loop=True, **kwargs)
            print("loss_check", loss_check)


        else:
            self.Define_Network()

        print("Epochs in training:", self.args.epochs)
        print("l_orthog", self.args.l_orthog)
        print("l2", self.args.l2_reg)
        print("l1_reg", self.args.l1_reg)
        print("Learning rate", self.args.lr)
        print("Organ", self.args.organ)

        test_losses = []
        train_losses = []

        for epoch in range(1, self.args.epochs + 1):

            # start_time = time.time()
            utils_class.train(self.args, self, self.model, self.train_loader, self.optimizer, epoch=epoch, **kwargs)
            # print("Time to run loss function", time.time() - start_time)

            if epoch % 10 == 0:

                test_loss = utils_class.check_accuracy(self, self.args, self.model, self.test_loader, **kwargs)
                print("test loss",test_loss )

                train_loss = utils_class.check_accuracy(self, self.args, self.model, self.train_loader, **kwargs)
                print("train loss",train_loss)

                print('Epoch: {}, Average loss: {}'.format(epoch, test_loss))
                try:
                    print('Singular values l2-norm: {}'.format(
                        utils_class.get_singular_values_norm(self.args, self.dataset, self.model)))
                except Exception as e:
                    print(e)
                    print("Could not compute Singular values")

                if test_loss < minimum_test_loss:
                    minimum_test_loss = test_loss

                    if self.args.path_model == None:

                        if self.args.path != None and "/" in self.args.path:
                            torch.save({
                                'epoch': self.args.epochs,
                                'args': self.args,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': test_loss,
                            }, self.args.path.rsplit('/', 1)[0] + "/" + "Best_model_final.tar")
                        else:
                            torch.save({
                                'epoch': self.args.epochs,
                                'args': self.args,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': test_loss,
                            }, "Best_model_final.tar")

                    else:
                        if self.args.path != None and "/" in self.args.path:
                            torch.save({
                                'epoch': self.args.epochs,
                                'args': self.args,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': test_loss,
                            }, self.args.path.rsplit('/', 1)[0] + "/" + "Best_model_final_2.tar")
                        else:
                            torch.save({
                                'epoch': self.args.epochs,
                                'args': self.args,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': test_loss,
                            }, "Best_model_final_2.tar")

                test_losses.append(test_loss)
                train_losses.append(train_loss)

                if len(test_losses) >= 2 and abs(test_losses[-2] - test_loss) < 1e-8:
                    print("No improvement in training", abs(test_losses[-1] - test_loss))
                    print("Training does not improve loss, ending network")
                    print("Test losses", test_losses[1:])
                    print("Train losses", train_losses[1:])
                    Break = True
                    break

                if len(test_losses) >= 2 and abs(test_losses[-2] - test_loss) < 1e-8:
                    print("No improvement in training", abs(test_losses[-1] - test_loss))
                    print("Training does not improve loss, ending network")
                    print("Test losses", test_losses[1:])
                    print("Train losses", train_losses[1:])
                    Break = True
                    break

                if test_loss > 10000000.:
                    print("Test Loss too high ending network")
                    print("Test losses", test_losses[1:])
                    print("Train losses", train_losses[1:])
                    Break = True
                    break

                elif np.isnan(test_loss):
                    print("Test losses", test_losses[1:])
                    print("Train losses", train_losses[1:])
                    print("Gradient has exploded, not a number")
                    Break = True
                    break

        # load the best saved model
        if self.args.path != None and "/" in self.args.path:
            try:
                checkpoint = torch.load(self.args.path.rsplit('/', 1)[0] + "/" + "Best_model_final.tar")
                self.args = checkpoint['args']
                self.args.epochs = checkpoint['epoch']

                self.model = models.AutoencoderTwoLayers(self.input_dim, self.args.hidden_size,
                                                         self.args.more_hidden_size).to(self.args.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                loss = checkpoint['loss']
            except:
                print("Could not load Best_model_final.tar with path")
                print("Likely too few epochs was trained")
        else:
            try:
                checkpoint = torch.load("Best_model_final.tar")
                self.args = checkpoint['args']
                self.args.epochs = checkpoint['epoch']
                self.model = models.AutoencoderTwoLayers(self.input_dim, self.args.hidden_size,
                                                         self.args.more_hidden_size).to(self.args.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.args.epochs = checkpoint['epoch']
                loss = checkpoint['loss']
            except:
                print("Could not load Best_model_final.tar")
                print("Likely too few epochs was trained")
        try:
            print("loss check", loss)
            #print("checkpoint", checkpoint)
        except:
            print("No optimal model loaded")

        if self.args.plot == True:
            try:
                utils_class.make_plot(
                    'lr: {:.4e}, l_orthog: {:.4e}'.format(self.args.lr, self.args.l_orthog), self.args,
                    self.dataset,
                    'PCA of model dataset {}'.format(self.args.dataset),
                    'figures/UMAP_original_{}.eps'.format(self.args.dataset), model=self.model, file=self.pdf,
                    plot='pca')
            except Exception as e:
                print(e)
                print("Could not make final PCA plot")

            try:
                utils_class.make_plot(
                    'lr: {:.4e}, l_orthog: {:.4e}'.format(self.args.lr, self.args.l_orthog), self.args,
                    self.dataset,
                    'UMAP of model dataset {}'.format(self.args.dataset),
                    'figures/UMAP_original optimization_{}.eps'.format(self.args.dataset), model=self.model,
                    file=self.pdf, plot='umap')

            except Exception as e:
                print(e)
                print("Could not make final UMAP plot")

            try:

                utils_class.plot_test_loss(test_losses, train_losses, title="Test loss", file=self.pdf)

            except Exception as e:
                print(e)
                print("Could not plot test losses")

            if best_hyperparameters != None:
                try:

                    utils_class.plot_test_loss(test_losses_optimization + test_losses,
                                               train_losses_optimization + train_losses,
                                               title="Test and traininigs loss", file=self.pdf)

                except Exception as e:
                    print(e)
                    print("Could not plot test losses")



        try:
            print("Test losses", test_losses)
            print("Train losses", train_losses)
        except Exception as e:
            print(e)



        try:
        #    print("heatmap", self.args)
            # print("pdf",self.pdf)

            utils_class.make_heatmap(self.args, self.dataset, self.model, file=self.pdf)

        except Exception as e:
            print(e)
        #    print("Could not make heatmap")

        if self.pdf != None:
            try:
                self.pdf.close()
                print("File closed")
            except:
                print("could not close pdf")

    def load_model(self):
        #print("In load model function")
        model_defined = False

        path_to_model = self.args.path_model



        #try:
        '''
        kwargs = utils_class.define_Kwargs(self.args.norm_type)

        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        '''

            


        print("path_model", self.args.path_model + "/" + "Best_model_final.tar")
        #checkpoint = torch.load(self.args.path_model + "/" + "Best_model_final.tar", map_location=lambda storage, loc: storage)
        #checkpoint= tarfile.open(name=self.args.path_model + "/" + "Best_model_final.tar", mode='r')
        #checkpoint = pickle.load(self.args.path_model + "/" + "Best_model_final.tar")
        checkpoint = torch.load(self.args.path_model + "/" + "Best_model_final.tar", map_location = self.args.device)
        #print("1 checkpoint['args'].device",checkpoint['args'].device)
        checkpoint['args'].device=self.args.device
        #print("2 checkpoint['args'].device",checkpoint['args'].device)


        #print(self.args.path_model + "/" + "Best_model_final.tar")
        self.args.epochs = checkpoint['epoch']
        #self.args = checkpoint['args']
        # test_losses_optimization = checkpoint['test_losses']
        # train_losses_optimization = checkpoint['train_losses']
        self.input_dim = self.dataset.expressions.shape[1]

        self.model = models.AutoencoderTwoLayers(self.input_dim, checkpoint['args'].hidden_size,
                                                 checkpoint['args'].more_hidden_size).to(self.args.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=checkpoint['args'].lr,
                                   momentum=checkpoint['args'].momentum,
                                   nesterov=True, weight_decay=checkpoint['args'].l2_reg)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.model_empty = models.AutoencoderTwoLayers(self.input_dim, checkpoint['args'].hidden_size,
                                                 checkpoint['args'].more_hidden_size).to(self.args.device)
        self.optimizer_empty = optim.SGD(self.model.parameters(), lr=checkpoint['args'].lr,
                                   momentum=checkpoint['args'].momentum,
                                   nesterov=True, weight_decay=checkpoint['args'].l2_reg)

        print("imput to the model")
        print(self.input_dim)


        self.checkpoint = checkpoint
        model_defined = True

        #except Exception as error:
         #   print(error)
         #   print("Could not load model Best_model_final.tar")

        if model_defined == False:

            try:

                kwargs = utils_class.define_Kwargs(self.args.norm_type)


                checkpoint = torch.load(self.args.path_model + "/Best_model.tar")

                checkpoint['args'].device = args.device
                print(checkpoint['args'])
                print(self.args.path_model + "/" + "Best_model.tar")

                kwargs = utils_class.define_Kwargs(self.args.norm_type)

                checkpoint = torch.load(self.args.path_model + "/" + "Best_model.tar")
                self.args.epochs = checkpoint['epoch']
                self.args = checkpoint['args']
                # test_losses_optimization = checkpoint['test_losses']
                # train_losses_optimization = checkpoint['train_losses']
                print("imput to the model")
                print(self.input_dim)
                print(checkpoint['args'].hidden_size)
                print(checkpoint['args'].more_hidden_size)

                self.model = models.AutoencoderTwoLayers(self.input_dim, checkpoint['args'].hidden_size,
                                                         checkpoint['args'].more_hidden_size).to(self.args.device)
                self.optimizer = optim.SGD(self.model.parameters(), lr=checkpoint['args'].lr,
                                           momentum=checkpoint['args'].momentum,
                                           nesterov=True, weight_decay=checkpoint['args'].l2_reg)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                model_defined = True

                self.checkpoint = checkpoint
                print("check point")
                print(checkpoint)




            except Exception as error:
                print(error)
                print("Could not load model Best_model.tar")
                print("exit script")
                quit(3)




        #plot 3-4
        if self.args.plot == True:
            try:
                utils_class.make_plot2(
                '', self.args, self.dataset,
                '{}, modelled dataset {}'.format(self.args.dataset, self.args.organ),
              file=self.pdf, plot='pca',model=self.model)

                utils_class.make_plot2(
                '', self.args, self.dataset,
                '{}, modelled dataset {}'.format(self.args.dataset, self.args.organ),
                 file=self.pdf, model=self.model, plot='tsne')
     


            except Exception as e:
                print(e)
                print("Could not make final UMAP plot")

        utils_class.make_plot2(
            '', self.args, self.dataset,
            '{}, modelled dataset {}'.format(self.args.dataset, self.args.organ),
            file=self.pdf, plot='pca', model=self.model)

        utils_class.make_plot2(
            '', self.args, self.dataset,
            '{}, modelled dataset {}'.format(self.args.dataset, self.args.organ),
            file=self.pdf, model=self.model, plot='tsne')

        loss = self.checkpoint['loss']
        #print("checkpoint", checkpoint)
        print("plot",self.args.plot)

        if self.args.plot == True:
            try:
                utils_class.make_plot2(
                    '', self.args,
                    self.dataset,
                    'PCA of modelled dataset {} from {} '.format(self.args.dataset,self.args.organ),
                    'figures/UMAP_original optimization_{}.eps'.format(self.args.dataset), model=self.model,
                    file=self.pdf, plot='pca')
            except:
               print("Could not make pca plot of loaded model")

            try:
                utils_class.make_plot2(
                '', self.args,
                self.dataset2,
                'UMAP of modelled dataset {} from {}'.format(self.args.dataset2,self.args.organ),
                'figures/UMAP_original optimization_{}.eps'.format(self.args.dataset), model=self.model,
                file=self.pdf, plot='tsne')


            except:
                print("Could not make UMAP of loaded model")


    def load_Heatmap(self):


        self.checkpoint['args'].path=self.args.path
        self.checkpoint['args']['test_dataset'] = self.args.test_dataset
        self.checkpoint['args']['one_cell'] = self.args.one_cell
        self.checkpoint['args']['one_cell'] = self.args.one_cell
        self.checkpoint['args']['signatures'] = self.args.signatures


           #if self.args.one_cell==True:
         #   if self.dataset2:
         #       utils_class.make_heatmap(self.checkpoint['args'], self.dataset2, self.model, file=self.pdf,cell_type=self.dataset2.subtypes )

        #elif self.dataset2:

         #   utils_class.make_heatmap(self.checkpoint['args'], self.dataset2, self.model, file=self.pdf)




        #try:
         #   print("heatmap", self.args)
         #   # print("pdf",self.pdf)

        print("printing arguments",self.args)

        try:
            utils_class.make_heatmap(checkpoint['args'], self.dataset, self.model, file=self.pdf)

        except:
            print("trial 1")

        try:
            utils_class.make_heatmap(self.args, self.dataset, self.model, file=self.pdf)

        except Exception as e:
            print(e)
            print("Could not make heatmap")

        #self.pdf = PdfPages(path_to_model + "/" + "heatmap" + ".pdf")

        #print("making heatmap in load model")



    def test_dataset(self):
        print("test_dataset",self.args.test_dataset)

        self.load_model()

        #Loading information from network


        #print("test_dataset", self.args.test_dataset)

        if self.checkpoint['args'].dataset!='Seurat' and self.checkpoint['args'].dataset!='Harmony' and  self.checkpoint['args'].dataset!='Merged_10X_FACS':
            if self.args.norm_type!= self.checkpoint['args'].norm_type:
                print("Loaded dataset was normalized using "+self.checkpoint['args'].norm_type)
                print("You Loaded the dataset using "+self.args.norm_type)
                print("They must be the same")
                quit(2)

        self.original_dataset= self.dataset
        self.original_subtypes = self.dataset.subtypes

        self.checkpoint['args']["one_cell"]=self.args.one_cell

        #print("1 checking dataset", self.checkpoint['args'].dataset)
        #print("2 checking dataset", self.args.dataset)
        print(self.args.dataset_test)

        if self.args.dataset_test == 'PBMC':
            self.dataset2 = datasets.PBMCDataset(args.norm_type)
        elif self.args.dataset_test == 'Paul':
            self.dataset2 = datasets.PaulDataset(args.norm_type)
        elif self.args.dataset_test == 'Velten':
            self.dataset2 = datasets.VeltenDataset(args.norm_type)
        elif self.args.dataset_test == 'Simulation':
            self.dataset2 = datasets.Simulation_data(args.file)
        elif self.args.dataset_test == 'Smart-seq2':
            self.dataset2 = datasets.Tabula_MurisDataset(self.checkpoint['args'],organ=self.args.test_dataset,original_dataset=self.original_dataset.expressions) #self.checkpoint['args']
        elif self.args.dataset_test == 'Human':
            self.dataset2 = datasets.Human(self.checkpoint['args'],organ=self.args.test_dataset,original_dataset=self.original_dataset.expressions)
            print("In human")




        elif self.args.dataset_test == 'Drop-seq':
            self.dataset2 = datasets.X_10(self.checkpoint['args'],organ=self.args.test_dataset,original_dataset=self.original_dataset.expressions)

        elif self.args.dataset_test == 'Seurat':
            #print("In seurat", self.checkpoint['args'].dataset)
            self.dataset2 = datasets.Seurat(self.checkpoint['args'],organ=self.args.test_dataset,original_dataset=self.original_dataset.expressions)

        elif self.args.dataset_test == "Harmony":
            print("In Harmony", self.checkpoint['args'].dataset)
            self.dataset2 = datasets.Harmony(self.checkpoint['args'],organ=self.args.test_dataset,original_dataset=self.original_dataset.expressions)



        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        #kwargs = utils_class.define_Kwargs(self.args.norm_type)
        #print("loader line 898")
        #print(len(self.dataset2))
        #print(self.args.test_batch_size)
        #print(int(0.99999 * len(self.dataset2)))
        self.loader_test = DataLoader(
            self.dataset2, batch_size=self.args.test_batch_size,
            sampler=datasets.ChunkSampler(len(self.dataset2), int(0.99999 * len(self.dataset2))),
            **kwargs)




        print("loss of original dataset")
        loss_of_data_1 = utils_class.check_accuracy(self, self.args, self.model, self.full_loader,
                                   Optimization_loop=True,
                                   **kwargs)
        print(loss_of_data_1 )

        print("loss of test dataset")
        loss_of_data_2 =utils_class.check_accuracy(self, self.args, self.model, self.loader_test,
                                   Optimization_loop=True,
                                   **kwargs)

        print(loss_of_data_2)

        #print("loaded test dataset", self.dataset2.expressions.shape)

        #print("Modelled using reduced dataser autoencoder")

        #print("Marrow data set")
        #print(self.original_dataset.subtypes.value_counts())
        
        #print("Modelled using reduced dataser autoencoder")
        #label_name="Encoded"
        #utils_class.K_NeighborsClassifier(label_name,self.args.test_dataset,self.original_dataset, self.original_dataset.subtypes,self.dataset2,self.dataset2.subtypes,self.checkpoint['args'],model=self.model,file=self.pdf,title="Modelled by autoencoder",path=self.args.path)

        #print("Modelled using randomly initialized model")
        #label_name = "Randomly_Encoded"
        #utils_class.K_NeighborsClassifier(label_name,self.args.test_dataset,self.original_dataset, self.original_dataset.subtypes, self.dataset2,
                                       #   self.dataset2.subtypes, self.checkpoint['args'], model=self.model_empty,
                                        #  file=self.pdf, title="Modelled by initialized autoencoder",path=self.args.path)

        #print("Modelled full dataset")
        #label_name = "Full_dataset"
        #utils_class.K_NeighborsClassifier(label_name,self.args.test_dataset, self.original_dataset, self.original_dataset.subtypes, self.dataset2, self.dataset2.subtypes, self.checkpoint['args'],file=self.pdf,title="Full dataset", path=self.args.path)

        #print("subtypes "+self.args.test_dataset+" :"+str(set(self.dataset2.subtypes)))
        #print("subtypes " + self.args.organ + " :" + str(set(self.dataset.subtypes)))



        #print("Original",self.original_dataset)
        #print("New", self.dataset2.expressions)

        print("printing args line 914", self.args)
        print("printing args line 915",self.checkpoint['args'])


        try:
            print(self.args)
            #utils_class.make_heatmap(self.args, self.dataset2, self.model, file=self.pdf)
        except Exception as e:
            print("could Not make Heat map 1")
            print(e)

        try:
            print("Ok")
            #utils_class.make_heatmap(self.checkpoint['args'], self.dataset2, self.model, file=self.pdf)

        except Exception as e:
            print(e)
            print("Could not make heatmap 2")

        torch.manual_seed(self.args.seed)
        '''
        self.input_dim = self.dataset.expressions.shape[1]
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.train_loader2 = DataLoader(
            self.dataset, batch_size=self.args.batch_size,
            sampler=datasets.ChunkSampler(int(self.args.train_percent * len(self.dataset))), **kwargs)
        self.test_loader2 = DataLoader(
            self.dataset, batch_size=self.args.test_batch_size,
            sampler=datasets.ChunkSampler(len(self.dataset), int(self.args.train_percent * len(self.dataset))),
            **kwargs)
        '''
        '''
        utils_class.make_plot(
            'lr: {:.4e}, l_orthog: {:.4e}'.format(self.args.lr, self.args.l_orthog), self.args,
            self.dataset,
            'UMAP of model dataset {}'.format(self.args.dataset),
            'figures/UMAP_original optimization_{}.eps'.format(self.args.dataset), model=self.model,
            file=self.pdf, plot='umap')
        '''

        #def make_plot2(subtitle, args, dataset, title, model=None, file=None, plot='umap', dataset2=None, dim=(9, 5)):



        #utils_class.make_plot2(
        #   '', self.args, self.dataset,
         #   '{}, trained on {}'.format(self.args.dataset,self.args.organ), model=self.model,
         #   file=self.pdf, plot='umap')
        #if self.args.one_cell==False:
        if self.args.plot == True:
            # -------------------------##
            utils_class.make_plot2(
                '', self.args, self.dataset2,
                '{}, original dataset {} {}'.format(self.args.dataset, self.args.test_dataset, self.args.dataset_test),
                file=self.pdf, plot='pca')

            utils_class.make_plot2(
                '', self.args, self.dataset2,
                '{}, original dataset {} {}'.format(self.args.dataset, self.args.test_dataset, self.args.dataset_test),
                file=self.pdf, plot='tsne')


           # utils_class.make_plot2(
            #    '', self.args, self.dataset2,
             #   '{}, modelled dataset {} {}'.format(self.args.dataset, self.args.test_dataset, self.args.dataset_test),
             #   file=self.pdf, plot='pca', model=self.model)

            #utils_class.make_plot2(
            #    '', self.args, self.dataset2,
             #   '{}, modelled dataset {} {}'.format(self.args.dataset, self.args.test_dataset, self.args.dataset_test),
             #   file=self.pdf, model=self.model, plot='tsne')

            utils_class.make_plot2(
                '', self.args, self.dataset2,
                '{}, trained on {} pred. using {}'.format(self.args.dataset, self.args.organ, self.args.test_dataset),
                model=self.model,
                file=self.pdf, plot='tsne')

            utils_class.make_plot2(
                '', self.args, self.dataset2,
                '{}, trained on {} pred. using {}'.format(self.args.dataset, self.args.organ, self.args.test_dataset),
                model=self.model,
                file=self.pdf, plot='tsne')

            #utils_class.make_plot2(
             #   '', self.args, self.dataset,
             #   '{}, trained on {} pred. using {} and {}'.
             #       format(self.args.dataset, self.args.organ, self.args.organ, self.args.test_dataset), model=self.model,
              #  file=self.pdf, plot='tsne', dataset2=self.dataset2)

        '''

        utils_class.make_plot(
            '', self.args, self.dataset,
            'UMAP of dataset {} trained on {} pred. using {} and {} modelled by Autoencoder'.format(self.args.dataset,
                                                                                                 self.args.organ,
                                                                                                    self.args.organ,
                                                                                                 self.args.test_dataset),
            '{}'.format(self.args.dataset), model=self.model,
            file=self.pdf, plot='umap', dataset2=self.dataset2, Channel=True)

        utils_class.make_plot(
            '', self.args, self.dataset,
            'UMAP of dataset {} trained on {} pred.using {} and {}  modelled by Autoencoder'.format(self.args.dataset,
                                                                                                 self.args.organ,
                                                                                                    self.args.organ,
                                                                                                 self.args.test_dataset),
            '{}'.format(self.args.dataset), model=self.model,
            file=self.pdf, plot='umap', dataset2=self.dataset2, Label=True)
        '''



        #model.eval()
        #data = torch.randn(1, 3, 24, 24)  # Load your data here, this is just dummy data
        #output = model(data)
        #prediction = torch.argmax(output)



    def Andet(self):
        kwargs = utils_class.define_Kwargs(self.args.norm_type)
        '''
        utils_class.make_plot(
            'lr: {:.4e}, l_orthog: {:.4e}'.format(self.args.lr, self.args.l_orthog), self.args,
            self.dataset,
            'UMAP of model dataset {}'.format(self.args.dataset),
            'figures/UMAP_original optimization_{}.eps'.format(self.args.dataset), model=self.model,
            file=self.pdf, plot='umap')


        try:
            print("Test losses", test_losses)
            print("Train losses", train_losses)
        except Exception as e:
            print(e)
        '''

        # utils_class.make_heatmap(self.args, self.dataset, self.model, file=self.pdf)

        # test_loss= utils_class.check_accuracy(self,self.args, self.model, self.test_loader, loss_func=F.poisson_nll_loss, **kwargs)
        # topk=["TP53", "TNF", "EGFR", "VEGFA", "APOE", "IL6", "TGFBI", "MTHFR"]
        # percent, marker_genes_found=utils_class.find_marker_genes(self.args, self.dataset, self.model, topk)
        # utils_class.make_heatmap(self.args, self.dataset, self.model,file=self.pdf)

        # utils_class.get_gene_set_dict(self.dataset, signatures=self.args.signatures)
        # model_encoder = models.AutoencoderTwoLayers_encoder(self.input_dim, self.args.hidden_size, self.args.more_hidden_size).to(self.args.device)
        # model_encoder = utils_class.get_common_model_part(self.model, model_encoder)

        # utils_class.get_gene_set_scores2(self.args,model_encoder, self.dataset,utils_class.get_gene_set_dict(self.dataset, signatures=self.args.signatures),condition=None)

        # utils_class.score_gene_set(self.dataset, self.input_dim, self.args, self.model, self.args.signatures, condition=None)

    def close_network(self):
        if self.pdf != None:
            try:
                self.pdf.close()
                print("File closed")
            except:
                print("could not close likely already closed pdf")


# {"name": "l1_reg", "type": "range", "bounds": [1e-25, 1e-5], "log_scale": True},
# {"name": "l2_reg", "type": "range", "bounds": [1e-25, 1e-5], "log_scale": True},
#{"name": "l2_reg", "type": "range", "bounds": [1e-25, 1e-5], "log_scale": True},

#

#{"name": "momentum", "type": "range", "bounds": [0.60, 0.98]}
 #           {"name": "l1_reg", "type": "range", "bounds": [1e-25, 1e-5], "log_scale": True},
  #          {"name": "l2_reg", "type": "range", "bounds": [1e-25, 1e-5], "log_scale": True},
#
#

def Call_Hyperparameter_optimization(ANN):
    torch.manual_seed(1)
    # print("Epochs in optimization loop:",self.args.epochs)
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-5, 1e-3], "log_scale": True},
            #{"name": "l_orthog", "type": "range", "bounds": [1e-25, 1e-5], "log_scale": True},
            {"name": "hidden_size", "type": "range", "bounds": [50, 140]},
            {"name": "more_hidden_size", "type": "range", "bounds": [35, 75]},









        ],
        evaluation_function=ANN.Optimize_hyperparameters_using_Ax_platform,
        objective_name='loss',
        minimize=True,
        total_trials=10,
    )

    # print(model)
    print("Best_parameters", best_parameters)
    print("values", values)
    print("experiment", experiment)
    # print("experiment.fetch_data()",experiment.fetch_data())
    data = experiment.fetch_data()
    df = data.df
    print(df)
    best_arm_name = df.arm_name[df['mean'] == df['mean'].min()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]
    best_arm
    print("Best arm", best_arm)
    # utils_class.plot_optimization(model,experiment)
    return best_parameters




ANN = Network()

print(ANN.args)

if ANN.args.test_dataset!=None:
    if ANN.args.path_model!=None:
        print("Loading test dataset")

        ANN.test_dataset()
        ###ANN.load_Heatmap()
        ANN.close_network()

    else:
        print("Missing path to model")
        print("Exiting script")
        quit(3)




elif ANN.args.path_model != None:
    print(ANN.args.path_model)
    print("Loading model")
    ANN.load_model()


    #ANN.load_Heatmap()
    # ANN.Train_Network()
    ANN.close_network()

elif ANN.args.optimize == False:
    print("Traning No Optimization")
    ANN.Define_Network()
    ANN.Train_Network()
    ANN.close_network()

else:
    print("Entering optimization loop")
    best_paramters_optimization_loop = Call_Hyperparameter_optimization(ANN)
    print(best_paramters_optimization_loop)
    ANN.Train_Network(best_hyperparameters=best_paramters_optimization_loop)
    ANN.close_network()

# ANN=Network()
# ANN.load_model()
# ANN.Train_Network()
# ANN.close_network()
#
# ANN.Andet()
# ANN.close_network()












