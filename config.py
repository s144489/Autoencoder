import torch
import munch #DICTIONARY LIKE STRUCTURE FOR SAVING THE VARIABLES
import numpy as np

print("config")

#print(argument.one_cell)

def config(argument):
    args = args = munch.Munch()
    args['batch_size'] = 1
    args['test_batch_size'] = 1 #Change loss function if changed.
    args['epochs'] = argument.epochs
    args['plot'] = argument.plot
    args['one_cell'] = argument.one_cell
    print("one_cell config",argument.one_cell)
    args['optimize'] = argument.optimize
    args['path_model']=argument.path_model
    args['test_dataset'] = argument.test_dataset
    args['dataset_test'] = argument.dataset_test
    args['l1_reg'] = 0
    args['l2_reg'] = 0    #weight decay
    args['hidden_size'] = 128
    args['more_hidden_size'] = 64
    args['train_percent'] = 0.95
    args['momentum'] = 0.9
    args['seed'] = 1
    args['log_interval'] = 500
    args['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args['dtype'] =torch.float32
    return args



def pbmc_config(argument):
    args = config(argument)
    args['lr'] = 1e-3
    args['l_orthog'] = 5e-2
    args['dataset'] = 'PBMC'
    args['signatures'] = 'signatures/DMAP_signatures_PBMC.txt'
    args['path'] = argument.path
    args['organ'] = argument.organ
    args['norm_type'] = argument.norm_type
    return args


def paul_config(argument):
    args = config(argument)
    args['lr'] = 1e-3
    args['l_orthog'] = 5e-2
    args['dataset'] = 'Paul'
    args['signatures'] = 'signatures/msigdb.v5.2.symbols_mouse.gmt.txt'
    args['path'] = argument.path
    args['organ'] = argument.organ
    args['norm_type'] = argument.norm_type
    return args


def velten_config(path,norm_type):
    args = config(argument)
    args['lr'] = 2e-11
    args['l_orthog'] = 3  #part of the geularization.
    args['dataset'] = 'Velten'
    args['signatures'] = 'signatures/DMAP_signatures_Velten.txt'
    args['path'] = argument.path
    args['organ'] = argument.organ
    args['norm_type'] = argument.norm_type
    return args


def Tabula_Muris_config(argument):
    args = config(argument)
    args['lr'] =1e-5
    args['l_orthog'] = 0
    args['dataset'] = 'Smart-seq2'
    args['signatures'] = 'signatures/msigdb.v5.2.symbols_mouse.gmt.txt'
    args['path'] = argument.path
    args['organ'] = argument.organ
    args['norm_type'] = argument.norm_type
    return args

def Harmony_config(argument):
    args = config(argument)
    args['lr'] =1e-5
    args['l_orthog'] = 1e-20 #[0, 0.0005, 0.005, 0.05, 0.5, 5]# 5e-2
    args['dataset'] = 'Harmony'
    args['signatures'] = 'signatures/msigdb.v5.2.symbols_mouse.gmt.txt'
    args['path'] = argument.path
    args['organ'] = argument.organ
    args['norm_type'] = argument.norm_type
    return args


def Seurat_config(argument):
    args = config(argument)
    args['lr'] =1e-5
    args['l_orthog'] = 1e-20 #[0, 0.0005, 0.005, 0.05, 0.5, 5]# 5e-2
    args['dataset'] = 'Seurat'
    args['signatures'] = 'signatures/msigdb.v5.2.symbols_mouse.gmt.txt'
    args['path'] = argument.path
    args['organ'] = argument.organ
    args['norm_type'] = argument.norm_type
    return args


def Human_config(argument):
    args = config(argument)
    args['lr'] =1e-5
    args['l_orthog'] = 1e-20 #[0, 0.0005, 0.005, 0.05, 0.5, 5]# 5e-2
    args['dataset'] = 'Seurat'
    args['signatures'] = 'signatures/msigdb.v5.2.symbols_mouse.gmt.txt'
    args['path'] = argument.path
    args['organ'] = argument.organ
    args['norm_type'] = argument.norm_type
    return args



def Simulation_config(argument):
    args = config(argument)
    args['dataset'] = 'Simulation'
    args['l_orthog'] = 0
    args['lr'] = 0.001
    args['path'] = argument.path
    args['norm_type'] = argument.norm_type
    args['signatures'] = 'signatures/msigdb.v5.2.symbols_mouse.gmt.txt'
    return args


def X_10_config(argument):
    args = config(argument)
    args['dataset'] = 'Drop-seq'
    args['lr'] = 2e-11
    args['l_orthog'] = 0
    args['path'] = argument.path
    args['norm_type']=argument.norm_type
    args['organ'] = argument.organ
    args['signatures'] = 'signatures/msigdb.v5.2.symbols_mouse.gmt.txt'
    return args


def toy_MLP_XOR_config():
    args = config()
    args['epochs'] = 1000
    args['train_percent'] = 0.8
    args['hidden_size'] = 200
    args['more_hidden_size'] = 4
    args['lr'] = 1e-3
    args['l2_reg'] = 0
    args['l1_reg'] = 0
    args['l_orthog'] = 0
    args['dataset'] = 'Toy_MLP'
    args['signatures'] = ''
    args['batch_size'] = 1
    args['test_batch_size'] = 500
    args['dropout'] = 0.2
    return args
