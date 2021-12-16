from torch.utils.data import Dataset, sampler
import random
import pandas as pd
import numpy.random as random
import numpy as np
import math


#import seaborn as sns
#print("seaborn")

#import matplotlib.pyplot as plt
# #from sklearn.preprocessing import MinMaxScaler
from scipy.io import mmread

path_data="/home/projects/cu_10160/people/s144489/Training/"


def data_normalization(args,data_frame,norm_type):


    if data_frame.isnull().values.any()==True:
        print("Data contains nan, System exit")
        quit(1)

    elif args.dataset=="Merged_10X_FACS":
        print("Data has been normalized using Harmony logCPM")
        return data_frame

    elif args.dataset=="Harmony" :
        print("Data has been normalized using Harmony logCPM")
        return data_frame

    elif args.dataset=="Seurat" :
        print("Data has been normalized using Seurat")
        return data_frame



    elif norm_type=="CPM":
        print("CMP normalization performed")



        data_frame= (data_frame*1/data_frame.sum(axis = 0, skipna = True))*10**6
        print(data_frame.head())

        return data_frame


    elif norm_type=="logCPM":
        print("log(CMP+1) normalization performed")




        data_frame= (data_frame/ data_frame.sum(axis=0)*10**6)


        data_frame=np.log(data_frame+1)

        #data_frame=data_frame.fillna(0)


        return data_frame

    elif norm_type=="logcounts":
        print("log(counts+1) normalization performed")
        data_frame=np.log(data_frame+1)

        return data_frame


    elif norm_type=="Order":
        print("Ordered")


        #print(data_frame.head)
        data_frame=data_frame.rank(method='max', axis=1)
        #scaler = MinMaxScaler()
        #data_frame=scaler.fit(data_frame)

        #print(data_frame.head)
        #data_frame = np.log(data_frame + 1)
        #print(data_frame.head)

        return data_frame

    else:
        print("No normalization performed")
        return data_frame





def Intersecting_datasets(original_dataset, new_dataset,one_cell=False):
    print("Intersecting datasets")
    print("Shape")
    print("Original", original_dataset.shape)
    print("New_dataset", new_dataset.shape)

    if one_cell==False:


        average_zeroes = int(((new_dataset<new_dataset.min().mean()).sum(axis=0)/ len(new_dataset.index) * 100).mean())


        df_new = new_dataset[
            list(set(original_dataset.columns) & set(new_dataset.columns))]

        print("len 1", len(list(set(original_dataset.columns) & set(new_dataset.columns))))
        print("len 2", len(list(set(original_dataset.columns))))
        print("df_new", df_new.shape)
        print("df_new.columns")
        print(df_new.columns[~df_new.columns.isin(list(set(original_dataset.columns) & set(new_dataset.columns)))])


        diff = list(set(original_dataset.columns).difference(set(new_dataset.columns)))
        print("len of diff",len(diff))
        print("len of diff + len1", len(diff)+len(list(set(original_dataset.columns) & set(new_dataset.columns))))



    else:
        new_dataset=new_dataset.to_frame()
        new_dataset=new_dataset.T
        #print(set(new_dataset.columns))




        df_new = new_dataset[
            list(set(original_dataset.columns) & set(new_dataset.columns))]
        #print("equal ", list(set(original_dataset.columns) & set(new_dataset.columns)))

        diff=list(set(original_dataset.columns).difference(set(new_dataset.columns)))






     #print("A.difference(B)",set(self.dataset.expressions.columns).difference(set(self.dataset2.expressions.columns)))



    #print([self.dataset2.expressions.min().mean()]*self.dataset.expressions.shape[0])

    #print("self.dataset2.expressions.shape[0]",self.dataset2.expressions.shape[0])

    #Create list of average percent of zeroes
    if one_cell == False:

        count=0
        if len(diff)>=1:
            for i in diff:
                count=count+1


                x = np.array([new_dataset.min().mean()] * len(df_new.index))

                amount_of_numbers_to_be_replaced = int(len(x) * (100 - average_zeroes) / 100)

                t = np.arange(len(x), dtype=int)
                index = random.choice(t, size=amount_of_numbers_to_be_replaced)

                x[index] = new_dataset.mean().min()




                df_new[i] = x
        print("count",count)
        print("if len(diff)>=1:",df_new.shape)
        print("amount of columns", len(df_new.columns))
        print("amount of columns", len(list(set(df_new.columns))))

        print("Check duplicated colums", df_new.columns[df_new.columns.duplicated()])
        print("Check duplicated colums", df_new.columns[~df_new.columns.duplicated()])
        print("Check duplicated colums", df_new.loc[:,~df_new.columns.duplicated()].shape)

        df_new =df_new.loc[:, ~df_new.columns.duplicated()]

    else:



        print("intersecting one cell")
        #print("diff",diff)
        print("data", new_dataset)

        print("mean", new_dataset.mean(axis=0))
        print("mean 2", new_dataset.mean(axis=1))
        #new_dataset=new_dataset.T


        for i in diff:
            #print(i)
            df_new[i] = new_dataset.min(axis=1)
            #print(df_new[i])

        print("df_new intersected",df_new)


    return df_new, list(df_new.columns)








class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset.
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.num_samples))

    def __len__(self):
        return self.num_samples


class PaulDataset(Dataset):

    def __init__(self,norm_type):
        self.expressions = pd.read_csv(path_data+'data/Paul.csv')
        self.subtypes = self.expressions['cluster']
        self.expressions = self.expressions.drop(columns=['sample', 'cluster'])
        #print(self.expressions.head())

        index = list(self.subtypes.index[self.subtypes == 'CMP CD41']) + \
                list(self.subtypes.index[self.subtypes == 'Cebpe control']) + \
                list(self.subtypes.index[self.subtypes == 'CMP Flt3+ Csf1r+']) + \
                list(self.subtypes.index[self.subtypes == 'Cebpa control']) + \
                list(self.subtypes.index[self.subtypes == 'CMP Irf8-GFP+ MHCII+'])

        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]
        names = self.expressions.columns.str.upper()
        self.expressions = pd.DataFrame(self.expressions.values, columns=names)

        self.subtypes = self.subtypes.iloc[index]

        gene_set_list = []
        with open('signatures/msigdb.v5.2.symbols_mouse.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(self.expressions.columns.isin(gene_set[1:]))


        index = np.unique(np.concatenate(gene_set_list, axis=None))

        self.expressions = self.expressions.iloc[:, index]
        print(self.expressions.info())


        scaler = MinMaxScaler()

        #Creating the shuffled dataframe
        self.expressions = pd.DataFrame(scaler.fit_transform(self.expressions.values), columns=names)


    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        expression = self.expressions.iloc[item, :].values
        return expression, subtype


class VeltenDataset(Dataset):

    def __init__(self,norm_type):
        self.expressions = pd.read_csv(path_data+'data/Velten.csv')
        self.subtypes = self.expressions['cluster']
        self.expressions = self.expressions.drop(columns=['sample', 'cluster'])

        index = list(self.expressions.index)
        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]
        self.subtypes = self.subtypes.iloc[index]
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]
        names = self.expressions.columns.str.upper()

        self.expressions = pd.DataFrame(self.expressions.values, columns=names)

        gene_set_list = []
        with open(path_data+'signatures/msigdb.v6.2.symbols.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(self.expressions.columns.str.upper().isin(gene_set[1:]))
        index = np.unique(np.concatenate(gene_set_list, axis=None))

        self.expressions = self.expressions.iloc[:, index]

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        expression = self.expressions.iloc[item, :].values
        return expression, subtype


class PBMCDataset(Dataset):

    def __init__(self,norm_type):
        self.expressions = pd.read_csv(path_data+'data/PBMC.csv')
        self.subtypes = self.expressions['clusters'].replace({1: "CD4 T cells", 2: "CD14+ Monocytes", 3: "B cells",
                                                              4: "CD8 T cells", 5: "FCGR3A+ Monocytes", 6: "NK cells",
                                                              7: "Dendritic cells", 8: "Megakaryocytes"})
        self.expressions = self.expressions.drop(columns=['clusters'])

        index = list(self.expressions.index)
        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]
        self.subtypes = self.subtypes.iloc[index]
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]
        names = self.expressions.columns.str.upper()

        self.expressions = pd.DataFrame(self.expressions.values, columns=names)

        gene_set_list = []
        with open(path_data+'signatures/msigdb.v6.2.symbols.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(self.expressions.columns.str.upper().isin(gene_set[1:]))
        index = np.unique(np.concatenate(gene_set_list, axis=None))

        self.expressions = self.expressions.iloc[:, index]

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        expression = self.expressions.iloc[item, :].values
        return expression, subtype


class ToyXORDataset(Dataset):
    def __init__(self):

        annotation = pd.read_csv(path_data+'data/revision_xor_annotation.csv')
        expressions = pd.read_csv(path_data+'data/revision_xor_counts.csv')
        self.marker_genes = pd.read_csv(path_data+'data/revision_xor_markergenes.csv')
        self.subtypes = annotation['x']

        colnames = expressions.columns
        expressions = minmax_scale(expressions)
        self.expressions = pd.DataFrame(expressions, columns=colnames)
        # self.expressions = self.expressions.iloc[:, 0:2]
        # sns.scatterplot(x=self.expressions.iloc[:, 0], y=self.expressions.iloc[:, 1], hue=self.subtypes)
        # plt.show()

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        expression = self.expressions.iloc[item, :].values
        return expression, subtype


class Tabula_MurisDataset(Dataset):

    def __init__(self,args,organ=None,original_dataset=None):
        print("Tabula muris ",args)
        try:
            if organ!=None:
                print()
                outfile = open(path_data+"data/FACS_Annotated/" + organ + "_Data_matrix_FACS.csv", 'r')

                self.expressions = pd.read_csv(outfile, index_col=[0])
                outfile.close()
            else:

                outfile = open(path_data+"data/FACS_Annotated/"+args.organ+"_Data_matrix_FACS.csv", 'r')

                self.expressions=pd.read_csv(outfile,index_col=[0])
                outfile.close()
        except:
            print("Could not find file")
            quit()

        #Setting the index to be an array of numbers in stead of the cell ID
        self.expressions=self.expressions.reset_index()

        #print("Before removing column",self.expressions.head())

        #self.expressions.to_csv(r"/home/projects/cu_10160/people/s144489/Training/" + args.organ + "_Data_matrix_FACS_new.csv", header=True)

        self.subtypes = self.expressions["CELL TYPE"]
        self.expressions=self.expressions.drop(columns=["CELL TYPE", "index"])
        #print("After removing column",self.expressions.head())

        #Shuffles all the samples Check
        index = list(self.expressions.index)
        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]
        #print(self.expressions.head(),"shuffled")
        self.subtypes = self.subtypes.iloc[index]



        #if organ==None:
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]
        names = self.expressions.columns.str.upper()

        #print(self.expressions.shape)

        #Creating the shuffled dataframe
        self.expressions = pd.DataFrame(self.expressions.values, columns=names)

        #Finding the HALLMARK genes for Mouse
        gene_set_list = []
        with open(path_data+'signatures/msigdb.v5.2.symbols_mouse.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(self.expressions.columns.isin(gene_set[1:]))
        index = np.unique(np.concatenate(gene_set_list, axis=None))
        self.expressions=self.expressions.iloc[:, index]

        if organ == None:
            count=(self.expressions == 0).astype(int).sum(axis=1)
            count=(count/(len(self.expressions.columns)-1))*100
            print("Average amount of zeroes pr. cell: "+str(round(count.mean(axis = 0),2))+" %")
            print(self.expressions.shape)
            print("Genes "+str(len(self.expressions.columns)-1))
            print("Cells "+str(len(self.expressions.index)))
            print("Amount of Cell types", len(set(self.subtypes)))
            #print("cell types count", self.subtypes)
            #print("cell types count", self.subtypes.value_counts())
            #print("cell types count", self.subtypes.value_counts()/self.subtypes.value_counts().sum())




        if args.one_cell and organ != None :
            random_number=np.random.choice(len(self.subtypes), 1)[0]
            print("random numer" ,random_number)
            self.expressions=self.expressions.iloc[random_number, :]
            #print( self.expressions)
            self.subtypes=self.subtypes.iloc[random_number]
            #print(self.subtypes)


        self.expressions=data_normalization(args,self.expressions,args.norm_type)
        #print("after normalization",self.expressions)


        if organ!=None:
            #print("before intersection",self.expressions)
            print("XX")
            self.expressions,names=Intersecting_datasets(original_dataset,self.expressions,one_cell=args.one_cell)

            self.expressions=pd.DataFrame(self.expressions, columns=names)
            self.expressions=self.expressions.astype(np.float32)
            #dtype=np.float32



        if organ!=None:
            print("XX")
            #print("coluns original",original_dataset.columns)
            #print("coluns new", self.expressions.columns)
            #print("after intersection", self.expressions)
            self.expressions=self.expressions[original_dataset.columns]



    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        #print(item)
        expression = self.expressions.iloc[item, :].values
        return expression, subtype




class Simulation_data(Dataset):

    def __init__(self,filename):

        print(filename)

        self.expressions=pd.read_csv(filename).T

        #print(self.expressions.head())


        #Setting the index to be an array of numbers in stead of the cell ID
        self.expressions=self.expressions.reset_index()
        print("Initial",self.expressions.head())


        self.subtypes = self.expressions["Cell type"]
        self.expressions=self.expressions.drop(columns=["Cell type ", "index"])


        #Shuffles all the samples Check
        index = list(self.expressions.index)
        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]
        #print(self.expressions.head(),"shuffled")
        self.subtypes = self.subtypes.iloc[index]
        #print(self.subtypes.head(),"shuffled")



        #Removing samples with values
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]
        names = self.expressions.columns.str.upper()

        #Creating the shuffled dataframe
        self.expressions = pd.DataFrame(scaler.fit_transform(self.expressions.values), columns=names)

        #print(self.expressions.head())


        #print(self.expressions.head())
        count=(self.expressions == 0).astype(int).sum(axis=1)
        print("Shape", self.expressions.shape)
        count=(count/(len(self.expressions.columns)))*100
        #print(count)
        #print(count.mean(axis = 0))
        print("Average amount of zeroes pr. cell: "+str(round(count.mean(axis = 0),2))+" %")
        print("Genes "+str(len(self.expressions.columns)))
        print("Cells "+str(len(self.expressions.index)))




    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        expression = self.expressions.iloc[item, :].values
        return expression, subtype



class X_10(Dataset):

    def __init__(self,args,organ=None,original_dataset=None):



        try:
            if organ!=None:

                outfile = open(path_data+"data/10_X_Merged_raw/" + organ + "_Data_matrix.csv", 'r')

                print("organ", organ)
                expressions = pd.read_csv(outfile)

                outfile.close()
            else:
                outfile = open(path_data+"data/10_X_Merged_raw/" + args.organ + "_Data_matrix.csv", 'r')

                print("organ", args.organ)
                expressions = pd.read_csv(outfile)

                outfile.close()
        except:
            print("Could not find file")
            quit()


        expressions = expressions.reset_index()
        self.subtypes = expressions["cell types"]
        # print(self.subtypes)
        # print("before dropping columns", expressions.head())
        expressions = expressions.drop(columns=["cell types", "cell", "channel"])
        # print("after dropping columns",expressions.head())

        # Shuffles all the samples Check
        index = list(expressions.index)
        random.shuffle(index)
        expressions = expressions.iloc[index, :]
        self.subtypes = self.subtypes.iloc[index]

        expressions = expressions.loc[:, (expressions > 2).any(axis=0)]
        names = expressions.columns.str.upper()

        # Creating the shuffled dataframe
        expressions = pd.DataFrame(expressions.values, columns=names)

        # Finding the HALLMARK genes for Mouse
        gene_set_list = []
        with open(path_data+'signatures/msigdb.v5.2.symbols_mouse.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(expressions.columns.isin(gene_set[1:]))
        index = np.unique(np.concatenate(gene_set_list, axis=None))

        self.expressions = expressions.iloc[:, index]

        count = (expressions == 0).astype(int).sum(axis=1)
        print("Shape", expressions.shape)
        count = (count / (len(expressions.columns) - 1)) * 100
        print("Average amount of zeroes pr. cell: " + str(round(count.mean(axis=0), 2)) + " %")
        print("Genes " + str(len(expressions.columns)))
        print("Cells " + str(len(expressions.index)))
        print("Amount of Cell types", len(set(self.subtypes)))


        self.expressions=data_normalization(args,self.expressions,args.norm_type)
        #print("after normalization",self.expressions)
        print(set(self.subtypes))



        if organ!=None:

            self.expressions,names=Intersecting_datasets(original_dataset,self.expressions)


            self.expressions=pd.DataFrame(self.expressions, columns=names) #dtype=np.float32

            self.expressions=self.expressions[original_dataset.columns]





    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        #print(item)
        expression = self.expressions.iloc[item, :].values
        return expression, subtype
        



class Harmony(Dataset):

    def __init__(self,args,organ=None,original_dataset=None):
        #loading data


        try:
            if organ!=None:
                print("organ", organ)
                print("data/Merge_10X_FACS_Batch_LogCPM_Harmony/")
                outfile = open(path_data+ "data/Merge_10X_FACS_Batch_LogCPM_Harmony/" + organ + "_data_matrix.csv", 'r')

                self.expressions = pd.read_csv(outfile, index_col=[0])

                outfile.close()



            else:
                print("organ", args.organ)
                print(path_data+ "data/Merge_10X_FACS_Batch_LogCPM_Harmony/")
                outfile = open(path_data+"data/Merge_10X_FACS_Batch_LogCPM_Harmony/" + args.organ + "_data_matrix.csv", 'r')

                self.expressions = pd.read_csv(outfile, index_col=[0])

                outfile.close()
        except:
            print("Could not find file")
            quit()


        try:
            if organ != None:
                print("organ", organ)
                self.expressions = pd.read_csv(path_data+
                    "data/Merge_10X_FACS_Batch_LogCPM_Harmony/" + organ + "_data_matrix" + ".csv",
                    index_col=[0])
            else:
                print("organ", args.organ)
                self.expressions = pd.read_csv(
                    "data/Merge_10X_FACS_Batch_LogCPM_Harmony/" + args.organ + "_data_matrix" + ".csv",
                    index_col=[0])


        except:
            print("Could not load data file")
            quit()

        #Setting the index to be an array of numbers in stead of the cell ID
        #print(self.expressions)
        self.expressions=self.expressions.reset_index()

        #print(self.expressions.head)

        #print("Before removing column",self.expressions.head())
        #print(self.expressions["CELL.TYPES"])
        self.subtypes = self.expressions["CELL.TYPES"]
        print(set(self.subtypes))
        #self.subtypes = self.expressions["CHANNEL"]
        #self.expressions=self.expressions.drop(columns=["index","CELL.TYPES", "CHANNEL", "CELL"])
        self.expressions = self.expressions.drop(columns=["index", "CELL.TYPES", "CHANNEL"])
        #print(self.expressions.head)

        #Shuffles all the samples Check
        index = list(self.expressions.index)
        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]
        #print(self.expressions.head(),"shuffled")
        self.subtypes = self.subtypes.iloc[index]
        #print(self.subtypes.head(),"shuffled")


        #Removing samples with values
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]
        names = self.expressions.columns.str.upper()

        #print(self.expressions.shape)

        #Creating the shuffled dataframe
        self.expressions = pd.DataFrame(self.expressions.values, columns=names)

        #Finding the HALLMARK genes for Mouse
        gene_set_list = []
        with open('signatures/msigdb.v5.2.symbols_mouse.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(self.expressions.columns.isin(gene_set[1:]))
        index = np.unique(np.concatenate(gene_set_list, axis=None))
        self.expressions=self.expressions.iloc[:, index]

        #print(self.expressions.head)


        #Printing the amount of genes,cells and zeroes.
        count=(self.expressions == 0).astype(int).sum(axis=1)
        count=(count/(len(self.expressions.columns)-1))*100
        print("Average amount of zeroes pr. cell: "+str(round(count.mean(axis = 0),2))+" %")
        print(self.expressions.shape)
        print("Genes "+str(len(self.expressions.columns)-1))
        print("Cells "+str(len(self.expressions.index)))
        print("Amount of Cell types", len(set(self.subtypes)))


        self.expressions=data_normalization(args,self.expressions,args.norm_type)
        #print("after normalization",self.expressions)



        if organ!=None:

            self.expressions,names=Intersecting_datasets(original_dataset,self.expressions)

            self.expressions=pd.DataFrame(self.expressions, columns=names) #dtype=np.float32




        if organ!=None:
            self.expressions=self.expressions[original_dataset.columns]


    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        #print(item)
        expression = self.expressions.iloc[item, :].values
        return expression, subtype


class Seurat(Dataset):

    def __init__(self,args,organ=None,original_dataset=None):
        #loading data

        try:
            if organ!=None:


                print("organ", organ)
                print(path_data)
                print(path_data+"data/Merge_10X_FACS_Batch_Seurat")

                outfile = open(path_data+ "data/Merge_10X_FACS_Batch_Seurat/" + organ + "_data_matrix" + ".csv", 'r')

                self.expressions = pd.read_csv(outfile,index_col=[0])

                outfile.close()
            else:
                print("organ", args.organ)
                print(path_data+ "data/Merge_10X_FACS_Batch_Seurat")
                outfile = open(path_data+"data/Merge_10X_FACS_Batch_Seurat/" + args.organ + "_data_matrix" + ".csv", 'r')

                self.expressions = pd.read_csv(outfile, index_col=[0])

                outfile.close()
        except:
            print("Could not find file")
            quit()

        #Setting the index to be an array of numbers in stead of the cell ID
        #print(self.expressions)
        self.expressions=self.expressions.reset_index()



        #print("Before removing column",self.expressions.head())
        #print(self.expressions["CELL.TYPES"])
        self.subtypes = self.expressions["CELL.TYPES"]
        #self.subtypes = self.expressions["CHANNEL"]
        #self.expressions=self.expressions.drop(columns=["index","CELL.TYPES", "CHANNEL", "CELL"])
        self.expressions = self.expressions.drop(columns=["index", "CELL.TYPES", "CHANNEL","CELL"])

        #Shuffles all the samples Check
        index = list(self.expressions.index)
        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]
        #print(self.expressions.head(),"shuffled")
        self.subtypes = self.subtypes.iloc[index]
        #print(self.subtypes.head(),"shuffled")


        #Removing samples with values
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]
        names = self.expressions.columns.str.upper()

        #print(self.expressions.shape)

        #Creating the shuffled dataframe
        self.expressions = pd.DataFrame(self.expressions.values, columns=names)

        #Finding the HALLMARK genes for Mouse
        gene_set_list = []
        with open(path_data+'signatures/msigdb.v5.2.symbols_mouse.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(self.expressions.columns.isin(gene_set[1:]))
        index = np.unique(np.concatenate(gene_set_list, axis=None))
        self.expressions=self.expressions.iloc[:, index]

        #print(self.expressions.head)


        #Printing the amount of genes,cells and zeroes.
        count=(self.expressions == 0).astype(int).sum(axis=1)
        count=(count/(len(self.expressions.columns)-1))*100
        print("Average amount of zeroes pr. cell: "+str(round(count.mean(axis = 0),2))+" %")
        print(self.expressions.shape)
        print("Genes "+str(len(self.expressions.columns)-1))
        print("Cells "+str(len(self.expressions.index)))
        print("Amount of Cell types", len(set(self.subtypes)))
        #print("cell types count", self.subtypes.value_counts())
        #print("cell types count", self.subtypes.value_counts() / self.subtypes.value_counts().sum())



        self.expressions=data_normalization(args,self.expressions,args.norm_type)
        #print("after normalization",self.expressions)

        print(set(self.subtypes))

        if organ!=None:

            self.expressions,names=Intersecting_datasets(original_dataset,self.expressions)

            self.expressions=pd.DataFrame(self.expressions, columns=names) #dtype=np.float32


        if organ!=None:
            self.expressions=self.expressions[original_dataset.columns]



    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        #print(item)
        expression = self.expressions.iloc[item, :].values
        return expression, subtype

class Human(Dataset):

    def __init__(self,args,organ=None,original_dataset=None):
        print("Human ",args)
        try:
            if organ!=None:
                print(organ)
                #outfile = open(path_data+"data/Human/" + organ + "_raw_counts.csv", 'r')
                outfile = open(path_data + "data/Human/" + organ + "_raw_counts_converted.csv", 'r')


                self.expressions = pd.read_csv(outfile, index_col=[0])
                outfile.close()
            else:
                #outfile = open(path_data + "data/Human/Marrow_raw_counts.csv", 'r')
                outfile = open(path_data+"data/Human/Marrow_raw_counts_converted.csv", 'r')

                self.expressions=pd.read_csv(outfile,index_col=[0])
                outfile.close()
        except:
            print("Could not find file")
            quit()

        #Setting the index to be an array of numbers in stead of the cell ID
        self.expressions=self.expressions.reset_index()


        #print("Before removing column",self.expressions.head())



        self.subtypes = self.expressions["CELL"]
        print(self.expressions["index"])
        self.expressions = self.expressions.drop(columns=["CELL", "index"])
        print("After removing column",self.expressions.head())
        print("After removing column", self.subtypes.head())
        #Shuffles all the samples Check
        index = list(self.expressions.index)
        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]
        #print(self.expressions.head(),"shuffled")
        self.subtypes = self.subtypes.iloc[index]



        #if organ==None:
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]
        names = self.expressions.columns.str.upper()

        #print(self.expressions.shape)

        #Creating the shuffled dataframe
        self.expressions = pd.DataFrame(self.expressions.values, columns=names)

        #Finding the HALLMARK genes for Mouse
        gene_set_list = []
        with open(path_data+'signatures/msigdb.v5.2.symbols_mouse.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(self.expressions.columns.isin(gene_set[1:]))
        index = np.unique(np.concatenate(gene_set_list, axis=None))
        self.expressions=self.expressions.iloc[:, index]
        print("After only taking Hallmark pathways")
        print(self.expressions.shape)

        if organ == None:
            count=(self.expressions == 0).astype(int).sum(axis=1)
            count=(count/(len(self.expressions.columns)-1))*100
            print("Average amount of zeroes pr. cell: "+str(round(count.mean(axis = 0),2))+" %")
            print(self.expressions.shape)
            print("Genes "+str(len(self.expressions.columns)-1))
            print("Cells "+str(len(self.expressions.index)))
            print("Amount of Cell types", len(set(self.subtypes)))
            #print("cell types count", self.subtypes)
            #print("cell types count", self.subtypes.value_counts())
            #print("cell types count", self.subtypes.value_counts()/self.subtypes.value_counts().sum())




        if args.one_cell and organ != None :
            random_number=np.random.choice(len(self.subtypes), 1)[0]
            print("random numer" ,random_number)
            self.expressions=self.expressions.iloc[random_number, :]
            #print( self.expressions)
            self.subtypes=self.subtypes.iloc[random_number]
            #print(self.subtypes)


        self.expressions=data_normalization(args,self.expressions,args.norm_type)
        print("after normalization",self.expressions.shape)


        if organ!=None:
            #print("before intersection",self.expressions)
            print("XX")
            self.expressions,names=Intersecting_datasets(original_dataset,self.expressions,one_cell=args.one_cell)
            self.expressions=pd.DataFrame(self.expressions, columns=names)
            self.expressions=self.expressions.astype(np.float32)
            print("After intersection", self.expressions.shape)
            print(len(self.subtypes.index))
            print(self.expressions.head())



        if organ!=None:
            print("XX")
            #print("coluns original",original_dataset.columns)
            #print("coluns new", self.expressions.columns)
            #print("after intersection", self.expressions)
            self.expressions=self.expressions[original_dataset.columns]



    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        #print(item)
        expression = self.expressions.iloc[item, :].values
        print("Creating data")
        print(expression.shape)
        print(len(subtype))
        return expression, subtype




'''
class Merged_10X_FACS(Dataset):

    def __init__(self,args):
        #loading data
        try:
        #k=pd.read_csv("data/Merged_10X_FACS/Bladder-counts.csv",nrows=1000)  #index_col = [0] to avoid flipping index
            #self.expressions=pd.read_csv("data/Merged_10X_FACS_raw/data_10X_FACS_"+args.organ+".csv",index_col=[0])
            self.expressions = pd.read_csv("data/Merge_10X_FACS_Batch_LogCPM_Harmony/"+args.organ+"_data_matrix" + ".csv",
                                       index_col=[0])
        except:
            print("Could not load data file")
            quit()

        #Setting the index to be an array of numbers in stead of the cell ID
        #print(self.expressions)
        self.expressions=self.expressions.reset_index()

        #print(self.expressions.head)

        #print("Before removing column",self.expressions.head())
        #print(self.expressions["CELL.TYPES"])
        self.subtypes = self.expressions["CELL.TYPES"]
        #self.subtypes = self.expressions["CHANNEL"]
        #self.expressions=self.expressions.drop(columns=["index","CELL.TYPES", "CHANNEL", "CELL"])
        self.expressions = self.expressions.drop(columns=["index", "CELL.TYPES", "CHANNEL"])
        #print(self.expressions.head)

        #Shuffles all the samples Check
        index = list(self.expressions.index)
        random.shuffle(index)
        self.expressions = self.expressions.iloc[index, :]
        #print(self.expressions.head(),"shuffled")
        self.subtypes = self.subtypes.iloc[index]
        #print(self.subtypes.head(),"shuffled")


        #Removing samples with values
        self.expressions = self.expressions.loc[:, (self.expressions > 2).any(axis=0)]
        names = self.expressions.columns.str.upper()

        #print(self.expressions.shape)

        #Creating the shuffled dataframe
        self.expressions = pd.DataFrame(self.expressions.values, columns=names)

        #Finding the HALLMARK genes for Mouse
        gene_set_list = []
        with open('signatures/msigdb.v5.2.symbols_mouse.gmt.txt') as go_file:
            for line in go_file.readlines():
                if line.startswith('HALLMARK_'):
                    gene_set = line.strip().split('\t')
                    gene_set_list += np.where(self.expressions.columns.isin(gene_set[1:]))
        index = np.unique(np.concatenate(gene_set_list, axis=None))
        self.expressions=self.expressions.iloc[:, index]

        #print(self.expressions.head)


        #Printing the amount of genes,cells and zeroes.
        count=(self.expressions == 0).astype(int).sum(axis=1)
        count=(count/(len(self.expressions.columns)-1))*100
        print("Average amount of zeroes pr. cell: "+str(round(count.mean(axis = 0),2))+" %")
        print(self.expressions.shape)
        print("Genes "+str(len(self.expressions.columns)-1))
        print("Cells "+str(len(self.expressions.index)))
        print("Amount of Cell types", len(set(self.subtypes)))


        #print(self.expressions.info())

        self.expressions=data_normalization(args,self.expressions,args.norm_type)
        #print(self.expressions.columns)

        #print("nancheck",self.expressions.isnull().any())

        #for col in list(self.expressions.loc[:,"AAAS"]):
        #    print(col)



    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, item):
        subtype = self.subtypes.iloc[item]
        #print(item)
        expression = self.expressions.iloc[item, :].values
        return expression, subtype
'''