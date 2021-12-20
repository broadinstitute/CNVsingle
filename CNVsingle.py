#@author igleshch
import scanpy as sc
import numpy as np
import anndata2ri
import pandas as pd
import seaborn as sns
import re

import scipy
import scipy.stats as stats

from scipy.sparse import dok_matrix #sparse matrix with dictionary keys to update

#TODO:fix
from scipy.sparse import csr_matrix

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
import matplotlib.pyplot as plt


class CNVsingle:
    """CLASS info
      FUNCTIONS:
          public:
            add_sample() - add new samples from individual to self.sample_list by making a new TumorSample instance
         private:
           _load_sample_ccf() - guess input file type for ccf data if not manually provided
      PROPERTIES:
          regular variables:
             self.
              sample_list
          @property methods: temporarily_removed - all mutations that are graylisted but not blacklisted in this sample
    """

    def __init__(self, het_pileup,adata=None, adata_hd5=None, gene_file="CNVsingle/refdata-cellranger-hg19-3.0.0/genes/genes.gtf",
            sample_id=None,neutral_ploidy=2,ambient_cont_level=0.01):

        self.sample_name = sample_id
        if not adata: adata = sc.read_h5ad(adata_hd5)
        self.adata=adata
        self.het_pileup =het_pileup

        # define
        self.cleaned_cn_by_cluster = {}

        self.clusters=sorted(list(set(adata.obs["clusters"])))

        # Dictionary for easy lookup later.
        self.low_coverage_mutations = {}
        self.ambient_cont_level = ambient_cont_level
        self.het_df=None
        self.parsed_mpileup_dict=self.parse_pileup()
        self.cell_barcode_list= self._extract_used_barcodes()
        self._make_sparse_read_cnt_mxs()

        self.gene_df=self._get_gene_list(gene_file)


        #self._preprocess_cn_data()

    @classmethod
    def merge_pileups():
        pass


    def _poiss_cn_data(self, cluster="6", normal_cluster="0"):
        genes=self.gene_df
        # 1 remove poorly covered cells
        data_filtered = self.adata[np.array(self.adata.X.sum(axis=1)).flatten() > 2000, :]
        # # of cells per gene
        count_nonzero = self.adata.X.astype(bool).sum(axis=0)
        good_cov = np.array(count_nonzero > 50).squeeze()

        data_filtered.var['Chrom'] = list(genes.Chrom)
        data_filtered.var['Position'] = list(genes.Position)
        data_filtered.var['Type'] = list(genes.Type)

        # remove genes with less than 50 cells registered
        data_filtered = data_filtered[:, good_cov]

        # remove genes in non autosomes+X+Y
        data_filtered = data_filtered[:, data_filtered.var.Chrom.isin(list(map(str, range(1, 23))) + ["X", "Y"])]

        # removing RP, MRP, MALAT etc
        # removing HLA ; protein_coding only
        remove = data_filtered.var_names.str.startswith('ZNF') + data_filtered.var_names.str.startswith(
            'HLA') + data_filtered.var_names.str.startswith('MALAT1') + data_filtered.var_names.str.startswith(
            'RP') + data_filtered.var_names.str.contains('^HB[^(P)]') + data_filtered.var_names.str.startswith(
            'MRPL') + data_filtered.var_names.str.startswith('MRPS')

        keep = np.invert(remove)
        data_filtered = data_filtered[:, keep]
        data_filtered = data_filtered[:, data_filtered.var.Type == "protein_coding"]

        #remove highly variable genes
        sc.pp.highly_variable_genes(data_filtered, flavor='seurat_v3', n_top_genes=100)

        data_filtered = data_filtered[:, ~data_filtered.var.highly_variable]

        # 1 normalize coverage to 10k per cell

        #sc.pp.normalize_total(data_filtered, target_sum=10000)



        #test poisson on genes
        bool_poiss_genes_normal=[]
        #cluster normal
        for gene in range(data_filtered.shape[1]):
             data = np.ravel(data_filtered.X[data_filtered.obs.clusters==str(normal_cluster),gene].todense())
             if np.mean(data)<0.1 or np.mean(data)>5:
                 bool_poiss_genes_normal.append(False)
                 continue
             x_plot=np.arange(0,15)
             delta_fit=scipy.special.logsumexp(np.log(np.histogram(data,bins=x_plot, density=True)[0]+1e-20)+np.log(stats.poisson.pmf(x_plot,np.mean(data))[0:-1]))
             if delta_fit<-0.3 :
                 bool_poiss_genes_normal.append(False)
                 #print (data_filtered.var.index[gene],data_filtered.var.iloc[gene].Chrom, delta_fit)
             else:
                 bool_poiss_genes_normal.append(True)

        bool_poiss_genes_cluster=[]
        print ("done")
        #cluster of interest
        for gene in range(data_filtered.shape[1]):
             data = np.ravel(data_filtered.X[data_filtered.obs.clusters==str(cluster),gene].todense())
             if np.mean(data)<0.1 or np.mean(data)>5: #unstable expressed genes
                 bool_poiss_genes_cluster.append(False)
                 continue
             x_plot=np.arange(0,15)
             delta_fit=scipy.special.logsumexp(np.log(np.histogram(data,bins=x_plot, density=True)[0]+1e-20)+np.log(stats.poisson.pmf(x_plot,np.mean(data))[0:-1]))
             if delta_fit<-0.3 or np.mean(data)<0.15:
                 bool_poiss_genes_cluster.append(False)
                 #print (data_filtered.var.index[gene],data_filtered.var.iloc[gene].Chrom, delta_fit)
             else:
                 bool_poiss_genes_cluster.append(True)

        #merge good genes
        bool_poiss_genes=np.array(bool_poiss_genes_normal)+np.array(bool_poiss_genes_cluster)

        #remove bad ones
        data_filtered=data_filtered[:,bool_poiss_genes]
        #normalize to poisson lamda of mean gene for both
        exp_matrix = np.array(data_filtered.X.todense())

        norm_exp_matrix=exp_matrix[data_filtered.obs.clusters==str(normal_cluster),:]
        norm_exp_matrix=np.mean(norm_exp_matrix, axis=0) #mean lambda across cells for each gene
        mean_gene=np.mean(norm_exp_matrix)
        norm_exp_matrix=norm_exp_matrix/mean_gene


        cl_exp_matrix = exp_matrix[data_filtered.obs.clusters == str(cluster), :]
        cl_exp_matrix = np.mean(cl_exp_matrix, axis=0)  # mean lambda across cells for each gene
        mean_gene=np.mean(norm_exp_matrix)
        cl_exp_matrix=cl_exp_matrix/mean_gene

        to_plot=cl_exp_matrix/norm_exp_matrix

        return data_filtered,to_plot


    def _preprocess_cn_data_cl(self,normal_cluster="0", case_cluster="6",min_val_cut=0.05,max_val_cut=5):
        from scipy.signal import savgol_filter
        genes = self.gene_df

        # 1 remove poorly covered cells
        data_filtered = self.adata[np.array(self.adata.X.sum(axis=1)).flatten() > 200, :]

        #data_filtered.var['Chrom'] = list(genes.Chrom)
        #data_filtered.var['Position'] = list(genes.Position)
        #data_filtered.var['Type'] = list(genes.Type)
        # # of cells per gene
        # count_nonzero = self.adata.X.astype(bool).sum(axis=0)
        # good_cov = np.array(count_nonzero > 50).squeeze()
        # remove poorly covered genes and non-poisson-like

  

        # remove genes with less than 50 cells registered
        # data_filtered = data_filtered[:, good_cov]

        # remove genes in non autosomes+X+Y
        data_filtered = data_filtered[:, data_filtered.var.Chrom.isin(list(map(str, range(1, 23))) )] #+ ["X", "Y"]

        # removing RP, MRP, MALAT etc
        # removing HLA ; protein_coding only
        remove = data_filtered.var_names.str.startswith('ZNF') + data_filtered.var_names.str.startswith(
            'HLA') + data_filtered.var_names.str.startswith('MALAT1') + data_filtered.var_names.str.startswith(
            'RP') + data_filtered.var_names.str.contains('^HB[^(P)]') + data_filtered.var_names.str.startswith(
            'MRPL') + data_filtered.var_names.str.startswith('MRPS')

        keep = np.invert(remove)
        data_filtered = data_filtered[:, keep]
        data_filtered = data_filtered[:, data_filtered.var.Type == "protein_coding"]

        # remove highly variable genes
        #sc.pp.highly_variable_genes(data_filtered, flavor='seurat_v3', n_top_genes=50)

        #data_filtered = data_filtered[:, ~data_filtered.var.highly_variable]

        # 1 normalize coverage to 7k per cell

        #sc.pp.normalize_total(data_filtered, target_sum=7000)

        #exp_matrix = data_filtered.X
        # geometric mean normalized
        #exp_matrix = exp_matrix / np.mean(exp_matrix, axis=0)

        #normalization will be against the means of the cluster - works well for poisson genes



        for cl in self.clusters:
            means_0 = np.array(np.mean(data_filtered[data_filtered.obs.clusters == normal_cluster].X, axis=0)).ravel()
            means_cl = np.array(np.mean(data_filtered[data_filtered.obs.clusters == cl].X, axis=0)).ravel()

            data_filtered_cl = data_filtered[:,
                        np.where((means_0 > min_val_cut) & (means_0 <= max_val_cut) & (means_cl > min_val_cut) & (means_cl <= max_val_cut))[0]]
            
            sc.pp.normalize_total(data_filtered_cl, target_sum=7000)
            
            #to_plot=np.log2(np.sum(smooth_data_np_hanning_2d(np.array(exp_matrix[data_filtered.obs.clusters==cl,:])), axis=0).squeeze())
            #to_plot = to_plot - scipy.special.logsumexp(to_plot)
            #to_plot=to_plot-np.mean(to_plot)
            #to_plot=to_plot-scipy.special.logsumexp(to_plot) #normalize to total 1
            #to_norm=np.mean(np.log2(smooth_data_np_hanning_2d(np.array(exp_matrix[data_filtered.obs.clusters==normal_cluster,:]))),axis=0).squeeze()
            #to_norm=to_norm-np.mean(np.array(to_norm))
            #to_norm=to_norm-scipy.special.logsumexp(to_norm) #normalize to total 1
            #to_plot=to_plot-to_norm
            #smooth by chromosome for tumor 
            merge_list=[]
            for chrom in range(1,23): 
                sub_cl=np.ravel(np.sum(data_filtered_cl[data_filtered_cl.obs.clusters == cl, data_filtered_cl.var.Chrom==str(chrom)].X, axis=0) )
                sub_cl=list(savgol_filter(sub_cl,41,1,mode="mirror")) 
                merge_list+=sub_cl
                
            data_filtered_cl.var["smooth_cn"]=merge_list
            
            cl_in = data_filtered_cl.var["smooth_cn"]#np.ravel(np.sum(data_filtered_cl.X[data_filtered_cl.obs.clusters == cl, :], axis=0))

            #cl_in=savgol_filter(cl_in,41,1)
            cl_in = cl_in / np.median(cl_in)

            cl_norm = np.ravel(np.sum(data_filtered_cl.X[data_filtered_cl.obs.clusters == normal_cluster, :], axis=0))
            cl_norm=savgol_filter(cl_norm,41,1)

            cl_norm = cl_norm / np.median(cl_norm)

            ratio = cl_in / cl_norm
            df= data_filtered_cl.var.copy()
            df["cn_ratio"]=ratio #smooth?
            self.cleaned_cn_by_cluster[cl]=df


    def _preprocess_cn_data(self,normal_cluster="0", case_cluster="6"):
        genes = self.gene_df

        # 1 remove poorly covered cells
        data_filtered = self.adata[np.array(self.adata.X.sum(axis=1)).flatten() > 200, :]

        #data_filtered.var['Chrom'] = list(genes.Chrom)
        #data_filtered.var['Position'] = list(genes.Position)
        #data_filtered.var['Type'] = list(genes.Type)
        # # of cells per gene
        # count_nonzero = self.adata.X.astype(bool).sum(axis=0)
        # good_cov = np.array(count_nonzero > 50).squeeze()
        # remove poorly covered genes and non-poisson-like

        means_0 = np.array(np.mean(data_filtered[data_filtered.obs.clusters == normal_cluster].X, axis=0)).ravel()
        means_6 = np.array(np.mean(data_filtered[data_filtered.obs.clusters == case_cluster].X, axis=0)).ravel()

        data_filtered = data_filtered[:,
                        np.where((means_0 > 0.1) & (means_0 <= 5) & (means_6 > 0.1) & (means_6 <= 5))[0]]

        # remove genes with less than 50 cells registered
        # data_filtered = data_filtered[:, good_cov]

        # remove genes in non autosomes+X+Y
        data_filtered = data_filtered[:, data_filtered.var.Chrom.isin(list(map(str, range(1, 23))) + ["X", "Y"])]

        # removing RP, MRP, MALAT etc
        # removing HLA ; protein_coding only
        remove = data_filtered.var_names.str.startswith('ZNF') + data_filtered.var_names.str.startswith(
            'HLA') + data_filtered.var_names.str.startswith('MALAT1') + data_filtered.var_names.str.startswith(
            'RP') + data_filtered.var_names.str.contains('^HB[^(P)]') + data_filtered.var_names.str.startswith(
            'MRPL') + data_filtered.var_names.str.startswith('MRPS')

        keep = np.invert(remove)
        data_filtered = data_filtered[:, keep]
        data_filtered = data_filtered[:, data_filtered.var.Type == "protein_coding"]

        # remove highly variable genes
        #sc.pp.highly_variable_genes(data_filtered, flavor='seurat_v3', n_top_genes=50)

        #data_filtered = data_filtered[:, ~data_filtered.var.highly_variable]

        # 1 normalize coverage to 7k per cell

        sc.pp.normalize_total(data_filtered, target_sum=7000)

        #exp_matrix = data_filtered.X
        # geometric mean normalized
        #exp_matrix = exp_matrix / np.mean(exp_matrix, axis=0)

        #normalization will be against the means of the cluster - works well for poisson genes

        #smoothing of data
        def smooth_data_np_hanning(arr, sm=np.hanning(11)):
            print(arr)
            return np.exp2(np.convolve(np.log2(arr), sm, mode="same"))

        #import scipy.signal
        def smooth_data_np_hanning_2d(arr): #2D smoothing
            hann = (np.hanning(31)[np.newaxis, :] * np.hanning(31)[np.newaxis, :].T)
            hann = hann / hann.sum()
            return scipy.signal.convolve2d(arr, hann, mode="same")

        for cl in self.clusters:
            #to_plot=np.log2(np.sum(smooth_data_np_hanning_2d(np.array(exp_matrix[data_filtered.obs.clusters==cl,:])), axis=0).squeeze())
            #to_plot = to_plot - scipy.special.logsumexp(to_plot)
            #to_plot=to_plot-np.mean(to_plot)
            #to_plot=to_plot-scipy.special.logsumexp(to_plot) #normalize to total 1
            #to_norm=np.mean(np.log2(smooth_data_np_hanning_2d(np.array(exp_matrix[data_filtered.obs.clusters==normal_cluster,:]))),axis=0).squeeze()
            #to_norm=to_norm-np.mean(np.array(to_norm))
            #to_norm=to_norm-scipy.special.logsumexp(to_norm) #normalize to total 1
            #to_plot=to_plot-to_norm
            cl_in = np.ravel(np.mean(data_filtered.X[data_filtered.obs.clusters == cl, :], axis=0))
            cl_in = cl_in / np.mean(cl_in)

            cl_norm = np.ravel(np.mean(data_filtered.X[data_filtered.obs.clusters == normal_cluster, :], axis=0))
            cl_norm = cl_norm / np.mean(cl_norm)

            ratio = cl_in / cl_norm
            df= data_filtered.var.copy()
            df["cn_ratio"]=ratio #smooth?
            self.cleaned_cn_by_cluster[cl]=df


    def _per_cell_cn_data(self, normal_cluster="-1", case_cluster="6",min_val_cut=0.05,max_val_cut=5):
        
        from scipy.signal import savgol_filter
        genes = self.gene_df

        # 1 remove poorly covered cells
        data_sc_level = self.adata[np.array(self.adata.X.sum(axis=1)).flatten() > 200, :].copy()

        #data_filtered.var['Chrom'] = list(genes.Chrom)
        #data_filtered.var['Position'] = list(genes.Position)
        #data_filtered.var['Type'] = list(genes.Type)
        # # of cells per gene
        # count_nonzero = self.adata.X.astype(bool).sum(axis=0)
        # good_cov = np.array(count_nonzero > 50).squeeze()
        # remove poorly covered genes and non-poisson-like

  

        # remove genes with less than 50 cells registered
        # data_filtered = data_filtered[:, good_cov]

        # remove genes in non autosomes+X+Y
        data_sc_level = data_sc_level[:, data_sc_level.var.Chrom.isin(list(map(str, range(1, 23))) )] #+ ["X", "Y"]

        # removing RP, MRP, MALAT etc
        # removing HLA ; protein_coding only
        remove = data_sc_level.var_names.str.startswith('ZNF') + data_sc_level.var_names.str.startswith(
            'HLA') + data_sc_level.var_names.str.startswith('MALAT1') + data_sc_level.var_names.str.startswith(
            'RP') + data_sc_level.var_names.str.contains('^HB[^(P)]') + data_sc_level.var_names.str.startswith(
            'MRPL') + data_sc_level.var_names.str.startswith('MRPS')

        keep = np.invert(remove)
        data_sc_level = data_sc_level[:, keep]
        data_sc_level = data_sc_level[:, data_sc_level.var.Type == "protein_coding"]

        # remove highly variable genes
        #sc.pp.highly_variable_genes(data_filtered, flavor='seurat_v3', n_top_genes=50)

        #data_filtered = data_filtered[:, ~data_filtered.var.highly_variable]

        # 1 normalize coverage to 7k per cell

        #sc.pp.normalize_total(data_filtered, target_sum=7000)

        #exp_matrix = data_filtered.X
        # geometric mean normalized
        #exp_matrix = exp_matrix / np.mean(exp_matrix, axis=0)
        means_0 = np.array(np.mean(data_sc_level[data_sc_level.obs.clusters == normal_cluster].X, axis=0)).ravel()
        means_cl = np.array(np.mean(data_sc_level[data_sc_level.obs.clusters != normal_cluster].X, axis=0)).ravel()

        data_sc_level = data_sc_level[:,
                        np.where((means_0 > min_val_cut) & (means_0 <= max_val_cut) & (means_cl > min_val_cut) & (means_cl <= max_val_cut))[0]]
            
        sc.pp.normalize_total(data_sc_level, target_sum=7000)
        
        #normalization will be against the means of the cluster - works well for poisson genes
        cl_norm = np.ravel(np.sum(data_sc_level.X[data_sc_level.obs.clusters == normal_cluster, :], axis=0))
        cl_norm=savgol_filter(cl_norm,91,1)

        cl_norm = cl_norm / np.mean(cl_norm)

        
        data_sc_level=data_sc_level[data_sc_level.obs.clusters != normal_cluster,:]

        cells_expr=[]
        
        for cell in range(len(data_sc_level.obs)):
            cell_exp = np.ravel(data_sc_level[cell, :].X.todense())
            cell_exp=savgol_filter(cell_exp,91,1)
          
            cell_exp=cell_exp/np.mean(cell_exp)
            ratio=cell_exp/cl_norm

            cells_expr.append(ratio)

        return (np.vstack(cells_expr), data_sc_level)
    
    def _per_cell_map_hmm_cn(self, normal_cluster="-1", hmm_cluster_map={}):
        
        data_sc_level = self.adata[np.array(self.adata.X.sum(axis=1)).flatten() > 200, :].copy()

        # remove genes in non autosomes+X+Y
        data_sc_level = data_sc_level[:, data_sc_level.var.Chrom.isin(list(map(str, range(1, 23))) )] #+ ["X", "Y"]

        # removing RP, MRP, MALAT etc
        # removing HLA ; protein_coding only
        remove = data_sc_level.var_names.str.startswith('ZNF') + data_sc_level.var_names.str.startswith(
            'HLA') + data_sc_level.var_names.str.startswith('MALAT1') + data_sc_level.var_names.str.startswith(
            'RP') + data_sc_level.var_names.str.contains('^HB[^(P)]') + data_sc_level.var_names.str.startswith(
            'MRPL') + data_sc_level.var_names.str.startswith('MRPS')

        keep = np.invert(remove)
        data_sc_level = data_sc_level[:, keep] 
        data_sc_level = data_sc_level[:, data_sc_level.var.Type == "protein_coding"]

        data_sc_level=data_sc_level[data_sc_level.obs.clusters != normal_cluster,:]

        cells_expr=[]
        hmm_cell_cl_map={}
        #generate gene-snv map array for each cluster
        for cl,hmm_df in hmm_cluster_map.items():
                hmm_row=[]
                for g in data_sc_level.var.iterrows(): #for each geen find the closest hmm value
                    chrom_res=hmm_df[hmm_df.Chrom==g[1].Chrom]
                    if len(chrom_res)<1:
                        hmm_row.append(2)
                        continue
                    
                    hmm_value=chrom_res.iloc[np.argmin(np.abs(chrom_res.Position.astype(int)-int(g[1].Position)))].hmm
                    hmm_row.append(hmm_value)
                hmm_cell_cl_map[cl]=np.array(hmm_row)
            
            
            
        
        for cell in data_sc_level.obs.index:
            hmm_res=hmm_cell_cl_map[data_sc_level.obs.loc[cell].clusters]
            cells_expr.append(hmm_res)

        return (np.vstack(cells_expr), data_sc_level)
    
    def _per_cell_hmm_cluster(self, normal_cluster="-1", case_cluster="6",min_val_cut=0.05,max_val_cut=5):
        
        from scipy.signal import savgol_filter
        genes = self.gene_df

        # 1 remove poorly covered cells
        data_sc_level = self.adata[np.array(self.adata.X.sum(axis=1)).flatten() > 200, :].copy()

        #data_filtered.var['Chrom'] = list(genes.Chrom)
        #data_filtered.var['Position'] = list(genes.Position)
        #data_filtered.var['Type'] = list(genes.Type)
        # # of cells per gene
        # count_nonzero = self.adata.X.astype(bool).sum(axis=0)
        # good_cov = np.array(count_nonzero > 50).squeeze()
        # remove poorly covered genes and non-poisson-like

  

        # remove genes with less than 50 cells registered
        # data_filtered = data_filtered[:, good_cov]

        # remove genes in non autosomes+X+Y
        data_sc_level = data_sc_level[:, data_sc_level.var.Chrom.isin(list(map(str, range(1, 23))) )] #+ ["X", "Y"]

        # removing RP, MRP, MALAT etc
        # removing HLA ; protein_coding only
        remove = data_sc_level.var_names.str.startswith('ZNF') + data_sc_level.var_names.str.startswith(
            'HLA') + data_sc_level.var_names.str.startswith('MALAT1') + data_sc_level.var_names.str.startswith(
            'RP') + data_sc_level.var_names.str.contains('^HB[^(P)]') + data_sc_level.var_names.str.startswith(
            'MRPL') + data_sc_level.var_names.str.startswith('MRPS')

        keep = np.invert(remove)
        data_sc_level = data_sc_level[:, keep]
        data_sc_level = data_sc_level[:, data_sc_level.var.Type == "protein_coding"]

        # remove highly variable genes
        #sc.pp.highly_variable_genes(data_filtered, flavor='seurat_v3', n_top_genes=50)

        #data_filtered = data_filtered[:, ~data_filtered.var.highly_variable]

        # 1 normalize coverage to 7k per cell

        #sc.pp.normalize_total(data_filtered, target_sum=7000)

        #exp_matrix = data_filtered.X
        # geometric mean normalized
        #exp_matrix = exp_matrix / np.mean(exp_matrix, axis=0)
        means_0 = np.array(np.mean(data_sc_level[data_sc_level.obs.clusters == normal_cluster].X, axis=0)).ravel()
        means_cl = np.array(np.mean(data_sc_level[data_sc_level.obs.clusters != normal_cluster].X, axis=0)).ravel()

        data_sc_level = data_sc_level[:,
                        np.where((means_0 > min_val_cut) & (means_0 <= max_val_cut) & (means_cl > min_val_cut) & (means_cl <= max_val_cut))[0]]
            
        
        
        #normalization will be against the means of the cluster - works well for poisson genes
        cl_norm = np.ravel(np.sum(data_sc_level.X[data_sc_level.obs.clusters == normal_cluster, :], axis=0))
        cl_norm=savgol_filter(cl_norm,41,1)

        cl_norm = cl_norm / np.mean(cl_norm)

        
        data_sc_level=data_sc_level[data_sc_level.obs.clusters != normal_cluster,:]

        cells_expr=[]
        
        for cell in range(len(data_sc_level.obs)):
            cell_exp = np.ravel(data_sc_level[cell, :].X.todense())
            cell_exp=savgol_filter(cell_exp,41,1)
          
            cell_exp=cell_exp/np.mean(cell_exp)
            ratio=cell_exp/cl_norm

            cells_expr.append(ratio)

        return (np.vstack(cells_expr), data_sc_level)

    def _get_gene_list(self,gene_file):
        gene_list = []
        for line in open(gene_file):
            if line[0] == "#": continue
            spl = line.strip().split("\t")
            if spl[2] != "gene": continue
            gene_dict = {"Gene": spl[8].split("gene_name")[1].split(";")[0].strip().strip('"'), "Chrom": spl[0],
                         "Position": int(spl[3]),
                         "Type": spl[8].split("gene_biotype")[1].strip().strip('"')}
            gene_list.append(gene_dict)
        return pd.DataFrame(gene_list)

    def _make_sparse_read_cnt_mxs(self):

        self.ALT_matrix = dok_matrix((len(self.cell_barcode_list), len(self.parsed_mpileup_dict)), dtype=np.float32)
        self.REF_matrix = dok_matrix((len(self.cell_barcode_list), len(self.parsed_mpileup_dict)), dtype=np.float32)
        for snp_idx, row_dict in enumerate(self.parsed_mpileup_dict):
            # chrom,pos,stnd,bases=line.strip().split("\t")
            # no evidence as het
            row_in = row_dict  # row[1].dropna()
            for cell in self.cell_barcode_list:
                if cell in row_in:
                    # print (cell, row_in[cell])
                    if row_in[cell][0]: self.REF_matrix[self.cell_barcode_list.index(cell), snp_idx] = row_in[cell][0]
                    if row_in[cell][1]: self.ALT_matrix[self.cell_barcode_list.index(cell), snp_idx] = row_in[cell][1]


    def _extract_used_barcodes(self):
        return list(self.adata.obs.index)

    def plot_cn_profile(self, normalize_factor=2,cluster="2"):
        import matplotlib.pyplot as plt
        from scipy.signal import savgol_filter

        mult = 3.230311e-10
        offset = [0.0000000, 0.0805157, 0.1590767, 0.2230440, 0.2847928, 0.3432341, 0.3985096, 0.4499163,
                  0.4971964, 0.5428127, 0.5865947, 0.6302060, 0.6734443, 0.7106477, 0.7453250, 0.7784458,
                  0.8076332, 0.8338618, 0.8590832, 0.8781837, 0.8985429, 0.9140903, 0.9306633, 0.9808205,
                  1.0000000]
        cent_pos_s = [121535434, 92326171, 90504854, 49660117, 46405641, 58830166, 58054331, 43838887, 47367679,
                      39254935, 51644205, 34856694, 16000000, 16000000, 17000000, 35335801, 22263006, 15460898,
                      24681782, 26369569, 11288129, 13000000, 58632012, 13104553]
        cent_pos_e = [124535434, 95326171, 93504854, 52660117, 49405641, 61830166, 61054331, 46838887, 50367679,
                      42254935, 54644205, 37856694, 19000000, 19000000, 20000000, 38335801, 25263006, 18460898,
                      27681782, 29369569, 14288129, 16000000, 61632012, 13104553]
        axis_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                       '18', '19', '20', '21', '22', 'X', 'Y']
        cent_pos_s = [sum(y) for y in zip(offset, [x * mult for x in cent_pos_s])]
        cent_pos_e = [sum(y) for y in zip(offset, [x * mult for x in cent_pos_e])]
        chr_mids = [(y - x) / 2 + x for x, y in zip(offset, offset[1:])]

        if 1:
            seg_pos = []
            data_filtered=self.cleaned_cn_by_cluster[cluster]
            # df2=df[df["Cell_count"]>10]
            for chrom, pos in zip(data_filtered.Chrom, data_filtered.Position):
                seg_pos.append(int(pos) * mult + offset[axis_labels.index(str(chrom))])

            a = 0
            from scipy.signal import savgol_filter
            
            #plt.plot(seg_pos,  savgol_filter(data_filtered.cn_ratio * normalize_factor,31,1), linestyle="None",  # /np.median(sub_targs).T)
            #         marker=".", color="blue", markersize=5, rasterized=True, alpha=0.7)
            plt.plot(seg_pos,  data_filtered.cn_ratio * normalize_factor, linestyle="None",  # /np.median(sub_targs).T)
                     marker=".", color="red", markersize=5, rasterized=True, alpha=0.7)
            #plt.plot(seg_pos,  savgol_filter(data_filtered.cn_ratio * normalize_factor,21,3), linestyle="None",  # /np.median(sub_targs).T)
            #         marker=".", color="green", markersize=5, rasterized=True, alpha=0.7)
            #plt.plot(seg_pos,  savgol_filter(data_filtered.cn_ratio * normalize_factor,41,1), linestyle="None",  # /np.median(sub_targs).T)
            #         marker=".", color="blue", markersize=5, rasterized=True, alpha=0.7)
            #plt.plot(seg_pos,  savgol_filter(data_filtered.cn_ratio * normalize_factor,41,3), linestyle="None",  # /np.median(sub_targs).T)
            #         marker=".", color="black", markersize=5, rasterized=True, alpha=0.7)
            #plt.plot(seg_pos,  savgol_filter(data_filtered.cn_ratio * normalize_factor,101,3), linestyle="None",  # /np.median(sub_targs).T)
            #         marker=".", color="orange", markersize=5, rasterized=True, alpha=0.7)
            
            #plt.plot(seg_pos, pd.DataFrame( data_filtered.cn_ratio * normalize_factor).rolling(5).mean(), label='Rollingavg', color="green")
            # plt.plot(seg_pos,dh_hets.AFN , linestyle="None",  #/np.median(sub_targs).T)
            #         marker=".", color="red", markersize=2,rasterized=True,alpha=0.2)

            # plt.plot(seg_pos,dh_hets.pcAFT , linestyle="None",  #/np.median(sub_targs).T)
            #         marker=".", color="green", markersize=2,rasterized=True,alpha=0.5)

            plt.ylim((0, 6))

            plt.ylabel('Copy Number')
            plt.xlabel('Chromosome')
            for i in range(0, 24):
                plt.axvline(cent_pos_s[i], ls='--', color='gray', linewidth=1, alpha=0.7, zorder=9)
                if i % 2 == 0:
                    plt.axvspan(offset[i], offset[i + 1], color='gray', alpha=0.35, ec='none')

            locs, labels = plt.xticks(chr_mids, axis_labels, fontsize=7)
            plt.setp(labels, rotation=30)
            plt.tick_params(axis='x', which='both', bottom='off', top='off')
            # plt.xlim((offset[0], offset[4]))

            figure = plt.gcf()
            figure.set_size_inches(15, 8)
            plt.show()


    def plot_hmm_result(self, het_df=None):
        import matplotlib.pyplot as plt

        mult = 3.230311e-10
        offset = [0.0000000, 0.0805157, 0.1590767, 0.2230440, 0.2847928, 0.3432341, 0.3985096, 0.4499163,
                  0.4971964, 0.5428127, 0.5865947, 0.6302060, 0.6734443, 0.7106477, 0.7453250, 0.7784458,
                  0.8076332, 0.8338618, 0.8590832, 0.8781837, 0.8985429, 0.9140903, 0.9306633, 0.9808205,
                  1.0000000]
        cent_pos_s = [121535434, 92326171, 90504854, 49660117, 46405641, 58830166, 58054331, 43838887, 47367679,
                      39254935, 51644205, 34856694, 16000000, 16000000, 17000000, 35335801, 22263006, 15460898,
                      24681782, 26369569, 11288129, 13000000, 58632012, 13104553]
        cent_pos_e = [124535434, 95326171, 93504854, 52660117, 49405641, 61830166, 61054331, 46838887, 50367679,
                      42254935, 54644205, 37856694, 19000000, 19000000, 20000000, 38335801, 25263006, 18460898,
                      27681782, 29369569, 14288129, 16000000, 61632012, 13104553]
        axis_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                       '18', '19', '20', '21', '22', 'X', 'Y']
        cent_pos_s = [sum(y) for y in zip(offset, [x * mult for x in cent_pos_s])]
        cent_pos_e = [sum(y) for y in zip(offset, [x * mult for x in cent_pos_e])]
        chr_mids = [(y - x) / 2 + x for x, y in zip(offset, offset[1:])]

        if 1:
            seg_pos = []
            # df2=df[df["Cell_count"]>10]
            for chrom, pos in zip(het_df.Chrom, het_df.Position):
                seg_pos.append(int(pos) * mult + offset[axis_labels.index(str(chrom))])

            a = 0
            plt.plot(seg_pos, het_df.hmm, linestyle="None",  # /np.median(sub_targs).T)
                     marker=".", color="red", markersize=5, rasterized=True, alpha=0.7)

                  # plt.plot(seg_pos,dh_hets.AFN , linestyle="None",  #/np.median(sub_targs).T)
            #         marker=".", color="red", markersize=2,rasterized=True,alpha=0.2)

            # plt.plot(seg_pos,dh_hets.pcAFT , linestyle="None",  #/np.median(sub_targs).T)
            #         marker=".", color="green", markersize=2,rasterized=True,alpha=0.5)

            plt.ylim((0, 6))

            plt.ylabel('Copy Number')
            plt.xlabel('Chromosome')
            for i in range(0, 24):
                plt.axvline(cent_pos_s[i], ls='--', color='gray', linewidth=1, alpha=0.7, zorder=9)
                if i % 2 == 0:
                    plt.axvspan(offset[i], offset[i + 1], color='gray', alpha=0.35, ec='none')

            locs, labels = plt.xticks(chr_mids, axis_labels, fontsize=7)
            plt.setp(labels, rotation=30)
            plt.tick_params(axis='x', which='both', bottom='off', top='off')
            # plt.xlim((offset[0], offset[4]))

            figure = plt.gcf()
            figure.set_size_inches(15, 8)
            plt.show()

    def plot_tiny_hmm_result(self,hmm, het_df=None,title="", ax=None):

            mult = 3.230311e-10
            offset = [0.0000000, 0.0805157, 0.1590767, 0.2230440, 0.2847928, 0.3432341, 0.3985096, 0.4499163,
                      0.4971964, 0.5428127, 0.5865947, 0.6302060, 0.6734443, 0.7106477, 0.7453250, 0.7784458,
                      0.8076332, 0.8338618, 0.8590832, 0.8781837, 0.8985429, 0.9140903, 0.9306633]#, 0.9808205,1.0000000]
            
            cent_pos_s = [121535434, 92326171, 90504854, 49660117, 46405641, 58830166, 58054331, 43838887, 47367679,
                          39254935, 51644205, 34856694, 16000000, 16000000, 17000000, 35335801, 22263006, 15460898,
                          24681782, 26369569, 11288129, 13000000]#, 58632012, 13104553]
            cent_pos_e = [124535434, 95326171, 93504854, 52660117, 49405641, 61830166, 61054331, 46838887, 50367679,
                          42254935, 54644205, 37856694, 19000000, 19000000, 20000000, 38335801, 25263006, 18460898,
                          27681782, 29369569, 14288129, 16000000]#, 61632012, 13104553]
            axis_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                           '18', '19', '20', '21', '22']#, 'X', 'Y']
            cent_pos_s = [sum(y) for y in zip(offset, [x * mult for x in cent_pos_s])]
            cent_pos_e = [sum(y) for y in zip(offset, [x * mult for x in cent_pos_e])]
            chr_mids = [(y - x) / 2 + x for x, y in zip(offset, offset[1:])]

            het_df_hmm = hmm[~hmm.Chrom.isin( ["X","Y"])] #filter X,Y
            
            if not ax:
                figure = plt.gcf()
                figure.set_size_inches(7, 0.5)
                ax=figure.gca()
                
            
            if 1:
                seg_pos = []
                # df2=df[df["Cell_count"]>10]
                for chrom, pos in zip(het_df_hmm.Chrom, het_df_hmm.Position):
                    seg_pos.append(int(pos) * mult + offset[axis_labels.index(str(chrom))])

                a = 0

                ax.scatter(seg_pos, het_df_hmm.hmm, c="green", s=5, alpha=0.5)  # hmm.hmm
                # plt.plot(seg_pos, het_df.hmm, linestyle="None",  # /np.median(sub_targs).T)
                #         marker=".", color=het_df.hmm, markersize=5, rasterized=True, alpha=0.7)

                # plt.plot(seg_pos,dh_hets.AFN , linestyle="None",  #/np.median(sub_targs).T)
                #         marker=".", color="red", markersize=2,rasterized=True,alpha=0.2)

                # plt.plot(seg_pos,dh_hets.pcAFT , linestyle="None",  #/np.median(sub_targs).T)
                #         marker=".", color="green", markersize=2,rasterized=True,alpha=0.5)

                ax.set_ylim((1.1, 6))

                # plt.ylabel('Copy Number')
                # plt.xlabel('Chromosome')
                for i in range(0, 22):
                    ax.axvline(cent_pos_s[i], ls='--', color='gray', linewidth=1, alpha=0.7, zorder=9)
                    if i % 2 == 0:
                        ax.axvspan(offset[i], offset[i + 1], color='gray', alpha=0.35, ec='none')

                labels = ax.set_xticklabels( axis_labels, fontsize=7)
                locs= ax.set_xticks(chr_mids)
                
                ax.set_title(title)
                
                # plt.setp(labels, rotation=30)
                ax.tick_params(axis='x', which='both', bottom='off', top='off')
                #plt.xlim((offset[0], offset[4]))

                #figure = plt.gcf()
                #figure.set_size_inches(7, 0.5)
                #plt.title(title)
                #plt.show()
                
    def plot_tiny_hmm_allelic_result(self,hmm, het_df=None,title="", ax=None):

            mult = 3.230311e-10
            offset = [0.0000000, 0.0805157, 0.1590767, 0.2230440, 0.2847928, 0.3432341, 0.3985096, 0.4499163,
                      0.4971964, 0.5428127, 0.5865947, 0.6302060, 0.6734443, 0.7106477, 0.7453250, 0.7784458,
                      0.8076332, 0.8338618, 0.8590832, 0.8781837, 0.8985429, 0.9140903, 0.9306633]#, 0.9808205,1.0000000]
            
            cent_pos_s = [121535434, 92326171, 90504854, 49660117, 46405641, 58830166, 58054331, 43838887, 47367679,
                          39254935, 51644205, 34856694, 16000000, 16000000, 17000000, 35335801, 22263006, 15460898,
                          24681782, 26369569, 11288129, 13000000]#, 58632012, 13104553]
            cent_pos_e = [124535434, 95326171, 93504854, 52660117, 49405641, 61830166, 61054331, 46838887, 50367679,
                          42254935, 54644205, 37856694, 19000000, 19000000, 20000000, 38335801, 25263006, 18460898,
                          27681782, 29369569, 14288129, 16000000]#, 61632012, 13104553]
            axis_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                           '18', '19', '20', '21', '22']#, 'X', 'Y']
            cent_pos_s = [sum(y) for y in zip(offset, [x * mult for x in cent_pos_s])]
            cent_pos_e = [sum(y) for y in zip(offset, [x * mult for x in cent_pos_e])]
            chr_mids = [(y - x) / 2 + x for x, y in zip(offset, offset[1:])]

            het_df_hmm = hmm[~hmm.Chrom.isin( ["X","Y"])] #filter X,Y
            
            if not ax:
                figure = plt.gcf()
                figure.set_size_inches(7, 0.5)
                ax=figure.gca()
                
            
            if 1:
                seg_pos = []
                # df2=df[df["Cell_count"]>10]
                for chrom, pos in zip(het_df_hmm.Chrom, het_df_hmm.Position):
                    seg_pos.append(int(pos) * mult + offset[axis_labels.index(str(chrom))])

                a = 0
                
                def color_recode(val_ar, def_col="red"):
                    return ["red" if x>1 else "blue" if x<1 else "purple" for x in val_ar]
                    
                
                ax.scatter(seg_pos, het_df_hmm.hmm_major, c=color_recode(het_df_hmm.hmm_major), marker="_",linewidths=5,s=20, alpha=0.5)
                ax.scatter(seg_pos, het_df_hmm.hmm_minor, c=color_recode(het_df_hmm.hmm_minor), marker="_",linewidths=5,s=20, alpha=0.5)  # hmm.hmm
                # plt.plot(seg_pos, het_df.hmm, linestyle="None",  # /np.median(sub_targs).T)
                #         marker=".", color=het_df.hmm, markersize=5, rasterized=True, alpha=0.7)

                # plt.plot(seg_pos,dh_hets.AFN , linestyle="None",  #/np.median(sub_targs).T)
                #         marker=".", color="red", markersize=2,rasterized=True,alpha=0.2)

                # plt.plot(seg_pos,dh_hets.pcAFT , linestyle="None",  #/np.median(sub_targs).T)
                #         marker=".", color="green", markersize=2,rasterized=True,alpha=0.5)

                ax.set_ylim((-0.5, 5.5))

                # plt.ylabel('Copy Number')
                # plt.xlabel('Chromosome')
                for i in range(0, 22):
                    ax.axvline(cent_pos_s[i], ls='--', color='gray', linewidth=1, alpha=0.7, zorder=9)
                    if i % 2 == 0:
                        ax.axvspan(offset[i], offset[i + 1], color='gray', alpha=0.35, ec='none')

                labels = ax.set_xticklabels( axis_labels, fontsize=7)
                locs= ax.set_xticks(chr_mids)
                
                ax.set_title(title)
                
                ax.set_xlim((0, 0.9306633)) #adjust boundaries at ends
                
                # plt.setp(labels, rotation=30)
                ax.tick_params(axis='x', which='both', bottom='off', top='off')
                #plt.xlim((offset[0], offset[4]))

                #figure = plt.gcf()
                #figure.set_size_inches(7, 0.5)
                #plt.title(title)
                #plt.show()
                
    def run_full_hmm_diploid(self,cluster=None,mean_cn=2,blacklist=[],clustering_name="clusters"):
        cnt = np.array(self.ALT_matrix.sum(axis=1)).flatten()
        cnt_high = cnt > 10

        def run_HMM_model( initial_prob, trans_prob, emiss_prob, observations):
            tfd = tfp.distributions

            initial_distribution = tfd.Categorical(probs=np.array(initial_prob, dtype=np.float32).flatten())

            #print(initial_prob.flatten())

            transition_distribution = tfd.Categorical(probs=np.array(trans_prob, dtype=np.float32).T)

            #print(np.array(trans_prob, dtype=np.float32).T)

            observation_distribution = CategoricalNew(logits=np.log(emiss_prob, dtype=np.float32))

            #print(np.log(emiss_prob, dtype=np.float32))

            model = tfd.HiddenMarkovModel(
                initial_distribution=initial_distribution,
                transition_distribution=transition_distribution,
                observation_distribution=observation_distribution,
                num_steps=len(observations))

            return model.posterior_mode(observations, mask=None, name='posterior_mode')


        for cluster in [cluster]:  # set(cell_barc_dict.values()):
            print("Cluster", cluster)

            cl_cell_ids = []

            for idx, cell in enumerate(self.cell_barcode_list):
                # print(cnt_high[idx])

                if cnt_high[idx] and self.adata.obs.loc[cell][clustering_name] == cluster:
                    # print ( cell_barc_dict[cell],idx)
                    cl_cell_ids.append(idx)

            cl_refs = csr_matrix(self.REF_matrix[cl_cell_ids, :].sum(axis=0))
            cl_alts = csr_matrix(self.ALT_matrix[cl_cell_ids, :].sum(axis=0))
            print("cell_count", len(cl_cell_ids))

            if len(cl_cell_ids) > 2:
                stacked = scipy.sparse.vstack([cl_refs, cl_alts])
                idx_stacked=np.ravel(stacked.sum(axis=0)>0) #.nonzero()[1]
                #print (idx_stacked)


                # alpha=np.array(stacked[0][:,idx_stacked].todense(),dtype=np.float64)+1.
                # beta=np.array(stacked[1][:,idx_stacked].todense(),dtype=np.float64)+1.
                # alpha=np.array(stacked[:,idx_stacked].todense(),dtype=np.float64).min(axis=0)+1. #replace alt, ref with min, max
                # beta=np.array(stacked[:,idx_stacked].todense(),dtype=np.float64).max(axis=0)+1.

                #remove for now
                #stacked_filt = stacked[:, idx_stacked][:,(stacked[:, idx_stacked].sum(axis=0) - 1).nonzero()[1]]  # remove nonzero and ones
                #scnd_idx = (stacked[:, idx_stacked].sum(axis=0) - 1).nonzero()[1]
                #print(stacked_filt.shape)
                #print(scnd_idx.shape)

                # alpha=np.array(stacked_filt.todense(),dtype=np.float64).min(axis=0)+1. #replace alt, ref with min, max
                # beta=np.array(stacked_filt.todense(),dtype=np.float64).max(axis=0)+1.
                hets_plt = self.het_df.iloc[idx_stacked]#.iloc[scnd_idx]
                #print(len(hets_plt), np.array(stacked[1].todense(), dtype=np.float64).shape)

                alpha = np.array(stacked[0].todense(), dtype=np.float64)[:,idx_stacked] + 1.
                beta = np.array(stacked[1].todense(), dtype=np.float64)[:,idx_stacked] + 1.

                # with tf.device('/gpu:1'):

                dist = tfd.Beta(alpha, beta)
                dist2 = tfd.Beta(beta, alpha)
                domain = tf.tile(
                    tf.expand_dims(tf.constant([0.03,  0.33, 0.5, ], dtype=tf.float64), axis=1),
                    tf.constant([1, alpha.shape[0]], tf.int32))

                def recode_state(state):
                    dict_map = {x: y for x, y in zip(range(0, 6), [1,  3,2 ]+[-1]*3)}  # 0/1,0/2,1/2,1/1,2/2,2/3,1/4
                    return dict_map[state]

                #print(hets_plt.index, len(set(hets_plt.index)))
                out_dist = dist.prob(domain) + dist2.prob(domain)
                b = tf.reduce_sum(out_dist, axis=0)

                out_dist = out_dist / b  # tf.reshape(b, (-1, 1)) #normalize

                # add copy-number LH
                cr_out_dist = None
                CR=self.cleaned_cn_by_cluster[cluster]

                for idx, snv in enumerate(hets_plt.iterrows()):
                    cr_near = CR[(CR.Chrom == snv[1].Chrom) & (
                        CR.Position.between(int(snv[1].Position) - 20000000, int(snv[1].Position) + 20000000))]
                    cr_normed = scipy.stats.norm.pdf(x=[1, 3, 2], loc=np.mean(cr_near.cn_ratio)*mean_cn ,
                                                     scale=0.8)
                    cr_normed = cr_normed / np.sum(cr_normed)/5 #scale to 20% impact
                    prod = out_dist[:, idx] * cr_normed
                    prod = prod / np.sum(prod)

                    if not cr_out_dist:
                        cr_out_dist = [prod]
                    else:
                        cr_out_dist.append(prod)

                out_dist = tf.stack(cr_out_dist, axis=1)
                print(out_dist.shape)

                emiss_prob = tf.concat([out_dist, tf.fill(out_dist.shape, tf.cast(1 / 20, tf.float64))], axis=0)

                # ,ALT_matrix[idx,:],REF_matrix[idx,:])

                initial_prob = np.full((6, 1), 1 / 6.)

                trans_prob = np.full((3, 6), 1 / 50.)
                epsilon_prob = np.full((3, 6), 10e-3)  # add epsilon error state

                np.fill_diagonal(trans_prob, 0.9)  # from itself to itself

                np.fill_diagonal(epsilon_prob, 0.001)  # between epsilon
                trans_prob = np.append(trans_prob, epsilon_prob, 0)
                for x in range(3):  # fill error_st back to regular
                    trans_prob[x, x + 3] = 0.8

                observations = tf.range(emiss_prob.shape[1])
                hmmr = run_HMM_model(initial_prob=initial_prob,
                                          trans_prob=trans_prob, emiss_prob=emiss_prob+1e-24, observations=observations)

                seq = hmmr
                print(len(seq))
                # hets_plt=hets.iloc[idx_stacked].iloc[scnd_idx] moved up
                hets_plt['hmm'] = list(map(recode_state,map(int,seq)))  # [:135249]
                #hets_plt = hets_plt[hets_plt.chrom != "X"]
                # hets_plt[hets_plt.hmm==1]=3

        return hets_plt
    def run_full_hmm(self,cluster=None,mean_cn=2,blacklist=[],clustering_name="clusters",min_cnt=0):
        cnt = np.array(self.ALT_matrix.sum(axis=1)).flatten()
        cnt_high = cnt > 10

        def run_HMM_model( initial_prob, trans_prob, emiss_prob, observations):
            tfd = tfp.distributions

            initial_distribution = tfd.Categorical(probs=np.array(initial_prob, dtype=np.float32).flatten())

            #print(initial_prob.flatten())

            transition_distribution = tfd.Categorical(probs=np.array(trans_prob, dtype=np.float32).T)

            #print(np.array(trans_prob, dtype=np.float32).T)

            observation_distribution = CategoricalNew(logits=np.log(emiss_prob, dtype=np.float32))

            #print(np.log(emiss_prob, dtype=np.float32))

            model = tfd.HiddenMarkovModel(
                initial_distribution=initial_distribution,
                transition_distribution=transition_distribution,
                observation_distribution=observation_distribution,
                num_steps=len(observations))

            return model.posterior_mode(observations, mask=None, name='posterior_mode')


        for cluster in [cluster]:  # set(cell_barc_dict.values()):
            print("Cluster", cluster)

            cl_cell_ids = []

            for idx, cell in enumerate(self.cell_barcode_list):
                # print(cnt_high[idx])

                if cnt_high[idx] and self.adata.obs.loc[cell][clustering_name] == cluster:
                    # print ( cell_barc_dict[cell],idx)
                    cl_cell_ids.append(idx)

            cl_refs = csr_matrix(self.REF_matrix[cl_cell_ids, :].sum(axis=0))
            cl_alts = csr_matrix(self.ALT_matrix[cl_cell_ids, :].sum(axis=0))
            print("cell_count", len(cl_cell_ids))

            if len(cl_cell_ids) > 2:
                stacked = scipy.sparse.vstack([cl_refs, cl_alts])
                idx_stacked=np.ravel(stacked.sum(axis=0)>min_cnt) #.nonzero()[1]
                #print (idx_stacked)


                # alpha=np.array(stacked[0][:,idx_stacked].todense(),dtype=np.float64)+1.
                # beta=np.array(stacked[1][:,idx_stacked].todense(),dtype=np.float64)+1.
                # alpha=np.array(stacked[:,idx_stacked].todense(),dtype=np.float64).min(axis=0)+1. #replace alt, ref with min, max
                # beta=np.array(stacked[:,idx_stacked].todense(),dtype=np.float64).max(axis=0)+1.

                #remove for now
                #stacked_filt = stacked[:, idx_stacked][:,(stacked[:, idx_stacked].sum(axis=0) - 1).nonzero()[1]]  # remove nonzero and ones
                #scnd_idx = (stacked[:, idx_stacked].sum(axis=0) - 1).nonzero()[1]
                #print(stacked_filt.shape)
                #print(scnd_idx.shape)

                # alpha=np.array(stacked_filt.todense(),dtype=np.float64).min(axis=0)+1. #replace alt, ref with min, max
                # beta=np.array(stacked_filt.todense(),dtype=np.float64).max(axis=0)+1.
                hets_plt = self.het_df.iloc[idx_stacked]#.iloc[scnd_idx]
                #print(len(hets_plt), np.array(stacked[1].todense(), dtype=np.float64).shape)

                alpha = np.array(stacked[0].todense(), dtype=np.float64)[:,idx_stacked] + 1.
                beta = np.array(stacked[1].todense(), dtype=np.float64)[:,idx_stacked] + 1.

                # with tf.device('/gpu:1'):

                dist = tfd.Beta(alpha, beta)
                dist2 = tfd.Beta(beta, alpha)
                domain = tf.tile(
                    tf.expand_dims(tf.constant([0.03, 0.02, 0.33, 0.5, 0.5, 0.40, 0.2], dtype=tf.float64), axis=1),
                    tf.constant([1, alpha.shape[0]], tf.int32))

                def recode_state(state):
                    dict_map = {x: y for x, y in zip(range(0, 14), [1, 2, 3, 2, 4, 5, 5]+[-1]*7)}  # 0/1,0/2,1/2,1/1,2/2,2/3,1/4
                    return dict_map[state]
                
                def recode_al_state_minor(state):
                    dict_map = {x: y for x, y in zip(range(0, 14), [0, 0, 1, 1, 2, 2, 1]+[-1]*7)}  # 0/1,0/2,1/2,1/1,2/2,2/3,1/4
                    return dict_map[state]
                
                def recode_al_state_major(state):
                    dict_map = {x: y for x, y in zip(range(0, 14), [1, 2, 2, 1, 2, 3, 4]+[-1]*7)}  # 0/1,0/2,1/2,1/1,2/2,2/3,1/4
                    return dict_map[state]
                
                #print(hets_plt.index, len(set(hets_plt.index)))
                out_dist = dist.prob(domain) + dist2.prob(domain)
                b = tf.reduce_sum(out_dist, axis=0)

                out_dist = out_dist / b  # tf.reshape(b, (-1, 1)) #normalize

                # add copy-number LH
                cr_out_dist = None
                CR=self.cleaned_cn_by_cluster[cluster]

                for idx, snv in enumerate(hets_plt.iterrows()):
                    cr_near = CR[(CR.Chrom == snv[1].Chrom) & (
                        CR.Position.between(int(snv[1].Position) - 20000000, int(snv[1].Position) + 20000000))]
                    cr_normed = scipy.stats.norm.pdf(x=[1, 2, 3, 2, 4, 5, 5], loc=np.mean(cr_near.cn_ratio)*mean_cn ,
                                                     scale=0.5)
                    cr_normed = cr_normed / np.sum(cr_normed)
                    prod = out_dist[:, idx] * cr_normed
                    prod = prod / np.sum(prod)

                    if not cr_out_dist:
                        cr_out_dist = [prod]
                    else:
                        cr_out_dist.append(prod)

                out_dist = tf.stack(cr_out_dist, axis=1)
                print(out_dist.shape)

                emiss_prob = tf.concat([out_dist, tf.fill(out_dist.shape, tf.cast(1 / 20, tf.float64))], axis=0)

                # ,ALT_matrix[idx,:],REF_matrix[idx,:])

                initial_prob = np.full((14, 1), 1 / 14.)

                trans_prob = np.full((7, 14), 1 / 50.)
                epsilon_prob = np.full((7, 14), 10e-3)  # add epsilon error state

                np.fill_diagonal(trans_prob, 0.9)  # from itself to itself

                np.fill_diagonal(epsilon_prob, 0.001)  # between epsilon
                trans_prob = np.append(trans_prob, epsilon_prob, 0)
                for x in range(7):  # fill error_st back to regular
                    trans_prob[x, x + 7] = 0.8

                observations = tf.range(emiss_prob.shape[1])
                hmmr = run_HMM_model(initial_prob=initial_prob,
                                          trans_prob=trans_prob, emiss_prob=emiss_prob+1e-24, observations=observations)

                seq = hmmr
                print(len(seq))
                # hets_plt=hets.iloc[idx_stacked].iloc[scnd_idx] moved up
                hets_plt['hmm'] = list(map(recode_state,map(int,seq)))  # [:135249]
                hets_plt['hmm_minor'] = list(map(recode_al_state_minor,map(int,seq)))
                hets_plt['hmm_major'] = list(map(recode_al_state_major,map(int,seq)))
                #hets_plt = hets_plt[hets_plt.chrom != "X"]
                # hets_plt[hets_plt.hmm==1]=3

        return hets_plt

    
    
    def run_full_hmm_cn_mode(self,cluster=None,mean_cn=2,blacklist=[],clustering_name="clusters"):
        cnt = np.array(self.ALT_matrix.sum(axis=1)).flatten()
        cnt_high = cnt > 10

        def run_HMM_model( initial_prob, trans_prob, emiss_prob, observations):
            tfd = tfp.distributions

            initial_distribution = tfd.Categorical(probs=np.array(initial_prob, dtype=np.float32).flatten())

            #print(initial_prob.flatten())

            transition_distribution = tfd.Categorical(probs=np.array(trans_prob, dtype=np.float32).T)

            #print(np.array(trans_prob, dtype=np.float32).T)

            observation_distribution = CategoricalNew(logits=np.log(emiss_prob, dtype=np.float32))

            #print(np.log(emiss_prob, dtype=np.float32))

            model = tfd.HiddenMarkovModel(
                initial_distribution=initial_distribution,
                transition_distribution=transition_distribution,
                observation_distribution=observation_distribution,
                num_steps=len(observations))

            return model.posterior_mode(observations, mask=None, name='posterior_mode')


        for cluster in [cluster]:  # set(cell_barc_dict.values()):
            print("Cluster", cluster)

            cl_cell_ids = []

            for idx, cell in enumerate(self.cell_barcode_list):
                # print(cnt_high[idx])

                if cnt_high[idx] and self.adata.obs.loc[cell][clustering_name] == cluster:
                    # print ( cell_barc_dict[cell],idx)
                    cl_cell_ids.append(idx)

            cl_refs = csr_matrix(self.REF_matrix[cl_cell_ids, :].sum(axis=0))
            cl_alts = csr_matrix(self.ALT_matrix[cl_cell_ids, :].sum(axis=0))
            print("cell_count", len(cl_cell_ids))

            if len(cl_cell_ids) > 2:
                stacked = scipy.sparse.vstack([cl_refs, cl_alts])
                idx_stacked=np.ravel(stacked.sum(axis=0)>=0) #.nonzero()[1] #include all SNPs makes uniform
                #print (idx_stacked)


                # alpha=np.array(stacked[0][:,idx_stacked].todense(),dtype=np.float64)+1.
                # beta=np.array(stacked[1][:,idx_stacked].todense(),dtype=np.float64)+1.
                # alpha=np.array(stacked[:,idx_stacked].todense(),dtype=np.float64).min(axis=0)+1. #replace alt, ref with min, max
                # beta=np.array(stacked[:,idx_stacked].todense(),dtype=np.float64).max(axis=0)+1.

                #remove for now
                #stacked_filt = stacked[:, idx_stacked][:,(stacked[:, idx_stacked].sum(axis=0) - 1).nonzero()[1]]  # remove nonzero and ones
                #scnd_idx = (stacked[:, idx_stacked].sum(axis=0) - 1).nonzero()[1]
                #print(stacked_filt.shape)
                #print(scnd_idx.shape)

                # alpha=np.array(stacked_filt.todense(),dtype=np.float64).min(axis=0)+1. #replace alt, ref with min, max
                # beta=np.array(stacked_filt.todense(),dtype=np.float64).max(axis=0)+1.
                hets_plt = self.het_df.iloc[idx_stacked]#.iloc[scnd_idx]
                #print(len(hets_plt), np.array(stacked[1].todense(), dtype=np.float64).shape)

                alpha = np.array(stacked[0].todense(), dtype=np.float64)[:,idx_stacked] + 1.
                beta = np.array(stacked[1].todense(), dtype=np.float64)[:,idx_stacked] + 1.

                # with tf.device('/gpu:1'):

                dist = tfd.Beta(alpha, beta)
                dist2 = tfd.Beta(beta, alpha)
                domain = tf.tile(
                    tf.expand_dims(tf.constant([0.03, 0.02, 0.30, 0.5, 0.5, 0.20, 0.1], dtype=tf.float64), axis=1),
                    tf.constant([1, alpha.shape[0]], tf.int32))

                def recode_state(state):
                    dict_map = {x: y for x, y in zip(range(0, 14), [1, 2, 3, 2, 4, 5, 5]+[-1]*7)}  # 0/1,0/2,1/2,1/1,2/2,2/3,1/4
                    return dict_map[state]

                #print(hets_plt.index, len(set(hets_plt.index)))
                out_dist = dist.prob(domain) + dist2.prob(domain)
                b = tf.reduce_sum(out_dist, axis=0)

                out_dist = out_dist / b  # tf.reshape(b, (-1, 1)) #normalize

                # add copy-number LH
                cr_out_dist = None
                CR=self.cleaned_cn_by_cluster[cluster]

                for idx, snv in enumerate(hets_plt.iterrows()):
                    cr_near = CR[(CR.Chrom == snv[1].Chrom) & (
                        CR.Position.between(int(snv[1].Position) - 20000000, int(snv[1].Position) + 20000000))]
                    cr_normed = scipy.stats.norm.pdf(x=[1, 2, 3, 2, 4, 5, 5], loc=np.mean(cr_near.cn_ratio)*mean_cn ,
                                                     scale=0.5)
                    cr_normed = cr_normed / np.sum(cr_normed)
                    prod = out_dist[:, idx] * cr_normed
                    prod = prod / np.sum(prod)

                    if not cr_out_dist:
                        cr_out_dist = [prod]
                    else:
                        cr_out_dist.append(prod)

                out_dist = tf.stack(cr_out_dist, axis=1)
                print(out_dist.shape)

                emiss_prob = tf.concat([out_dist, tf.fill(out_dist.shape, tf.cast(1 / 20, tf.float64))], axis=0)

                # ,ALT_matrix[idx,:],REF_matrix[idx,:])

                initial_prob = np.full((14, 1), 1 / 14.)

                trans_prob = np.full((7, 14), 1 / 50.)
                epsilon_prob = np.full((7, 14), 10e-3)  # add epsilon error state

                np.fill_diagonal(trans_prob, 0.9)  # from itself to itself

                np.fill_diagonal(epsilon_prob, 0.001)  # between epsilon
                trans_prob = np.append(trans_prob, epsilon_prob, 0)
                for x in range(7):  # fill error_st back to regular
                    trans_prob[x, x + 7] = 0.8

                observations = tf.range(emiss_prob.shape[1])
                hmmr = run_HMM_model(initial_prob=initial_prob,
                                          trans_prob=trans_prob, emiss_prob=emiss_prob+1e-24, observations=observations)

                seq = hmmr
                print(len(seq))
                # hets_plt=hets.iloc[idx_stacked].iloc[scnd_idx] moved up
                hets_plt['hmm'] = list(map(recode_state,map(int,seq)))  # [:135249]
                #hets_plt = hets_plt[hets_plt.chrom != "X"]
                # hets_plt[hets_plt.hmm==1]=3

        return hets_plt    
    
    def run_HMM_model(self,initial_prob, trans_prob, emiss_prob, observations):
        tfd = tfp.distributions

        initial_distribution = tfd.Categorical(probs=initial_prob)

        print(initial_distribution)


        transition_distribution = tfd.Categorical(probs=np.array(trans_prob, dtype=np.float32).T)

        print(transition_distribution)

        observation_distribution = CategoricalNew(logits=np.log(emiss_prob, dtype=np.float32))

        print(observation_distribution)

        model = tfd.HiddenMarkovModel(
            initial_distribution=initial_distribution,
            transition_distribution=transition_distribution,
            observation_distribution=observation_distribution,
            num_steps=len(observations))


        return model.posterior_mode(observations, mask=None, name='posterior_mode') #viterbi path


    def run_hmm_hets_only(self,cluster=None,clustering_name="clusters"):

        print("Cluster", cluster)

        cl_cell_ids = list(self.adata.obs[self.adata.obs[clustering_name]==cluster].index )
        cl_refs = csr_matrix(REF_matrix[cl_cell_ids, :].sum(axis=0))
        cl_alts = csr_matrix(ALT_matrix[cl_cell_ids, :].sum(axis=0))
        print("cell_count", len(cl_cell_ids))

        if len(cl_cell_ids) <=2 : print("Cluster too small, less than 3 cells")
        if len(cl_cell_ids) > 2:
            stacked = scipy.sparse.vstack([cl_refs, cl_alts])
            idx_stacked = stacked.nonzero()[1]

            # alpha=np.array(stacked[0][:,idx_stacked].todense(),dtype=np.float64)+1.
            # beta=np.array(stacked[1][:,idx_stacked].todense(),dtype=np.float64)+1.
            # alpha=np.array(stacked[:,idx_stacked].todense(),dtype=np.float64).min(axis=0)+1. #replace alt, ref with min, max
            # beta=np.array(stacked[:,idx_stacked].todense(),dtype=np.float64).max(axis=0)+1.

            stacked_filt = stacked[:, idx_stacked][:,
                           (stacked[:, idx_stacked].sum(axis=0) - 1).nonzero()[1]]  # remove nonzero and ones
            scnd_idx = (stacked[:, idx_stacked].sum(axis=0) - 1).nonzero()[1]

            # alpha=np.array(stacked_filt.todense(),dtype=np.float64).min(axis=0)+1. #replace alt, ref with min, max
            # beta=np.array(stacked_filt.todense(),dtype=np.float64).max(axis=0)+1.

            alpha = np.array(stacked_filt[0].todense(), dtype=np.float64) + 1.
            beta = np.array(stacked_filt[1].todense(), dtype=np.float64) + 1.

            # with tf.device('/gpu:1'):

            dist = tfd.Beta(alpha, beta)
            dist2 = tfd.Beta(beta, alpha)
            domain = tf.tile(tf.expand_dims(tf.constant([0.02, 0.5, 0.30, 0.20], dtype=tf.float64), axis=1),
                             tf.constant([1, alpha.shape[0]], tf.int32))

            def recode_state(state):
                dict_map = {1: 1, 2: 4, 3: 3, 4: 5}
                return dict_map[state]

            out_dist = dist.prob(domain) + dist2.prob(domain)
            b = tf.reduce_sum(out_dist, axis=0)

            out_dist = out_dist / b  # tf.reshape(b, (-1, 1)) #normalize
            emiss_prob = tf.concat([out_dist, tf.fill(out_dist.shape, tf.cast(1 / 20, tf.float64))], axis=0)

            # ,ALT_matrix[idx,:],REF_matrix[idx,:])

            initial_prob = np.full((8, 1), 1 / 8.)

            trans_prob = np.full((4, 8), 1 / 100.)
            epsilon_prob = np.full((4, 8), 10e-4)  # add epsilon error state

            np.fill_diagonal(trans_prob, 0.9)  # from itself to itself

            np.fill_diagonal(epsilon_prob, 0.001)  # between epsilon
            trans_prob = np.append(trans_prob, epsilon_prob, 0)
            for x in range(4):  # fill error_st back to regular
                trans_prob[x, x + 4] = 0.8

            hmmr = HMM(initial_prob=initial_prob, trans_prob=trans_prob, obs_prob=emiss_prob)

            observations = tf.range(emiss_prob.shape[1])  # [:30000]
            seq = viterbi_decode(hmmr, observations)  # [0:20000]
            print(len(seq))
            hets_plt = hets.iloc[idx_stacked].iloc[scnd_idx]
            hets_plt['hmm'] = seq  # [:135249]
            hets_plt = hets_plt[hets_plt.chrom != "X"]
            # hets_plt[hets_plt.hmm==1]=3

    @staticmethod
    def _rebin_to_n(y, n):
        f = interp1d(np.linspace(0, 1, len(y)), y)
        return list(f(np.linspace(0, 1, n)))

    @property
    def temporarily_removed(self):
        return set(self.low_coverage_mutations.values())

    def prepare_inputs(self):
        return


    def parse_pileup(self):

        bases = set("ATGCatgc")
        bases_snps = set(",.ATGCatgc")

        het_pileup_df = pd.read_csv(self.het_pileup,
                                 names=["Chrom", "Position", "Reference", "Count", "Pileup", "Qual", "CB", "UB"],
                                 sep="\t")

        all_sites = []
        all_full = []

        for row in het_pileup_df.iterrows():

            pile = row[1]["Pileup"]
            spl = set(list(pile))
            pile = pile.replace(",", ".")
            pile = re.sub(r"[ATGCatgc]", 'x', pile)

            # if  ">" in spl : spl.remove(">")
            # if "<" in spl: spl.remove("<")
            #cntr = Counter(pile)
            if len(set(spl).intersection(bases_snps)) > 0:
                # print (cntr,row[1]["Chrom"],row[1]["Pileup"],spl)
                res = self.parse_mpileup_row(row[1])
                if res and len(res) > 5 and (res["ref"]+res["alt"])<7000 and res["AF"]>0.1 and res["AF"]<0.9:
                    # print(res["Variant"],len(res)-4,res["AF"])
                    res["Chrom"], res["Position"], res["Reference"], res["Cell_count"] = res["Variant"].split("_") + [
                        len(res) - 4]
                    resout = {}
                    resout["Chrom"], resout["Position"], resout["Reference"], resout["Cell_count"], resout["AF"], \
                    resout["ref"], resout["alt"] = res["Chrom"], res["Position"], res["Reference"], res["Cell_count"], \
                                                   res["AF"], res["ref"], res["alt"]
                    all_sites.append(resout)
                    all_full.append(res)

        self.het_df=pd.DataFrame(all_sites)

        return all_full

    @classmethod
    def parse_mpileup_row(self,row):
        bases=set("ATGCatgc")
        bases_snps=set(",.ATGCatgc")


        #each variant
        dict_out={}
        cells=row['CB'].split(",")
        umi=row['UB'].split(",")
        calls=list(row['Pileup'])
        #if set(calls).intersection(bases_snps)==0: continue
        dict_out["Variant"]="_".join(map (str,list(row[["Chrom","Position","Reference"]])))
        dict_out["ref"]=0
        dict_out["alt"]=0
        not_unique=set()
        for cell,um,var in zip(cells,umi,calls):
            #if var not in bases_ref: continue

            if (cell,um) not in  not_unique:
                if var in bases:
                    if cell not in dict_out: dict_out[cell]=[0,0]


                    dict_out[cell][1]+=1 #add count alt
                    dict_out["alt"]+=1
                if var in [",","."]:
                    if cell not in dict_out: dict_out[cell]=[0,0]

                    dict_out[cell][0]+=1 #add count ref
                    dict_out["ref"]+=1
            not_unique.add( (cell,um))
        if (dict_out['alt']+dict_out['ref'])==0:return None
        dict_out["AF"]=dict_out['alt']/(dict_out['alt']+dict_out['ref'])

        #if len(dict_out)>1: dict_full.append(dict_full)

        return dict_out


class CategoricalNew(tfd.Categorical):
  def _log_prob(self, k):
    logits = self.logits_parameter()
    #if self.validate_args:
    #  k = distribution_util.embed_check_integer_casting_closed(
      #    k, target_dtype=self.dtype)
    #k, logits = _broadcast_cat_event_and_params(
     #   k, logits, base_dtype=dtype_util.base_dtype(self.dtype))
    #print(logits,k)
    return np.array(logits).T#[:,k]