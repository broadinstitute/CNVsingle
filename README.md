# CNVsingle
single cell/nuclei allele specific copy number


Usage:

cnv=CNVsingle("sc.het_pileup.tsv",adata=data) # load per cell allele counts tsv in pileup format

cnv._preprocess_cn_data_cl(normal_clusters_id)

cnv.plot_cn_profile(cnv,cluster=str(cl))

hmm=CNVsingle.run_full_hmm(cnv,str(cl),norm_cl)
CNVsingle.plot_hmm_result(cnv, hmm)

CNVsingle.plot_tiny_hmm_result(cnv, hmm, title="Cluster"+str(cl_idx))

#per cell heatmap results
res=CNVsingle._per_cell_cn_data(cnv,normal_clusters_id)
