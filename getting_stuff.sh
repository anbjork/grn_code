
# Did not rerun this as script after, so not fully tested
# In case the exact links expire, you can find the web page to 
# download from by following links from the publication.
# URL to human interface downloads page at time of writing:
# https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387

# # Yeah, this did not work. Claude thinks I need a browser session for it
# # to work. Verified that file sizes are right in browser, then
# # (3.13_glob) anbjork@eris:~/shortcuts/replogle_33/data$ cp -r replogle/ ~/projects/replogle_round_2/data/
# # to get then here from a previous download. Not sure how I did it the first time.
# # Either I circumvented this code manually, or figshare has changed behavior since.
# wget https://plus.figshare.com/ndownloader/files/35773217
# mkdir -p data/replogle
# mv 35773217 data/replogle/K562_gwps_normalized_bulk_01.h5ad
#
# wget https://plus.figshare.com/ndownloader/files/35774443
# mv 35774443 data/replogle/K562_gwps_raw_bulk_01.h5ad



# Same here, if links expire you can find your way following 
# links from the paper

mkdir -p data/beeline/networks
mkdir data/beeline/data

wget -O data/beeline/networks/BEELINE-Networks.zip 'https://zenodo.org/records/3701939/files/BEELINE-Networks.zip?download=1'
wget -O data/beeline/data/BEELINE-data.zip 'https://zenodo.org/records/3701939/files/BEELINE-data.zip?download=1'

cd data/beeline/data/
unzip BEELINE-data.zip
cd ../networks/
unzip BEELINE-Networks.zip
cd ../../../


# Remove unused synthetic and curated beeline data, just to clean up
# output of tree data
rm -r data/beeline/data/BEELINE-data/inputs/Curated/
rm -r data/beeline/data/BEELINE-data/inputs/Synthetic/



# # Get the raw single cell data, ie not pseudo bulk
# wget https://plus.figshare.com/ndownloader/files/35775507
#




