set -e

git clone git@bitbucket.org:sonnhammergrni/genesnake.git

cd genesnake
pip install -e .
cd -

git clone git@bitbucket.org:sonnhammergrni/genespider.git
cd genespider
# Changes to suitable commit;
# The one for the publication, before sneak noise change
# Since then, have also branched, and fixed noise for control cells
git checkout dcb208e17da01af65cfe0b92e59a822c03288361
cd ../

pip install numpy pandas matplotlib anton_util h5py

pip install -e .

