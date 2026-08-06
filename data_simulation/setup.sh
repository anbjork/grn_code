
rm -rf genespider
rm -rf genesnake

git clone git@bitbucket.org:sonnhammergrni/genespider.git
cd genespider
# Changes to suitable commit;
# The one for the publication, before sneak noise change
git checkout 37ad0c97adbfe52892040de74f26d90acd9b7cce
cd ../

git clone git@bitbucket.org:sonnhammergrni/genesnake.git
cd genesnake/
# Pinning for reproducibility, since no proper versioning
git checkout 72a16a0e5b91e528880b742553f0161db79c81f6
pip install -e .
cd ../

