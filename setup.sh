
cd src/grn_code/
cp pipeline_configuration_template.py pipeline_configuration.py
cd -

git clone git@bitbucket.org:sonnhammergrni/genesnake.git

cd genesnake
pip install -e .
cd -

cd src/grn_code/data_simulation
bash setup.sh
cd -

pip install numpy pandas matplotlib anton_util h5py

pip install -e .

