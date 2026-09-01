
cd src/grn_code/
cp pipeline_configuration_template.py pipeline_configuration.py
cd -

git clone https://antonbjork@bitbucket.org/sonnhammergrni/genesnake.git

cd genesnake
pip install -e .
cd -


pip install numpy pandas matplotlib anton_util h5py

pip install -e .

