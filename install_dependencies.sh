sudo dpkg --configure -a
sudo apt-get -f install
sudo apt-get --fix-missing install
sudo apt-get clean
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade
sudo apt-get install build-essential
sudo apt-get install cmake
sudo apt-get install libboost-all-dev
sudo apt-get install python-dev
sudo apt-get install python-pandas
sudo apt-get install unzip
sudo apt-get clean
sudo apt-get autoremove

rm -f get-pip.py
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
rm get-pip.py

sudo pip install numpy
sudo pip install cython
sudo pip install scipy
sudo pip install scikit-learn
sudo pip install jupyter
sudo pip install xgboost
sudo pip install networkx
sudo pip install numba
sudo pip install futures

sudo easy_install joblib

sudo rm -f master.zip
sudo rm -f -r Boost.NumPy-master

wget https://github.com/ndarray/Boost.NumPy/archive/master.zip
unzip master.zip
cd Boost.NumPy-master
cmake .
./configure
make
sudo make install
cd ../
sudo rm master.zip
sudo rm -r Boost.NumPy-master

# cd /usr/lib/x86_64-linux-gnu/
# sudo ln -s libboost_python-py27.so.1.62.0 libboost_python-py27.so.1.58.0


sudo rm -f master.zip
sudo rm -f -r sparsehash-master

wget https://github.com/sparsehash/sparsehash/archive/master.zip
unzip master.zip
cd sparsehash-master
./configure
./make
sudo make install
cd ../
sudo rm master.zip
sudo rm -r sparsehash-master

wget https://github.com/hyperopt/hyperopt/archive/master.zip
unzip master.zip
cd hyperopt-master
mv hyperopt ../
cd ../
sudo rm master.zip
sudo rm -r hyperopt-master

wget https://github.com/danielhomola/boruta_py/archive/master.zip
unzip master.zip
cd boruta_py-master
sudo python setup.py install
cd ../
sudo rm master.zip
sudo rm -r boruta_py-master

wget https://github.com/numba/numba/archive/master.zip
unzip master.zip
cd numba-master
sudo python setup.py install
cd ../
sudo rm master.zip
sudo rm -r numba-master


################

# sudo pip install networkx==1.11


