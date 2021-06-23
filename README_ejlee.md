# NCF_official_ejlee
This repository is from `hexiangnan/neural_collaborative_filtering`.

### Related Materials
* Paper: [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf)
* Repository: [hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering)

### Installation
* theano
```
$ sudo pip install theano
```
RESULT
```
Collecting theano
  Downloading Theano-1.0.5.tar.gz (2.8 MB)
     |████████████████████████████████| 2.8 MB 4.2 MB/s 
Requirement already satisfied: numpy>=1.9.1 in /usr/local/lib/python3.8/dist-packages (from theano) (1.20.3)
Collecting scipy>=0.14
  Downloading scipy-1.7.0-cp38-cp38-manylinux_2_5_x86_64.manylinux1_x86_64.whl (28.4 MB)
     |████████████████████████████████| 28.4 MB 533 kB/s 
Requirement already satisfied: six>=1.9.0 in /usr/lib/python3/dist-packages (from theano) (1.14.0)
Building wheels for collected packages: theano
  Building wheel for theano (setup.py) ... done
  Created wheel for theano: filename=Theano-1.0.5-py3-none-any.whl size=2668094 sha256=e3acec86cf89322a6bcf8c5899a94cf972eb4e86f33de761cff68c60dd821851
  Stored in directory: /root/.cache/pip/wheels/84/cb/19/235b5b10d89b4621f685112f8762681570a9fa14dc1ce904d9
Successfully built theano
Installing collected packages: scipy, theano
Successfully installed scipy-1.7.0 theano-1.0.5
```
* keras: `initializations` to `initializer`
```
2021-06-23 11:26:26.535104: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
Traceback (most recent call last):
  File "NeuMF.py", line 14, in <module>
    from keras import initializations
ImportError: cannot import name 'initializations' from 'keras' 
```
Replacing 'initializations' to 'initializer' in 'NeuMF.py' file.

Done.

