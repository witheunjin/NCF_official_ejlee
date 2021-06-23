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
### `NeuMF.py`
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

* keras: 'l1l2' to 'l1_l2'
```
2021-06-23 11:31:20.855886: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
Traceback (most recent call last):
  File "NeuMF.py", line 15, in <module>
    from keras.regularizers import l1, l2, l1l2
ImportError: cannot import name 'l1l2' from 'keras.regularizers'
```
Replacing 'l1l2' to 'l1_l2'

* keras: 'Merge' to 'merge'
```
2021-06-23 11:33:59.369824: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
Traceback (most recent call last):
  File "NeuMF.py", line 18, in <module>
    from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
ImportError: cannot import name 'Merge' from 'keras.layers'
```
There is 'merge' already, so just delete 'Merge'.

* keras: 'keras.optimizers; to 'tensorflow.keras.optimizers'
```
2021-06-23 12:28:25.533745: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
Traceback (most recent call last):
  File "NeuMF.py", line 19, in <module>
    from keras.optimizers import Adagrad, Adam, SGD, RMSprop
ImportError: cannot import name 'Adagrad' from 'keras.optimizers' 
```
This error message is also for Adam, SGD, RMSprop

To fix this, replace 'keras.optimizers' to 'tensorflow.keras.optimizers' like below.
```python3
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
```

### `MLP.py`
* keras: 'activity_l2' to 'l2'
```
2021-06-23 13:29:29.738927: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
Traceback (most recent call last):
  File "NeuMF.py", line 27, in <module>
    import GMF, MLP
  File "/home/ygkim/NCF_official_ejlee/MLP.py", line 16, in <module>
    from keras.regularizers import l2, activity_l2
ImportError: cannot import name 'activity_l2' from 'keras.regularizers'
```

* keras: Delete Graph from `keras.models`
```
2021-06-23 13:31:33.508606: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
Traceback (most recent call last):
  File "NeuMF.py", line 27, in <module>
    import GMF, MLP
  File "/home/ygkim/NCF_official_ejlee/MLP.py", line 17, in <module>
    from keras.models import Sequential, Graph, Model
ImportError: cannot import name 'Graph' from 'keras.models'
```
Delete `Graph` because the latest version of keras has removed Graph module from models.

 
