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

* TypeError
```
TypeError: ('Keyword argument not understood:', 'init')
```
`GMF.py`

Before
```
prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)
```
After
```
prediction = Dense(1, activation='sigmoid', name='prediction')(predict_vector)
```

`MLP.py`

Before 
```
prediction = Dense(1, activation='sigmoid',init='lecun_uniform',name='prediction')(vector)
```
After
```
prediction = Dense(1,activation='sigmoid',name='prediction')(vector)
```

`NeuMF.py`

Before
```
prediction = Dense(1,activaiton='sigmoid',init='lecun_uniform',name="prediction")(predict_vector)
```
After
```
prediction = Dense(1,activation='sigmoid',name="prediction")(predict_vector)
```

```
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  init = init_normal, W_regularizer = l2(reg_mf), input_length=1)
```

```
2021-06-23 14:00:35.220002: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
NeuMF arguments: Namespace(batch_size=256, dataset='ml-1m', epochs=100, layers='[64,32,16,8]', learner='adam', lr=0.001, mf_pretrain='', mlp_pretrain='', num_factors=8, num_neg=4, out=1, path='Data/', reg_layers='[0,0,0,0]', reg_mf=0, verbose=1) 
Load data done [11.1 s]. #user=6040, #item=3706, #train=994169, #test=6040
Traceback (most recent call last):
  File "NeuMF.py", line 181, in <module>
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
  File "NeuMF.py", line 76, in get_model
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user', init=init_normal, W_regularizer = l2(reg_mf), input_length=1)
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/layers/embeddings.py", line 111, in __init__
    super(Embedding, self).__init__(**kwargs)
  File "/home/ygkim/.local/lib/python3.8/site-packages/tensorflow/python/training/tracking/base.py", line 522, in _method_wrapper
    result = method(self, *args, **kwargs)
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/engine/base_layer.py", line 323, in __init__
    generic_utils.validate_kwargs(kwargs, allowed_kwargs)
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/utils/generic_utils.py", line 1134, in validate_kwargs
    raise TypeError(error_message, kwarg)
TypeError: ('Keyword argument not understood:', 'init')
```
SOLUTION: init to embeddings_initializer | W_regularizer to embeddings_regularizer
```
tf.keras.layers.Embedding(
    input_dim, output_dim, embeddings_initializer='uniform',
    embeddings_regularizer=None, activity_regularizer=None,
    embeddings_constraint=None, mask_zero=False, input_length=None, **kwargs
)
```
BEFORE
```
prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(predict_vector)
```
`tf.keras.layers.Dense` - Parameters
```python3
tf.keras.layers.Dense(
    units, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```
* 'init' to 'kernel_initializer'

RESULT : `TypeError: init_normal() got an unexpected keyword argument 'dtype'`

```
Traceback (most recent call last):
  File "NeuMF.py", line 181, in <module>
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
  File "NeuMF.py", line 83, in get_model
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/engine/base_layer.py", line 945, in __call__
    return self._functional_construction_call(inputs, args, kwargs,
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/engine/base_layer.py", line 1083, in _functional_construction_call
    outputs = self._keras_tensor_symbolic_call(
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/engine/base_layer.py", line 816, in _keras_tensor_symbolic_call
    return self._infer_output_signature(inputs, args, kwargs, input_masks)
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/engine/base_layer.py", line 854, in _infer_output_signature
    self._maybe_build(inputs)
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/engine/base_layer.py", line 2601, in _maybe_build
    self.build(input_shapes)  # pylint:disable=not-callable
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/utils/tf_utils.py", line 258, in wrapper
    output_shape = fn(instance, input_shape)
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/layers/embeddings.py", line 141, in build
    self.embeddings = self.add_weight(
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/engine/base_layer.py", line 615, in add_weight
    variable = self._add_variable_with_custom_getter(
  File "/home/ygkim/.local/lib/python3.8/site-packages/tensorflow/python/training/tracking/base.py", line 810, in _add_variable_with_custom_getter
    new_variable = getter(
  File "/home/ygkim/.local/lib/python3.8/site-packages/keras/engine/base_layer_utils.py", line 115, in make_variable
    return tf.compat.v1.Variable(
  File "/home/ygkim/.local/lib/python3.8/site-packages/tensorflow/python/ops/variables.py", line 260, in __call__
    return cls._variable_v1_call(*args, **kwargs)
  File "/home/ygkim/.local/lib/python3.8/site-packages/tensorflow/python/ops/variables.py", line 206, in _variable_v1_call
    return previous_getter(
  File "/home/ygkim/.local/lib/python3.8/site-packages/tensorflow/python/ops/variables.py", line 199, in <lambda>
    previous_getter = lambda **kwargs: default_variable_creator(None, **kwargs)
  File "/home/ygkim/.local/lib/python3.8/site-packages/tensorflow/python/ops/variable_scope.py", line 2612, in default_variable_creator
    return resource_variable_ops.ResourceVariable(
  File "/home/ygkim/.local/lib/python3.8/site-packages/tensorflow/python/ops/variables.py", line 264, in __call__
    return super(VariableMetaclass, cls).__call__(*args, **kwargs)
  File "/home/ygkim/.local/lib/python3.8/site-packages/tensorflow/python/ops/resource_variable_ops.py", line 1584, in __init__
    self._init_from_args(
  File "/home/ygkim/.local/lib/python3.8/site-packages/tensorflow/python/ops/resource_variable_ops.py", line 1722, in _init_from_args
    initial_value = initial_value()
TypeError: init_normal() got an unexpected keyword argument 'dtype'

```

`GMF.py`
 - Before
```python3
62	MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding', init = init_normal, W_regularizer = l2(regs[0]), input_length=1)
```
`GMF.py` - After ('init' to 'embeddings_initializer' | 'W_regularizer' to 'embeddings_regularizer')

```python3
 62     MF_Embedding_User = Embedding(input_dim = num_users, output_dim = latent_dim, name = 'user_embedding', embeddings_initializer = init_normal, embeddings_regularizer = l2(regs[0]), input_length=1)
 63     MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = latent_dim, name = 'item_embedding', embeddings_initializer = init_normal, embeddings_regularizer = l2(regs[1]), input_length=1)
 ```
 
`NeuMF.py` - After ('init' to 'embeddings_initializer' | 'W_regularizer' to 'embeddings_regularizer')
```python3
 75     # Embedding layer
 76     MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user', embeddings_initializer=init_normal, embeddings_regularizer = l2(reg_mf), input_length=1)
 77     MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item', embeddings_initializer=init_normal, embeddings_regularizer = l2(reg_mf), input_length=1)
 78 
 79     MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = "mlp_embedding_user", embeddings_initializer=init_normal, embeddings_regularizer = l2(reg_layers[0]), input_length=1)
 80     MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'mlp_embedding_item', embeddings_initializer=init_normal, embeddings_regularizer = l2(reg_layers[0]), input_length=1)
```

`MLP.py` - After ('init' to 'embeddings_initializer' | 'W_regularizer' to 'embeddings_regularizer')
```
 66     MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = 'user_embedding',embeddings_initializer=init_normal, embeddings_regularizer = l2(reg_layers[0]), input_length=1)
 67     MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'item_embedding', embeddings_initializer = init_normal, embeddings_regularizer = l2(reg_layers[0]), input_length=1)
```


SAME ERROR

`MLP.py`
 - Before
 ```python3
 56 def init_normal(shape, name=None):
 57     return initializations.normal(shape, scale=0.01, name=name)
 ```
 - After
  ```python3
 56 def init_normal(shape, name=None):
 57     return initializers.normal(shape, scale=0.01, name=name)
 ```
 
 `GMF.py` - Before 
 ```python3
 54 def init_normal(shape, name=None):
 55     return initializations.normal(shape, scale=0.01, name=name)
 ```
 - After
 ```python3
  54 def init_normal(shape, name=None):
  55     return initializers.normal(shape, scale=0.01, name=name)
 ```
 
 NOT RESOLVED.. OTL
 
