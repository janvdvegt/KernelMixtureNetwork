# KMN

A Kernel Mixture Network implementation based on [Ambrogioni et al. 2017](https://arxiv.org/abs/1705.07111) with some minor tweaks 
(kernel center clustering and trainable scales). 
We provide notebooks with both a low-level implementation in [TensorFlow](https://www.tensorflow.org/), 
as well as a plug-and-play estimator class with [Keras](https://keras.io/) and [Edward](edwardlib.org).
For more technical details, see Jan van der Vegt's blog post on [Kernel Mixture Networks](https://janvdvegt.github.io/2017/06/04/Kernel-Mixture-Networks.html) 
or ["How to obtain advanced probabilistic predictions for your data science use case"](http://www.bigdatarepublic.nl/kernel-mixture-networks/) 
for a high-level summary.
