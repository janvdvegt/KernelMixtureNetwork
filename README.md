# KMN

A Kernel Mixture Network implementation based on [Ambrogioni et al. 2017](https://arxiv.org/abs/1705.07111) with some minor tweaks 
(kernel center clustering and trainable scales). 
We provide notebooks with both a low-level implementation in [TensorFlow](https://www.tensorflow.org/), 
as well as a plug-and-play estimator class with [Keras](https://keras.io/) and [Edward](edwardlib.org).
For more technical details, see Jan van der Vegt's blog post on [Kernel Mixture Networks](https://janvdvegt.github.io/2017/06/07/Kernel-Mixture-Networks.html) 
or ["How to obtain advanced probabilistic predictions for your data science use case"](http://www.bigdatarepublic.nl/kernel-mixture-networks/) 
for a high-level summary.

# KernelMixtureNetwork Class

This class API allows you to plug in your own network together with the placeholder for the input, and uses kernels based on your training data to condition probability densities based on your input. What this class also does what is not discussed in the paper is allow you to train the bandwidth of your kernels. Currently only Gaussian kernels are supported but the class is easily extended. It is not meant as a package but just a reference on how to use this technique using TensorFlow, Edward and Keras.
