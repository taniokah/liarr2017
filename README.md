# Super Easy Way of Building Image Search with Keras

Introduction of Deep Learning for Information Retrieval Researchers in __LIARR 2017__. (11, Aug. 2017)

## 1. ABSTRACT

This paper provides detailed suggestions to create an Image Search Engine with Deep Learning.  ere are still few a empts with Deep Learning on a search engine. Here is a good idea of an extremely easy way of building an image search with Elasticsearch and Keras on Jupyter Notebook. So, it is demonstrated how an image search engine can be created where Keras is used to extract features from images, and Elasticsearch is used for indexing and retrieval. 

To demonstrate how easily this can be achieved, Jupyter notebook is used with two open source libraries, Keras and Elasticsearch.  en, the image search engine can search relevant images by images or keywords. In our approach, keras is regarded as analysis which is a process of converting an image into tokens or terms which are added to the inverted index for searching.

## 2. INSTALLATION

We have to install some libraries: Python, Elasticsearch, Keras, Jupyter Notebook, and TensorFlow. Jupyter notebook and TensorFlow are optional, because Jupyter notebook is used for my demonstration and Theano or CNTK is available for Kearas instead of TensorFlow.

My demo environment is MacBook Pro (Retina, 15-inch, Mid 2015). OS X El Capitan (version 10.11), Python 2.7.13 <Anaconda 4.3.1 (x86_64)>, Elasticsearch 5.4.1, Keras 2.0.4, Jupyter Notebook 4.2.1, and TensorFlow 1.1.0 are installed. I guess Python 3+ is OK and Elasticsearch 2+ is also OK. Of course, Linux and Windows are both OK.

### 2.1 Python

https://www.python.org/

### 2.2 Elasticsearch

https://www.elastic.co/

### 2.3 Keras

https://keras.io/

### 2.4 Jupyter Notebook

http://jupyter.org/

### 2.5 TensorFlow

https://www.tensorflow.org/

## 3. DEMONSTRATION

First of all, Kaggle dogs-vs-cats dataset should be downloaded on your machine.
The indexing function is composed of Keras and Elasticsearch, which indexes dog and cat images as a search target. Keras plays a role of feature extraction.  Then, Elasticsearch has an index for image file retrieval.

### 3.1 Download Images

It's a competition to classify whether images contain either a dog or a cat. This is easy for humans, dogs, and cats, but computers will find it a bit more difficult.
The training archive contains 25,000 images of dogs and cats. Train your algorithm on these files and predict the labels for test1.zip.

https://www.kaggle.com/c/dogs-vs-cats/data

You can download test1.zip and train.zip.

### 3.2 Load Libraries for Keras

### 3.3 Function Declaration for Keras

### 3.4 Predict Test

### 3.5 Load Libraries for Elasticsearch

### 3.6 Setting for Elasticsearch

### 3.7 Function Declaration for Search

### 3.8 Index Images

### 3.9 Search Test

<p align="left">
  <img src="search-cat.png" width="480"/>
</p>

