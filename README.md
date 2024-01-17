# pytorch-CNN-pneumonia-detection

The objective of this work is to find a model capable of detecting and classifying between healthy patients and patients with bacterial or viral pneumonia. For the model creation process, Transfer Learning will be employed to apply well-known and pretrained state-of-the-art architectures, assess the results, and choose the model that performs best.

For this project, I choose [PyTorch](https://pytorch.org/), an open-source framework for Python developed by Facebook's AI Research Lab (FAIR) and based on the Torch library, offering an ecosystem of tools and libraries that facilitate experimentation with neural networks using graphics processing units (GPUs).

The dataset to be used for the development of the classification model can be obtained from [Kaggle](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images) and contains a set of X-ray radiograph images from pediatric patients aged one to five years.


## Overall Objective:

To build a CNN model capable of achieving a high level of classification using a set of chest X-ray images, distinguishing among:

* Cases without pneumonia.
* Cases with viral pneumonia.
* Cases with bacterial pneumonia.

## Specific objectives: 

* Deepen the knowledge of the application of machine learning techniques for image classification.
* Research and use a Python library or framework for the creation of image classification models.
* Understand the preprocessing of data for models dealing with images.
* Applying various methods to optimize and improve models.
* Apply Transfer Learning to leverage feature extraction from a model trained on a large number of images, evaluate, and choose the model that best suits our classification problem.


## References

* https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
* LeCun, Yann; Léon Bottou; Yoshua Bengio; Patrick Haffner (1998). ["Gradient-based
learning applied to document recognition"](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).
* Shetty, A.A.; Hegde, N.T.; Vaz, A.C.; Srinivasan, C.R. Deep Learning Methodologies
for Diagnosis of Respiratory Disorders from Chest X-ray Images: A Comparative
Study. Comput. Sci. Math. Forum 2022, 2, 20. [en línia] [consulta: 27 de març de
2022] Disponible a: https://doi.org/10.3390/IOCA2021-10900
