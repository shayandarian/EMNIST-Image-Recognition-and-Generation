This project is from 2024.<br/>
Topics: Convolutional Neural Networks (CNNs) and Generative Adversial Networks (GANs).<br/><br/>
Since the original .ipynb file with all the outputs displayed was 55.8 MB in size, and GitHub only allows uploads with a maximum size of 25 MB, the uploaded .ipynb file has all outputs cleared.<br/>
A 31 page PDF file is also uploaded showing some of the outputs, but not everything is visible in this PDF file either.<br/>
If you would like to see the project with all of its outputs on display, download the .ipynb file and the ["emnist_letter.npz"](https://drive.google.com/file/d/1waU2u-d4joIGRzuZHVessHdb66-pUg7m/view?usp=sharing) dataset, upload the .ipynb file to Google Colab, add the dataset, and then run each section in Google Colab.
# Datasets:
The Extended MNIST or [EMNIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset) expands on the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) commonly used as a benchmark, adding handwritten letters as well as additional samples of handwritten digits.<br/>
There are several "splits" of the data by various characteristics. This project used the "EMNIST Letters" dataset, which contained values split into 27 classes, one unused (class 0) and one for each letter in the English alphabet.<br/>
Some classes in this dataset can be challenging to recognize, because each class contains images of both upper-case and lower-case letters. For example, while 'C' and 'c' are very similar in appearance, 'A' and 'a' are quite different.<br/>
The file ["emnist_letter.npz"](https://drive.google.com/file/d/1waU2u-d4joIGRzuZHVessHdb66-pUg7m/view?usp=sharing) contains EMNIST Letters in a format that can be opened with the "numpy.load()" method.<br/>
The data contains six arrays: 'train_images', 'train_labels', 'validate_images', 'validate_labels', 'test_images', and 'test_labels'.<br/>
The values have been adjusted from the original EMNIST dataset in order to match the MNIST examples included with Keras:
  - The images have been transposed and scaled to floating point.
  - The labels have been one-hot encoded.
  
While portions of the EMNIST dataset are available in a variety of variations and formats from other sources, this project uses the data in the "emnist_letter.npz "file linked above.

# Tasks
## 1. Dense Neural Network:
  - A dense feed-forward network was constructed to classify images from EMNIST Letters. The network met the specifications of having at least three hidden layers, the appropriate number of inputs for the dataset, and appropriate activation functions, loss function, and number of outputs for the task.
  - The performance of this deep neural network on the test set was compared with the performance of the OPIUM-based classifier described in the EMNIST paper.
## 2. Convolutional Neural Network:
  - A convolutional neural network was constructed for the same task. The architecture of the network was similar to the dense neural network that was mentioned in task 1, with three primary hidden layers (two convolutional layers and one dense layer). Additional layers for processes such as pooling, dropout, and batch normalization were also included.
  - The performance of this convolutional neural network was compared with the performance of the dense neural network mentioned in task 1.
## 3. TensorBoard:
  - Even with a GPU or TPU, the training process for the convolutional neural network was significantly slower than for the dense neural network. This meant that experiments took longer, and mistakes would be costly.
  - While plotting a learning curve when training has finished can help diagnose problems, ideally we want to to see updates during the training process. In order to avoid dead-ends while adjusting and tuning the model, TensorFlow includes the [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) tool and the [TensorBoard notebook extension](https://www.tensorflow.org/tensorboard/tensorboard_in_notebooks).
  - While the examples in the links above show Keras models, [PyTorch supports TensorBoard as well](https://docs.pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html).
  - TensorBoard support was added to some of the models, and the TensorBoard extension was added to the .ipynb file in order to visualize the training process.
  - If you get a 403 error when trying to use TensorBoard in Google Colab, you may need to [enable third-party cookies](https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox/65221220#65221220).
## 4. Generative Adversarial Network:
  - A Deep Convolutional Generative Adversarial Network (GAN) was constructed to generate new letter images based on images from the EMNIST Letters dataset. The generated images were compared with real images from the original dataset.
  - For Keras users, the [DCGAN to generate face images](https://keras.io/examples/generative/dcgan_overriding_train_step/) tutorial may be helpful to look at.
  - For PyTorch users, the [DCGAN Tutorial](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) may be helpful to look at.
