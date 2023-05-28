# Fine Tuning
Fine-tuning is a technique that allows us to train a model for a specific task using a pre-trained model that has learned general features from a large dataset. For example, suppose we want to create a model that can classify different types of flowers. Instead of training a model from scratch using thousands of images of flowers, we can use a pre-trained model that has been trained on a large dataset of natural images, such as ImageNet. This pre-trained model already knows how to recognise basic shapes, colours, textures, and patterns that are common in natural images. We can then fine-tune this model by replacing the last layer with a new layer that has the number of classes we want (e.g., roses, tulips, sunflowers, etc.) and training it on a smaller dataset of flower images. This way, the model will learn to adjust its weights and biases to fit the new task, while keeping most of the features learned from the previous task.

## The Benefits of Fine-Tuning in Vision Classification with Fastai

Fine-tuning is a very useful in vision classification problems as it allows you to leverages pre-trained models to achieve better performance with less training time. The main benefits of fine tuning can be seen below.

### 1. Transfer Learning

Fine-tuning enables transfer learning, allowing you to leverage the knowledge learned by pre-trained models on large-scale datasets, such as ImageNet. These models have already learned useful features and patterns that can be generalized to different vision classification tasks. By fine-tuning, you can adapt these pre-trained models to new datasets, and achieve improved results.

### 2. Reduced Training Time

Training deep learning models from scratch can be computationally expensive and time-consuming, especially with large-scale datasets. Fine-tuning mitigates this issue by starting with a pre-trained model, which has already learned low-level features. This initialization speeds up the training process as the model only needs to adjust the parameters to the specific dataset to be able to be used on the model.

### 3. Improved Generalization

Fine-tuning allows models to generalize better to new data by leveraging pre-trained models' learned representations. The pre-trained models have already learned common features, such as edges, textures, and object parts, that are beneficial for vision tasks in general. Fine-tuning enables the model to adapt these generic features to the specific characteristics of the target dataset, leading to improved generalization performance.

## How to apply fine-tuning in Fastai

In Fastai, the process of fine-tuning a vision classification model typically involves the following steps:

1. **Data Preparation**: Prepare your dataset using Fastai's `ImageDataLoaders` or `DataBlock` APIs.

2. **Model Creation**: Use a pre-defined architecture (e.g., ResNet, DenseNet) within the creation of your `vision_learner`.

3. **Tune the model**: Using the `fine_tune()` method of `vision_learner`, we can adjust the weights of the pre-trained model to work on our particular dataset.


---
## Example
```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```
Output:
epoch	| train_loss | valid_loss	| error_rate |	time
0	| 0.606847	| 0.776637	| 0.240000	| 00:09
epoch	| train_loss	| valid_loss	| error_rate	| time
0	| 0.114463	| 0.045450	| 0.026667	| 00:03
1	| 0.065541	| 0.009315	| 0.000000	| 00:02
2	| 0.043300	| 0.007967	| 0.000000	| 00:02
