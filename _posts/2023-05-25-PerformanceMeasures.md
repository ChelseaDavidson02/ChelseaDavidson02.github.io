## Assessing Performance of a Vision Learner Model in Fastai

When training a `vision_learner` model in Fastai, it's crucial to assess its performance to understand how well it is performing on your specific data set. Fastai provides several methods and metrics to evaluate and analyze the model's performance. Here are some common approaches:

### 1. Accuracy

Accuracy is a widely used metric to assess the performance of a vision model. It measures the proportion of correctly classified images in the dataset. Fastai provides a convenient way to calculate accuracy using the `accuracy` function. Here's an example:

```python
from fastai.vision.all import accuracy

# Calculate accuracy on the validation dataset
accuracy = accuracy(vision_learner.model, dataloader.valid)
```

In the code above, `vision_learner.model` represents the trained model, and `dataloader.valid` is the validation dataset. The `accuracy` function compares the model's predictions with the true labels and calculates the accuracy score.

### 2. Confusion Matrix

A confusion matrix provides a more detailed analysis of the model's performance by showing the distribution of predicted labels against the true labels. It helps identify classes that may be more challenging for the model to classify accurately. Fastai's `plot_confusion_matrix` function enables you to generate a confusion matrix. Here's an example:

```python
from fastai.vision.all import plot_confusion_matrix

# Generate confusion matrix on the validation dataset
plot_confusion_matrix(vision_learner.model, dataloader.valid)
```
Example output:
<img width="449" alt="image" src="https://github.com/ChelseaDavidson02/ChelseaDavidson02.github.io/assets/84437493/145decaf-d0ff-4a50-afa3-c9a98f7bf76c">

### 2. t-SNE plots
A t-SNE plot is a visualization technique commonly used for high-dimensionality problems (multi-class problems) to see relationships between data points.

In a t-SNE plot, each data point is represented as a point in a two-dimensional scatter plot, with the position of the point determined by its lower-dimensional coordinates. Data points of the same class should appear clustered together with the relative distances reflecting lack of confidence in the prediction. Misclassified points are shown in amongst a cluster of another class.
Here is an example of how a t-SNE plot is created:
```python
preds, targets = learn.get_preds(dl=test_data)
preds_np = preds.numpy()
targets_np = targets.numpy()
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(preds_np)
x = tsne_embeddings[:, 0]
y = tsne_embeddings[:, 1]
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i, animal in enumerate(searches):
    animal_indices = targets_np == i
    plt.scatter(x[animal_indices], y[animal_indices], label=animal)
plt.legend()
plt.show()
```
Output:

![image](https://github.com/ChelseaDavidson02/ChelseaDavidson02.github.io/assets/84437493/b3684a44-48f8-4a47-a619-816b074b3d54)


All these methods help to better understand your models performance and are useful tools when developing a model.
