# Image match search using K Nearest Neighbors and Correspondence to other classes

## Problem statement

The objective of this task is to develop an algorithm that enables the comparison of images using two methods: **Nearest Neighbours** and **Correspondence to other classes**. The algorithm should be capable of identifying the most similar images from the base dataset for a given test image.

The input to the algorithm is an image, for which it is necessary to identify the images that are most similar in content within the database. The algorithm outputs the five most suitable images from the database, ordered by their degree of correspondence to the input image. This enables users to rapidly identify images with similar content for further research.

Three iterations are presented, which differ in the methods of solving the Nearest Neighbors and Correspondence to other classes algorithms:

1. In **the first iteration**, both the first and second algorithms, when analyzing the distance and proximity of two datasets, consider all classes in the base dataset, despite the class of the test image from the test class.
2. In **the second iteration**, the approaches of the two algorithms diverge significantly:
    - the *Nearest Neighbours* approach evaluates the proximity of the image to the images in the base dataset of the same class. This means that all other classes that do not match the class of the test image are not taken into account.
    - In the *Correspondence to other classes* approach, we evaluate the similarity of the test image to images in a different class. We don't evaluate the same classes as the test image.
3. In **the third iteration**, the following changes have been implemented::
    - **New images of the test dataset** have been incorporated. In the third iteration, the images of the rooms are designed to differ as much as possible from one another. Some images are ambiguous. For example, there is a photograph in the dataset that shows only a ceiling and a lamp. This is done to observe the behaviour of the algorithms and to determine whether they will identify similar signs on completely unfamiliar and dissimilar images from the base dataset.
    - **An interactive component has also been added**, allowing users to upload their own image and apply each of the methods, with the option of viewing the top five closest images from the base dataset.

## 1st Iteration

### The stages of task completion

1. **Preprocessing of the provided data**, which involves bringing them to a single format.
2. **Loading a pre-trained ResNet model** with 50 layers.
3. **Removing the last classification layers** in order to leave only the layers necessary for feature extraction.
4. **Converting images** from the test and base sets ** into tensors**  by feeding them through the ResNet model and obtaining feature vectors. Two algorithms are employed to identify the most similar images from the base set:
    - Nearest Neighbours.
    - Correspondence to other classes.
5. **The results** are returned in the form of the top five matches for each image.

### Stage 1. Preprocessing data from the base and test datasets

The base dataset was created for the purposes of the study, comprising 225 images divided into 5 classes (subfolders in the main folder). These classes were defined as follows: *"Kitchen"*, *"Living room"*, *"Bathroom"*, *"Bedroom"* and *"Wardrobe"*. A similar structure was used to create the test dataset, containing a total of 25 images, with 5 images for each class.

Images from the test dataset were used for comparison with images from the base dataset.

At this stage, the necessary directories for image preprocessing are created, including folders for image classes. The images are reduced to one size (1028 by 768 pixels) and the folders are organised.

### Stage 2. Loading the pre-trained ResNet50 model

**ResNet-50** is a deep convolutional neural network pre-trained on a large image dataset. It has a proven ability to classify images with high accuracy. This model was chosen because it has a deep architecture. This allows it to extract high-level features from images, making it good at capturing complex patterns and features.

### Stage 3. Removing the last classification layers in the model

The ResNet-50 model is used to extract features from images, not to classify them. This is done by removing the last classification layers and using the remaining layers to obtain representations (features) of images using `model = torch.nn.Sequential(*(list(model.children())[:-1])`, where:

1. `model.children()` returns a generator containing all child layers of the model.
2. `list(model.children())` converts the generator to a list so that indexing and slices can be used.
3. `[:-1]` performs a slice of the list, leaving all the elements to the last (excluding the last element).
4. `torch.nn.Sequential(*...)` creates a new sequence of PyTorch layers that will contain all the layers of the model except the last one.

This approach is often used when the last layer of the model is responsible for classification.

### Stage 4. Converting images from the test and base sets into tensors and obtaining feature vectors

Previously processed images are converted back into tensors and fed into the ResNet model to select features for comparison (see below).

`# Feature extraction
def extract_features(image_tensor):
    with torch.no_grad():
        features = model(image_tensor)
    features = features.squeeze().numpy()
    return features`

1. `image_tensor' is an image tensor that is passed as input to the model.
2. `with torch.no_grad():` - This disables the requirement of gradients during calculations. This is useful because we don't train the model, just use it to extract features.
3. `features = model(image_tensor)`: The model processes the image tensor and returns its features.
4. `features.squeeze().numpy()` - the features are compressed and converted into a NumPy array to get a one-dimensional array.

### Stage 5. Applying algorithms to find the most similar images from the base set

#### Using the Nearest Neighbor Algorithm

The **Nearest Neighbors** algorithm is a machine learning method that is used to find the nearest objects in the feature space. For a new object, the algorithm searches for the objects closest to it from the training dataset. This is based on the assumption that objects with similar attributes often belong to the same class or have similar properties.

In the Image match search task, the algorithm is used to search for the 5 most similar images from the image database. The implementation of the algorithm is presented below.

Query attributes (`query_features`), database attributes (`database_features`), database labels (`database_labels`) and the optional parameter `k`, which indicates the number of neighbors to be found (in our case - 5), are accepted as input.
    - Creates the `Nearest Neighbors` object using the `cosine` metric (cosine similarity).
    - Trains the model based on database attributes using the `fit` method.
    - Uses the `kneighbors` method to find the `k` nearest neighbors for query features.
    - Returns the distances (`distances`) and indexes (`indices`) of these nearest neighbors.

The Accuracy metric was used to determine whether the model correctly classifies the base image with which the test image is compared. The metric is displayed for each test image and the top 5 base images closest to it. Examples of the results obtained for the "Kitchen" class (Fig.1)

![kithen_example](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/kitchen_example.png)

![kithen_rank](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/kitchen_rank.png)

and for the "Bathroom" class (Fig.3)

![bathroom_example](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/bathroom_example.png)

![bathroom_rank](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/bathroom_rank.png)

**As a result** in the first approach, the proximity of the image to the images in the database of any class was evaluated. The accuracy of this comparison was then compared, so that the total accuracy indicator will always be 100%. The model correctly identified the top 5 most "close" images.

#### Using Correspondence to other classes

The 'Correspondence to other classes' method means that each image from the test set can match images from the database of other classes, not just its own. The logic of using the second approach is as follows:

1. For each image from the test set, the similarities between its features and the features of the images in the database are calculated using cosine proximity. Cosine similarity is a way of measuring how similar two things are in a multidimensional space. This is perfect for us to solve the problem.
2. Next, we calculate how many images match with other classes. We do this by dividing the number of images that match with classes other than the test class by the total number of images we have (parameter `k`).
3. The database is searched for images with the greatest similarities and the calculated accuracy of the match. The closest images are also returned.

The results for the "Bedroom" class (Fig. 4) are below.

![bedroom_example](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/bedroom_example.png)

![bedroom_rank](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/bedroom_rank.png)

and "Wardrobe" (Fig.5) achieved a 94% accuracy rate.

![wardrobe_example](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/wardrobe_example.png)

![wardrobe_rank](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/wardrobe_rank.png)

**As a result**, the second approach permits the assessment of the degree to which the image corresponds to any of the classes, as well as the measurement of the accuracy of this correspondence. It employs a matching method based on cosine similarity, in contrast to the previous approach, which utilised the nearest neighbour method with cosine distance.

## 2nd Iteration

In this iteration, we also load all the libraries and process the test and base datasets. The changes relate to the Nearest Neighbors and Correspondence to other classes methods themselves (5th and 6th stages).

### Stage 5. Applying algorithms to find image matches

#### Using the Nearest Neighbor Algorithm

This algorithm employs an approach to identifying the nearest neighbours, constrained to images from the base dataset belonging to the same class as the test image. This is achieved through the use of the additional argument `query_label`, which represents the class label of the test image. This argument is employed to limit the nearest neighbour search to images of the same class within the image database.

Consequently, we also obtain an array of image indexes from the database with the smallest distance, as well as the rendered images themselves (Fig. 6-7). In this iteration, we do not utilise Accuracy metric, as it will not yield any insights.

![bedroom_2_example](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/bedroom_2_example.png)

![bedroom_2_rank](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/bedroom_2_rank.png)

**P.S.**: It is notable that in the presented example, the algorithm selected only those images in which the camera tilt and angle were almost identical. This indicates that the algorithm was able to successfully complete its assigned task.

#### Using Correspondence to other classes

We use this method to find images that don't match the test image. This helps us understand why images are considered "close" (Fig.8-9). We also don't calculate accuracy because it's unnecessary. In this iteration, it is also unnecessary to calculate the accuracy.

![bathroom_2_example](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/bathroom_2_example.png)

![bathroom_2_rank](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/bathroom_2_rank.png)

**P.S.**: From the presented example, it can be observed that all the images, despite their initial lack of visual resemblance, share a common feature: the presence of flat surfaces, such as tables, cabinets, kitchen cabinets, and washing machines. This indicates that the algorithm is functioning as intended, identifying similar features across the images.



## 3rd Iteration

In the 3rd iteration, we removed the test and base datasets preprocessing stage. Data processing occurs at the stage of writing the algorithm and selecting feature vectors (3rd stage in the 3rd iteration code).

The base data set has been augmented to include **252 images**, without the necessity to categorise them according to distinct room classes. **New images of the test dataset** have been incorporated and designed to differ as much as possible from one another (Fig 10).

![test_examples](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/test_examples.png)


#### Using the Nearest Neighbor Algorithm

Let's investigate the results obtained using this method on new test images (Fig. 11-12) is now required.

![3rs_it_nn_ex](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/nn_rank.png)

![3rs_it_nn_rank](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/nn_example.png)

**P.S.**: It is noteworthy that in the presented example, the algorithm selected only those images in which the table is depicted (with the exception of the final image, that can be explained to an insufficiently large dataset). This indicates that the feature vector correctly determined that the objects on the test and base datasets are a table.

#### Rank User Given Image Using Nearest Neighbors

In this stage **an interactive component has also been added**, allowing users to upload their own image and apply each of the methods, with the option of viewing the top five closest images from the base dataset (Fig. 13-14).

![3rs_it_nn_user_ex](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/nn_user_rank.png)

![3rs_it_nn_user_rank](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/nn_user_example.png)

**P.S.**: Similarly, it can be observed that the kitchen set is consistently present in all the images, which serves to unite them visually.

#### Using Correspondence to other classes

Let's explore the result obtained using Correspondence to other classes method on new test images (Fig. 15-16).

![3rs_it_cor_ex](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/cor_rank.png)

![3rs_it_cor_rank](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/cor_example.png)

**P.S.**: Similarly, it can be observed that the refrigerator is a constant presence in all the images, uniting them visually.

#### Rank User Given Image Using Correspondence to other classes

The results of the experiment conducted with the user-provided image (Fig. 17-18) are presented below.

![3rs_it_cor_user_ex](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/cor_user_rank.png)

![3rs_it_cor_user_rank](https://github.com/totminaekaterina/Image-match-search/blob/main/imgs/cor_user_example.png)

**P.S.**: This is an intriguing example of how the model identified the feature vector and how the bed transformed into a sofa. However, this can be attributed to the fact that the bed depicted in the test image has a shape comparable to the dvians from the base dataset.


# Conclusion

Both algorithmic approaches are effective methods for evaluating the performance of an image matching model, despite their focus on various aspects of similarities and differences between images. While the first method focuses on identifying the most similar images in the database without taking into account their class, the second method also takes into account the correspondence of classes between images, which provides additional information about the similarity.

The significance of this code extends beyond the scope of indoor interiors, as its methodology can be applied to various fields.
