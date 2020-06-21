# VLAD-implementation-vs-BoVW-on-CIFAR-10
Implementation of VLAD and build KNN, logistic regression model to classify 10 categories from CIFAR-10 dataset.
Results compared with that of BoVW.

The first classification technique is using BoVW (Bag of visual words) approach. 
• Convert the RGB image to a gray image.
• Use SIFT/SURF technique to get the key points and descriptors.
• As we have variable no.of descriptors for each image to get a fixed vector
to perform k means algorithm on that descriptor sets of the image with n
clusters and compute histogram for n centers.
• The frequency list of these n centers is the final feature vector.

The reasons for low accuracy with BoVW:
• Under fitting i.e., the feature vector, is not rich enough to capture features of all
ten classes.
• This is one of the reasons for better performance binary classification over ten
class classification.
• The small size of the image causes fewer descriptors/key points of the image;
these descriptors of the image give rise to the feature vector, which gives the
frequency of each cluster.

The next classification technique is VLAD (Vector of Locally Aggregated
Descriptors), which is an extension to BOVW.

The vectors obtained using this method are similar to the Fisher vectors. But the
aggregated representation of vectors is more simple to compute.These are the steps
followed for classification using VLAD.
• Convert the RGB image to a gray image.
• Use SIFT/SURF technique to get the key points and descriptors.
• After we obtain all descriptors of the images, we perform K-means clustering on
the descriptor list. Here each data point is d dimensional vector (i.e., length of the
descriptor)
• We, then, subtract each data point with the center of its cluster and then add
descriptors of the image which belong to the same label. This gives k no.of d
dimensional vectors for each image. We get a k*d vector as a feature vector for each
image. Hence we call it VLAD the vector of locally aggregated descriptors.

