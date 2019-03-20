# PASCAL VOC 
## Object Classification
### 50.039 Theory and Practice of Deep Learning

The goal of this project is to recognize objects from a number of visual object classes in realistic scenes. There are 20 object classes:
1. Person
2. Bird, cat, cow, dog, horse, sheep
3. Aeroplane, bicycle, boat, bus, car, motorbike, train
4. Bottle, chair, dining table, potted plant, sofa, tv/ monitor

Specifically for the classification task, the goal is, for each of the classes predict the presence/ absence of at least one object of that class in a test image.

## Data
We will use Pascal VOC 2012 dataset for this project and the latest version of pytorch has Pascal VOC dataset class built-in. For the purpose of this project, we will only use training set and validation set of Pascal VOC. The ground truth annotation for the dataset contains the following information,
* Class: the object class. I.e. car or person
* Bounding box: an axis-aligned rectangle specifying the extent of the object visible in the image
* View: ‘frontal’ , ‘rear’, ‘left’ or right
* Difficult: an object marked as difficult indicates that the object is considered difficult to recognize without substantial use of context.
* Truncated: an object marked as ‘truncated’ indicates that the bounding box specified for the object does not correspond to the full extent of the object.
* Occluded: an object marked as ‘occluded’ indicates that a significant portion of the  subject image is within the bounding box occluded by another object.

For our task, we regarded all ‘difficult’ marked objects as negative examples.

## Loss function

The task is multi-label classification for 20 object classes, which is analogous to creating 20 object detectors, 1 for every class. Hence we have used binary cross entropy (with logits loss) as our loss function. The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right],

    where :math:`N` is the batch size. If reduce is ``True``, then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if size\_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if size\_average} = \text{False}.
        \end{cases}



In pytorch, the loss function is torch.nn.BCEWithLogitsLoss( ). Do note that this function provides numerical stability over the sequence of sigmoid followed by binary cross entropy. 

