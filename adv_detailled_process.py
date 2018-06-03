
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.misc     import imsave
from keras          import metrics
from PIL            import Image

from keras.models                      import Model
from keras.applications.vgg16          import VGG16
from keras.applications.vgg16          import decode_predictions
from keras.utils.np_utils              import to_categorical

import keras.backend     as K
import numpy             as np
import matplotlib.pyplot as plt


# In[2]:


import warnings

warnings.filterwarnings('ignore')


# In[3]:


def limit_mem():
    cfg                          = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config = cfg))


# In[4]:


limit_mem()


# In[5]:


def create_title(category, proba):
    return '"%s" %.1f%% confidence' % (category.replace('_', ' '), proba * 100) 


# In[6]:


def generate_title(model, array):
    prediction = model.predict(array)
    _, category, proba = decode_predictions(prediction)[0][0]
    
    return create_title(category, proba)


# In[7]:


filename = 'images/brocoli.jpg'


# # Goal
# 
# The goal of this notebook is to implement the "fast gradient sign method" presented in [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572). This method is used to modify classical samples that a deep neural network trained classification will fail to classify properly.
# 
# ![fgsm idea](http://karpathy.github.io/assets/break/breakconv.png "Fast Gradient Sign Method")
# 
# The idea of this method is to take a sample, ask the network to classify it, compute the gradient of the loss in function of the input pixels and update the picture by a small amount in the direction of the gradient. This direction is opposite to the one that would increase the score for the correct class.
# 
# # VGG loading

# In[8]:


original_pic   = Image.open(filename).resize((224, 224))
original_array = np.expand_dims(np.array(original_pic), 0)
original_pic


# We transform the image according to the VGG16 requirements. It consists in substracting the mean value for each channel and reversing the order of the channels. 

# In[9]:


imagenet_mean      = np.array([123.68, 116.779, 103.939], dtype = np.float32)
preprocess         = lambda x: (x - imagenet_mean)[:, :, :, ::-1]
deprocess          = lambda x: (x[:, :, :, ::-1] + imagenet_mean)
preprocessed_array = preprocess(original_array)
model              = VGG16()


# Now let's ask the network what he sees in the picture. 

# In[10]:


pred = model.predict(preprocessed_array)
print(*[p[1:] for p in decode_predictions(pred)[0]], sep = '\n')


# # Adversarial example generation
# 
# Now we try to confuse the network by modifying the brocoli picture. We first compute the derivatives of the loss function according to the pixels of the image.

# In[11]:


target_idx      = model.predict(original_array).argmax()
target          = to_categorical(target_idx, 1000)
target_variable = K.variable(target)
loss            = metrics.categorical_crossentropy(model.output, target_variable)
gradients       = K.gradients(loss, model.input)
get_grad_values = K.function([model.input], gradients)
grad_values     = get_grad_values([original_array])[0]


# As mentioned in the article, we will just use the signs of the derivatives.

# In[12]:


grad_signs = np.sign(grad_values)


# Now we multiply these values by a very small number and add them to the pixel values of the original image. The conversion to uint8 is necessary to get a proper display using imshow ([Stackoverflow](https://stackoverflow.com/questions/39925420/bizzare-matplotlib-behaviour-in-displaying-images-cast-as-floats)). We also clip the values to the interval [0, 255] to get a valid image.

# In[13]:


epsilon         = 4
perturbation    = grad_signs * epsilon
modified_array  = preprocessed_array + perturbation
deprocess_array = np.clip(deprocess(modified_array), 0., 255.).astype(np.uint8)


# In[14]:


title_original     = generate_title(model, preprocessed_array)
title_perturbation = generate_title(model, perturbation)
title_modified     = generate_title(model, modified_array)


# In[15]:


plt.figure(figsize = (17, 17))
plt.subplot(1, 3, 1)
plt.imshow(original_array[0])
plt.title(title_original)
plt.subplot(1, 3, 2)
plt.imshow(perturbation[0])
plt.title(title_perturbation)
plt.subplot(1, 3, 3)
plt.imshow(deprocess_array[0])
plt.title(title_modified)


# In[16]:


print('mean value of perturbation:', perturbation.mean())
pred = model.predict(modified_array)
print(*[p[1:] for p in decode_predictions(pred)[0]], sep = '\n')

