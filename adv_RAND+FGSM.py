
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.misc     import imsave
from keras          import metrics
from PIL            import Image

from keras.models                import Model
from keras.applications.vgg16    import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16    import decode_predictions
from keras.utils.np_utils        import to_categorical

import keras.backend     as K
import numpy             as np
import matplotlib.pyplot as plt

import os


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


folder  = 'images/'
epsilon = 0.03
alpha   = 0.008


# # Goal
# 
# The goal of this notebook is to implement the `RAND+FGSM` presented in [Ensemble adversarial training: Attacks and Defenses](https://arxiv.org/abs/1705.07204). This method is used to modify classical samples that a deep neural network trained classification will fail to classify properly.
# 
# The idea of this method is to take a sample, ask the network to classify it, compute a small random perbutation and the gradient of the loss in function of the input pixels and update the picture by a small amount in these directions. The random perturbation gets the image to a place where the loss function is smooth and the gradients make the classification fail. This methods works even on adversarially trained networks.
# 
# # Adversarial example generation
# 
# This notebook presents a factorized version of the generation process. For a step-by-step explanation, see the other notebook in this repository.

# In[6]:


imagenet_mean = np.array([123.68, 116.779, 103.939], dtype = np.float32)
preprocess    = lambda x: (x - imagenet_mean)[:, :, :, ::-1]
deprocess     = lambda x: (x[:, :, :, ::-1] + imagenet_mean)

def get_gradient_signs(model, original_array):
    target_idx      = model.predict(original_array).argmax()
    target          = to_categorical(target_idx, 1000)
    target_variable = K.variable(target)
    loss            = metrics.categorical_crossentropy(model.output, target_variable)
    gradients       = K.gradients(loss, model.input)
    get_grad_values = K.function([model.input], gradients)
    grad_values     = get_grad_values([original_array])[0]
    grad_signs      = np.sign(grad_values)
    
    return grad_signs
    
def pertubate_image(preprocessed_array, random_perturbation, gradient_perturbation):
    modified_array  = preprocessed_array + random_perturbation + gradient_perturbation
    deprocess_array = np.clip(deprocess(modified_array), 0., 255.).astype(np.uint8)
    
    return deprocess_array

def generate_titles(display_model, preprocessed_array, perturbation, modified_array):
    title_original     = generate_title(display_model, preprocessed_array)
    title_perturbation = generate_title(display_model, perturbation)
    title_modified     = generate_title(display_model, modified_array)
    
    return title_original, title_perturbation, title_modified

def generate_adversarial_example(pertubation_model, original_array, alpha, epsilon):
#    random_noise          = np.random.uniform(-1, 1, original_array.shape)
    random_noise          = np.random.normal(0, 1, original_array.shape)
    random_perturbation   = 127.5 * alpha * random_noise 
    gradient_signs        = get_gradient_signs(pertubation_model, original_array)
    gradient_perturbation = 127.5 * gradient_signs * (epsilon - alpha)
    modified_image        = pertubate_image(original_array, random_perturbation, gradient_perturbation)
    
    return modified_image, random_perturbation + gradient_perturbation

def load_image(filename):
    original_pic   = Image.open(filename).resize((224, 224))
    original_array = np.expand_dims(np.array(original_pic), 0)

    return original_array
    
def create_title(category, proba):
    return '"%s" %.1f%% confidence' % (category.replace('_', ' '), proba * 100) 

def generate_title(model, array):
    prediction = model.predict(array)
    _, category, proba = decode_predictions(prediction)[0][0]
    
    return create_title(category, proba)
    
def generate_adversarial_examples(folder, title, perturbation_model, display_model = None, epsilon = 0.3, alpha = 0.05):
    if not display_model:
        display_model = perturbation_model

    filenames   = os.listdir(folder)
    line_number = len(filenames)
    plt.figure(figsize = (15, 10 * line_number))
    
    for line, filename in enumerate(filenames):
        original_array               = load_image(folder + filename)
        preprocessed_array           = preprocess(original_array)    
        modified_image, perturbation = generate_adversarial_example(perturbation_model, preprocessed_array, alpha, epsilon)
        preprocess_modified          = preprocess(modified_image)
        orig_tit, pert_tit, modi_tit = generate_titles(display_model, preprocessed_array, perturbation, preprocess_modified)

        plt.subplot(line_number, 3, 3 * line + 1)
        plt.imshow(original_array[0])
        plt.title(orig_tit)
        plt.subplot(line_number, 3, 3 * line + 2)
        plt.imshow(perturbation[0])
        plt.title(pert_tit)
        plt.subplot(line_number, 3, 3 * line + 3)
        plt.imshow(modified_image[0])
        plt.title(modi_tit)
        
    plt.suptitle(title)
    plt.tight_layout(pad = 4)


# In[7]:


vgg16    = VGG16()
resnet50 = ResNet50()


# In the following figures, the left column contains the original images, the middle one the perturbation and the right one the generated adversarial sample. The result of the classification is displayed above each image.

# In[8]:


generator_parameters = {
    'folder': folder,
    'title': 'Perturbation using VGG16, classification using VGG16', 
    'perturbation_model': vgg16, 
    'display_model': vgg16,
    'epsilon': epsilon,
    'alpha': alpha
}
generate_adversarial_examples(**generator_parameters)


# Using the Fast Gradient Sign Method allows us to generate new images from existing samples that the network will fail to classify.   
