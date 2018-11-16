
# coding: utf-8

# In[1]:


from keras.applications.inception_resnet_v2 import InceptionResNetV2
import sys
import os
from keras.layers import *
from keras.optimizers import *
from keras.applications import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as k
from keras import optimizers
import tensorflow as tf
import os
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=0.4):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(get_session())

# fix seed for reproducible results (only works on CPU, not GPU)
seed = 9
np.random.seed(seed=seed)
tf.set_random_seed(seed=seed)

# hyper parameters for model
nb_classes = 2  # number of classes
based_model_last_block_layer_number = 1     # value is based on based model selected.
img_width, img_height = 352, 288  # change based on the shape/structure of your images
batch_size = 8  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).
nb_epoch = 10  # number of iteration the algorithm gets trained.
learn_rate = 1e-2  # sgd learning rate
momentum = 0.9  # sgd momentum to avoid local minimum
rotation_range = 15  # how aggressive will be the data augmentation/transformation
shear_range = 0.2
zoom_range = 0.2
cval = 0.2



data_dir = 'sample'
train_data_dir = 'sample/train'  # Inside, each	 class should have it's own folder
validation_data_dir = 'sample/validation'  # each class should have it's own folder
model_path = '/data/mzy/pl_overload_detection/model'


# Pre-Trained CNN Model using imagenet dataset for pre-trained weights
base_model = InceptionResNetV2(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(base_model.input, predictions)


for layer in base_model.layers:
    layer.trainable = False

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=rotation_range,
                                   shear_range=shear_range,
                                   zoom_range=zoom_range,
                                   cval=cval,
                                   horizontal_flip=False,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

os.makedirs(os.path.join(os.path.abspath(train_data_dir), '../preview'), exist_ok=True)



train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')



validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size=(img_width, img_height),
                                                              batch_size=batch_size,
                                                              class_mode='categorical')


model.compile(optimizer='nadam',
              loss='categorical_crossentropy',  # categorical_crossentropy if multi-class classifier
              metrics=['accuracy'])


top_weights_path = os.path.join(os.path.abspath(model_path), 'temp1.h5')


callbacks_list = [ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),EarlyStopping(monitor='val_acc', patience=3, verbose=0)]


model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.samples,
                    epochs=nb_epoch / 2,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples,
                    callbacks=callbacks_list)

for layer in model.layers[:based_model_last_block_layer_number]:
    layer.trainable = False
for layer in model.layers[based_model_last_block_layer_number:]:
    layer.trainable = True

model.compile(optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

final_weights_path = os.path.join(os.path.abspath(model_path), 'temp2.h5')
callbacks_list = [
    ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=3, verbose=0)
]

# fine-tune the model
model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.samples,
                    epochs=nb_epoch,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples,
                    callbacks=callbacks_list)
# save model
model.save(model_path + '/res.h5') # compile the model with a SGD/momentum optimizer

k.clear_session()