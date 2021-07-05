from keras_segmentation.models.unet import resnet50_unet
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from imgaug import augmenters as iaa
from tensorflow.compat.v1 import ConfigProto, Session
model = resnet50_unet(n_classes=3)
log_dir = "/home/ubuntu/unet_BigSet" + "/logs/fit/"
config = ConfigProto()
config.gpu_options.allow_growth = True
session = Session(config=config)
def custom_augmentation():
    return  iaa.Sequential(
        [
        #
        # Apply the following augmenters to most images.
        #
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.5),

        # iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        iaa.Sometimes(
                    0.5,
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # Same as sharpen, but for an embossing effect.
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.Add((-10, 10), per_channel=0.5),

                # Change brightness of images (50-150% of original value).
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        # iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # # Apply affine transformations to each image.
        # # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45),
            #shear=(-8, 8)
        )
    ],random_order=True)
# When using custom callbacks, the default checkpoint saver is removed
# callbacks = [
#     ModelCheckpoint(
#                 filepath="/home/ubuntu/unet_BigSet/" + ".{epoch:05d}",
#                 save_weights_only=True,
#                 verbose=True
#             ),
#     tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# ]


model.summary()
model.train(
    train_images =  "/home/ubuntu/dataset_seg/images_prepped_train/",
    train_annotations = "/home/ubuntu/dataset_seg/annotations_prepped_train/",
    checkpoints_path = "/home/ubuntu/trained_weights/" ,
    #val_images="/home/ubuntu/dataset_seg/images_prepped_test/",
    #val_annotations="/home/ubuntu/dataset_seg/annotations_prepped_test/",
    batch_size=4,
    epochs=70,
    validate=False,
    # callbacks=callbacks,
    optimizer_name="adamax"
    #do_augment=True, # enable augmentation 
    #custom_augmentation=custom_augmentation
)