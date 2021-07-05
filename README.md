<h2>Setting up</h2>

<p>

First clone repo and run this command inside project folder:<br/>

'''
docker build -t tf-seg-rapida. 
'''

Then run the image and mount directories for dataset and weights.

'''
docker run -it \
  --mount src="/path/to/images",target=/home/ubuntu/dataset_seg/images_prepped_train/,type=bind \
  --mount src="/path/to/annotations/",target=/home/ubuntu/dataset_seg/annotations_prepped_train/,type=bind \
  --mount src="/path/to/weights/",target=/home/ubuntu/trained_weights/,type=bind \
  --gpus all \
  tf-seg-rapida
'''

</p>