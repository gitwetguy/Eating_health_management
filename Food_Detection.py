import tensorflow as tf
import tensorflow_hub as hub
# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import cv2
import pandas as pd

# For measuring the inference time.
import time

# Print Tensorflow version
print("Tensorflow Version: {}".format(tf.__version__))
# Check available GPU devices.
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())

## Example use

### Helper functions for downloading images and for visualization.

#Visualization code adapted from [TF object detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py) for the simplest required functionality.

def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)


def download_and_resize_image(url, new_width=256, new_height=256,
                              display=False):
  _, filename = tempfile.mkstemp(suffix=".jpg")
  response = urlopen(url)
  image_data = response.read()
  image_data = BytesIO(image_data)
  pil_image = Image.open(image_data)
  pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
  pil_image_rgb = pil_image.convert("RGB")
  pil_image_rgb.save(filename, format="JPEG", quality=90)
  print("Image downloaded to %s." % filename)
  if display:
    display_image(pil_image)
  return filename

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=10,
                               display_str_list=()):
  """Adds a bounding box to an image."""
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
             (left, top)],
            width=thickness,
            fill=color)

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = top + total_display_str_height
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]:
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                    (left + text_width, text_bottom)],
                   fill=color)
    draw.text((left + margin, text_bottom - text_height - margin),
              display_str,
              fill="black",
              font=font)
    text_bottom -= text_height - 2 * margin

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.5):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
                              25)
  except IOError:
    print("Font not found, using default font.")
    font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                     int(100 * scores[i]))
      color = colors[hash(class_names[i]) % len(colors)]
      image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
      draw_bounding_box_on_image(
          image_pil,
          ymin,
          xmin,
          ymax,
          xmax,
          color,
          font,
          display_str_list=[display_str])
      np.copyto(image, np.array(image_pil))
  return image

## Apply module

#Load a public image from Open Images v4, save locally, and display.

#Pick an object detection module and apply on the downloaded image. Modules:
#* **FasterRCNN+InceptionResNet V2**: high accuracy,
#* **ssd+mobilenet V2**: small and fast.


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def run_detector(detector, path):
    img = load_img(path)

    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()

    result = {key:value.numpy() for key,value in result.items()}

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time-start_time)

    image_with_boxes = draw_boxes(
      img.numpy(), result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])

    display_image(image_with_boxes)
    return result

def crop_roi_food(path,res,index,conf=.5):
    if res["detection_scores"][index] >= conf:
        ymin, xmin, ymax, xmax = tuple(res["detection_boxes"][index])
        img = load_img(path)
        im_width = img.shape[1]
        im_height = img.shape[0]
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
        obj_image = img[int(top):int(bottom),int(left):int(right)]
        obj_image = np.array(obj_image)
        obj_image = cv2.resize(obj_image, (250, 250), interpolation=cv2.INTER_AREA)
        
        plt.imshow(obj_image)
        plt.show()
        return  obj_image

if __name__=="__main__":
    
    import glob
    import pandas as pd
    #test_list = glob.glob("./test_img/*.jpg")
    test_list= ["./test_img/20151127_120446.jpg",
                "./test_img/20151127_120755.jpg",
                "./test_img/20151127_122010.jpg"]
    
    try:
        import tensorflow_hub as hub
        import tensorflow as tf
        module_handle = "D:/pythonwork/Eating_health_management/faster_rcnn_openimages_v4_inception_resnet_v2_1"
        ResNet_V2_50 = 'https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5'

        print("Loading {} Model...".format(module_handle.split("/")[-1]))
        detector = hub.load(module_handle).signatures['default']
        print("Load {} Model Successfully".format(module_handle.split("/")[-1]))

        print("Loading {} Model...".format(ResNet_V2_50.split("/")[-3]))
        model_ResNet = tf.keras.Sequential([
        hub.KerasLayer(ResNet_V2_50, trainable = False, input_shape = (250,250,3), name = 'Resnet_V2_50'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(101, activation = 'softmax', name = 'Output_layer')
        ])
        model_ResNet.load_weights("./best_model/20220117-032102_Resnet_V2_50_best_weight.hdf5")
        print("Load {} Model Successfully".format(ResNet_V2_50.split("/")[-3]))



    except:
        print("Load Model Failed")


    class_df = pd.read_csv("dict.csv")
    while True:
        
        if len(test_list) == 0:
            
            print("Waiting...")
       
        else:
            for i,path in enumerate(test_list[:]):
                obj_list = []
                print("Inference No.{} Food Image".format(i+1))
                res = run_detector(detector, path)
                plt.show()
                obj_len = res["detection_scores"][res["detection_scores"]>=0.5].shape[0]
                for i in range(obj_len):
                    obj_list.append([crop_roi_food(path,res,i)])
                obj_arr = np.concatenate(obj_list)
                classfi_res = model_ResNet.predict(obj_arr)
                for i in range(classfi_res.shape[0]):
                    
                    print(class_df[class_df["value"]==np.argmax(classfi_res[i])]["Class name"].values[0],": ",round(classfi_res[i,np.argmax(classfi_res[i])]*100,3),"%")
                    #print("pasta",": ",round(classfi_res[i,np.argmax(classfi_res[i])]*100,3),"%")
                print("\n")
            test_list = []

