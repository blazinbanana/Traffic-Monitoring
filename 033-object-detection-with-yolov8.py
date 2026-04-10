#!/usr/bin/env python
# coding: utf-8

# ## 3.3 Image Detection with YOLO

# ### Getting Ready

# Let's get ready for this lesson by importing the packages we need.

# In[1]:


import sys
from collections import Counter
from pathlib import Path

import PIL
import torch
import torchvision
import ultralytics
from IPython.display import Video
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import make_grid
from ultralytics import YOLO


# Next, we'll print out the version numbers for our libraries, including Python. We want to make sure that anyone who reviews our work knows exactly what software we used in case they want to reproduce our analysis.

# In[2]:


print("Platform:", sys.platform)
print("Python version:", sys.version)
print("---")
print("PIL version : ", PIL.__version__)
print("torch version : ", torch.__version__)
print("torchvision version : ", torchvision.__version__)
print("ultralytics version : ", ultralytics.__version__)


# ### Image detection with YOLO

# In the previous project, we solved the problem of detecting if an image contains one object from a set of objects. For example, does the image contain a hog or not. But we know that images rarely just contain one object. Images often contain different objects and multiples of the same object. Imagine an image of a traffic scene that contains cars, trucks, pedestrians, traffic signs, and more. We need to use an object detection model. Object detection is the process of identifying and locating objects in an image or video.
# 
# We won't train an object detection algorithm from scratch but instead use a pretrained model. We'll use the YOLO (You Only Look Once) model. It's popular for its speed and accuracy. Lucky for us, the `ultralytics` package contains the YOLO model. We'll use the YOLO version 8 pretrained model.

# In[3]:


yolo = YOLO(task="detect", model="yolov8s.pt")


# What classes can this pretrained model detect? That's stored in `yolo.names`.

# In[4]:


yolo.names


# It's a dictionary that maps an integer to the class label. Let's see what 5 corresponds to.

# In[5]:


yolo.names[5]


# **Task 3.3.1:** Determine the class that's assigned to integer 23?

# In[6]:


class_assigned_to_23 = yolo.names[23]
print(f"{class_assigned_to_23} corresponds to 23")


# Our task involves identifying objects from traffic video feeds. There are several objects we want to detect that are not included in the classes from the pretrained YOLO model. These classes are defined below.

# In[7]:


classes_not_in_yolo = [
    "ambulance",
    "army vehicle",
    "auto rickshaw",
    "garbagevan",
    "human hauler",
    "minibus",
    "minivan",
    "pickup",
    "policecar",
    "rickshaw",
    "scooter",
    "suv",
    "taxi",
    "three wheelers (CNG)",
    "van",
    "wheelbarrow",
]


# Let's double check that "ambulance" is not in the YOLO classes.

# In[8]:


"ambulance" not in yolo.names.values()


# **Task 3.3.2:** Double check that "army vehicle" is not in the YOLO classes.

# In[9]:


is_army_vehicle_inlcuded = "army vehicle" not in yolo.names.values()
print(is_army_vehicle_inlcuded)


# In a later lesson, we'll retrain the YOLO model to include the missing classes. For this lesson, we are OK with what's already provided. We are most interested in the first 13 classes. Those classes are objects often found in traffic. 
# 
# Let's use the YOLO model to identify objects in one frame of our video data. We'll use `Path` provided by `pathlib`.

# In[10]:


data_dir = Path("data_video", "extracted_frames")
image_path = data_dir / "frame_1050.jpg"

result = yolo(image_path)


# What is `result`?

# In[11]:


print(f"Type of result: {type(result)}")
print(f"Length of result: {len(result)}")


# Looks like it's a list of length 1. We'll explore its contents shortly.

# **Task 3.3.3:** Run the YOLO model on `frame_2575.jpg`.

# In[12]:


image_path_task = data_dir / "frame_2575.jpg"
result_task = yolo(image_path_task)

print(type(result_task))


# There's another way to use the YOLO model. It's to use the object's `.predict` method. The advantage is that it's clearer what we're doing and allows us to overwrite any default values when predicting. For example, we can control the confidence value of the resulting bounding boxes. Let's use the `.predict` method and specify a 50% threshold for the bounding box and save the results to disk as a text file.

# In[14]:


result = yolo.predict(image_path, conf=0.5, save=True, save_txt=True)


# The results are contained in the created `runs` directory.

# **Task 3.3.4:** Use the `predict` method for `frame_2575.jpg`. Make sure you use a 50% confidence threshold and save the results as a text file.

# In[15]:


result_task = yolo.predict(image_path_task, conf=0.5, save=True, save_txt=True)


# ### Results From Running YOLO

# `result[0]` contains a special object with the results of the prediction stored as attributes.

# In[16]:


result[0]


# We'll break these results down further.
# 
# `.boxes` contains the data for the bounding boxes. These bounding boxes are the main things we want from object detection. These boxes are then used to create a box around the detected objects.

# In[17]:


result[0].boxes


# We'll need to further unpack what's inside the `.boxes` attribute. The `.cls` attribute contains the classes of each of the objects detected. It's a PyTorch tensor. The length of the tensor is the number of objects detected.

# In[18]:


print(result[0].boxes.cls)
print(f"Number of objects detected: {len(result[0].boxes.cls)}")


# Recall that these numbers are mapped to the name of the classes. For example, 0 corresponds to "person".

# **Task 3.3.5:** Determine the number of detected objects in `frame_2575.jpg`.

# In[19]:


number_of_detected_objs = len(result_task[0].boxes.cls)
print(f"Number of objects detected in frame_2575.jpg: {number_of_detected_objs}")


# Now let's see what the objects we detected. The keys of `yolo.names` are integers so we'll need to cast the floats in `result[0].boxes.cls` to integers.

# In[20]:


object_counts = Counter([yolo.names[int(cls)] for cls in result[0].boxes.cls])
object_counts


# The "car" class was the most common, followed by "person".

# **Task 3.3.6:** Determine the most common class and the number of times it was detected in `frame_2575.jpg`.

# In[21]:


object_counts_task = Counter([yolo.names[int(cls)] for cls in result_task[0].boxes.cls])

most_common_class, count_of_class = object_counts_task.most_common(n=1)[0]
print(f"Most common class: {most_common_class}")
print(f"Number of detected {most_common_class}: {count_of_class}")


# Another important attribute is `.conf` which has the confidence of the detected bounding boxes. The confidence is stored in a PyTorch tensor. We should expect this tensor's length to match the number we saw earlier.

# In[22]:


print(result[0].boxes.conf)
print(f"Number of objects detected: {len(result[0].boxes.conf)}")


# **Task 3.3.7:** Check the length of the confidence tensor of `result_task` to verify this number matches to what was observed earlier.

# In[23]:


length_of_confidence_tensor = len(result_task[0].boxes.conf)
print(f"Number of objects detected: {length_of_confidence_tensor}")


# When calling `.predict`, we set the confidence threshold to 50%. That is why all values in the confidence tensor is greater than 0.5. How many of the bounding boxes have a confidence value greater than 75%? For frame `frame_1050.jpg`, that would be:

# In[24]:


number_of_confident_objects = (result[0].boxes.conf > 0.75).sum().item()
print(f"Number of objects detected with 50% confidence: {number_of_confident_objects}")


# **Task 3.3.8:** Calculate the number of objects that were detected in `frame_2575.jpg` with 75% confidence.

# In[25]:


number_of_confident_objects_task = (result_task[0].boxes.conf > 0.75).sum().item()

print(
    f"Number of objects detected in frame_2575.jpg with 50% confidence: {number_of_confident_objects_task}"
)


# The `.data` attribute contains the raw detection data. We won't be using it as there are attributes with the bounding box data in an easier to use form. `.orig_shape` is just the original shape of the input. The attribute `is_track` indicates whether object tracking has been turned on. This is useful when we want to track an object across multiple frames. What follows next are the attributes that store the processed bounding boxes. They are provided in four different forms. All these forms describe the box using four values. The different forms will help us if we are using a tool where the bound box can only be one particular format.
# 
# We'll go through all four of them.

# `.xywh` is a tensor with four columns for each row. Each row represents one box. The first and second column is the x and y coordinates of the top-left corner of the box, respectively. The third and fourth columns are width and height, respectively.

# In[26]:


result[0].boxes.xywh


# `.xywhn` is very similar to `.xywh` but these coordinates have been normalized by the image size. We can remind ourselves of the original shape with `.orig_shape`.

# In[27]:


result[0].orig_shape


# This means the image is 360 pixels high and 640 pixels wide. Let's examine one row of the normalized bounding box.

# In[28]:


result[0].boxes.xywhn[0]


# Now we can use the original shape to verify that indeed `.xywhn` is normalized.

# In[29]:


result[0].boxes.xywh[0] / torch.Tensor([640, 360, 640, 360]).to("cuda")


# That matches from what we saw earlier.

# **Task 3.3.9:** Print out the original shape of `frame_2575.jpg`.

# In[30]:


original_shape_task = result_task[0].orig_shape
print(f"Original shape of frame_2574.jpg: {original_shape_task}")


# **Task 3.3.10:** Print out the normalized `xywh` bounding box for the first object of `frame_2575.jpg`.

# In[31]:


normalized_xywh = result_task[0].boxes.xywhn
print(f"Normalized xywh bounding box for frame_2575.jpg: {normalized_xywh[0]}")


# **Task 3.3.11:** Normalize the bounding box using the original shape of the `frame_2575.jpg`.

# In[32]:


normalized_xywh_task = result_task[0].boxes.xywh[0] / torch.Tensor([640, 360, 640, 360]).to("cuda")
print(f"Normalized xywh bounding box for frame_2575.jpg: {normalized_xywh[0]}")


# The third provided bounding box form is `.xyxy`. This form contains two coordinates, the (x, y) coordinate for the top left corner and the (x, y) coordinate of the bottom right corner.

# In[33]:


result[0].boxes.xyxy


# The last form is `.xyxyn` which is the normalized form of `.xyxy`.

# In[34]:


result[0].boxes.xyxyn


# We've explored the most important attributes of the `.boxes` attribute of the returned result object. Now let's return the remaining important attributes. `.save_dir` is just the location where we've saved the resulting bounding boxes. We'll use the method `exists` of a `Path` object to make sure the location actually exists.

# In[35]:


location_of_results = Path(result[0].save_dir)

print(f"Results saved to {location_of_results}")
location_of_results.exists()


# **Task 3.3.12:** Determine the location for the results of `frame_2575.jpg`.

# In[36]:


location_of_results_task = Path(result_task[0].save_dir)
print(f"Results for frame_2575.jpg saved to {location_of_results_task}")


# Finally, `.speed` is a dictionary for the time it took to run the preprocessing, inference (prediction), and postprocessing steps. These times are measured in milliseconds. A good rule of thumb is that times less than 100 milliseconds are experienced as instantaneous.

# In[37]:


result[0].speed


# In[38]:


print(f"Total time in milliseconds: {sum(result[0].speed.values())}")


# **Task 3.3.13:** Calculate the total time object detection took for `frame_2575.jpg`.

# In[39]:


total_time = sum(result_task[0].speed.values())
print(f"Total time in milliseconds: {total_time}")


# ### Displaying the Bounding Boxes

# By saving our results, we've created an image file with the bounding boxes drawn in.

# In[40]:


Image.open(location_of_results / "frame_1050.jpg")


# Notice how each class uses a different color, the labels are displayed, along with the confidence of the bounding box.

# **Task 3.3.14:** Display image `frame_2575.jpg` with its drawn bounding boxes.

# In[43]:


Image.open(data_dir / "frame_2575.jpg")


# In[42]:


# Display image frame_2575.jpg with the bounding boxes
Image.open(location_of_results_task / "frame_2575.jpg")


# The bounding boxes were saved as a text file.

# In[44]:


with (location_of_results / "labels" / "frame_1050.txt").open("r") as f:
    print(f.read())


# The first column is the class, followed by 4 columns defining the bounding box.

# **Task 3.3.15:** Display the text file results for the bounding box for `frame_2575.jpg`.

# In[45]:


with (location_of_results_task / "labels" / "frame_2575.txt").open("r") as f:
    print(f.read())


# ### Using YOLO on Multiple Images and Video Source

# We're ready to move on to using YOLO for identifying objects across multiple images. For convenience, we'll define a function that accepts a directory of images and displays them in a grid.

# In[46]:


def display_sample_images(dir_path, sample=5):
    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path

    image_list = []
    # Sort the images to ensure they are processed in order
    images = sorted(dir_path.glob("*.jpg"))
    if not images:
        return None

    # Iterate over the first 'sample' images
    for img_path in images[:sample]:
        img = read_image(str(img_path))
        resize_transform = transforms.Resize((240, 240))
        img = resize_transform(img)
        image_list.append(img)

    # Organize the grid to have 'sample' images per row
    Grid = make_grid(image_list, nrow=5)
    # Convert the tensor grid to a PIL Image for display
    img = torchvision.transforms.ToPILImage()(Grid)
    return img


# With this function defined, let's use it for the first 25 frames we extracted from the video.

# In[47]:


display_sample_images(data_dir, sample=25)


# **Task 3.3.16:** Use `display_sample_images` to display the first ten frames in a grid.

# In[48]:


# Display the first ten images
display_sample_images(data_dir, sample=10)


# We'll create a list of the path of 25 images from the extracted frames.

# In[49]:


images_path = list(data_dir.iterdir())[:25]
images_path


# **Task 3.3.17:** Create a list of the _last_ ten frames as listed by `data_dir.iterdir()`.

# In[51]:


images_path_task = list(data_dir.iterdir())[-10:]

print(f"Number of frames in list: {len(images_path_task)}")
images_path_task


# We'll once again use `yolo.predict` but this time we'll make use of two additional arguments to control where the results are saved. By using `project` and `name`, the saved results will be in `project/name`.

# In[52]:


results = yolo.predict(
    images_path,
    conf=0.5,
    save=True,
    save_txt=True,
    project=Path("runs", "detect"),
    name="multiple_frames",
)


# In[53]:


print(results[0].save_dir)


# You can see how the output includes a summary of the results for each of the 25 frames.

# **Task 3.3.18:** Use `yolo.predict` on `images_path_task`. Save the results to `runs/detect/multiple_frames_task`.

# In[54]:


results_task = yolo.predict(
    images_path_task,
    conf=0.5,
    save=True,
    save_txt=True,
    project=Path("runs","detect"),
    name="multiple_frames_task",
    
)

print(f"\nResults from task saved to: {results_task[0].save_dir}")


# With our `display_sample_images` function, we can display the results.

# In[56]:


image = display_sample_images(results[0].save_dir, sample=25)
image


# You can see how YOLO did a good job at detecting the different objects.

# **Task 3.3.19:** Display the images with the bounding boxes with `display_sample_images` for the results generated in the previous task. Make sure to set `sample` to 10.

# In[57]:


image_task = display_sample_images(results_task[0].save_dir, sample=10)


# In[58]:


image_task


# Now let's try to use YOLO on a video source instead of the frames extracted from a video. The cell below displays the video.

# In[59]:


video_path = Path("data_video", "dhaka_traffic.mp4")
Video(video_path)


# In[75]:


Video(video_path, embed=True, mimetype="video/mp4")


# To speed things up, we're going to truncate our video and run YOLO against the truncated version. We'll use `ffmpeg`, a command line tool for video and audio editing. The part that controls the timestamps for truncation are the numbers that follow `-ss` and `-to`. The number after `-ss` is the starting timestamp and `-to` is the ending timestamp. The value `output/dhaka_traffic_truncated.mp4` is the path of the created file.

# <div class="alert alert-warning" style="color: #000">
#     <p>
#         <strong>Warning! Notebook updated!</strong>
#     </p>
#     <p>
#         In the video, the instructor saves the truncated video in the <code>data_video</code> directory, but in this notebook we will save the video in the <code>output</code> directory.
#     </p>
# </div>
# 

# In[60]:


get_ipython().system('ffmpeg -ss 00:00:00 -to 00:00:30 -y -i $video_path -c copy output/dhaka_traffic_truncated.mp4')


# In[61]:


video_truncated_path = Path("output", "dhaka_traffic_truncated.mp4")
Video(video_truncated_path)


# In[74]:


Video(video_truncated_path, embed=True, mimetype="video/mp4")


# **Task 3.3.20:** Truncate the same video as above but from the `00:00:30` to `00:01:00` timestamp and name the video `output/dhaka_traffic_truncated_task.mp4`.

# In[62]:


get_ipython().system('ffmpeg -ss 00:00:30 -to 00:01:00 -y -i $video_path -c copy output/dhaka_traffic_truncated_task.mp4')

video_truncated_path_task = Path("output", "dhaka_traffic_truncated_task.mp4")
Video(video_truncated_path_task)


# In[73]:


Video(video_truncated_path_task, embed=True, mimetype="video/mp4")


# To use YOLO on a video source, we just need to tell it the location of the video and set `stream` to True.

# In[63]:


results_video = yolo.predict(
    video_truncated_path,
    conf=0.5,
    save=True,
    stream=True,
    project=Path("runs", "detect"),
    name="video_source",
)


# Unlike before, the returned value of `yolo.predict` is a generator rather than a list. Detection happens only as we iterate over the generator, giving us control over when the actual computation takes place.

# In[64]:


for result in results_video:
    continue


# Since we saved the results, YOLO created a video. In the next section, we'll look at the video that YOLO produces.

# ### Using YOLO in the Command Line

# YOLO has a command line interface. This is great if we are working with shell scripts. You can see how its usage is very similar to what we saw earlier.

# In[65]:


get_ipython().system('yolo task=detect mode=predict conf=0.5 model=yolov8s.pt source=$video_truncated_path project="runs/detect" name="command_line" > /dev/null')


# <div class="alert alert-info" role="alert">
# If you look at the end, you'll see <code>> /dev/null</code>. This redirects the output of running YOLO from the screen to the device called null. This device is basically a black hole where it destroys anything written to it. The output that would've appeared would've been the same as we saw earlier when running YOLO in Python.
# </div>

# YOLO will create a video of the source video with the bounding boxes. Before we display the video, we'll need to convert it to an mp4 as that format provides better compression. Better compression leads to a smaller file size. The notebook environment might have issues with playing large files. Once again, `ffmpeg` is the tool to use.

# In[66]:


get_ipython().system('ffmpeg -y -i runs/detect/command_line/dhaka_traffic_truncated.avi output.mp4')


# In[67]:


Video("output.mp4")


# In[69]:


from IPython.display import Video


# In[70]:


Video("output.mp4", embed=True, mimetype="video/mp4")


# Notice how the video contains the bounding boxes on the detected objects.

# **Task 3.3.21:** Use YOLO in the command line for the truncated video. You'll need to change `source` to be `$video_truncated_path_task` and the `name` to be `command_line_task`.

# In[68]:


get_ipython().system('yolo task=detect mode=predict conf=0.5 model=yolov8s.pt source=$video_truncated_path_task project="runs/detect" name="command_line_task" > /dev/null')

# This will convert your video to mp4 and display it in the notebook
get_ipython().system('ffmpeg -y -i runs/detect/command_line_task/dhaka_traffic_truncated_task.avi output_task.mp4')
Video("output_task.mp4")


# In[71]:


Video("output_task.mp4", embed=True, mimetype="video/mp4")


# ---
# &#169; 2024 by [WorldQuant University](https://www.wqu.edu/)
