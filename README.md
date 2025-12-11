# DRACO
Diffusion-based Robot Arm Controller


## Demo Videos

[drawing hello](videos/hello_draw.mov)

[hello](videos/hello.mov)

[drawing a smiley face](videos/smiley_draw.mov)

[smiley face](videos/smiley.mov)

[drawing a car](videos/car_draw.mov)

[car](videos/car.mov)

[drawing a scenery](videos/scenery_draw.mov)

[scenery](videos/scenery.mov)

## Execution

### 0. Clone The Repo
First you can cd in the $src$ folder in a ROS2 workspace and clone our repo by `git clone https://github.com/skaraen/DRACO.git`.

### 1. Train The Flow Matching Model
Given that we cannot upload the trained model because of the file size is more than 200MiB. If you want to use our repo, we recommand you to train the model first using our data in the folder $data$ by running `python3 flow_matching_model/train_fl_pose.py` after cd to $DRACO$. After this, we strong recommand you to move the model after 1,000 epochs' training with the path `DRACO/flow_matching_model/checkpoint_epoch_1000.pt` so that you don't need to change anything in the other scripts.

### 2. Run The Whole Pipeline
After moving the model after 1,000 epochs' training with the path `DRACO/flow_matching_model/checkpoint_epoch_1000.pt`, you can just run `bash pipeline.sh` after cd to $DRACO$.

### 3. Note On The Performance
One thing need to notice is that, if your internet is lagging, the drawing on the canvas is affected by this and will have much less points of the drawing than it should have. So please make sure your internet is good if you want to get a relatively precise picture.

### 4. Note On The Scripts

