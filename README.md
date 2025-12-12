# DRACO
Flow matching-based Robot Arm Controller

## Project description

The goal of this project is to enable a robot arm to reproduce user-drawn traces in real time using a flow matching generative model. A user draws a 2D trace in a simple GUI, and the system converts it into smooth joint-angle trajectories that allow the robot arm to draw the same trace on a whiteboard. This project is interesting because it combines real-time human input, generative modeling, and robot control. Traditional inverse kinematics and trajectory smoothing methods often produce jerky motion or fail for curved paths. Vision-based approaches depend heavily on lighting and camera calibration. Flow matching provides a fast, data-driven way to generate continuous trajectories directly in joint space, which makes it well-suited for real-time drawing. When the user draws a stroke in the GUI, the robot moves its arm to recreate the same motion with a marker.

<img width="664" height="368" alt="Screenshot 2025-12-11 at 4 12 27 PM" src="https://github.com/user-attachments/assets/df57845a-6336-498b-9800-8d0130b2dac2" />


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

## Prior Research and Identified Gaps

Recent advances in diffusion models have shown remarkable success in robotic manipulation, particularly in trajectory planning tasks [1], and they can effectively learn complex robot manipulation behaviors by representing action distributions as conditional diffusion models [2]. Diffusion models stand out with their ability to model multi-modal distributions and their robustness to high-dimensional input and output spaces [1], making them well-suited for generating smooth robot trajectories. Prior work, such as Motion Planning Diffusion, has demonstrated that diffusion models can effectively encode trajectory distributions and sample collision-free paths for robot arms [3]. Traditional approaches to robot-arm task completion typically rely on a combination of inverse kinematics and trajectory smoothing techniques [4]. Also, recent researcher found that deep reinforcement learning can have the ability to map the visual inputs to robot’s actions [5] as well, which shows that their effectiveness in contact-rich manipulation tasks. 

Although there has been significant research on the application of diffusion models in robotics, several limitations remain regarding generalizability. Most diffusion-based systems rely heavily on the quality and diversity of their training data, which constrains their ability to handle outlier or unseen scenarios [1]. While expanding the dataset can improve generalization, it proportionally increases the computational cost of training. 

Another major limitation lies in the inference speed of diffusion models. Their iterative denoising process requires multiple sampling steps, making real-time control and continuous motion generation computationally expensive. This latency challenge restricts their applicability in time-sensitive robotic tasks, highlighting the need for faster generative methods such as flow matching, which replaces multi-step denoising with a single continuous transformation. Flow matching greatly reduces sampling time, making it far more suitable for interactive applications like real-time trajectory generation and robot drawing.

## System Architecture
### 1. Canvas Drawing Interface

**Purpose**: Capture 2D drawing strokes from user input

**Key Files**:
- `canvas/canvas.py`

**Functionality**:
- Provides a Tkinter-based GUI for drawing on a 400×400 pixel canvas
- Captures mouse movements as 2D points `(x, y)` in canvas coordinates (centered at origin)
- Organizes points into segments (continuous strokes)
- Saves each segment as a separate `.npz` file with shape `(N, 2)` where N is the number of points

**Key Code Components**:
- `Whiteboard` class: Main GUI application
- `start_draw()`, `draw()`, `stop_draw()`: Mouse event handlers that record points
- `_calculate_centered_coords()`: Converts screen coordinates to centered coordinate system
- `save_poses()`: Saves segments to `traces/{trace_name}/XXX_cposes.npz`

**Output**: Canvas pose files (`*_cposes.npz`) containing 2D drawing points

---

### 2. Coordinate Transformation Module

**Purpose**: Transform 2D canvas coordinates to 3D real-world end-effector poses

**Key Files**:
- `pose/pose/point_scaler.py`

**Functionality**:
- Reads canvas pose files (`*_cposes.npz`) containing 2D points
- Uses LiDAR to detect the drawing board distance
- Transforms 2D canvas points to 3D real-world poses (7D: `[x, y, z, qx, qy, qz, qw]`)
- Accounts for marker length offset and board orientation
- Saves transformed poses as `*_rposes.npz` files

**Key Code Components**:
- `PointScaler` class: ROS2 node that performs coordinate transformation
- `lidar_callback()`: Receives LiDAR scan to detect board distance
- `find_board()`: Determines board distance using LiDAR at angle 0°
- `compute_real_coords()`: 
  - Scales canvas coordinates (200 pixels) to real-world dimensions (0.07m)
  - Projects points onto the detected board plane
  - Applies marker length offset (`M = 0.06m`) along the direction vector
  - Computes quaternion orientation pointing along the direction from base to target point
- `write_points()`: Saves transformed poses to `*_rposes.npz`

**Output**: Real-world pose files (`*_rposes.npz`) with shape `(N, 7)` containing `[x, y, z, qx, qy, qz, qw]`

---

### 3. Flow Matching Model Architecture

**Purpose**: The flow matching model learns a continuous vector field that maps noise to valid joint-angle configurations for points on the whiteboard. We collect training data by sampling joint angles on the Turtlebot arm and recording the resulting 3D claw positions. Because the data come from real robots, the model learns to account for mechanical noise, minor inaccuracies, and other effects that a pure inverse kinematics simulation would not capture. During inference, the model generates joint angles with lower latency than diffusion models, enabling real-time drawing.

**Key Files**:
- `flow_matching_model/train_fl_pose.py` (model definition)
- `transformer/core.py` (attention mechanism)
- `transformer/modules.py` (encoder layers)
- `transformer/__init__.py` (model factory)

**Architecture**: Transformer Encoder

**Key Components**:

#### 3.1 ConditionedVelocityNet (`train_fl_pose.py:81-162`)
- **Input Projections**:
  - `joint_proj`: Projects joint angles (4D) → `d_model` (512D)
  - `pose_proj`: Projects pose (7D) → `d_model` (512D)
  - `time_mlp`: Projects time `t` (1D) → `d_model` (512D) using SiLU activation

- **Token Construction**:
  - Creates 3 tokens: `[joint_token, pose_token, time_token]` each of shape `(batch, 1, d_model)`
  - Concatenates tokens: `(batch, 3, d_model)`

- **Transformer Encoder**:
  - 6 encoder layers (`num_layers=6`)
  - 8 attention heads (`num_heads=8`)
  - Model dimension: 512 (`d_model=512`)
  - Feed-forward dimension: 2048 (`d_ff=2048`)
  - Dropout: 0.1

- **Output Projection**:
  - Takes first token (joint token) output
  - Projects through MLP: `d_model → d_model → joint_dim` (4D)
  - Predicts velocity in joint space

#### 3.2 Transformer Components (`transformer/`)
- **MultiHeadedAttention** (`core.py:49-99`): Scaled dot-product attention with multiple heads
- **PositionwiseFeedForward** (`core.py:102-116`): Two-layer MLP with ReLU
- **EncoderLayer** (`modules.py:46-59`): Self-attention + feed-forward with residual connections
- **Encoder** (`modules.py:13-25`): Stack of N encoder layers with layer normalization


---

### 4. Flow Matching Training Pipeline

**Purpose**: Train the velocity network to learn pose → joint angle mapping

**Key Files**:
- `flow_matching_model/train_fl_pose.py`

**Training Process**:

#### 4.1 Data Loading (`train_fl_pose.py:40-57`)
- **JointToPoseDataset**: Loads `.pt` files containing `(joint, pose)` tuples
- Training data: `data/joint_to_pose_train.pt`
- Test data: `data/joint_to_pose_test.pt`

#### 4.2 Normalization (`train_fl_pose.py:60-78`)
- Computes per-dimension min/max over joint angles
- Normalizes joints to `[-1, 1]` range for training
- Stores normalization stats in checkpoint for inference

#### 4.3 Flow Matching Path (`train_fl_pose.py:383-384`)
- Uses `AffineProbPath` with `CondOTScheduler`
- Defines probability path: `x_t = (1-t)·x_0 + t·x_1` where:
  - `x_0`: Random noise (Gaussian)
  - `x_1`: Normalized joint angles
  - `t`: Random time in `[0,1]`

#### 4.4 Training Loop (`train_fl_pose.py:184-229`)
For each batch:
1. Sample random time `t ~ Uniform(0,1)`
2. Normalize joint angles: `x_1 = normalize(joint)`
3. Sample noise: `x_0 ~ N(0, I)`
4. Interpolate: `x_t = path.sample(x_0, x_1, t)`
5. Compute ground-truth velocity: `dx_t` from path
6. Predict velocity: `v_θ = model(x_t, t, pose)`
7. Compute loss: `MSE(v_θ, dx_t)`
8. Backpropagate and update

**Key Functions**:
- `train_epoch()`: One training epoch
- `evaluate()`: Test set evaluation
- `compute_joint_minmax_stats()`: Compute normalization statistics

**Output**: Trained checkpoint (`checkpoint_epoch_1000.pt`) containing:
- Model weights
- Joint normalization stats (`joint_min`, `joint_max`)
- Training metadata

---

### 5. Flow Matching Inference (Sampling)

**Purpose**: Generate joint angles from poses using trained model

**Key Files**:
- `flow_matching_model/sample_joint_from_pose.py`

**Sampling Process**:

#### 5.1 Model Loading (`sample_joint_from_pose.py:196-223`)
- Loads checkpoint and rebuilds `ConditionedVelocityNet`
- Wraps model in `VelocityModelWrapper` for flow_matching solver compatibility
- Creates `ODESolver` with wrapped model

#### 5.2 ODE Integration (`sample_joint_from_pose.py:268-278`)
- **Initialization**: Start from random noise `x_0 ~ N(0, I)` in normalized joint space
- **Time Grid**: `t ∈ [0, 1]` discretized into `num_steps=100` steps
- **ODE Solving**: Integrates velocity field from `t=0` to `t=1`: $\frac{dx_t}{dt}=v_\theta(x_t,t,pose)$
- **Denormalization**: Converts normalized `x_1` back to original joint space: $joint = (x_1 + 1) \times 0.5 \times scale + joint_{min}$

  **Key Functions**:
- `load_poses()`: Loads poses from `.npy` or `.npz` files
- `process_pose_directory()`: Batch processes all `*_rposes.npz` files in a trace directory
- `fm_sample()` (from `train_fl_pose.py:274-312`): Core sampling function

**Input**: Real-world pose files (`*_rposes.npz`) with shape `(N, 7)`
**Output**: Joint angle files (`*_joints.npy`) with shape `(N, 4)`

---

### 6. ROS2 Robot Control

**Purpose**: Publish joint angles to robot arm via ROS2 topics

**Key Files**:
- `pose/pose/push_the_angles.py`
- `pose/pose/reset_arm.py`

#### 6.1 Joint Angle Publisher (`push_the_angles.py`)

**Functionality**:
- Loads joint angle files (`*_joints.npy`) from trace directory
- Publishes joint angles sequentially to `/tb{ROS_DOMAIN_ID}/target_joint_angles` topic
- Inserts home pose transitions between segments for smooth movement

**Key Code Components**:
- `AnglePublisher` class: ROS2 node for publishing joint angles
- `__init__()`: 
  - Loads all `*_joints.npy` files from trace directory
  - Prepends/appends home pose (`[0, -1.494, 0.508, 1.033]`) before/after each segment
  - Creates 10-step home pose transitions for smooth motion
- `publish_next_angle()`: Timer callback (10 Hz) that publishes next joint angle set
- Publishes `ArmJointAngles` messages with 4 joint values

  
#### 6.2 Arm Reset (`reset_arm.py`)

**Functionality**:
- Simple ROS2 node that publishes home pose once
- Used at the end of pipeline to reset arm position

**Key Code Components**:
- `ResetRobot` class: Publishes home pose on initialization
- `reset_arm()`: Publishes `[0, -1.494, 0.508, 1.033]` joint angles


## Challenges

The biggest challenge was calibration. If the robot thought the board was even a few millimeters closer or further than it actually was, the marker would either not write at all or get crushed against the wall. We had to carefully tune the marker length offset in point_scaler.py and rely on the LiDAR data to get this right. Another challenge was that the Flow Matching model takes time to run because it has to solve an ODE for every single point, so we could not do this in real time.

## Future Work
In the future, we would like to to use the camera in the system. The robot would back up and see what it has drawn and correct its own mistakes if the line is too faint or in the wrong place. We also want to speed up the model inference. If we can make the Flow Matching faster, we could let the user draw and have the robot copy them instantly, rather than waiting for the files to process. Currently, we limit the drawing area to keep the model stable. We would want to retrain the Flow Matching model on a larger dataset that covers the full reach of the robot arm. This would let us to increase the size of the projected canvas and create much larger detailed drawings.

## Takeaways
- Generative Control: Flow Matching works. It finds smooth, natural robot poses that standard math formulas often miss.

- Physical Precision: We learned that tiny physical errors, like the marker being 1mm too short or slightly tilted, can completely ruin a drawing or cause the robot to fully miss the board and not draw at all.

- Speed Trade-off: Flow Matching creates smoother motion, but it is slow. Because it has to solve a complex equation for every single point in the drawing, it takes much longer to calculate than standard formulas.

## References

[1] Rosa Wolf, Yitian Shi, Sheng Liu, and Rania Rayyes. Diffusion models for robotic manipulation: a survey. Frontiers in Robotics and AI, Volume 12 - 2025, 2025.
[2] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion, 2024.
[3] Joao Carvalho, An T. Le, Mark Baierl, Dorothea Koert, and Jan Peters. Motion planning diffusion: Learning and planning of robot motions with diffusion models, 2024.
[4] Youdong Chen, Ling Li, and Xudong Ji. Smooth and accurate trajectory planning for industrial robots. Advances in Mechanical Engineering, 6:342137, 2014.
[5] Jiang Hua, Liangcai Zeng, Gongfa Li, and Zhaojie Ju. Learning for a robot: Deep
reinforcement learning, imitation learning, transfer learning. Sensors, 21(4), 2021.



