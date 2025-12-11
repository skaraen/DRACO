# DRACO
Flow matching-based Robot Arm Controller

## Project description

DRACO connects digital sketching to physical robotics. The system allows a user to draw on a computer screen, and a robot with 4 joints copies the drawing onto a whiteboard using a marker.

The project is unique because it replaces traditional Inverse Kinematics (IK) with Flow Matching. Instead of using standard formulas to calculate angles, we use this machine learning technique to generate the robot's motion. The system converts 2D screen pixels into real-world targets and uses the model to guide the arm. 

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

**Purpose**: Neural network that predicts joint-angle velocity conditioned on pose

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




