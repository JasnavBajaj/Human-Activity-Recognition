# Human Activity Recognition with OpenCV

This project implements human activity recognition in video streams using a pre-trained deep learning model and OpenCV's DNN module. It processes input from either a video file or a webcam, classifies human activities, and overlays the predicted activity on each frame. Optionally, the processed video can be saved to disk.

## Features

- **Real-time Recognition:** Process video streams frame-by-frame to recognize human activities.
- **Flexible Input:** Supports both upload video files and live webcam streams.
- **GPU Support:** Optionally utilize GPU acceleration (with CUDA) for faster processing.
- **Output Video:** Save the annotated video output to a file.
- **Easy Configuration:** Control model path, labels, input/output, display options, and GPU usage via command-line arguments.

## Requirements

- **Python 3.x**
- **OpenCV:** Ensure your OpenCV installation includes the DNN module. For GPU support, use a CUDA-enabled build.
- **NumPy**
- **imutils**

_Optional: [Git Large File Storage (Git LFS)](https://git-lfs.github.com/) if your model file exceeds GitHub's 100 MB limit._

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/JasnavBajaj/Human-Activity-Recognition.git
   cd Human-Activity-Recognition-with-OpenCV
   ```
2. **(Optional) Create and Activate a Virtual Environment:**

   ```bash
   python -m venv myenv
   source myenv/bin/activate      # On Windows use: myenv\Scripts\activate
   ```
3. **Install Dependencies:**

   ```bash
   pip install numpy opencv-python imutils
   ```

## Usage 

Run the script from the command line by specifying the required arguments:

```bash
python activity_recognition.py -m path/to/model -c path/to/classes.txt [-i path/to/input_video] [-o path/to/output_video] [-d 1] [-g 0]
```

## Command-Line Arguments
- `-m, --model`
  **Required** - Path to the pre-trained deep learning model file.
