# Human Activity Recognition with OpenCV

This project implements human activity recognition in video streams using a pre-trained deep learning model and OpenCV's DNN module. It processes input from either a video file or a webcam, classifies human activities, and overlays the predicted activity on each frame. Optionally, the processed video can be saved to disk.

---

## Features

- **Real-time Recognition:** Process video streams frame-by-frame to recognize human activities.
- **Flexible Input:** Supports both upload video files and live webcam streams.
- **GPU Support:** Optionally utilize GPU acceleration (with CUDA) for faster processing.
- **Output Video:** Save the annotated video output to a file.
- **Easy Configuration:** Control model path, labels, input/output, display options, and GPU usage via command-line arguments.

---

## Requirements

- **Python 3.x**
- **OpenCV:** Ensure your OpenCV installation includes the DNN module. For GPU support, use a CUDA-enabled build.
- **NumPy**
- **imutils**

_Optional: [Git Large File Storage (Git LFS)](https://git-lfs.github.com/) if your model file exceeds GitHub's 100 MB limit._

---

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
---

## Usage 

Run the script from the command line by specifying the required arguments:

```bash
python activity_recognition.py -m path/to/model -c path/to/classes.txt [-i path/to/input_video] [-o path/to/output_video] [-d 1] [-g 0]
```

---

## Command-Line Arguments
- `-m, --model`
   - **Required** - Path to the pre-trained deep learning model file.
- `-c, --classes`
   - **Required** - Path to the text file containing class labels (one per line).
- `-i, --input`
   - Optional - Path to an input video file. If omitted, the webcam is used.
- `-o, --output`
   - Optional. Path to save the output video with annotated activity labels.
- `-d, --display`
   - Optional - Set to `1` to display the output frames in real time; set to `0` to disable. Default is `1`
- `-g, --gpu`
   - Optional - Set to `1` to enable GPU processing (requires CUDA-enabled OpenCV); set to `0` to use the CPU. Default is `0`
 
  ### Example
   
   ```bash
   python activity_recognition.py -m models/resnet-34_kinetics.onnx -c models/labels.txt -i input_video.mp4 -o output_video.mp4 -d 1 -g 0
   ```
---

## How it works

1. **Argument Parsing:** - The script uses argparse to handle command-line arguments, ensuring that the model and class labels paths are provided.
2. **Loading Model and Labels:** - It reads class labels from a file and loads the deep learning model using OpenCV’s DNN module. If GPU processing is enabled, the script configures the backend and target accordingly.
3. **Video Stream Processing:** - Frames are read in batches from the video stream. Each batch is processed by resizing, converting to a blob, and passing through the model to predict the activity.
4. **Displaying and Writing Frames:** - The recognized activity label is overlayed on the frames. The frames are then displayed and/or written to an output video file if specified.

## Troubleshooting 

1. **Large Files Issue:** - If your model file is larger than 100 MB, ensure it’s managed via Git LFS. For more information, see Git LFS.
2. **GPU Errors:** - Confirm that your OpenCV build supports CUDA if you enable GPU processing `(-g 1)`.
3. **Video Capture Issues:** - If no frames are read from the video source, verify the correctness of the video file path or ensure that your webcam is functioning properly.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- OpenCV
- imutils
- Git LFS
