import numpy as np
import argparse
import imutils
import sys
import cv2

def parse_arguments():
    """Parses command-line arguments."""
    argv = argparse.ArgumentParser()
    argv.add_argument("-m", "--model", required=True, help="specify path to pre-trained model")
    argv.add_argument("-c", "--classes", required=True, help="specify path to class labels file")
    argv.add_argument("-i", "--input", type=str, default="", help="specify path to video file")
    argv.add_argument("-o", "--output", type=str, default="", help="path to output video file")
    argv.add_argument("-d", "--display", type=int, default=1, help="to display output frame or not")
    argv.add_argument("-g", "--gpu", type=int, default=0, help="whether or not it should use GPU")
    return vars(argv.parse_args())

def load_activity_labels(path):
    """Loads activity labels."""
    return open(path).read().strip().split("\n")

def load_model(model_path, use_gpu):
    """Loads the deep learning model."""
    print("Loading the deep learning model for human activity recognition")
    gp = cv2.dnn.readNet(model_path)
    
    if use_gpu > 0:
        print("Setting preferable backend and target to CUDA...")
        gp.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        gp.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return gp

def get_video_stream(input_path):
    """Gets the video stream."""
    print("Accessing the video stream...")
    return cv2.VideoCapture(input_path if input_path else 0)

def initialize_writer(output_path, frame, fps):
    """Initializes the video writer."""
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (frame.shape[1], frame.shape[0]), True)
    return None

def process_frames(vs, model, labels, sample_duration, sample_size):
    """Processes frames for activity recognition."""
    frames = []
    originals = []
    
    for _ in range(sample_duration):
        grabbed, frame = vs.read()
        if not grabbed:
            print("[INFO] No frame read from the stream - Exiting...")
            sys.exit(0)
        originals.append(frame)
        frame = imutils.resize(frame, width=400)
        frames.append(frame)
    
    blob = cv2.dnn.blobFromImages(frames, 1.0, (sample_size, sample_size), (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)
    
    model.setInput(blob)
    outputs = model.forward()
    label = labels[np.argmax(outputs)]
    return originals, label

def display_and_write_frames(originals, label, writer, display):
    """Displays and writes frames with activity labels."""
    for frame in originals:
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if display > 0:
            cv2.imshow("Activity Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
        
        if writer is not None:
            writer.write(frame)
    return True

def main():
    args = parse_arguments()
    ACT = load_activity_labels(args["classes"])
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112
    
    model = load_model(args["model"], args["gpu"])
    vs = get_video_stream(args["input"])
    fps = vs.get(cv2.CAP_PROP_FPS)
    writer = None

    while True:
        originals, label = process_frames(vs, model, ACT, SAMPLE_DURATION, SAMPLE_SIZE)
        
        if args["output"] != "" and writer is None:
            writer = initialize_writer(args["output"], originals[0], fps)
        
        if not display_and_write_frames(originals, label, writer, args["display"]):
            break

    vs.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
