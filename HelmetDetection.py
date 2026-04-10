from tkinter import *
import tkinter
from tkinter import filedialog, messagebox
import numpy as np
import cv2 as cv
import os
import pytesseract as tess
from tensorflow.keras.models import model_from_json

# If using Windows, set your Tesseract path here:
tess.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Import YOLO detection module (ensure this exists in your project)
from yoloDetection import detectObject, displayImage

main = tkinter.Tk()
main.title("Helmet & Number Plate Detection")
main.geometry("850x750")

global filename
global loaded_model
global class_labels
global cnn_model
global cnn_layer_names
global option
global frame_count
global frame_count_out

frame_count = 0
frame_count_out = 0
option = 0

confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

labels_value = []

# Load label mappings
with open("Models/labels.txt", "r") as file:
    for line in file:
        labels_value.append(line.strip())

# Load CNN model for number plate recognition
with open('Models/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    plate_detecter = model_from_json(loaded_model_json)
plate_detecter.load_weights("Models/model_weights.h5")

# Load YOLO classes
classesFile = "Models/obj.names"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load YOLO detection model
modelConfiguration = "Models/yolov3-obj.cfg"
modelWeights = "Models/yolov3-obj_2400.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

def loadLibraries():
    """Load YOLOv3 model"""
    global class_labels, cnn_model, cnn_layer_names
    with open('yolov3model/yolov3-labels', 'r') as f:
        class_labels = f.read().strip().split('\n')
    print(f"Loaded {len(class_labels)} YOLOv3 labels")
    cnn_model = cv.dnn.readNetFromDarknet('yolov3model/yolov3.cfg', 'yolov3model/yolov3.weights')
    layer_names = cnn_model.getLayerNames()
    cnn_layer_names = [layer_names[i - 1] for i in cnn_model.getUnconnectedOutLayers()]

def upload():
    """Upload an image"""
    global filename
    filename = filedialog.askopenfilename(initialdir="bikes", title="Select Image")
    if filename:
        messagebox.showinfo("Selected", f"Loaded file: {filename}")

def detectBike():
    global option
    option = 0
    indexno = 0
    label_colors = (0, 255, 0)
    try:
        image = cv.imread(filename)
        image_height, image_width = image.shape[:2]
    except:
        messagebox.showerror("Error", "Invalid image path")
        return
    image, ops = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels, indexno)
    displayImage(image, 0)
    option = 1 if ops == 1 else 0

def drawPred(classId, conf, left, top, right, bottom, frame, option):
    global frame_count
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    label_name, _ = label.split(':')
    if label_name == 'Helmet' and conf > 0.5:
        if option == 0 and conf < 0.9:
            cv.putText(frame, "Helmet Not Detected", (10, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            img = cv.imread(filename)
            img = cv.resize(img, (64, 64))
            im2arr = np.array(img).reshape(1, 64, 64, 3).astype('float32') / 255
            preds = plate_detecter.predict(im2arr)
            predict = np.argmax(preds)
            textarea.insert(END, f"Helmet Missing! Number plate: {labels_value[predict]}\n")
        else:
            cv.putText(frame, "Helmet Detected", (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return True

def postprocess(frame, outs, option):
    frameHeight, frameWidth = frame.shape[:2]
    classIds, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i
        box = boxes[i]
        left, top, width, height = box
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height, frame, option)

def detectHelmet():
    textarea.delete('1.0', END)
    if option == 1:
        frame = cv.imread(filename)
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(frame, outs, 0)
        cv.imshow("Helmet Detection", frame)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        messagebox.showinfo("Info", "Please detect bike & person first!")

# 🔹 NEW: Video Recognition Function
import math

def videoHelmetDetect():
    videofile = filedialog.askopenfilename(initialdir="videos", title="Select Video File")
    if not videofile:
        messagebox.showerror("Error", "No video selected")
        return

    cap = cv.VideoCapture(videofile)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video file")
        return

    textarea.delete('1.0', END)
    textarea.insert(END, "🎥 Starting video processing... please wait.\n")
    main.update()

    # Video info
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Output video writer
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out_path = "output_detected.avi"
    video_writer = cv.VideoWriter(out_path, fourcc, fps, (width, height))

    skip_rate = 5  # process every 5th frame
    frame_no = 0
    detected_plates = set()
    tracked_vehicles = []  # stores (x, y, status)

    # GUI progress
    progress_label = Label(main, text="Progress: 0%", font=('times', 13, 'bold'), bg='light coral')
    progress_label.place(x=350, y=340)
    main.update()

    def recognize_number_plate(frame):
        """Detect number plate text using trained CNN model."""
        img = cv.resize(frame, (64, 64))
        im2arr = np.array(img).reshape(1, 64, 64, 3).astype('float32') / 255
        preds = plate_detecter.predict(im2arr)
        predict = np.argmax(preds)
        return labels_value[predict]

    def is_new_vehicle(x, y, tracked_list, threshold=100):
        """Check if a new vehicle is far from all previous detections."""
        for (px, py, _) in tracked_list:
            dist = math.hypot(px - x, py - y)
            if dist < threshold:
                return False
        return True

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1

        # Skip frames to speed up
        if frame_no % skip_rate != 0:
            video_writer.write(frame)
            continue

        blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))

        frameHeight, frameWidth = frame.shape[:2]
        classIds, confidences, boxes = [], [], []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    w = int(detection[2] * frameWidth)
                    h = int(detection[3] * frameHeight)
                    left = int(center_x - w / 2)
                    top = int(center_y - h / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, w, h, center_x, center_y])

        indices = cv.dnn.NMSBoxes(
            [b[:4] for b in boxes], confidences, confThreshold, nmsThreshold
        )

        for i in indices:
            i = i
            left, top, w, h, cx, cy = boxes[i]
            classId = classIds[i]
            label = classes[classId]
            conf = confidences[i]

            # Skip duplicate detections for same area
            if not is_new_vehicle(cx, cy, tracked_vehicles):
                continue

            tracked_status = "Helmet Worn" if label.lower() == "helmet" and conf > 0.7 else "Helmet Missing"
            tracked_vehicles.append((cx, cy, tracked_status))

            # Draw bounding box
            color = (0, 255, 0) if tracked_status == "Helmet Worn" else (0, 0, 255)
            cv.rectangle(frame, (left, top), (left + w, top + h), color, 2)
            cv.putText(frame, tracked_status, (left, top - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Log detection in interface
            if tracked_status == "Helmet Worn":
                textarea.insert(END, f"🟢 Vehicle Detected: Helmet Worn (Frame {frame_no})\n")
            else:
                textarea.insert(END, f"🔴 Vehicle Detected: Helmet Missing (Frame {frame_no})\n")
                # Try number plate recognition only when helmet missing
                try:
                    y1 = min(top + h + 80, frameHeight)
                    x1 = max(left - 30, 0)
                    x2 = min(left + w + 30, frameWidth)
                    y2 = min(y1, frameHeight)
                    plate_roi = frame[top + h:y1, x1:x2]
                    if plate_roi.size > 0:
                        plate_text = recognize_number_plate(plate_roi)
                        if plate_text not in detected_plates:
                            detected_plates.add(plate_text)
                            textarea.insert(END, f"🚗 Number Plate: {plate_text}\n")
                except Exception as e:
                    print("Plate detection skipped:", e)

            textarea.see(END)
            main.update()

        video_writer.write(frame)

        # Update progress
        if frame_no % (skip_rate * 5) == 0:
            progress = int((frame_no / total_frames) * 100)
            progress_label.config(text=f"Progress: {progress}% ({frame_no}/{total_frames} frames)")
            main.update()

    cap.release()
    video_writer.release()

    # Summary
    progress_label.config(text="✅ Processing complete!")
    textarea.insert(END, "\n✅ Video processing completed!\n")
    textarea.insert(END, f"Output saved to: {out_path}\n\n")
    textarea.insert(END, "🧾 Summary of Detected Number Plates:\n")
    for plate in detected_plates:
        textarea.insert(END, f" - {plate}\n")
    main.update()

    messagebox.showinfo("Done", "Processing completed! Showing processed video...")

    # Show processed video
    processed_video = cv.VideoCapture(out_path)
    screen_width = main.winfo_screenwidth()
    screen_height = main.winfo_screenheight()

    while True:
        ret, frame = processed_video.read()
        if not ret:
            break
        h, w, _ = frame.shape
        scale = min(screen_width / w, screen_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_frame = cv.resize(frame, (new_w, new_h))
        cv.imshow("Processed Video (Press Q to Close)", resized_frame)
        if cv.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    processed_video.release()
    cv.destroyAllWindows()


def exitApp():
    main.destroy()

# ================== GUI ==================
font = ('times', 16, 'bold')
title = Label(main, text='Helmet & Number Plate Detection System', bg='lavender blush', fg='DarkOrchid1', font=font)
title.pack(pady=10)

font1 = ('times', 14, 'bold')
Button(main, text="Upload Image", command=upload, font=font1).place(x=200, y=100)
Button(main, text="Detect Motor Bike & Person", command=detectBike, font=font1).place(x=200, y=150)
Button(main, text="Detect Helmet", command=detectHelmet, font=font1).place(x=200, y=200)
Button(main, text="Detect from Video", command=videoHelmetDetect, font=font1, bg="lightblue").place(x=200, y=250)
Button(main, text="Exit", command=exitApp, font=font1, bg="lightcoral").place(x=200, y=300)

font2 = ('times', 12, 'bold')
textarea = Text(main, height=15, width=70, font=font2)
textarea.place(x=30, y=370)

loadLibraries()
main.config(bg='light coral')
main.mainloop()
