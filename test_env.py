import cv2, torch, ultralytics, pymavlink
from ultralytics import YOLO

print("OpenCV", cv2.__version__, "| torch", torch.__version__,
      "| ultralytics", ultralytics.__version__, "| CUDA:", torch.cuda.is_available())
model = YOLO("models/yolov8n.pt")
r = model("https://ultralytics.com/images/bus.jpg")[0]
print("Detected:", [model.names[int(c)] for c in r.boxes.cls])
cv2.imshow("Step 1 test - press any key", r.plot()); cv2.waitKey(0); cv2.destroyAllWindows()