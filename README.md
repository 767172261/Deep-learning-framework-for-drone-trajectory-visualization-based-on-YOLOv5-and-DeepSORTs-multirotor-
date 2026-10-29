基于YOLOv5和DeepSORT的多旋翼无人机检测与跟踪系统 Deepsort算法跟踪来实现无人机运动轨迹可视化
@[toc]

yolov5+deepsort实现基于视频的多旋翼无人机检测与跟踪跟踪、
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e03454dcc43d4db6b371cc3113026a67.png)

运行原理：主要步骤
1.YOIOV5无人机视觉检测
1.1训练无人机数据集
1.2无人机检测结果
2.Deepsort跟踪
2.1算法原理
2.2无人机跟踪现实
2.3无人机运动轨迹可视化
2.4目标质心点保存
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9524fdbb2ee14ab0a530e06f37e6f123.png)

基于YOLOv5和DeepSORT的多旋翼无人机检测与跟踪系统，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1e5f8573438d475db703d15e812f81c8.png)

### 1. YOLOv5无人机视觉检测

#### 1.1 训练无人机数据集

首先，确保已经安装了YOLOv5所需的依赖项，并且同学的环境已经配置好。

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

然后，准备你的无人机数据集都可以了。同学，假设，假设你的数据集结构如下：

```
dataset/
    images/
        train/
        val/
    labels/
        train/
        val/
    data.yaml
```

`data.yaml` 文件内容示例：

```yaml
train: ../dataset/images/train
val: ../dataset/images/val

nc: 1  # 类别数量
names: ['drone']  # 类别名称
```

接下来，同学们，使用YOLOv5进行训练：

```bash
python train.py --img 640 --batch 16 --epochs 100 --data dataset/data.yaml --weights yolov5s.pt --cache
```

#### 1.2 无人机检测结果

训练完成后，你可以使用训练好的模型进行推理：

```bash
python detect.py --source dataset/images/val --weights runs/train/exp/weights/best.pt --conf 0.25
```

### 2. DeepSORT跟踪

#### 2.1 算法原理

DeepSORT 是一种用于多目标跟踪的算法，结合了卡尔曼滤波器和匈牙利算法来关联检测框并生成轨迹。其主要步骤包括：

1. **特征提取**：从检测框中提取外观特征。
2. **卡尔曼滤波**：预测目标的位置和速度。
3. **关联匹配**：使用匈牙利算法将检测框与跟踪对象进行匹配。
4. **更新轨迹**：根据匹配结果更新每个目标的轨迹。

#### 2.2 无人机跟踪现实

为了在视频中实现无人机跟踪，同学你需要集成YOLOv5和DeepSORT。以下是完整的代码示例：

```python
import cv2
import torch
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import non_max_suppression, scale_coords

# 加载YOLOv5模型
weights = 'runs/train/exp/weights/best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights, map_location=device)
model.to(device).eval()

# 加载DeepSORT
cfg = get_config()
cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT, max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

# 视频路径
video_path = 'mov7.mp4'

# 创建视频捕获对象
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv5进行检测
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    for i, det in enumerate(pred):  # detections per image
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            # 使用DeepSORT进行跟踪
            outputs = deepsort.update(det.cpu(), frame)

            # 绘制跟踪结果
            for output in outputs:
                bbox_2d = output[:4]
                id = output[-1]
                cv2.rectangle(frame, (int(bbox_2d[0]), int(bbox_2d[1])), (int(bbox_2d[2]), int(bbox_2d[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {id}", (int(bbox_2d[0]), int(bbox_2d[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Drone Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### 2.3 无人机运动轨迹可视化

为了可视化无人机的运动轨迹，你可以记录每个目标的历史位置并在每一帧上绘制轨迹线：

```python
tracks = {}

for output in outputs:
    bbox_2d = output[:4]
    id = output[-1]
    center = ((bbox_2d[0] + bbox_2d[2]) // 2, (bbox_2d[1] + bbox_2d[3]) // 2)

    if id not in tracks:
        tracks[id] = []
    tracks[id].append(center)

    for i in range(1, len(tracks[id])):
        point1 = tracks[id][i - 1]
        point2 = tracks[id][i]
        cv2.line(frame, point1, point2, (0, 0, 255), 2)
```

#### 2.4 目标质心点保存

为了保存目标的质心点，你可以将它们写入一个文本文件：

```python
with open('centroid_points.txt', 'w') as f:
    for id, points in tracks.items():
        for point in points:
            f.write(f"ID: {id}, Point: ({point[0]}, {point[1]})\n")
```

### 总结

构建完整的基于YOLOv5和DeepSORT的多旋翼无人机检测与跟踪系统。实时检测无人机，跟踪无人机的运动轨迹并保存质心点信息。
