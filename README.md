# undergraduate_project2

## Data Training
* LabelIMG Tool
* 3 classes : kickboard, person, helmet (with 331 images)

## Algorithm
1. Person on kickboard?
  * Manhattan Distance를 사용하여 특정 거리 안으로 들어오면 킥보드에 탑승하고 있다고 판단
  * 이때 특정 거리는 kickboard의 size를 기준으로 함. (킥보드 가로/2 + 킥보드 세로)
  * If |center(kickboard)-center(person}|<kickboard size: "in kickboard!"
                                                  
2. Safe or Not?
  * 헬멧과 사람 사이의 IoU (=Area of Overlap/Area of Union)를 이용하여 0.03이 넘을 경우 헬멧을 착용하고 있다고 판단
  * 사람의 머리와 전체 몸의 비율이 대부분 일정하다고 가정한 결과.

3. Optimization
  * Yolov4-tiny model's TensorRT 사용 (FP 16)
  * https://github.com/jkjung-avt/tensorrt_demos


### Algorithm's Advantage
* Detection을 No helmet, safe, two people과 같이 전체적으로 하는 경우보다 정확도 향상
* Train Image 수가 줄어듦 (1694 -> 331)

\begin{table}[]
\begin{tabular}{cccc}
         & yolov4 & yolov4-tiny & yolov4-tiny with Algorithm \\
FPS      & 3.5    & 9           & 9                          \\
Accuracy & Low    & Very Low    & High                      
\end{tabular}
\end{table}

![Demo Video](./demo2.gif)