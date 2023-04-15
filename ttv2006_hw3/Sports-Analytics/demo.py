import numpy as np 
import cv2

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms as T

from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
from deep_sort.tools import generate_detections as gdet


def main(video_path, output_path, label_class):
    # Initialize PyTorch Faster R-CNN
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model_threshold = 0.7
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()
    print("Initialize pre-trained Faster R-CNN model...")
    
    # Initialize Deep SORT
    max_cosine_distance = 0.7
    nn_budget = None
    model_filename = "resources/networks/mars-small128.pb"
    
    preprocess = T.Compose([T.ToTensor()])
    
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    print("Initialize Deep SORT...")
    
    vid = cv2.VideoCapture(video_path)  # detect on path
    
    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4
    
    frame_num = 1
    
    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break

        image = original_frame  # extract frame from video
        
        batch = [preprocess(image)] # process image into tensor
        
        prediction = model(batch)[0]    # extract prediction from faster R-CNN
        
        # extract labels and boxes for the image
        labels = [weights.meta["categories"][i] for i in prediction["labels"]]
        boxes = [[min(i[0], i[2]), min(i[1], i[3]), abs(i[0] - i[2]), abs(i[1] - i[3])] for i in list(prediction["boxes"].detach().numpy())]
        scores = list(prediction["scores"].detach().numpy())
        
        # filter out any boxes and labels that do not belong to the input label classes
        filter = []
        for i in range(len(labels)):
            if labels[i] in label_class and scores[i] >= model_threshold:
                filter.append(i)  
        filter = np.array(filter)
        
        # obtain all detections for the current frame
        labels = np.array(labels)[filter]
        boxes = np.array(boxes)[filter]
        scores = np.array(scores)[filter]
        features = np.array(encoder(image, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, labels, features)]
        
        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)
        
        # Obtain info from the tracks
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue 
            bbox = track.to_tlbr() # Get the corrected/predicted bounding box
            class_name = track.get_class()  # Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track
            
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 1)
            cv2.putText(image, "{class_name}:{tracking_id}".format(class_name=str(class_name), tracking_id=str(tracking_id)), 
                        (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    
        out.write(image)
        print("Frame {} processed ...".format(frame_num))
        frame_num += 1
        
    out.release()
    vid.release()
    
    
if __name__ == '__main__':
    video_path = "Football_match.mp4"
    output_path = "output/output.mp4"
    labels = ['person', 'sports ball']
    
    main(video_path=video_path, output_path=output_path, label_class=labels)
    
    print("Output video complete. Locate at {}".format(output_path))