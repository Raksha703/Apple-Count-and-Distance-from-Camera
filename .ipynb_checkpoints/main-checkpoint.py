import cv2
import math


net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)

class_names = []
with open("dnn_model/classes.txt" , "r") as f:
    for cla in f:
        cla = cla.strip()
        class_names.append(cla)
print(class_names)

cap = cv2.VideoCapture("apple.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH,720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1280)


#cv2.circle(frame,pt,5,(0,0,0),-1)

a = 0
o = 0
count = 0
center_points_prev_frame = []

tracking_objects = {}
tracking_id = 0

while True:
    ret,frame = cap.read()
    count += 1

    center_points_cur_frame = []


    (class_ids,scores,bboxes)= model.detect(frame,confThreshold=0.3,nmsThreshold=.4)
    for class_id,score,bboxe in zip(class_ids,scores,bboxes):
        x,y,w,h = bboxe
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx,cy))
        class_name = class_names[class_id]
        #print(class_name)
        


        cv2.putText(frame, class_name, (x, y -10), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #if class_name == "apple":
            #a = a+1
            #cv2.putText(frame,"APPLE =" + str(a),(30,60),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)


    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                if distance < 20:
                    tracking_objects[tracking_id] = pt
                    tracking_id += 1

    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()
        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue
            
            if not object_exists:
                tracking_objects.pop(object_id)
        
        for pt in center_points_cur_frame:
            tracking_objects[tracking_id] = pt
            tracking_id += 1


    for id,pt in tracking_objects.items():
        pt_dup = id
        total = []
        if pt != pt_dup:
            total.append(id) 
        #cv2.circle(frame,pt,5,(0,0,0),-1)
        cv2.putText(frame, "count : " + str(total), (30,60), 0, 1, (0, 0, 255), 2)

    print("tracking")
    print(tracking_objects)

        






    cv2.imshow("Frame",frame)

    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(30)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()