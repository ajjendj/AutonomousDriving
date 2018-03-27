from styx_msgs.msg import TrafficLight
import tensorflow as tf
from PIL import Image
import numpy as np
from utils import visualization_utils as vis_util
from matplotlib import pyplot as plt
import rospy
import os

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        
        self.category_index = {1: {'id':1, 'name': 'Red'}, 2: {'id':2, 'name': 'Yellow'}, 3: {'id':3, 'name': 'Green'}}

	CKPT = '/home/student/Desktop/System-Integration/ros/src/tl_detector/light_classification/frozen_inference_graph.pb'

	self.detection_graph = tf.Graph()
	
	with self.detection_graph.as_default():
    
  	    od_graph_def = tf.GraphDef()

	    if os.path.exists(CKPT):
  	    	with tf.gfile.GFile(CKPT, 'rb') as fid:
    		   serialized_graph = fid.read()
    		   od_graph_def.ParseFromString(serialized_graph)
    		   tf.import_graph_def(od_graph_def, name='')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction


	with self.detection_graph.as_default():
	    with tf.Session(graph=self.detection_graph) as sess:
		# Definite input and output Tensors for detection_graph
		image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		
		# Each box represents a part of the image where a particular object was detected.
		detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

		# expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image, axis=0)

		# apply detector
		(boxes, scores, classes, num) = sess.run(
		      [detection_boxes, detection_scores, detection_classes, num_detections],
		      feed_dict={image_tensor: image_np_expanded})

		scores = np.squeeze(scores)
		classes = np.squeeze(classes).astype(np.int32)
		
		cls = classes[0]
		score = scores[0]

		if cls > 0 and score > 0.5:
		   rospy.loginfo('detected and classified light as state {}'.format(cls-1))
		   return cls-1

		"""IMAGE_SIZE = (12, 8)

		vis_util.visualize_boxes_and_labels_on_image_array(
		      image,
		      np.squeeze(boxes),
		      np.squeeze(classes).astype(np.int32),
		      np.squeeze(scores),
		      self.category_index,
		      use_normalized_coordinates=True,
		      line_thickness=8) 
		plt.figure(figsize=IMAGE_SIZE)
		plt.imshow(image)
		plt.show() """



        return TrafficLight.UNKNOWN


