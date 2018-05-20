;;;; ann.lisp

(in-package #:ann)

;; A connection specifies weights between neurons
;; of different layers
(defstruct connection
  (weight       0.0 :type single-float) ;; bug?
  (delta-weight 0.0 :type single-float))

;; Representation for neuron data
(defstruct neuron
  index
  output-weights
  (output   0.0 :type single-float)
  (gradient 0.0 :type single-float))

;; Overall network training rate
(defvar *neuron-eta* 0.15)

;; Momentum; multiplier of the last weight change
(defvar *neuron-alpha* 0.5)

;; Represents an actual neural network.
(defstruct ann
  layers
  (error                  0.0 :type single-float)
  (recent-avg-error       0.0 :type single-float)
  (recent-avg-smth-factor 0.0 :type single-float))

;; Transfer function
(defun transfer-function (x)
  (tanh x))

;; Transfer function derivative
(defun transfer-function-derivative (x)
  (- 1.0 (* x x)))


;;; ==============

;; Though I decided to implement this from the general part
;; to the most specific part, I will be using a wishful thinking
;; approach, so I'll begin by building the neural network, and then
;; getting to stuff I need.

(defun build-neuron (outputs index)
  "Builds a single neuron, given the amount of outputs and the current index on the layer."
  (make-neuron :index index
	       :output-weights (make-array outputs
					   :element-type 'connection
					   :initial-contents
					   (loop for x from 0 to (1- outputs)
					      collect (make-connection :weight (random 1.0))))))
	       

(defun build-ann (topology)
  "Builds a new neural network. topology is a list containing the network topology."
  ;; Assume a topology of at least two layers (input and output). Otherwise, yield
  ;; an error
  (when (< (length topology) 2)
    (error "Invalid number of layers. ANN must have at least an input and output layer."))
  (let ((net (make-array (length topology)
			 ;; This also ensures an extra bias neuron per layer
			 :initial-contents
			 (loop for num in topology collect (make-array (1+ num)))))
	(outputs (mapcar #'1+ (append (cdr topology) '(0)))))
    ;; Iterate through layer
    (loop for layer being the elements of net
       for layer-size in topology
       for number-of-outputs in outputs
       ;; Create every neuron
       do (loop for index from 0 to layer-size ; extra one here is bias
	     do (setf (aref layer index)
		      (build-neuron number-of-outputs
				    index)))
       ;; Last neuron of layer should have a bias of 1.0
	 (setf (neuron-output (aref layer layer-size)) 1.0))
    (make-ann :layers net)))


;;; ==============

;; All we're looking for is a way to feed forward the input values and obtain a
;; result; and then, to correct then, we backpropagate expected values.

;; For that we'll define methods, just to make sure we're using our neural network.

;;; ==============

;;; Necessary for feeding-forward the information

(defmacro get-ann-layer (net layer-index)
  `(aref (ann-layers ,net) ,layer-index))

(defmacro get-ann-number-of-layers (net)
  `(length (ann-layers ,net)))

(defmethod feed-forward ((neuron neuron) (previous-layer vector))
  "Feeds forward the input values from the last layer to the current neuron."
  (setf (neuron-output neuron)
	(transfer-function
	 (reduce #'+
		 (loop for prev-neuron being the elements of previous-layer
		    collect (* (neuron-output prev-neuron)
			       (connection-weight (aref (neuron-output-weights prev-neuron)
							(neuron-index neuron)))))))))

(defmethod feed-forward ((net ann) input-values)
  "Feeds forward the input values for a neural network.
The given input values must have the same number of neurons as the first layer
of the neural network, minus the bias neuron."
  ;; Make sure our input values are exactly what we want
  (when (not (= (length input-values)
		(1- (length (get-ann-layer net 0)))))
    (error "Invalid number of input values for the ANN."))
  ;; Assign / latch the input values to the input neurons
  (loop for neuron being the elements of (get-ann-layer net 0)
     for input in input-values
     do (setf (neuron-output neuron) input))
  ;; Forward-propagate the values on each neuron
  (loop for layer-index from 1 below (get-ann-number-of-layers net)
       for previous-layer-index = (1- layer-index)
     do (loop for neuron being the elements of (get-ann-layer net layer-index)
	   do (feed-forward neuron (get-ann-layer net previous-layer-index)))))


;;; =============

;;; Necessary for back-propagating the results

(defmacro get-ann-last-layer (net)
  `(let ((layer-index (1- (get-ann-number-of-layers ,net))))
     (aref (ann-layers ,net) layer-index)))

(defmethod calculate-output-gradient ((neuron neuron) target)
  "Calculates the output gradient for the current neuron."
  (* (- target (neuron-output neuron)) ; delta
     (transfer-function-derivative (neuron-output neuron))))

(defmethod calculate-hidden-gradient ((neuron neuron) next-layer)
  "Calculates the gradient of a neuron on a hidden layer."
  (let ((sum-of-layers-weighted-gradients
	 (reduce #'+
		 (loop for next-layer-neuron being the elements of next-layer
		    for index from 0 below (length next-layer)
		    collect (* (connection-weight (aref (neuron-output-weights neuron)
							index))
			       (neuron-gradient next-layer-neuron))))))
    (* sum-of-layers-weighted-gradients
       (transfer-function-derivative (neuron-output neuron)))))

(defmethod update-input-weights ((neuron neuron) previous-layer)
  ;; The weights which need to be updated are on the connections
  ;; of the previous layer's containers
  (loop for previous-neuron being the elements of previous-layer
     for old-delta-weight = (connection-delta-weight
			     (aref (neuron-output-weights previous-neuron)
				   (neuron-index neuron)))
     for new-delta-weight = (+ (* *neuron-eta* ; overall net learning rate
				  (neuron-output previous-neuron)
				  (neuron-gradient neuron))
			       (* *neuron-alpha* ; learning momentum
				  old-delta-weight))
     do (setf (connection-delta-weight (aref (neuron-output-weights previous-neuron)
					     (neuron-index neuron)))
	      new-delta-weight)
       (incf (connection-weight (aref (neuron-output-weights previous-neuron)
				      (neuron-index neuron)))
	     new-delta-weight)))

(defmethod back-propagate ((net ann) target-values)
  "Backpropagates the target values to effectively make the ANN learn.
The given output values must have the same number of neurons as the last layer
of the neural network, minus the bias neuron."
  ;; Make sure our output values are exactly what we want
  (when (not (= (length target-values)
		(1- (length (get-ann-last-layer net)))))
    (error "Invalid number of target values for ANN backpropagation."))
  ;; Calculate overall ANN error using root mean square error of output layer
  (setf (ann-error net)
	(let ((sum-of-squared-deltas
	       (reduce #'+
		       (loop for neuron being the elements of (get-ann-last-layer net)
			  for target in target-values
			  collect (let ((delta (- target (neuron-output neuron))))
				    (* delta delta))))))
	  (sqrt (/ sum-of-squared-deltas
		   (1- (length (get-ann-last-layer net)))))))
  ;; Implement a recent average measurement, just for kicks
  (setf (ann-recent-avg-error net)
	(/ (* (ann-recent-avg-error net)
	      (+ (ann-recent-avg-smth-factor net)
		 (ann-error net)))
	   (1+ (ann-recent-avg-smth-factor net))))
  ;; Calculate output layer gradients
  (loop for neuron being the elements of (get-ann-last-layer net)
     for target in target-values
     do (setf (neuron-gradient neuron)
	      (calculate-output-gradient neuron target)))
  ;; Calculate gradients on hidden layers
  ;; Reverse iteration from rightmost hidden layer to first hidden layer
  (loop for layer-index from (- (get-ann-number-of-layers net) 2) downto 1
     for next-layer-index = (1+ layer-index)
     do (loop for neuron being the elements of (get-ann-layer net layer-index)
	   do (setf (neuron-gradient neuron)
		    (calculate-hidden-gradient neuron
					       (get-ann-layer net next-layer-index)))))
  ;; Update connection weights for hidden layers
  (loop for layer-index from (1- (get-ann-number-of-layers net)) downto 1
     for previous-layer-index = (1- layer-index)
     do (loop for neuron being the elements of (get-ann-layer net layer-index)
	   do (update-input-weights neuron (get-ann-layer net previous-layer-index)))))


;;; =============

;;; The following are mostly related to collecting results and training.

(defmethod collect-results ((net ann))
  "Collects the result of the last operation on the neural network.
Returns a list containing all the results but the bias."
  (butlast (loop for neuron being the elements of (get-ann-last-layer net)
	      collect (neuron-output neuron))))


;;; =============

;;; The following are mostly related to training the ANN.

(defmethod run-training ((net ann) (input-test-cases list) (target-test-cases list) &optional (show-output nil))
  "Performs an automated training on the neural network, and also counts the time taken.
Requires several test cases, where the input list is a list of lists, where each sublist
is a different input case for the exact amount of input neurons; the test targets also
follow the same principle, however they must pair up their number with the output neurons.
All amounts for these test cases must disconsider the bias neuron of each layer."
  (time (loop for test-input in input-test-cases
	   for test-target in target-test-cases
	   do (feed-forward net test-input)
	     (back-propagate net test-target)
	     (when show-output
	       (format t "Case: ~a, Target: ~a, Answer: ~a~&"
		       test-input
		       test-target
		       (collect-results net))))))
