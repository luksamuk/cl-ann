;;;; ann-test.lisp

(in-package #:ann-test)

;;; ================

;;; The following test is for building a neural network which can accurately perform
;;; the XOR operation.
;;; From my previous calculations on another implementations, about 20,000 iterations
;;; of tests using random data should be enough.

;;; For any test, we need to
;;; a) Effectively build our neural network;
;;; b) Create test data, be it random or not.
;;;    Test data consists of two lists, of input and target values, where the first
;;;    has sublists with the same amount of input neurons (less the bias neuron), and
;;;    the second has the same amount of output neurons (less the bias neuron as well);
;;; c) Train the network, using the generated test data, by feeding-forward the input
;;;    test cases and back-propagating the expected target values;
;;; d) We then run a few test cases on our own, specially treating the output so that
;;;    we can present a better interface to the user; and finally
;;; e) We put the generated test network at the user's disposition by using some kind
;;;    of interface for that. Here, I figured it would be better to do this by exposing
;;;    the symbols holding the test networks, since I know that, if you're programming
;;;    in Lisp, you probably already know what you're doing, so I don't have to
;;;    underestimate you with any weird encapsulation.

;;; Test networks
(defparameter *xor-ann* nil)

;;; Training cases
(defparameter *xor-test-cases* nil)

(defun xor-begin ()
  "Initializes the test for the simulation of EXCLUSIVE-OR operation."
  ;; The XOR neural net uses a topology of two input neurons,
  ;; one hidden layer with four neurons, and a single output.
  (format t "Creating the neural network~&")
  (setf *xor-ann* (ann:build-ann '(2 4 1)))
  ;; Generate test cases
  (format t "Generating test cases~&")
  (setf *xor-test-cases* (list nil nil))
  (loop for x from 0 below 20000
     for test-case = (list (coerce (floor (random 2.0)) 'float)
			   (coerce (floor (random 2.0)) 'float))
     ;; Manually deducing the values with no clever arithmetic
     ;; because I'm not confident that I can do it on Common
     ;; Lisp. So bear with me.
     for result =
       (list (cond ((and (= (car test-case) 0.0)
			 (= (cadr test-case) 0.0))
		    0.0)
		   ((and (= (car test-case) 1.0)
			 (= (cadr test-case) 0.0))
		    1.0)
		   ((and (= (car test-case) 0.0)
			 (= (cadr test-case) 1.0))
		    1.0)
		   ((and (= (car test-case) 1.0)
			 (= (cadr test-case) 1.0))
		    0.0)))
     do (push test-case (car *xor-test-cases*))
       (push result (cadr *xor-test-cases*))))

(defun xor-train (&optional (show-input nil))
  "Train the neural network which simulates the XOR operation."
  (when (and *xor-ann* *xor-test-cases*)
    (format t "Training the network using ~a test cases~&"
	    (list-length (cadr *xor-test-cases*)))
    (ann:run-training *xor-ann*
		      (car *xor-test-cases*)
		      (cadr *xor-test-cases*)
		      show-input)
    (setf *xor-test-cases* nil)))

(defun xor-finish ()
  "Performs a final checking of the results on the XOR neural network."
  (when *xor-ann*
    (format t "Performing final test: evaluate all four possible test cases.~&")
    (labels ((interpret-result (value)
	       (cond ((<= value 0.25)  "certainly false")
		     ((<= value 0.375) "probably false")
		     ((<= value 0.625) "uncertain")
		     ((<= value 0.75)  "probably true")
		     (t                "certainly true"))))
      (format t "Interpreting results for common operations:~&")
      (ann:feed-forward *xor-ann* '(0.0 0.0))
      (let ((result (car (ann:collect-results *xor-ann*))))
	(format t "(0.0 0.0) => ~a ~a~&"
		(interpret-result result)
		result))
      (ann:feed-forward *xor-ann* '(1.0 0.0))
      (let ((result (car (ann:collect-results *xor-ann*))))
	(format t "(1.0 0.0) => ~a ~a~&"
		(interpret-result result)
		result))
      (ann:feed-forward *xor-ann* '(0.0 1.0))
      (let ((result (car (ann:collect-results *xor-ann*))))
	(format t "(0.0 1.0) => ~a ~a~&"
		(interpret-result result)
		result))
      (ann:feed-forward *xor-ann* '(1.0 1.0))
      (let ((result (car (ann:collect-results *xor-ann*))))
	(format t "(1.0 1.0) => ~a ~a~&"
		(interpret-result result)
		result)))))

(defun xor-run-test (&optional (show-input nil))
  "Performs the whole test cycle on the XOR neural network at once."
  (format t "Performing artificial neural network test: learning the XOR operation.~&")
  (when *xor-ann*
    (format t "WARNING: this will rebuild ann-test:*xor-ann*.~&"))
  (xor-begin)
  (xor-train show-input)
  (xor-finish)
  (format t "Testing finished.~&Use ann-test:*xor-ann* for further experiments.~&"))
