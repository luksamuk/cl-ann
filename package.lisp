;;;; package.lisp

(defpackage #:ann
  (:use #:cl)
  (:export :*neuron-eta*
	   :*neuron-alpha*
	   :ann
	   :neuron
	   :connection
	   :build-ann
	   :feed-forward
	   :back-propagate
	   :collect-results
	   :run-training))

(defpackage #:ann-test
  (:use #:cl #:ann)
  (:export :*xor-ann*
	   :xor-begin
	   :xor-train
	   :xor-finish
	   :xor-run-test))