;;;; cl-ann.asd

(asdf:defsystem #:cl-ann
  :description "Artificial Neural Network implementation"
  :author "Lucas Vieira <lucasvieira@lisp.com.br>"
  :license "MIT"
  :serial t
  :components ((:file "package")
               (:file "ann")
	       (:file "ann-test")))

