;;;; -*- mode: Lisp -*-

(asdf:defsystem #:cl-safetensors
  :license "MIT"
  :version "0.1.0"
  :name "CL-SAFETENSORS"
  :author "Andrew C. Mayo <acmayo399@gmail.com>"
  :mailto "acmayo399@gmail.com"
  :homepage "https://github.com/andrmayo/cl-safetensors"
  :bug-tracker "https://github.com/andrmayo/cl-safetensors/issues"
  :source-control (:git "https://github.com/andrmayo/cl-safetensors.git")
  :description "Utilities for reading and writing .safetensors files."
  :long-description
  #.(uiop:read-file-string
      (uiop:subpathname *load-pathname* "README.md"))
  :depends-on (#:alexandria ; utilities library
	       #:babel ; UTF-8 byte-to-string
	       #:cffi ; C foreign pointers
	       #:cl-cuda ; library for CUDA
	       #:mgl-mat ; multi-dimensional array library
	       #:mmap ; Memory mapping
	       #:shasht) ; JSON parser
  :components ((:module "src"
		:serial t
		:components ((:file "package")
			     (:file "safetensors"))))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-safetensors/test"))))

(asdf:defsystem #:cl-safetensors/test
  :license "MIT"
  :author "Andrew Mayo <acmayo399@gmail.com>"
  :mailto "acmayo399@gmail.com"
  :description "Test system for CL-SAFETENSORS."
  :depends-on (#:CL-SAFETENSORS)
  :components ((:module "test"
		:serial t
		:components ((:file "package") (:file "test-safetensors"))))
  :perform (asdf:test-op (o s)
			 (uiop:symbol-call '#:cl-safetensors-test '#:test)))
