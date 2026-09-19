(defpackage #:cl-safetensors-test
  (:use #:cl #:cl-safetensors)
  (:export #:test))

(in-package :cl-safetensors-test)

(defun test ()
  (test-read-u64-to-u32-le)
  (test-write-u64-le)
  (test-extract-header-data)
  (test-create-header-data)
  (test-read-write-roundtrip)
  (format t "All tests passed.~%"))
