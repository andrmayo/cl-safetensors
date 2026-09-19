;;;; Test utilities

(in-package :cl-safetensors-test)

(defun get-paths ()
  (let ((safetensor-path
	 (asdf:system-relative-pathname
	  :cl-safetensors "test/fixtures/example_mlp.safetensors"))
	(json-path
	 (asdf:system-relative-pathname
	  :cl-safetensors "test/fixtures/example_mlp.json")))
    (values safetensor-path json-path)))

(defun test-read-u64-to-u32-le ()
  (multiple-value-bind
   (safetensor-path json-path)
   (get-paths)
   (let ((header-size-from-read-u64
	  (with-open-file (safetensor-stream safetensor-path :element-type '(unsigned-byte 8))
			  (cl-safetensors::read-u64-to-u32-le safetensor-stream)))
	 (header-size-from-json
	  (with-open-file (json-stream json-path)
			  (gethash "header_size"
				   (shasht:read-json json-stream)))))
     (assert (= header-size-from-read-u64 header-size-from-json)))))

(defun test-write-u64-le ()
  (let* ((val 16)
	 (le-byte-arr #(16 0 0 0 0 0 0 0))
	 (output (uiop:with-temporary-file (:stream stream
						    :element-type '(unsigned-byte 8)
						    :pathname path)
					   (cl-safetensors::write-u64-le val stream)
					   (finish-output stream)
					   (alexandria:read-file-into-byte-vector path))))
    (assert (equalp le-byte-arr output))))


(defun test-extract-header-data ()
  (multiple-value-bind
   (safetensor-path json-path)
   (get-paths)
   (multiple-value-bind (mmap-ptr fd mmap-size) (mmap:mmap safetensor-path)
			(unwind-protect
			    (with-open-file (stream safetensor-path
						    :element-type '(unsigned-byte 8))
					    (let*  ((header-size (cl-safetensors::read-u64-to-u32-le
								  stream))
						    (header-bytes (make-array
								   header-size
								   :element-type
								   '(unsigned-byte 8))))
					      (read-sequence header-bytes stream)
					      (let ((tensor-meta-data
						     (cl-safetensors::extract-header-data header-bytes))
						    (json-meta-data
						     (with-open-file (json-stream json-path)
								     (shasht:read-json
								      json-stream))))
						(setf (gethash "header_size" tensor-meta-data)
						      header-size)
						(assert (equalp tensor-meta-data json-meta-data)))))
			  ;; Free mmap buffer after copying floats into mgl-mat memory
			  (when mmap-ptr
			    (mmap:munmap mmap-ptr fd mmap-size))))))



(defun test-create-header-data ()
  (multiple-value-bind (safetensor-path json-path) (get-paths)
		       (let* ((mats-table (cl-safetensors::load-safetensors safetensor-path))
			      (names (sort (alexandria:hash-table-keys mats-table) #'string<))
			      (json-arr-from-mats (nth-value 1
							     (cl-safetensors::create-header-data
							      mats-table names)))
			      (json-arr-from-json
			       (let* ((parsed
				       (with-open-file
					(in-stream json-path)
					(shasht:read-json in-stream)))
				      (sorted-alist (progn
						      (remhash
						       "header_size" parsed)
						      (sort
						       (alexandria:hash-table-alist parsed)
						       #'string< :key #'car)))
				      (minified (let ((*print-pretty* nil)
						      (shasht:*write-alist-as-object* t))
						  (shasht:write-json sorted-alist nil)))
				      (octets (babel:string-to-octets minified :encoding :utf-8)))
				 octets)))
			 (setf json-arr-from-json
			       (cl-safetensors::pad-safetensors-header (length json-arr-from-json)
						      json-arr-from-json))
			 (assert (equalp json-arr-from-mats json-arr-from-json)))))

(defun assert-equal-header-bytecounts (save-output file-bytes)
  (let* ((output-bytes (subseq save-output 0 8))
	 (file-bytes (subseq file-bytes 0 8))
	 (output-val (let ((val 0))
		       (dotimes (i 8 val)
			 (setf val (logior val (ash (aref output-bytes i) (* i 8)))))))
	 (file-val (let ((val 0))
		     (dotimes (i 8 val)
		       (setf val (logior val (ash (aref file-bytes i) (* i 8))))))))
    (assert (= output-val file-val))))

;; this should be fine for testing with small .safetensors files
(defun test-read-write-roundtrip ()
  (let* ((safetensor-path (get-paths))
	 (mats-table (cl-safetensors::load-safetensors safetensor-path))
	 (save-output (uiop:with-temporary-file (:stream stream
							 :element-type '(unsigned-byte 8)
							 :pathname path)
						(cl-safetensors::write-safetensors mats-table stream)
						(finish-output stream)
						(alexandria:read-file-into-byte-vector path)))
	 (file-bytes (alexandria:read-file-into-byte-vector safetensor-path)))
    (assert-equal-header-bytecounts save-output file-bytes)
    (assert (equalp save-output file-bytes))))

;; TODO: test with empty tensors (with 1 dimension being 0), since safetensors allows this
;; TODO: test with 0-rank tensors
;; TODO: maybe validate that byte buffer is entirely indexed, without holes
;; (prevents polygot files)
;; TODO: more importantly, validate that mgl-mat objects actually have the right dimensions
