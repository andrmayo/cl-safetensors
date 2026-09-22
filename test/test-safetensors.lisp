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

(defun read-write-roundtrip (safetensor-path)
  (let* ((mats-table (cl-safetensors::load-safetensors safetensor-path))
	 (save-output (uiop:with-temporary-file (:stream stream
							 :element-type '(unsigned-byte 8)
							 :pathname path)
						(cl-safetensors::write-safetensors mats-table stream)
						(finish-output stream)
						(alexandria:read-file-into-byte-vector path)))
	 (file-bytes (alexandria:read-file-into-byte-vector safetensor-path)))
    (assert-equal-header-bytecounts save-output file-bytes)
    (assert (equalp save-output file-bytes))))

(defun test-read-write-roundtrip ()
  (let ((safetensor-path (get-paths)))
    (read-write-roundtrip safetensor-path)))

(defun test-small-attention-model ()
  (let ((attention-model-path
          (asdf:system-relative-pathname
	   :cl-safetensors "test/fixtures/small_attention_model.safetensors")))
    (read-write-roundtrip attention-model-path)))

(defun test-empty-tensor-model ()
  (let ((empty-model-path
	  (asdf:system-relative-pathname
	   :cl-safetensors "test/fixtures/empty_model.safetensors")))
    (read-write-roundtrip empty-model-path)))

(defun test-zero-rank-tensor ()
  (let ((zero-rank-model-path
	  (asdf:system-relative-pathname
	   :cl-safetensors "test/fixtures/zero_rank.safetensors")))
    (read-write-roundtrip zero-rank-model-path)))


(defun validation-helper (path)
  (multiple-value-bind (mmap-ptr fd mmap-size) (mmap:mmap path)
    (check-type mmap-size (and unsigned-byte fixnum))
    (unwind-protect
	 (with-open-file (stream path :element-type '(unsigned-byte 8))
	   (let* ((header-size (cl-safetensors::read-u64-to-u32-le stream))
		  (header-bytes
		    (make-array header-size :element-type '(unsigned-byte 8))))
	     (read-sequence header-bytes stream)
	     (let ((header-data (cl-safetensors::extract-header-data header-bytes)))
	       (values header-data mmap-size header-size))))
      (when mmap-ptr
	(mmap:munmap mmap-ptr fd mmap-size)))))


(defun catch-trailing-bytes ()
  (let* ((trailing-byte-path
	   (asdf:system-relative-pathname
	    :cl-safetensors
	    "test/fixtures/trailing_bytes.safetensors"))
	 (error-message
	   (with-output-to-string (output)
	     (let ((*error-output* output))
	       (multiple-value-bind (header-data mmap-size header-size)
		   (validation-helper trailing-byte-path)
		 (assert
		  (cl-safetensors::safetensors-invalid-p header-data mmap-size header-size)))))))
    ;; effectively checks that there's just one error message
    (assert (= 1 (count #\Newline error-message)))
    ;; effectively checks that it's the right error message
    (assert (uiop:string-prefix-p "Tensor offset" error-message))))

(defun catch-overflow-bytes ()
  (let* ((overflow-byte-path
	   (asdf:system-relative-pathname
	    :cl-safetensors
	    "test/fixtures/overflow_bytes.safetensors"))
	 (error-message
	   (with-output-to-string (output)
	     (let ((*error-output* output))
	       (multiple-value-bind (header-data mmap-size header-size)
		   (validation-helper overflow-byte-path)
		 (assert
		  (cl-safetensors::safetensors-invalid-p header-data mmap-size header-size)))))))
    ;; effectively checks that there's just one error message
    (assert (= 1 (count #\Newline error-message)))
    ;; effectively checks that it's the right error message
    (assert (uiop:string-prefix-p "Actual size of tensors" error-message))))

(defun catch-overlap-bytes ()
  (let* ((overlap-byte-path
	   (asdf:system-relative-pathname
	    :cl-safetensors
	    "test/fixtures/overlap_bytes.safetensors"))
	 (error-message
	   (with-output-to-string (output)
	     (let ((*error-output* output))
	       (multiple-value-bind (header-data mmap-size header-size)
		   (validation-helper overlap-byte-path)
		 (assert
		  (cl-safetensors::safetensors-invalid-p header-data mmap-size header-size)))))))
    (assert (= 1 (count #\Newline error-message)))
    (assert (uiop:string-prefix-p "Safetensors file has overlapping byte offsets" error-message))))

(defun catch-gap-bytes ()
  (let* ((gap-byte-path
	   (asdf:system-relative-pathname
	    :cl-safetensors
	    "test/fixtures/gap_bytes.safetensors"))
	 (error-message
	   (with-output-to-string (output)
	     (let ((*error-output* output))
	       (multiple-value-bind (header-data mmap-size header-size)
		   (validation-helper gap-byte-path)
		 (assert
		  (cl-safetensors::safetensors-invalid-p header-data mmap-size header-size)))))))
    (assert (= 1 (count #\Newline error-message)))
    (assert (uiop:string-prefix-p "Safetensors file has a gap in byte offsets" error-message))))

(defun test-safetensors-invalid-p ()
  (catch-trailing-bytes)
  (catch-overflow-bytes)
  (catch-overlap-bytes)
  (catch-gap-bytes))
