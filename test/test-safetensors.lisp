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

;;; Checking that mat objects read from test/fixtures/example_mlp.safetensors
;;; behave correctly as mgl-mat:mat objects, by running a forward pass through
;;; the MLP (nn.Linear -> ReLU -> nn.Linear -> ReLU -> nn.Linear -> Sigmoid)
;;; using mgl-mat operations (GEMM!, MREF) on the mat objects and comparing
;;; against reference output computed by loading the same weights into an
;;; equivalent PyTorch model and running it on the same input.

(defparameter +mlp-test-input+
  #(1.92691529d0 1.48728406d0 0.900717199d0 -2.10552096d0 0.678418458d0
    -1.23454487d0 -0.0430674776d0 -1.60466695d0 -0.752135277d0 1.64872301d0
    -0.392478645d0 -1.40360713d0 -0.727881312d0 -0.559430182d0 -0.768838882d0
    0.76244539d0 1.64231694d0 -0.159597471d0 -0.497397542d0 0.439589262d0
    -0.758131146d0 1.07831764d0 0.800800562d0 1.68062055d0 1.27912438d0
    1.29642284d0 0.61046648d0 1.33473778d0 -0.23162432d0 0.041759491d0
    -0.251575291d0 0.859858513d0 -1.38467371d0 -0.871236145d0 -0.223365918d0
    1.71736145d0 0.31888032d0 -0.424518973d0 0.305720925d0 -0.774592519d0
    -1.55757248d0 0.995636106d0 -0.879785836d0 -0.601142049d0 -1.27415121d0
    2.12278509d0 -1.23465312d0 -0.487913877d0 -0.913823009d0 -0.658137262d0
    0.0780238733d0 0.525808752d0 -0.48799172d0 1.19136906d0 -0.81400764d0
    -0.735992789d0 -1.40324783d0 0.0360036679d0 -0.0634772703d0 0.675614893d0
    -0.0978068933d0 1.844594d0 -1.18453741d0 1.38354933d0))

(defparameter +mlp-test-expected-output+
  #(0.498995781d0 0.520689189d0 0.524289727d0 0.526735544d0 0.574269414d0
    0.51565814d0 0.532075405d0 0.515236318d0))

(defun mlp-linear (x weight bias)
  "Applies an affine layer X @ WEIGHT^T + BIAS, where X is a 1xIN mat, WEIGHT
  is an OUTxIN mat, and BIAS is a length-OUT mat, returning a 1xOUT mat."
  (let* ((out (first (mgl-mat:mat-dimensions weight)))
	 (result (mgl-mat:make-mat (list 1 out) :ctype :float)))
    (mgl-mat:gemm! 1.0 x weight 0.0 result :transpose-b? t)
    (dotimes (i out)
      (incf (mgl-mat:mref result 0 i) (mgl-mat:mref bias i)))
    result))

(defun mlp-relu! (mat)
  (dotimes (i (mgl-mat:mat-size mat))
    (when (< (mgl-mat:row-major-mref mat i) 0.0)
      (setf (mgl-mat:row-major-mref mat i) 0.0)))
  mat)

(defun mlp-sigmoid! (mat)
  (dotimes (i (mgl-mat:mat-size mat))
    (setf (mgl-mat:row-major-mref mat i)
	  (/ 1.0 (+ 1.0 (exp (- (mgl-mat:row-major-mref mat i)))))))
  mat)

(defun test-mat-forward-pass ()
  "Loads test/fixtures/example_mlp.safetensors and checks that the resulting
  mgl-mat:mat objects behave correctly under real mgl-mat operations by
  running a forward pass and comparing against PyTorch's output for the same
  weights and input."
  (let* ((mats-table
	  (cl-safetensors:load-safetensors
	   (asdf:system-relative-pathname
	    :cl-safetensors "test/fixtures/example_mlp.safetensors")))
	 (x (mgl-mat:make-mat (list 1 64) :ctype :float)))
    (dotimes (i 64)
      (setf (mgl-mat:mref x 0 i) (coerce (aref +mlp-test-input+ i) 'single-float)))
    (let* ((h1 (mlp-relu! (mlp-linear x (gethash "0.weight" mats-table)
					  (gethash "0.bias" mats-table))))
	   (h2 (mlp-relu! (mlp-linear h1 (gethash "2.weight" mats-table)
					   (gethash "2.bias" mats-table))))
	   (y (mlp-sigmoid! (mlp-linear h2 (gethash "4.weight" mats-table)
					     (gethash "4.bias" mats-table)))))
      (dotimes (i 8)
	(assert (< (abs (- (mgl-mat:mref y 0 i)
			    (coerce (aref +mlp-test-expected-output+ i) 'single-float)))
		   1d-4))))))
