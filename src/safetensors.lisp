;;;; This file is for converting a .safetensors file to a simple hash-table, and writing
;;;; a .safetensors file from a hash-table. The hash-table that gets read here should be consumed
;;;; by a function defined elsewhere to a suitable data structure for operations with mgl packages.
;;;; Hash-table deserialization target has format {<tensor_name>: mgl-mat:mat},
;;;; where <tensor_name> is read from .safetensors file, and mgl-mat:mat object is created by
;;;; reading mmap'd tensors using header offsets and constructing mgl-mat:mat objects from them.

(in-package :cl-safetensors)


;;; Reading and writing 8-byte integer value at start of file

(declaim (ftype (function (stream) (unsigned-byte 32)) read-u64-to-u32-le)
         (inline read-u64-to-u32-le))

;; we need u62 or less for this to fit in a fixnum
(defun read-u64-to-u32-le (stream)
  "Reads an 8-byte little-endian unsigned integer from stream."
  (declare (optimize (speed 3) (safety 1)))
  (let ((val 0))
    (declare (type (and fixnum (unsigned-byte 32)) val))
    (dotimes (i 8 val)
      ;; enforce last four bytes being 0
      (let ((cur-byte (read-byte stream)))
        (unless (or (< i 4) (zerop cur-byte))
          (error
           "read-u64-to-u32: value exceeds (unsigned-byte 32) range,
            byte ~d at position ~d is nonzero." cur-byte i))
        ;; suppress warning related to i not being known at compile time
        (locally (declare (sb-ext:muffle-conditions sb-ext:compiler-note))
                 (setf (ldb (byte 8 (* i 8)) val) cur-byte))))))

(declaim (ftype (function ((unsigned-byte 32) stream) (values)) write-u64-le)
         (inline write-u64-le))

(defun write-u64-le (val write-stream)
  (declare (optimize (speed 3) (safety 1)))
  (dotimes (i 8)
    (write-byte (ldb (byte 8 (* i 8)) val) write-stream))
  (values))

;;; Handling header data

(declaim (ftype (function ((vector (unsigned-byte 8))) hash-table) extract-header-data)
         (inline extract-header-data))

(defun extract-header-data (header-bytes)
  (shasht:read-json (babel:octets-to-string header-bytes :encoding :utf-8)))


(declaim (ftype (function (mgl-mat:mat) (and (unsigned-byte 64) fixnum)) ctype-byte-count)
         (inline ctype-byte-count))

;; calculating this is more reliable than using mgl-mat::vec-n-bytes,
;; which, aside from not being exported, also won't play nice with displaced/shared MAT objects
(defun ctype-byte-count (mat)
  (declare (optimize (speed 3) (safety 1)))
  (the (and (unsigned-byte 64) fixnum)
       (* (the (and (unsigned-byte 64) fixnum) (mgl-mat:mat-size mat))
          (the (and (unsigned-byte 64) fixnum) (cffi:foreign-type-size (mgl-mat:mat-ctype mat))))))

(defun get-dtype-string (dtype-keyword)
  ;; surround the type string with quotation marks
  (format nil "~s"
          (cond ((eq dtype-keyword :float) "F32")
                ((eq dtype-keyword :double) "F64")
                (t (error "Unsupported tensor type `~a`." dtype-keyword)))))


(declaim (ftype (function (mgl-mat:mat (unsigned-byte 64))
                          (values (unsigned-byte 64) list))
                get-header-fields)
         (inline get-header-fields))

(defun get-header-fields (mat start-byte)
  (let* ((new-offset (+ start-byte (ctype-byte-count mat)))
         (dtype (cons "dtype" (get-dtype-string (mgl-mat:mat-ctype mat))))
         (shape (cons "shape" (format nil "[~{~a~^,~}]" (mgl-mat:mat-dimensions mat))))
         (data-offsets
          (cons "data_offsets" (format nil "[~d,~d]" start-byte new-offset))))
    (values new-offset (list dtype shape data-offsets))))


;; pads header for mmemory alignment, ensuring it's aligned with 8-byte segments
(declaim (ftype (function (hash-table (simple-array (unsigned-byte 8) (*)))
                          (values (simple-array (unsigned-byte 8) (*)) fixnum))))

(defun pad-safetensors-header (header-size header-arr)
  (let* ((header-size-mod (mod header-size 8))
         (padded-header-size
          (if (= header-size-mod 0) header-size (+ header-size (- 8 header-size-mod))))
         (padded-header-arr
          (make-array padded-header-size :element-type '(unsigned-byte 8) :initial-element #x20)))
    (replace padded-header-arr header-arr)
    (values padded-header-arr padded-header-size)))

(declaim (ftype (function (hash-table list)
                          (values hash-table (simple-array (unsigned-byte 8) (*))))
                create-header-data))

(defun create-header-data (mats-table names)
  "`names` gives the order in which names appear in the header.
  The size of the header in UTF-8 bytes is given by `header_size` key in the returned hash map.
  Return values are a simple-array of 8-bit unsigned values representing the json string encoded
  in utf-8."
  (declare (optimize (debug 3)))
  (let ((header-data (make-hash-table :test #'equal))
        (offset-cursor 0)
        (first-elt t))
    (declare (type (unsigned-byte 64) offset-cursor))
    (let ((whole-string
	   (with-output-to-string (string-stream)
				  (write-string "{" string-stream)
				  ;; add opening of minified json header
				  (dolist (name names)
				    (if first-elt
					(setf first-elt nil)
				      (write-string "," string-stream))
				    ;; add comma delimiter
				    (write-string (format nil "~s:{" name)
						  string-stream)
				    (multiple-value-bind (new-offset fields)
							 (get-header-fields
							  (gethash name mats-table)
							  offset-cursor)
							 (setf offset-cursor new-offset)
							 (let ((cur-header-fields (setf
										   (gethash
										    name
										    header-data)
										   (make-hash-table
										    :test #'equal)))
							       (first-field t))
							   (dolist (field-pair fields)
							     (if first-field
								 (setf first-field nil)
							       (write-string "," string-stream))
							     (let ((field-key (car field-pair))
								   (field-value (cdr field-pair)))
							       (setf
								(gethash
								 field-key
								 cur-header-fields)
								field-value)
							       (write-string
								(format nil "~s:~a"
									field-key field-value)
								string-stream)))))
				    (write-string "}" string-stream))
				  (write-string "}" string-stream))))
      (let* ((header-arr (babel:string-to-octets whole-string :encoding :utf-8))
             (header-size (length header-arr)))
        (multiple-value-bind
         (padded-header-arr padded-header-size)
         (pad-safetensors-header header-size header-arr)
         (setf (gethash "header_size" header-data) padded-header-size)
         (values header-data padded-header-arr))))))


;;; Validating contents to avoid polyglot exploits

(declaim (ftype (function (hash-table fixnum (unsigned-byte 32)) boolean))
	 (inline trailing-bytes-p))

(defun trailing-or-overflow-bytes-p (header-data file-size header-size)
  (let ((tensors-size (the fixnum (- file-size header-size 8)))
	(max-offset-end 0))
    (alexandria:maphash-values
     (lambda (meta-data)
       (let* ((offsets (gethash "data_offsets" meta-data))
	      (offset-end (aref offsets 1)))
	 (when (> offset-end tensors-size)
	   (format *error-output* "Tensor offset ~a exceeds size of tensors in file.~%"
		   (gethash "data_offsets" meta-data))
	   (return-from trailing-or-overflow-bytes-p t))
	 (setf max-offset-end (max offset-end max-offset-end))))
     header-data)
    (when (/= max-offset-end tensors-size)
      (format *error-output* "Actual size of tensors (~a) exceeds last offset value (~a).~%"
	      tensors-size max-offset-end)
      (return-from trailing-or-overflow-bytes-p t))
    nil))

(declaim (ftype (function (hash-table) boolean) trailing-byte-p)
	 (inline overlap-or-gap-in-bytes-p))

(defun overlap-or-gap-in-bytes-p (header-data)
  (let ((offset-ranges (make-array (hash-table-count header-data) :adjustable t :fill-pointer 0)))
    (alexandria:maphash-values
     (lambda (meta-data)
       (vector-push-extend
	(gethash "data_offsets" meta-data) offset-ranges))
     header-data)
    (setf offset-ranges (sort offset-ranges #'< :key #'car))
    (loop for i from 1 below (length offset-ranges)
	  do (let ((last-offset-end (aref (aref offset-ranges (1- i)) 1))
		   (cur-offset-start (aref (aref offset-ranges i) 0)))
	       (when (< cur-offset-start last-offset-end)
		 (format *error-output*
			 "Safetensors file has overlapping byte offsets for offsets ~a and ~a.~%"
			 (aref offset-ranges (1- i)) (aref offset-ranges i))
		 (return t))
	       (when (> cur-offset-start last-offset-end)
		 (format *error-output*
			 "Safetensors file has a gap in byte offsets for offsets ~a and ~a.~%"
			 (aref offset-ranges (1- i)) (aref offset-ranges i))
		 (return t)))
	     finally (return nil))))


(declaim (ftype (function (hash-table (and unsigned-byte fixnum) (unsigned-byte 32)) boolean)
		safetensors-invalid-p))

(defun safetensors-invalid-p (header-data mmap-size header-size)
  "Runs data validation functions with `path'."
  (cond
    ((trailing-or-overflow-bytes-p header-data mmap-size header-size) t)
    ((overlap-or-gap-in-bytes-p header-data) t)
    (t nil)))

;;; Reading and writing .safetensors files

;; currently supports float (fl32) and double (fl64), since mgl supports these types
;; if this changes, add more types here
(declaim (ftype (function (string) (or keyword null)) check-dtype))
(declaim (inline check-dtype))

(defun check-dtype (dtype)
  (cond
   ((string= dtype "F32") :float)
   ((string= dtype "F64") :double)
   (t nil)))

(declaim (ftype (function
                 (hash-table (unsigned-byte 32) cffi:foreign-pointer boolean)
                 hash-table)
		handle-tensor-data))

;; helper function for processing safetensor

;; meta-data: header from safetensor file
;; data-base: byte count for header integer + header itself
;; mmap-ptr: pointer into virtual memory for safetensor
;; cuda-p: whether cuda is available and should be used
(defun handle-tensor-data (meta-data data-base mmap-ptr cuda-p)
  (declare (optimize (debug 3)))
  (let ((tensor-map (make-hash-table :test 'equal)))
    (maphash
     (lambda (name meta-data)
       (unless (string= name "__metadata__")
         (alexandria:if-let ((offsets (gethash "data_offsets" meta-data))
                             (shape (gethash "shape" meta-data))
                             (dtype (gethash "dtype" meta-data)))
                            (progn (unless (check-dtype dtype)
                                     (error
                                      "Tensor ~a has dtype ~a, mgl-mat requires F32."
                                      name dtype))
                                   (let* ((start (+ data-base (aref offsets 0)))
                                          (src-ptr (cffi:inc-pointer mmap-ptr start))
                                          (dimensions (coerce shape 'list))
                                          (total-elems (reduce #'* dimensions))
                                          (float-type (check-dtype dtype))
                                          ;; allocate mgl-mat
                                          ;; allows lazy transfer whenever used within with-cuda*
                                          (mat (mgl-mat:make-mat dimensions
                                                                 :ctype float-type
                                                                 :cuda-enabled cuda-p)))
                                     ;; copy raw floats directly into mgl-mat backing memory
                                     (mgl-mat:with-facet
                                      (dest (mat 'mgl-mat:foreign-array :direction :output))
                                      (cffi:foreign-funcall "memcpy"
                                                            :pointer (mgl-mat::offset-pointer dest)
                                                            :pointer src-ptr
                                                            :size (* total-elems 4) :pointer))
                                     ;; directory in hash table
                                     (setf (gethash name tensor-map) mat)))
                            ;; else branch if if-let
                            (error
                             "safetensor[~a] header does not have all of `data_offsets`, `shape`,
and `dtype`"
                             name))))
     ;; hash-table for maphash
     meta-data)
    tensor-map))

(declaim (ftype (function ((or string pathname) &key (:cuda-p boolean)) hash-table)
                load-safetensors-mats))

;; returns hash-table that maps names to mgl-mat matrices
(defun load-safetensors (path &key (cuda-p nil))
  "Parses PATH and returns a hash table mapping tensor names to mgl-mat:mat objects."
  (multiple-value-bind (mmap-ptr fd mmap-size) (mmap:mmap path)
    (check-type mmap-size (and unsigned-byte fixnum))
    (unwind-protect
	 (with-open-file (stream path :element-type '(unsigned-byte 8))
	   (let*  ((header-size (read-u64-to-u32-le stream))
		   (header-bytes
		     (make-array
		      header-size
		      :element-type '(unsigned-byte 8))))
	     (read-sequence header-bytes stream)
	     (let ((meta-data (extract-header-data header-bytes))
		   ;; 8 bytes for unsigned 64 bit integer
		   (data-base (+ 8 header-size)))
	       (when (safetensors-invalid-p
		      meta-data mmap-size header-size)
		 (error "Invalid data read from ~a." path))
	       (handle-tensor-data
		meta-data
		data-base
		mmap-ptr
		cuda-p)))) ; return value
      ;; Free mmap buffer after copying floats into mgl-mat memory
      (when mmap-ptr
	(mmap:munmap mmap-ptr fd mmap-size)))))

;; loosely modelled after mgl/src/core.lisp SAVE-STATE, but without use of generics.
;; Ultimately a write method that inherits from the generic mgl-core:write-state*
;; is desirable, but it probably makes sense to define this in a dedicated package
;; for handling transformers that calls this function,
;; and which uses a dedicated type for transformer weights
;; something like:

;; (defmethod (mgl-core:write-state* (transformer transformer) stream context)
;;     (write-safetensors (make-state-hash-table transformer) stream))

;; save-safetensors is an alternative standalone to the approach ^
;; .safetensors file is written in alphabetical order of the names


(declaim (ftype (function
		 ((or string pathname) hash-table &key (:if-exists keyword) (:ensure boolean))
		 hash-table)
		save-safetensors))

(defun save-safetensors (filename mats-table &key (if-exists :error) (ensure t))
  "Save hash-table MATS-TABLE of names to mat objects to FILENAME. If ENSURE,
  ENSURE-DIRECTORIES-EXIST is called on FILENAME. IF-EXISTS is passed
  on to OPEN. Return MATS-TABLE."
  (when ensure
    (ensure-directories-exist filename))
  (with-open-file (stream filename :direction :output
				   :element-type '(unsigned-byte 8)
				   :if-does-not-exist :create
				   :if-exists if-exists)
    (write-safetensors mats-table stream)))

;; TODO: perhaps add quanitization options to downsize (and, for completeness' sake)
;; upsize floating point bits.

(declaim (ftype (function (hash-table stream) (values)) write-safetensors))

(defun write-safetensors (mats-table stream)
  "Function to write a hash-table in format {<tensor-name>:<mgl-mat:mat tensor>} to stream
in safetensors serialization format. Ensures header is padded to align with 8-byte segments,
and writes tensors in order determined by an alphabetical sort (not alphanumeric) of tensor
names."
  (let ((names (sort (alexandria:hash-table-keys mats-table) #'string<)))
    (multiple-value-bind (header-data header-arr) (create-header-data mats-table names)
      (when (>= (gethash "header_size" header-data) (expt 2 32))
	(error "header_size is implausibly large (= ~d)." (gethash "header_size" header-data)))
      (write-u64-le (gethash "header_size" header-data) stream)
      (write-sequence header-arr stream)
      (dolist (name names)
	(let* ((mat (gethash name mats-table))
	       (mat-size (ctype-byte-count mat)))
	  (mgl-mat:with-facet (src
			       (mat 'mgl-mat:foreign-array :direction :input))
	    (write-sequence (cffi:foreign-array-to-lisp
			     (mgl-mat::offset-pointer src)
			     (list :array :uint8 mat-size))
			    stream))))
      (values))))
