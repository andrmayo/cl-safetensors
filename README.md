# CL-Safetensors

Utilities for reading and writing .safetensors files.

# Overview

For now, this is a minimal library for deserializing and serializing
`.safetensors` files to hash-tables of `mgl-mat:mat` objects. The idea here is
just that these hash-tables serve as an intermediate representation before
reading them into some specific Common Lisp representation of the model, e.g.
`mgl:bpn`.

# Usage

The API consists of just three function, `LOAD-SAFETENSORS`, `SAVE-SAFETENSORS`,
and `WRITE-SAFETENSORS`.

## Writing

`LOAD-SAFETENSORS` simply requires the path of the file to read, and can
optionally take a boolean `CUDA-P` argument that gets passed to the
`MGL-MAT:MAKE-MAT` function, for instance:

```Lisp
(defparameter *dir*
  (uiop:pathname-parent-directory-pathname (uiop:current-lisp-file-pathname)))
(defparameter *load-path* (uiop:merge-pathnames "load_file.safetensors" *dir*))
;; cuda-p is nil by default
(defparameter *mat-tensors* (load-safetensors *load-path*))
;; if we want mgl-mat:make-mat to use cuda
(defparameter *mat-tensors*
  (load-safetensors *load-path* :cuda-p t))
```

## Reading

When saving a .safetensors file from a hash-table mapping strings to
`MGL-MAT:MAT` objects (`MATS-TABLE`), there are optional keyword arguments
`IF-EXISTS` and `ENSURE`. `IF-EXISTS` gets passed to `WITH-OPEN-FILE`, and if
`ENSURE` is `T`, then `(ensure-directories-exist filename)` gets called. This is
the basic usage:

```Lisp
(defparameter *save-path* (uiop:merge-pathnames "save_file.safetensors"))
(save-safetensors *save-path* mats-table)

```

Alternatively, if you want to write to a stream, you can use

```Lisp
(write-safetensors mats-table stream)
```

It struck me as less valuable to provide a direct API for reading from a stream,
but it might be worth adding this.

## Tests

To run unit tests, you can either call the `TEST` function defined in
`test/package.lisp`, or (if you have `make`) simply call `make test` from the
project root.

# State of the Project

A more complete version of this library would make reading files to
`mgl-mat:mat` objects one of several options. Most obviously, it would be good
(and simple) to be able to read and write from Common Lisp arrays, as well as to
handle column-major as well as row-major (`mgl-mat` is row-major), and perhaps
even to coerce floating-point precision. My immediate needs are only for
`mgl-mat:mat` representations, hence the current functionality.

Ideally, operations with `mgl-mat` would be done by writing methods for the
`mgl-mat` `read-state*` `write-state*` generics. But for now, simple hash-tables
mapping names (strings) to `mgl-mat:mat` objects is what the API provides.

# License

Licensed under the MIT license.
