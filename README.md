# CL-Safetensors

## About the Project

For now, this is a minimal library for deserializing and serializing
`.safetensors` files to hash-tables of `mgl-mat:mat` objects. The idea here is
just that these hash-tables serve as an intermediate representation before
reading them into some specific Common Lisp representation of the model, e.g.
`mgl:bpn`.

Ideally, this would be done by writing methods for the `mgl-mat` `read-state*`
`write-state*` generics. But for now, simple hash-tables with

## State of the Project

For now, this simply gives you what you need to make use of neural networks
saved to `.safetensors` files (most likely trained using relevant Python
libraries) and work with them in Common Lisp. There's a lot of additional
functionality (and safety features) that could stand to be added, for instance
modifying quantization, especially given that currently the `mgl` ecosystem only
works with fl32 and fl64 formats.
