.PHONY: enter-env cl-setup check-roswell check-qlot debian-setup test

enter-env:
	qlot exec ros run -- eval '(ql:quickload :cl-safetensors)'

cl-setup: check-roswell check-qlot
	qlot install

check-roswell:
	@if ! command -v ros >/dev/null 2>&1; then \
		echo "Error: Roswell is required for the cl-setup target"; \
		exit 1; \
	fi

check-qlot:
	@if ! command -v qlot >/dev/null 2>&1; then \
		echo "Error: Qlot is required for the cl-setup target"; \
		exit 1; \
	fi

clean-qlot:
	rm -rf ".qlot/"

debian-setup: cl-setup
	@if dpkg-query --show -f='${Status}\n' nvidia-cuda-toolkit; then \
		echo "nvidia-cuda-toolkit already available"; \
	else \
		sudo apt-get install nvidia-cuda-toolkit; \
	fi
	@if dpkg-query --show -f='${Status}\n' libblas-dev; then \
		echo "libblas-dev already available"; \
	else \
		sudo apt-get install libblas-dev; \
	fi
	@if dpkg-query --show -f='${Status}\n' liblapack-dev; then \
		echo "liblapack-dev already available"; \
	else \
		sudo apt-get install liblapack-dev; \
	fi

test:
	qlot exec sbcl --non-interactive --eval \
		'(setf sb-ext:*muffled-warnings* (quote (or sb-kernel:redefinition-warning asdf/parse-defsystem:bad-system-name)))' \
		--eval '(asdf:test-system :cl-safetensors)'
