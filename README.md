:warning: *This project is no longer supported. Users are encouraged to move to* [Concrete Numpy](https://github.com/zama-ai/concrete-numpy) :warning:

# HNP examples

This repository is a collection of examples for Homomorphic Numpy (HNP), a prototype
released by [Zama](https://zama.ai) to turn Numpy programs into their homomorphic equivalent.

## Installation

HNP is currently only available as a docker image. To install it just use:

```console
docker pull zamafhe/hnp
```

## Running examples

Default entrypoint for the image is to launch a jupyter notebook inside the HNP environment
on port 8888 so that you could run easily examples from this repository:

```console
git clone https://github.com/zama-ai/hnp-examples
cd hnp-examples
docker run -p 8888:8888 -v "$(pwd)"/examples:/data zamafhe/hnp
```

Then on stdout you should see the notebook link:

```console
[...]
http://127.0.0.1:8888/?token=[...]
```

Every files you add in the `examples` directory will be shown in the default notebook directory
inside the container: `/data`.

If you are satisfied with your current example do not hesitate to open a PR to share it with the community.

## Documentation

HNP documentation is available [here](https://docs.zama.ai/hnp)

If you have any question, leave us a message [here](https://community.zama.ai)
