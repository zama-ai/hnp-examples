 hnp-examples

This repository is a collection of examples for Homomorphic Numpy (HNP), a prototype
released by [Zama](https://zama.ai) to turn Numpy programs into their homomorphic equivalent.

## Installation

HNP is currently only available as a docker image. To install it just use:

```
docker pull zamafhe/hnp
```

## Running examples

Default entrypoint for the image is to launch a jupyter notebook inside the HNP environment
on port 8888 so that you could run easily examples from this repository:

```
git clone https://github.com/zama-ai/hnp-examples
cd hnp-examples
docker run -p 8888:8888 -v "$(pwd)"/examples:/data zamafhe/hnp
```

Then on stdout you should see the notebook link:

```
[...]
http://127.0.0.1:8888/?token=[...]
```

Every files you add in the examples folder will be shown in the default notebook directory (`/data`).

If you are satisfied with your current example do not hesitate to open a PR to share it with the community.
