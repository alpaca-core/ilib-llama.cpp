# llama.cpp

This is a wrapper of llama.cpp implemented as per the discussion [Integration of llama.cpp and whisper.cpp](https://github.com/alpaca-core/alpaca-core/discussions/5):

* Use the llama.cpp C interface in llama.h
* Reimplement the common library

## Build

> [!IMPORTANT]
> When cloning this repo, don't forget to fetch the submodules.
> * Either: `$ git clone https://github.com/alpaca-core/ac-local.git --recurse-submodules`
> * Or:
>    * `$ git clone https://github.com/alpaca-core/ac-local.git`
>    * `$ cd ac-local`
>    * `$ git submodule update --init --recursive`
