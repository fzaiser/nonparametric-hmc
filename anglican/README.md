# Anglican Experiments

This folder is for producing the Anglican samples for comparison with our algorithm.

It is a fork of the probabilistic programming language [Anglican](https://bitbucket.org/probprog/anglican/)’s [examples repository](https://bitbucket.org/probprog/anglican-examples/src/master/) with browser-based interaction via [Gorilla REPL](http://gorilla-repl.org/).

## Install

1. Install the Java Runtime Environment **Version 8** (on Ubuntu: `sudo apt install openjdk-8-jdk`) and select it (on Ubuntu: `sudo update-alternatives --config java`). Note that unfortunately, Anglican does not support newer Java versions.
2. Download and install [Leiningen](http://leiningen.org/).

## Usage

1. In this folder, run the command `lein gorilla`.
2. Open the indicated local website in a browser.
3. Hit `Alt+G Alt+L` and select `icml.clj` to load and run the worksheet.
4. Hit `Shift+Enter` to evaluate each code segment in order. The final computations will take several hours.

The generated samples will be stored in the `samples/` directory.
Please refer to the parent `../README.md` for how to evaluate the results.

*Note:* Anglican does not seem to provide a way to set the random seed.
Therefore, each execution can be different, which may explain slight discrepancies with the numbers reported in the paper.
