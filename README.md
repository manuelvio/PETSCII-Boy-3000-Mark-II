# PETSCII-Boy 3000 Mark-II

A neural network powered digit recognizer, written in C, runs on a Commodore 64

![Screenshot](https://github.com/user-attachments/assets/557ee16e-a63b-496c-8a69-3457cc09fc4c)

## Introduction

The aim of this project is training and running a neural network in order to recognize handwritten digits. This is a very common and introductory task in neural network literature and it can be easily completed using modern languages and systems, but can become a challenge when implemented in a '80s machine.

It also tries to be as clear and as straightforward as possible in order to be understood by a neophyte (as I am).

The core functions (train and predict) are translated from https://github.com/dlidstrom/NeuralNetworkInAllLangs C# implementation, while some utility units are inspired from the https://github.com/KarolS/millfork counterpart.

This is a C ([Oscar64](https://github.com/drmortalwombat/oscar64)) port of [PETSCII-Boy 3000](https://github.com/manuelvio/PETSCII-Boy-3000), which was written in Mad Pascal. 

## Requirements

No particular hardware is needed. A floppy disk drive and a joystick is enough, if you want to write a digit more freely you can use a light pen.

No REU, no SuperCpu, just a stock C64.

## Training data

The dataset used for training is a compressed version of the one available at https://archive.ics.uci.edu/dataset/178/semeion+handwritten+digit. The original file structure contained 1593 rows of 256 + 10 values: each row represented a handwritten digit in a 16x16 pixel matrix, where every pixel could be either on or off. In the dataset the two states were written as a 1.0 or a 0.0 respectively.

The remaining ten values (either 1 or 0) described the represented digit: a 1 in the fourth position meant that the row was representing a 5, and so on.
