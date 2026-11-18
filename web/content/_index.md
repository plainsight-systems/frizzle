---
title: Frizzle
lede: Reach inside a language model, pin its neurons, and read what it says instead.
note: Gemma 3 1B runs entirely in your browser on WebGPU. Ask it something, ride through its 26 layers, turn a neuron up or down, and watch the answer change course.
---

## What you can do

Ask the model something and it writes a reply. Pick any word of that reply
and ride inside: you drive through the model's 26 layers and watch its next
word take shape, from nonsense to the answer.

Pin any of its neurons to a value of your choosing and ride again. The road
forks at the layer where your pin changes the model's mind, and Frizzle
writes the new reply from there, from the same seed, beside the one the
model first wrote. Every difference between the two is yours.

For a closer look, the Deck lays out all 6,912 neurons of every layer as
cards you can flip through.

New here? [Take the ride](tutorial/): one question, from Paris to Berlin.

## What it needs

- A browser with WebGPU. Frizzle is built and tried in Chrome on a Mac;
  other browsers with WebGPU may work but are untried. If yours has none,
  Frizzle says so.
- A one-time download of 722 MB, the model file, kept on your device for
  next time.
- Nothing else. Nothing runs on a server: the model, the engine and every
  value you see stay in your browser.

## How it is built

The engine is C++ compiled to WebAssembly, running every step of the model
on your GPU. The page is React and three.js. The model is Google's Gemma 3
1B, instruction-tuned, in its 4-bit Q4_0 build. The source is on
[GitHub](https://github.com/plainsight-systems/frizzle).
