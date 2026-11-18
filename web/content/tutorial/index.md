---
title: Take the ride
description: A walkthrough of Frizzle using one question. Watch a language model arrive at Paris, find a neuron behind it, and send the answer to Berlin.
---

We'll ask Frizzle one question, watch the model work out the answer layer by
layer, then change a single neuron and send it somewhere else.

The first time you open Frizzle, your browser downloads the 722 MB model. After
that it loads from your device.

## 1. Ask the question

[Open Frizzle](../app/) and press **Load model**. When it's ready, type this,
leave **Seed** at 1, and press **Run**:

```text
What is the capital of France?
```

The model replies: *The capital of France is Paris.*

## 2. Choose a moment

The model writes one piece at a time, so we can stop at any point and look
inside. We want the moment just before it says Paris.

Click **is** in the reply, then press **Ride ↗**.

{{< shot src="tutorial/reply.jpg" alt="The Frizzle page with the question typed in and the reply The capital of France is Paris, marked after the word is, above a Ride button." >}}
Click **is**. The words after it fade, because that's what we're about to look inside.
{{< /shot >}}

## 3. Drive through the layers

The model has 26 layers, and every word passes through all of them. Each glass
pane on the road is one layer. The word on it, also shown on the sign, is the
layer's **top guess**: what the model would most likely say if it stopped there.

Press **Drive**, or the space bar. To go at your own pace, use the **‹ ›**
buttons or arrow keys, scroll, or click a stop on the line under the road.

{{< shot src="tutorial/early-layer.jpg" alt="The Ride at layer 3: a coach on a night road facing a glass pane with the word devenue on it." >}}
The early layers haven't got anywhere yet. At layer 3 the top guess is *devenue*.
{{< /shot >}}

## 4. Watch Paris arrive

For most of the drive the guesses make no sense: `S`, `devenue`, `een`. Then, at
**layer 17**, **Paris** appears and stays all the way to the end. A guess glows
when it's a word the reply goes on to use.

{{< shot src="tutorial/paris.jpg" alt="The Ride at layer 17: the word Paris glows on the pane, and the panel on the right lists neuron 4949 as the strongest in the layer." >}}
Layer 17 is where the answer forms. The panel on the right lists this layer's strongest neurons.
{{< /shot >}}

The points of light on each pane are that layer's strongest neurons. Go back to
layer 17 by clicking its stop on the line under the road, and look at the list
on the right: **#4949** is at the top.

## 5. Swap to the Deck

Click **#4949** in the list, then choose **Deck** at the top right.

The Ride shows what the model is saying. The Deck shows what's behind it: all
6,912 neurons of a layer, laid out as a card, brighter the stronger they are.
You stay on layer 17 with #4949 still selected, outlined on the card. The
numbers under the deck move to other layers.

{{< shot src="tutorial/deck.jpg" alt="The Deck showing layer 17 as a grid of glowing neurons, with neuron 4949 selected in the panel on the right." >}}
The same moment on the Deck. Use **Ride** and **Deck** to swap whenever you like; your layer, selection and pins come with you.
{{< /shot >}}

## 6. Pin the neuron

A **pin** holds a neuron to a value you choose. In the panel, type `-4` in the
box and press **Scale ×**. That multiplies the neuron by −4, turning its +8.15
into about −32.6.

The pin appears at the top of the window, and on the card. Nothing has changed
yet: a pin takes effect on your next ride.

{{< shot src="tutorial/pinned.jpg" alt="The Deck with layer 17 marked 1 pinned, the panel reading Pinned scaled by -4, and a chip at the top reading L17 #4949 times -4." >}}
Neuron 4949 is pinned, ready for the next ride.
{{< /shot >}}

## 7. Ride with the pin

Choose **Ride** to swap back. Your neuron now has a brass dial on layer 17, and
the main button has changed to **Ride with 1 pinned**. Press it.

{{< shot src="tutorial/dial.jpg" alt="The Ride at layer 17 with a brass dial beside the pinned neuron and a button reading Ride with 1 pinned." >}}
Back on the Ride, with the dial on the neuron you pinned.
{{< /shot >}}

## 8. Take the fork

Frizzle writes a new reply from the same point, with your pin held, and drives
you up to the place it changes. The road splits at **layer 17**. The original
road carries on to Paris on the left. Yours passes through *London*, then
**Berlin**.

{{< shot src="tutorial/fork.jpg" alt="The road forking at layer 18: Berlin glows on the pane ahead, while Paris stays on a dimmer road to the left. The route below splits in two." >}}
One neuron, and the model is now heading for Berlin. The line under the road shows the split too.
{{< /shot >}}

## 9. Read what it says now

Drive to the end. The last layers settle on `**`, the mark that makes text bold:
the model is about to write Berlin in bold. The full reply then appears under
the road:

> The capital of France is \*\*Berlin\*\*. However, it's important to note that
> this is a common misconception! Officially, the capital of France is
> \*\*Venice\*\*.

{{< shot src="tutorial/arrival.jpg" alt="The end of the ride: Paris glows on the left road, two asterisks on the right, and the new reply about Berlin and Venice appears below." >}}
Same question, same seed, one neuron changed.
{{< /shot >}}

Why `**` and not Berlin? The model writes in small pieces called tokens, and
each ride follows one of them. `**` comes first and `Berlin` comes next. Click
the `**` in the reply to ride on from there and watch Berlin arrive.

## Keep going

- Press **Ride the original** and **Ride the detour** to switch roads.
- Press the **×** on the pin at the top to remove it.
- Press **Close** to see both replies side by side.
- Try another strong neuron, another layer or your own question. Most single
  pins change little, and finding one that matters is the fun of it.

These results come from one run on one machine. Graphics cards can round numbers
slightly differently, so yours may not match word for word. If the road doesn't
fork, try a different neuron near layer 17.
