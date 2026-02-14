<h1 align="center">PicoGPT: GPT in a QR Code</h1>

<p align="center">
    <img src="qrcode.png" alt="QR Code for The Backdooms">
</p>
<p align="center" style="font-size: small; font-weight: lighter;">
    Yes, it's small enough to fit inside a QR code. You can <a href="https://scanqr.org/">scan it</a> to try it out.
</p>

## Origin

Andrej Karpathy tweeted this on X today, talking about [MicroGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

<p align="center">
    <img src="assets/karpathy-tweet.png" alt="QR Code for The Backdooms">
</p>

It was genuinely so cool but then I saw something that made me laugh out loud

<p align="center">
    <img src="assets/gist-preview.png" alt="QR Code for The Backdooms">
</p>

IT WASN'T MINIFIED

So uh yeah, that's all this is, I minified it further and then went a step further...

## JavaScript Port

I ported the entire thing to **JavaScript** for native browser execution! The minified version at [picogpt.qr.html](picogpt.qr.html) is what goes into the QR code.

It works in **JUST 39 LINES** of pure JavaScript!

For context, these 39 lines include:

- A custom Autograd engine
- Multi-head attention (MHA)
- Feed-forward MLP with GeLU² approximation
- AdamW Optimizer with cosine learning rate schedule
- Training & Inference loops
- Seeded PRNG (xoshiro128**)

### Architecture

```
Layers:    1
Heads:     4  
Embedding: 16
Context:   8
MLP dim:   64
Params:    4,064
```

## QR Code Generation

The `qrgen.py` script is adapted from my **Doom running inside a QR code** project: [The Backdooms](https://github.com/kuberwastaken/backdooms).

The QR code contains a compressed HTML payload using the browser's native `DecompressionStream` API. When scanned:

1. The browser decompresses the gzipped base64 payload
2. Renders the full HTML page with the GPT implementation
3. Trains the model right in your browser!

The entire training loop, inference, and UI fits in a **version-40-L QR code** (2953 bytes).

## Legacy Python Version

The original minified Python implementation is preserved in the `legacy/` folder. It was 64 lines of minified code, but JavaScript allows for native browser execution without any dependencies so that's where we're at.

It's still pretty cool though, you can read it in [legacy/picogpt.py](legacy/picogpt.py).

As for the naming choice, iykyk
