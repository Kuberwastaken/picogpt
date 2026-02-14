#!/usr/bin/env python3
"""Generate a QR code that carries a compressed HTML payload via DecompressionStream."""
import zlib, base64, sys, os, qrcode

WRAPPER = (
    '<script type=module>'
    'document.open();document.write(await new Response('
    'new Response(Uint8Array.from(atob("{b64}"),c=>c.charCodeAt(0)))'
    '.body.pipeThrough(new DecompressionStream("gzip"))).text());'
    'document.close()'
    '</script>'
)

def main():
    input_file  = sys.argv[1] if len(sys.argv) > 1 else "picogpt.qr.html"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "qrcode.png"

    with open(input_file, "rb") as f:
        raw = f.read()

    comp = zlib.compressobj(level=9, wbits=31)
    gz   = comp.compress(raw) + comp.flush()
    b64  = base64.b64encode(gz).decode("ascii")

    boot = WRAPPER.format(b64=b64)
    uri  = "data:text/html," + boot
    size = len(uri.encode("utf-8"))

    print(f"Raw: {len(raw)}  Gzip: {len(gz)}  Base64: {len(b64)}  URI: {size}")

    if size > 2953:
        print(f"⚠  {size} bytes exceeds version-40-L capacity (2953)!")
    else:
        print(f"✓  Fits in version-40-L ({2953 - size} bytes to spare)")

    # Save the payload for reference
    with open("payload.txt", "w") as f:
        f.write(uri)

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(uri)
    try:
        qr.make(fit=True)
    except ValueError:
        qr = qrcode.QRCode(version=40, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
        qr.add_data(uri)
        qr.make(fit=False)

    qr.make_image(fill_color="black", back_color="white").save(output_file)
    print(f"QR saved: {output_file} (version {qr.version})")
    print("Scan → open in browser → GPT trains in your browser!")

if __name__ == "__main__":
    main()