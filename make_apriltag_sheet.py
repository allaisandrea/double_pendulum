#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["reportlab>=4.0"]
# ///
"""Build a printable PDF sheet of AprilTags (tag36h11) at exact physical sizes.

Tags are drawn as vector rectangles, so they stay crisp at any printer DPI.

Size convention: the quoted (nominal) size fixes the module pitch at
nominal/10 -- that is, the tag as it would be with one module of white
padding per side. tag36h11 is 8 modules across, so the black square is
always 0.8x nominal whatever the padding. Padding then widens the cut
square around it:

    1 module  ->  cut square = 1.0x nominal
    2 modules ->  cut square = 1.2x nominal

One page is emitted per padding setting, so the same tag at the same
physical scale appears on each page with a different cut margin. The black
square is what a pose estimator wants as `tag_size`; every caption gives it.

Usage:
    ./make_apriltag_sheet.py                        # ids 0 1 2, 15..50 mm, pads 1 and 2
    ./make_apriltag_sheet.py --ids 4 5 6 --pads 2 --page letter --out tags.pdf
"""

import argparse

from reportlab.lib.pagesizes import A4, LETTER
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# tag36h11 codewords (36 data bits each), from the AprilTag 3 family definition.
TAG36H11_CODES = {
    0: 0xD5D628584,
    1: 0xD97F18B49,
    2: 0xDD280910E,
    3: 0xE479E9C98,
    4: 0xEBCBCA822,
    5: 0xF31DAB3AC,
    6: 0x056A5D085,
    7: 0x10652E1D4,
    8: 0x22B1DFEAD,
    9: 0x265AD0472,
}

DATA_DIM = 6  # tag36h11 carries 6x6 data bits
TAG_DIM = DATA_DIM + 2  # plus the 1-module black border on every side
NOMINAL_DIM = TAG_DIM + 2  # modules spanned by the nominal size: sets the pitch

CUT_GRAY = 0.65  # light enough not to read as a tag edge to the detector


def tag_bits(tag_id):
    """Return TAG_DIM x TAG_DIM booleans; True means a black module."""
    try:
        code = TAG36H11_CODES[tag_id]
    except KeyError:
        raise SystemExit(
            f"tag id {tag_id} not in the embedded table "
            f"(available: {sorted(TAG36H11_CODES)})"
        )
    grid = [[True] * TAG_DIM for _ in range(TAG_DIM)]
    for i in range(DATA_DIM * DATA_DIM):
        bit = (code >> (DATA_DIM * DATA_DIM - 1 - i)) & 1
        row, col = divmod(i, DATA_DIM)
        grid[row + 1][col + 1] = not bit  # a set bit is white
    return grid


def geometry(nominal_mm, pad_modules):
    """Module pitch, black-square edge and cut-square edge, all in mm."""
    module = nominal_mm / NOMINAL_DIM
    return module, TAG_DIM * module, (TAG_DIM + 2 * pad_modules) * module


def draw_tag(c, x, y, nominal_mm, tag_id, pad_modules):
    """Draw one tag; (x, y) is the lower-left corner of the cut square."""
    module_mm, black_mm, cut_mm = geometry(nominal_mm, pad_modules)
    module, black, cut = module_mm * mm, black_mm * mm, cut_mm * mm
    grid = tag_bits(tag_id)

    # White padding, then the cut line on its boundary.
    c.setFillColorRGB(1, 1, 1)
    c.rect(x, y, cut, cut, stroke=0, fill=1)
    c.setStrokeColorRGB(CUT_GRAY, CUT_GRAY, CUT_GRAY)
    c.setLineWidth(0.25)
    c.rect(x, y, cut, cut, stroke=1, fill=0)

    # One solid black square, then the white modules painted on top with a
    # hairline overlap: abutting fills would otherwise leave visible seams.
    bx, by = x + pad_modules * module, y + pad_modules * module
    c.setFillColorRGB(0, 0, 0)
    c.rect(bx, by, black, black, stroke=0, fill=1)
    eps = 0.02 * module
    c.setFillColorRGB(1, 1, 1)
    for row in range(TAG_DIM):
        for col in range(TAG_DIM):
            if not grid[row][col]:
                # row 0 is the top row of the tag, PDF y grows upwards
                c.rect(bx + col * module - eps,
                       by + (TAG_DIM - 1 - row) * module - eps,
                       module + 2 * eps, module + 2 * eps, stroke=0, fill=1)


def draw_ruler(c, x, y, length_mm=100):
    """A labelled reference line for checking that the print scale is 1:1."""
    c.setLineWidth(0.5)
    c.setStrokeColorRGB(0, 0, 0)
    c.line(x, y, x + length_mm * mm, y)
    for i in range(length_mm // 10 + 1):
        tick = 3 * mm if i % 5 == 0 else 1.5 * mm
        c.line(x + i * 10 * mm, y, x + i * 10 * mm, y + tick)
    c.setFont("Helvetica", 7)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(x + length_mm * mm + 3 * mm, y - 0.5,
                 f"scale check: this line must measure exactly {length_mm} mm")


def build(path, ids, sizes, pads, pagesize, family="tag36h11"):
    page_w, page_h = pagesize
    margin = 11 * mm
    caption_h, gap = 5 * mm, 6 * mm
    col_w = (page_w - 2 * margin) / len(ids)
    sizes = sorted(sizes, reverse=True)

    c = canvas.Canvas(path, pagesize=pagesize)
    c.setTitle(f"AprilTag {family} calibration sheet")
    first = True

    for pad in pads:
        def new_page():
            nonlocal first
            if not first:
                c.showPage()
            first = False
            c.setFillColorRGB(0, 0, 0)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, page_h - margin,
                         f"AprilTag {family} — ids {', '.join(map(str, ids))}"
                         f" — {pad} module border")
            c.setFont("Helvetica", 8)
            ratio = (TAG_DIM + 2 * pad) / NOMINAL_DIM
            c.drawString(
                margin, page_h - margin - 5 * mm,
                "Print at 100% / 'Actual size' — do not scale to fit.  "
                f"Cut on the grey line: {ratio:g}x the nominal size.  "
                "Black square (use this as tag_size) = 0.8x nominal.")
            draw_ruler(c, margin, margin)
            return page_h - margin - 12 * mm

        top = new_page()
        for nominal in sizes:
            _, black_mm, cut_mm = geometry(nominal, pad)
            block_h = cut_mm * mm + caption_h
            if top - block_h < margin + 12 * mm:
                top = new_page()
            base = top - cut_mm * mm  # y of the cut square's bottom edge
            for i, tag_id in enumerate(ids):
                x = margin + i * col_w
                draw_tag(c, x, base, nominal, tag_id, pad)
                c.setFillColorRGB(0, 0, 0)
                c.setFont("Helvetica", 6.5)
                c.drawString(x, base - 3.5 * mm,
                             f"id {tag_id}  ·  {nominal:g} mm nominal  ·  "
                             f"tag {black_mm:g} mm  ·  cut {cut_mm:g} mm")
            top -= block_h + gap

    c.save()
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2],
                   help="tag36h11 ids (default: 0 1 2)")
    p.add_argument("--sizes", type=float, nargs="+",
                   default=[15, 20, 30, 40, 50],
                   help="nominal edge lengths in mm (default: 15 20 30 40 50)")
    p.add_argument("--pads", type=int, nargs="+", default=[1, 2],
                   help="white border in modules; one page each (default: 1 2)")
    p.add_argument("--page", choices=["a4", "letter"], default="a4")
    p.add_argument("--out", default="apriltags.pdf")
    a = p.parse_args()
    out = build(a.out, a.ids, a.sizes, a.pads,
                A4 if a.page == "a4" else LETTER)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
