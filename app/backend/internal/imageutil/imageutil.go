package imageutil

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/draw"
	"image/jpeg"
	"image/png"
	"io"
	"math"

	"github.com/disintegration/imaging"
	"github.com/fogleman/gg"
	"github.com/rwcarlsen/goexif/exif"
)

var (
	imagenetMean = [3]float32{0.485, 0.456, 0.406}
	imagenetStd  = [3]float32{0.229, 0.224, 0.225}
)

// Letterbox resizes src to fit within a size×size square while preserving aspect ratio,
// padding the remainder with grey (114,114,114). Returns the padded image, the scale
// factor applied, and the left/top padding offsets.
func Letterbox(src image.Image, size int) (image.Image, float64, int, int) {
	b := src.Bounds()
	origW, origH := b.Dx(), b.Dy()

	scale := float64(size) / math.Max(float64(origW), float64(origH))
	newW := int(math.Round(float64(origW) * scale))
	newH := int(math.Round(float64(origH) * scale))

	resized := imaging.Resize(src, newW, newH, imaging.Lanczos)

	padLeft := (size - newW) / 2
	padTop := (size - newH) / 2

	grey := image.NewNRGBA(image.Rect(0, 0, size, size))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := grey.PixOffset(x, y)
			grey.Pix[i] = 114
			grey.Pix[i+1] = 114
			grey.Pix[i+2] = 114
			grey.Pix[i+3] = 255
		}
	}
	draw.Draw(grey, image.Rect(padLeft, padTop, padLeft+newW, padTop+newH), resized, image.Point{}, draw.Over)

	return grey, scale, padLeft, padTop
}

// ToTensorCHW converts an image to a CHW float32 slice with pixel values in [0,1].
func ToTensorCHW(img image.Image) []float32 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()

	nrgba := toNRGBA(img)
	data := make([]float32, 3*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := nrgba.PixOffset(b.Min.X+x, b.Min.Y+y)
			data[0*h*w+y*w+x] = float32(nrgba.Pix[off]) / 255.0
			data[1*h*w+y*w+x] = float32(nrgba.Pix[off+1]) / 255.0
			data[2*h*w+y*w+x] = float32(nrgba.Pix[off+2]) / 255.0
		}
	}
	return data
}

// ToTensorCHWNormalized converts an image to a CHW float32 slice normalised with ImageNet mean/std.
func ToTensorCHWNormalized(img image.Image) []float32 {
	data := ToTensorCHW(img)
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	n := h * w
	for c := 0; c < 3; c++ {
		for i := 0; i < n; i++ {
			data[c*n+i] = (data[c*n+i] - imagenetMean[c]) / imagenetStd[c]
		}
	}
	return data
}

// DrawBBox draws a thick red axis-aligned rectangle on img and returns a new image.
func DrawBBox(img image.Image, x1, y1, x2, y2 int) image.Image {
	dc := gg.NewContextForImage(img)
	dc.SetRGBA(1, 0, 0, 1)
	dc.SetLineWidth(4)
	dc.DrawRectangle(float64(x1), float64(y1), float64(x2-x1), float64(y2-y1))
	dc.Stroke()
	return dc.Image()
}

// EncodeJPEGBase64 encodes an image as JPEG and returns a base64 string.
func EncodeJPEGBase64(img image.Image) (string, error) {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

// DecodeImage decodes a JPEG or PNG from r and corrects any EXIF orientation so
// the returned image is always in display orientation. This matters for photos
// taken by Android (and iOS) cameras, which store raw sensor data and encode the
// required rotation in the EXIF Orientation tag rather than rotating the pixels.
func DecodeImage(r io.Reader) (image.Image, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return correctEXIFOrientation(img, bytes.NewReader(data)), nil
}

// correctEXIFOrientation reads the EXIF Orientation tag from r and rotates /
// flips img so that its pixels match the intended display orientation.
// If r contains no EXIF data or the tag is absent, img is returned unchanged.
func correctEXIFOrientation(img image.Image, r io.Reader) image.Image {
	x, err := exif.Decode(r)
	if err != nil {
		return img
	}
	tag, err := x.Get(exif.Orientation)
	if err != nil {
		return img
	}
	orientation, err := tag.Int(0)
	if err != nil {
		return img
	}
	// EXIF orientation values and the transform needed to reach normal display:
	//   1 – no-op
	//   2 – flip horizontal
	//   3 – rotate 180°
	//   4 – flip vertical
	//   5 – transpose  (rotate 90° CW + flip horizontal)
	//   6 – rotate 90° CW
	//   7 – transverse (rotate 90° CCW + flip horizontal)
	//   8 – rotate 90° CCW
	switch orientation {
	case 2:
		return imaging.FlipH(img)
	case 3:
		return imaging.Rotate180(img)
	case 4:
		return imaging.FlipV(img)
	case 5:
		return imaging.FlipH(imaging.Rotate270(img))
	case 6:
		return imaging.Rotate270(img)
	case 7:
		return imaging.FlipH(imaging.Rotate90(img))
	case 8:
		return imaging.Rotate90(img)
	}
	return img
}

// ResizeSquare resizes src to size×size (stretching, not letterboxing).
func ResizeSquare(src image.Image, size int) image.Image {
	return imaging.Resize(src, size, size, imaging.Lanczos)
}

// MaskBBox finds the tight bounding box of pixels where mask > threshold.
// mask is a row-major float32 slice of shape [h][w].
// Returns x1, y1, x2, y2 inclusive, and ok=false if no pixel exceeded the threshold.
func MaskBBox(mask []float32, w, h int, threshold float32) (x1, y1, x2, y2 int, ok bool) {
	x1, y1, x2, y2 = w, h, 0, 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if mask[y*w+x] > threshold {
				if x < x1 {
					x1 = x
				}
				if y < y1 {
					y1 = y
				}
				if x > x2 {
					x2 = x
				}
				if y > y2 {
					y2 = y
				}
				ok = true
			}
		}
	}
	return
}

// CropImage returns the sub-image defined by [x1,x2)×[y1,y2) in original pixels,
// scaled from a maskW×maskH mask space to the original image dimensions.
func CropImageFromMaskBBox(img image.Image, maskX1, maskY1, maskX2, maskY2, maskW, maskH int) image.Image {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()

	x1 := int(float64(maskX1) * float64(origW) / float64(maskW))
	y1 := int(float64(maskY1) * float64(origH) / float64(maskH))
	x2 := int(math.Ceil(float64(maskX2+1) * float64(origW) / float64(maskW)))
	y2 := int(math.Ceil(float64(maskY2+1) * float64(origH) / float64(maskH)))

	if x2 > origW {
		x2 = origW
	}
	if y2 > origH {
		y2 = origH
	}
	if x1 < 0 {
		x1 = 0
	}
	if y1 < 0 {
		y1 = 0
	}

	return imaging.Crop(img, image.Rect(x1, y1, x2, y2))
}

// EncodeJPEG encodes an image as JPEG bytes.
func EncodeJPEG(img image.Image) ([]byte, error) {
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 90}); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// EncodePNG encodes an image as PNG bytes.
func EncodePNG(img image.Image) ([]byte, error) {
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func toNRGBA(img image.Image) *image.NRGBA {
	if n, ok := img.(*image.NRGBA); ok {
		return n
	}
	b := img.Bounds()
	dst := image.NewNRGBA(b)
	draw.Draw(dst, b, img, b.Min, draw.Over)
	return dst
}
