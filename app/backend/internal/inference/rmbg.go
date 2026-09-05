package inference

import (
	"fmt"
	"image"
	"log"
	"os"
	"time"

	"github.com/iss_group_24_app/internal/imageutil"
	ort "github.com/yalue/onnxruntime_go"
)

const rmbgSize = 1024

// RMBG wraps the RMBG-1.4 salient-object-detection model.
//
// A CPU session is always kept alive.  When CUDA is enabled a GPU session is
// also created from in-memory model bytes and kept alive for fast uploads.
// DropGPU / RestoreGPU allow the pipeline to temporarily release all GPU
// memory held by this model so the Localizer can use it.
type RMBG struct {
	cpuSession   *ort.DynamicAdvancedSession
	gpuSession   *ort.DynamicAdvancedSession // nil when GPU is not in use
	modelBytes   []byte
	cudaDeviceID int
	useCUDA      bool
}

func NewRMBG(modelPath string, opts *ort.SessionOptions) (*RMBG, error) {
	cpuSession, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"output"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("rmbg: load CPU session: %w", err)
	}

	r := &RMBG{cpuSession: cpuSession}

	if opts != nil {
		data, err := os.ReadFile(modelPath)
		if err != nil {
			cpuSession.Destroy()
			return nil, fmt.Errorf("rmbg: read model bytes: %w", err)
		}
		r.modelBytes = data
		r.useCUDA = true

		gpuSession, err := ort.NewDynamicAdvancedSessionWithONNXData(
			data,
			[]string{"input"},
			[]string{"output"},
			opts,
		)
		if err != nil {
			log.Printf("rmbg: GPU session failed (%v) — using CPU", err)
		} else {
			r.gpuSession = gpuSession
		}
	}

	return r, nil
}

func (r *RMBG) Destroy() {
	r.cpuSession.Destroy()
	if r.gpuSession != nil {
		r.gpuSession.Destroy()
	}
}

// DropGPU destroys the GPU session and returns its VRAM to CUDA.
func (r *RMBG) DropGPU() {
	if r.gpuSession != nil {
		r.gpuSession.Destroy()
		r.gpuSession = nil
		log.Println("rmbg: GPU session dropped")
	}
}

// RestoreGPU recreates the GPU session from the in-memory model bytes.
func (r *RMBG) RestoreGPU(cudaDeviceID int) {
	if !r.useCUDA || r.gpuSession != nil {
		return
	}
	opts, session, err := newGPUSessionFromBytes(r.modelBytes,
		[]string{"input"}, []string{"output"}, cudaDeviceID)
	if opts != nil {
		opts.Destroy()
	}
	if err != nil {
		log.Printf("rmbg: GPU session restore failed (%v) — staying on CPU", err)
		return
	}
	r.gpuSession = session
	log.Println("rmbg: GPU session restored")
}

// CropSalientObject runs inference and returns the cropped image plus the
// pure model inference duration.
func (r *RMBG) CropSalientObject(img image.Image) (image.Image, time.Duration, error) {
	session := r.activeSession()

	resized := imageutil.ResizeSquare(img, rmbgSize)
	inputData := imageutil.ToTensorCHWNormalized(resized)

	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, rmbgSize, rmbgSize), inputData)
	if err != nil {
		return nil, 0, fmt.Errorf("rmbg: input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputData := make([]float32, 1*1*rmbgSize*rmbgSize)
	outputTensor, err := ort.NewTensor(ort.NewShape(1, 1, rmbgSize, rmbgSize), outputData)
	if err != nil {
		return nil, 0, fmt.Errorf("rmbg: output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	start := time.Now()
	if err := session.Run(
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
	); err != nil {
		return nil, 0, fmt.Errorf("rmbg: inference: %w", err)
	}
	elapsed := time.Since(start)

	mask := outputTensor.GetData()
	x1, y1, x2, y2, ok := imageutil.MaskBBox(mask, rmbgSize, rmbgSize, 0.5)
	if !ok {
		return img, elapsed, nil
	}
	return imageutil.CropImageFromMaskBBox(img, x1, y1, x2, y2, rmbgSize, rmbgSize), elapsed, nil
}

func (r *RMBG) activeSession() *ort.DynamicAdvancedSession {
	if r.gpuSession != nil {
		return r.gpuSession
	}
	return r.cpuSession
}
