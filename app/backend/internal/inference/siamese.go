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

const (
	siameseImgSize = 518
	siameseKMax    = 10
)

// Siamese wraps the multi-shot existence-check model.
//
// Same GPU drop/restore pattern as RMBG: a CPU session is always alive, the
// GPU session can be released before the Localizer runs and recreated after.
type Siamese struct {
	cpuSession   *ort.DynamicAdvancedSession
	gpuSession   *ort.DynamicAdvancedSession
	modelBytes   []byte
	cudaDeviceID int
	useCUDA      bool
}

func NewSiamese(modelPath string, opts *ort.SessionOptions) (*Siamese, error) {
	cpuSession, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"support_imgs", "support_mask", "query_img"},
		[]string{"existence_prob"},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("siamese: load CPU session: %w", err)
	}

	s := &Siamese{cpuSession: cpuSession}

	if opts != nil {
		data, err := os.ReadFile(modelPath)
		if err != nil {
			cpuSession.Destroy()
			return nil, fmt.Errorf("siamese: read model bytes: %w", err)
		}
		s.modelBytes = data
		s.useCUDA = true

		gpuSession, err := ort.NewDynamicAdvancedSessionWithONNXData(
			data,
			[]string{"support_imgs", "support_mask", "query_img"},
			[]string{"existence_prob"},
			opts,
		)
		if err != nil {
			log.Printf("siamese: GPU session failed (%v) — using CPU", err)
		} else {
			s.gpuSession = gpuSession
		}
	}

	return s, nil
}

func (s *Siamese) Destroy() {
	s.cpuSession.Destroy()
	if s.gpuSession != nil {
		s.gpuSession.Destroy()
	}
}

// DropGPU destroys the GPU session and returns its VRAM to CUDA.
func (s *Siamese) DropGPU() {
	if s.gpuSession != nil {
		s.gpuSession.Destroy()
		s.gpuSession = nil
		log.Println("siamese: GPU session dropped")
	}
}

// RestoreGPU recreates the GPU session from in-memory model bytes.
func (s *Siamese) RestoreGPU(cudaDeviceID int) {
	if !s.useCUDA || s.gpuSession != nil {
		return
	}
	opts, session, err := newGPUSessionFromBytes(s.modelBytes,
		[]string{"support_imgs", "support_mask", "query_img"},
		[]string{"existence_prob"}, cudaDeviceID)
	if opts != nil {
		opts.Destroy()
	}
	if err != nil {
		log.Printf("siamese: GPU session restore failed (%v) — staying on CPU", err)
		return
	}
	s.gpuSession = session
	log.Println("siamese: GPU session restored")
}

// CheckExistence returns the existence probability and the pure inference duration.
func (s *Siamese) CheckExistence(supportImgs []image.Image, queryImg image.Image) (float32, time.Duration, error) {
	const (
		sz   = siameseImgSize
		kMax = siameseKMax
	)

	supportData := make([]float32, kMax*3*sz*sz)
	maskData := make([]float32, kMax)

	k := len(supportImgs)
	if k > kMax {
		k = kMax
	}
	for i := 0; i < k; i++ {
		lb, _, _, _ := imageutil.Letterbox(supportImgs[i], sz)
		chw := imageutil.ToTensorCHW(lb)
		copy(supportData[i*3*sz*sz:], chw)
		maskData[i] = 1.0
	}

	lbQuery, _, _, _ := imageutil.Letterbox(queryImg, sz)
	queryData := imageutil.ToTensorCHW(lbQuery)

	suppTensor, err := ort.NewTensor(ort.NewShape(1, kMax, 3, sz, sz), supportData)
	if err != nil {
		return 0, 0, fmt.Errorf("siamese: support tensor: %w", err)
	}
	defer suppTensor.Destroy()

	maskTensor, err := ort.NewTensor(ort.NewShape(1, kMax), maskData)
	if err != nil {
		return 0, 0, fmt.Errorf("siamese: mask tensor: %w", err)
	}
	defer maskTensor.Destroy()

	queryTensor, err := ort.NewTensor(ort.NewShape(1, 3, sz, sz), queryData)
	if err != nil {
		return 0, 0, fmt.Errorf("siamese: query tensor: %w", err)
	}
	defer queryTensor.Destroy()

	outData := make([]float32, 1)
	outTensor, err := ort.NewTensor(ort.NewShape(1), outData)
	if err != nil {
		return 0, 0, fmt.Errorf("siamese: output tensor: %w", err)
	}
	defer outTensor.Destroy()

	session := s.activeSession()

	start := time.Now()
	if err := session.Run(
		[]ort.ArbitraryTensor{suppTensor, maskTensor, queryTensor},
		[]ort.ArbitraryTensor{outTensor},
	); err != nil {
		return 0, 0, fmt.Errorf("siamese: inference: %w", err)
	}
	elapsed := time.Since(start)

	return outTensor.GetData()[0], elapsed, nil
}

func (s *Siamese) activeSession() *ort.DynamicAdvancedSession {
	if s.gpuSession != nil {
		return s.gpuSession
	}
	return s.cpuSession
}
