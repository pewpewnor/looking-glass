package inference

import (
	"fmt"
	"image"
	"log"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/iss_group_24_app/internal/imageutil"
	"github.com/iss_group_24_app/internal/storage"
	ort "github.com/yalue/onnxruntime_go"
)

// Config holds the three model paths, thresholds, and runtime settings.
type Config struct {
	RMBGPath         string
	SiamesePath      string
	LocalizerPath    string
	SiameseThreshold float32
	LocalizerAbstain float32
	OrtLibPath       string
	TimingLogPath    string
	CUDAEnabled      bool
	CUDADeviceID     int
	// Localizer runs in a Python worker subprocess.
	PythonPath          string // interpreter, e.g. "python3.14"
	LocalizerScriptPath string // path to localizer_worker.py
}

// QueryResult is returned by Pipeline.Query.
type QueryResult struct {
	Found          bool
	ImageBase64    string
	SiameseProb    float32
	LocalizerScore float32
}

// Pipeline loads and owns all three models plus the timing log.
type Pipeline struct {
	rmbg      *RMBG
	siamese   *Siamese
	localizer *Localizer
	store     *storage.Store
	cfg       Config

	timingLog *log.Logger
	logFile   *os.File

	rmbgMu     sync.Mutex
	rmbgLastMs map[string]float64
}

// NewPipeline initialises the ONNX environment, loads RMBG + Siamese on GPU,
// and starts the Python localizer worker.
func NewPipeline(cfg Config, store *storage.Store) (*Pipeline, error) {
	if cfg.OrtLibPath != "" {
		ort.SetSharedLibraryPath(cfg.OrtLibPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("onnxruntime init: %w", err)
	}

	sessionOpts, err := buildSessionOptions(cfg)
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	if sessionOpts != nil {
		defer sessionOpts.Destroy()
	}

	rmbgModel, err := NewRMBG(cfg.RMBGPath, sessionOpts)
	if err != nil {
		return nil, fmt.Errorf("load rmbg: %w", err)
	}
	log.Println("RMBG model loaded")

	siameseModel, err := NewSiamese(cfg.SiamesePath, sessionOpts)
	if err != nil {
		rmbgModel.Destroy()
		return nil, fmt.Errorf("load siamese: %w", err)
	}
	log.Println("Siamese model loaded")

	localizerModel, err := NewLocalizer(cfg.PythonPath, cfg.LocalizerScriptPath, cfg.LocalizerPath, cfg.CUDADeviceID)
	if err != nil {
		rmbgModel.Destroy()
		siameseModel.Destroy()
		return nil, fmt.Errorf("start localizer worker: %w", err)
	}
	log.Println("Localizer Python worker started")

	logPath := cfg.TimingLogPath
	if logPath == "" {
		logPath = "pipeline_timing.log"
	}
	lf, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		rmbgModel.Destroy()
		siameseModel.Destroy()
		localizerModel.Destroy()
		return nil, fmt.Errorf("open timing log %q: %w", logPath, err)
	}
	timingLogger := log.New(lf, "", log.Ldate|log.Ltime)

	p := &Pipeline{
		rmbg:       rmbgModel,
		siamese:    siameseModel,
		localizer:  localizerModel,
		store:      store,
		cfg:        cfg,
		timingLog:  timingLogger,
		logFile:    lf,
		rmbgLastMs: make(map[string]float64),
	}
	timingLogger.Printf("=== server started ===")
	log.Printf("Timing log: %s", logPath)
	return p, nil
}

// Destroy releases all sessions, kills the Python worker, and closes the log.
func (p *Pipeline) Destroy() {
	p.rmbg.Destroy()
	p.siamese.Destroy()
	p.localizer.Destroy()
	ort.DestroyEnvironment()
	p.logFile.Close()
}

// ProcessAndSaveSupportImages crops salient objects via RMBG and saves them.
func (p *Pipeline) ProcessAndSaveSupportImages(imgs []image.Image, categoryName string) (int, error) {
	saved := 0
	var rmbgTotal time.Duration

	for i, img := range imgs {
		cropped, dur, err := p.rmbg.CropSalientObject(img)
		if err != nil {
			log.Printf("rmbg crop image %d: %v (skipping)", i, err)
			continue
		}
		rmbgTotal += dur
		if _, err := p.store.SaveCroppedImage(categoryName, cropped); err != nil {
			return saved, fmt.Errorf("save image %d: %w", i, err)
		}
		saved++
	}

	if saved > 0 {
		rmbgAvg := rmbgTotal / time.Duration(saved)
		p.timingLog.Printf(
			"[UPLOAD]  category=%-20s  images=%d  rmbg_per_image=%6.1fms  rmbg_total=%6.1fms",
			categoryName, saved, ms(rmbgAvg), ms(rmbgTotal),
		)
		p.rmbgMu.Lock()
		p.rmbgLastMs[categoryName] = ms(rmbgTotal)
		p.rmbgMu.Unlock()
	}
	return saved, nil
}

// Query runs the full pipeline: Siamese (Go/ONNX) → Localizer (Python/CUDA).
func (p *Pipeline) Query(
	categoryName string,
	queryImg image.Image,
	siameseThresh float32,
	abstainThresh float32,
) (*QueryResult, error) {
	if siameseThresh <= 0 {
		siameseThresh = p.cfg.SiameseThreshold
	}
	if abstainThresh <= 0 {
		abstainThresh = p.cfg.LocalizerAbstain
	}

	supportImgs, err := p.store.LoadCategoryImages(categoryName)
	if err != nil {
		return nil, fmt.Errorf("load support images: %w", err)
	}
	if len(supportImgs) == 0 {
		return nil, fmt.Errorf("category %q has no support images", categoryName)
	}

	ql := queryLogEntry{
		category:      categoryName,
		siameseThresh: siameseThresh,
		abstainThresh: abstainThresh,
	}

	// --- Stage 1: Siamese existence check (Go ONNX, GPU via RMBG/Siamese sessions) ---
	prob, siameseDur, err := p.siamese.CheckExistence(supportImgs, queryImg)
	if err != nil {
		return nil, fmt.Errorf("siamese: %w", err)
	}
	ql.siameseProb = prob
	ql.siameseDur = siameseDur
	ql.siameseExists = prob >= siameseThresh

	if !ql.siameseExists {
		p.writeQueryLog(ql)
		return &QueryResult{Found: false, SiameseProb: prob}, nil
	}

	// --- Stage 2: Localizer (Python worker, separate CUDA context) ---
	// Encode the query image to JPEG bytes for the Python worker.
	queryJPEG, err := imageutil.EncodeJPEG(queryImg)
	if err != nil {
		return nil, fmt.Errorf("encode query image: %w", err)
	}
	supportDir := p.store.CategoryDir(categoryName)

	loc, localizerDur, err := p.localizer.Localize(supportDir, queryJPEG)
	if err != nil {
		return nil, fmt.Errorf("localizer: %w", err)
	}

	ql.localizerRan = true
	ql.localizerBgProb = loc.BgProb
	ql.localizerScore = loc.Score
	ql.localizerDur = localizerDur
	ql.localizerFound = loc.BgProb < abstainThresh

	if !ql.localizerFound {
		p.writeQueryLog(ql)
		return &QueryResult{Found: false, SiameseProb: prob}, nil
	}

	ql.finalFound = true
	p.writeQueryLog(ql)

	annotated := imageutil.DrawBBox(queryImg, loc.X1, loc.Y1, loc.X2, loc.Y2)
	b64, err := imageutil.EncodeJPEGBase64(annotated)
	if err != nil {
		return nil, fmt.Errorf("encode result: %w", err)
	}

	return &QueryResult{
		Found:          true,
		ImageBase64:    b64,
		SiameseProb:    prob,
		LocalizerScore: loc.Score,
	}, nil
}

// ---- logging ----------------------------------------------------------------

type queryLogEntry struct {
	category      string
	siameseThresh float32
	abstainThresh float32
	siameseProb   float32
	siameseExists bool
	siameseDur    time.Duration
	localizerRan    bool
	localizerBgProb float32
	localizerScore  float32
	localizerFound  bool
	localizerDur    time.Duration
	finalFound      bool
}

func (p *Pipeline) writeQueryLog(ql queryLogEntry) {
	siameseDecision := "NOT_EXISTS"
	if ql.siameseExists {
		siameseDecision = "EXISTS"
	}

	localizerPart := "localizer=SKIPPED"
	if ql.localizerRan {
		dec := "NOT_FOUND"
		if ql.localizerFound {
			dec = "FOUND"
		}
		localizerPart = fmt.Sprintf(
			"localizer: bg_prob=%.4f score=%.4f abstain_thresh=%.4f → %s",
			ql.localizerBgProb, ql.localizerScore, ql.abstainThresh, dec,
		)
	}

	finalStr := "NOT_FOUND"
	if ql.finalFound {
		finalStr = "FOUND"
	}

	resultLine := fmt.Sprintf(
		"[QUERY RESULT] category=%-20s | siamese: prob=%.4f thresh=%.4f → %s | %s | result=%s",
		ql.category, ql.siameseProb, ql.siameseThresh, siameseDecision, localizerPart, finalStr,
	)
	p.timingLog.Print(resultLine)
	log.Print(resultLine)

	siameseMs := ms(ql.siameseDur)
	localizerMs := ms(ql.localizerDur)
	siamLocMs := siameseMs + localizerMs

	p.rmbgMu.Lock()
	rmbgMs := p.rmbgLastMs[ql.category]
	p.rmbgMu.Unlock()

	timingLine := fmt.Sprintf(
		"[QUERY TIMING] category=%-20s | rmbg=%6.1fms  siamese=%6.1fms  localizer=%6.1fms  siamese+localizer=%6.1fms  rmbg+siamese+localizer=%6.1fms",
		ql.category, rmbgMs, siameseMs, localizerMs, siamLocMs, rmbgMs+siamLocMs,
	)
	p.timingLog.Print(timingLine)
	log.Print(timingLine)
}

func ms(d time.Duration) float64 {
	return float64(d.Microseconds()) / 1000.0
}

// ---- helpers ----------------------------------------------------------------

func newGPUSessionFromBytes(
	modelBytes []byte,
	inputNames, outputNames []string,
	cudaDeviceID int,
) (*ort.SessionOptions, *ort.DynamicAdvancedSession, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, nil, fmt.Errorf("session options: %w", err)
	}
	cudaOpts, err := ort.NewCUDAProviderOptions()
	if err != nil {
		opts.Destroy()
		return nil, nil, fmt.Errorf("cuda provider options: %w", err)
	}
	defer cudaOpts.Destroy()
	if err := cudaOpts.Update(map[string]string{
		"device_id":             strconv.Itoa(cudaDeviceID),
		"arena_extend_strategy": "kSameAsRequested",
	}); err != nil {
		opts.Destroy()
		return nil, nil, fmt.Errorf("cuda options: %w", err)
	}
	if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
		opts.Destroy()
		return nil, nil, fmt.Errorf("append cuda: %w", err)
	}
	session, err := ort.NewDynamicAdvancedSessionWithONNXData(modelBytes, inputNames, outputNames, opts)
	if err != nil {
		opts.Destroy()
		return nil, nil, err
	}
	return opts, session, nil
}

func buildSessionOptions(cfg Config) (*ort.SessionOptions, error) {
	if !cfg.CUDAEnabled {
		log.Println("CUDA disabled in config — using CPU execution provider")
		return nil, nil
	}
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}
	cudaOpts, err := ort.NewCUDAProviderOptions()
	if err != nil {
		opts.Destroy()
		log.Printf("WARNING: CUDA provider unavailable (%v) — falling back to CPU", err)
		return nil, nil
	}
	defer cudaOpts.Destroy()
	if err := cudaOpts.Update(map[string]string{
		"device_id":             strconv.Itoa(cfg.CUDADeviceID),
		"arena_extend_strategy": "kSameAsRequested",
	}); err != nil {
		opts.Destroy()
		log.Printf("WARNING: CUDA options update failed (%v) — falling back to CPU", err)
		return nil, nil
	}
	if err := opts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
		opts.Destroy()
		log.Printf("WARNING: append CUDA provider failed (%v) — falling back to CPU", err)
		return nil, nil
	}
	log.Printf("CUDA execution provider enabled (device_id=%d)", cfg.CUDADeviceID)
	return opts, nil
}
