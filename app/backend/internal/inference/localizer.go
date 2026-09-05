package inference

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"sync"
	"time"
)

// LocalizerResult holds the un-letterboxed bounding box in original pixel coordinates.
type LocalizerResult struct {
	X1, Y1, X2, Y2 int
	Score           float32
	BgProb          float32
}

// Localizer delegates inference to a persistent Python worker process.
// The worker loads localizer.onnx once at startup with CUDA enabled and stays
// alive for the server lifetime. Each Localize call sends one JSON line to
// stdin and reads one JSON line from stdout — zero per-call model-load cost.
type Localizer struct {
	mu   sync.Mutex
	cmd  *exec.Cmd
	enc  *json.Encoder
	scan *bufio.Scanner
}

func NewLocalizer(pythonPath, scriptPath, modelPath string, cudaDeviceID int) (*Localizer, error) {
	cmd := exec.Command(pythonPath, scriptPath,
		"--model", modelPath,
		"--device-id", fmt.Sprintf("%d", cudaDeviceID),
	)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("localizer: stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("localizer: stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("localizer: stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("localizer: start python worker: %w", err)
	}

	l := &Localizer{
		cmd:  cmd,
		enc:  json.NewEncoder(stdin),
		scan: bufio.NewScanner(stdout),
	}

	// Mirror Python stderr to Go log so CUDA / ORT messages are visible.
	go streamStderr(stderr)

	// Block until the worker signals it is ready (model loaded).
	if err := l.awaitReady(); err != nil {
		cmd.Process.Kill()
		cmd.Wait()
		return nil, fmt.Errorf("localizer worker startup: %w", err)
	}
	return l, nil
}

func (l *Localizer) Destroy() {
	if l.cmd != nil && l.cmd.Process != nil {
		l.cmd.Process.Kill()
		l.cmd.Wait()
	}
}

// Localize sends a request to the Python worker and returns the result.
// supportDir is the on-disk directory of pre-cropped support images.
// queryImageBytes is a JPEG-encoded query image written to a temp file.
func (l *Localizer) Localize(supportDir string, queryImageBytes []byte) (*LocalizerResult, time.Duration, error) {
	tmp, err := os.CreateTemp("", "lg_query_*.jpg")
	if err != nil {
		return nil, 0, fmt.Errorf("localizer: temp file: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(queryImageBytes); err != nil {
		tmp.Close()
		return nil, 0, fmt.Errorf("localizer: write temp: %w", err)
	}
	tmp.Close()

	l.mu.Lock()
	defer l.mu.Unlock()

	req := map[string]string{"support_dir": supportDir, "query_image": tmpPath}

	start := time.Now()
	if err := l.enc.Encode(req); err != nil {
		return nil, 0, fmt.Errorf("localizer: send request: %w", err)
	}
	if !l.scan.Scan() {
		return nil, 0, fmt.Errorf("localizer: worker stdout closed unexpectedly")
	}
	elapsed := time.Since(start)

	var resp struct {
		X1     int     `json:"x1"`
		Y1     int     `json:"y1"`
		X2     int     `json:"x2"`
		Y2     int     `json:"y2"`
		Score  float32 `json:"score"`
		BgProb float32 `json:"bg_prob"`
		Error  string  `json:"error"`
	}
	if err := json.Unmarshal(l.scan.Bytes(), &resp); err != nil {
		return nil, elapsed, fmt.Errorf("localizer: parse response: %w", err)
	}
	if resp.Error != "" {
		return nil, elapsed, fmt.Errorf("localizer worker: %s", resp.Error)
	}

	return &LocalizerResult{
		X1: resp.X1, Y1: resp.Y1, X2: resp.X2, Y2: resp.Y2,
		Score: resp.Score, BgProb: resp.BgProb,
	}, elapsed, nil
}

func (l *Localizer) awaitReady() error {
	if !l.scan.Scan() {
		return fmt.Errorf("worker closed before sending ready signal")
	}
	var msg map[string]interface{}
	if err := json.Unmarshal(l.scan.Bytes(), &msg); err != nil {
		return fmt.Errorf("parse ready message: %w", err)
	}
	if errMsg, ok := msg["error"].(string); ok {
		return fmt.Errorf("%s", errMsg)
	}
	log.Printf("Localizer Python worker ready — providers: %v", msg["providers"])
	return nil
}

func streamStderr(r io.ReadCloser) {
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		log.Printf("[localizer.py] %s", sc.Text())
	}
}
