package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/iss_group_24_app/internal/api"
	"github.com/iss_group_24_app/internal/inference"
	"github.com/iss_group_24_app/internal/storage"
)

type rawConfig struct {
	Server struct {
		Port int `json:"port"`
	} `json:"server"`
	Models struct {
		RMBGPath      string `json:"rmbg_path"`
		SiamesePath   string `json:"siamese_path"`
		LocalizerPath string `json:"localizer_path"`
	} `json:"models"`
	Thresholds struct {
		SiameseExistence float32 `json:"siamese_existence"`
		LocalizerAbstain float32 `json:"localizer_abstain"`
	} `json:"thresholds"`
	DataDir       string `json:"data_dir"`
	OrtLibPath    string `json:"ort_lib_path"`
	TimingLogPath string `json:"timing_log_path"`
	CUDA          struct {
		Enabled  bool `json:"enabled"`
		DeviceID int  `json:"device_id"`
	} `json:"cuda"`
	Localizer struct {
		PythonPath string `json:"python_path"`
		ScriptPath string `json:"script_path"`
	} `json:"localizer"`
}

func main() {
	cfgPath := "config.json"
	if len(os.Args) > 1 {
		cfgPath = os.Args[1]
	}

	raw, err := os.ReadFile(cfgPath)
	if err != nil {
		log.Fatalf("read config: %v", err)
	}
	var cfg rawConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		log.Fatalf("parse config: %v", err)
	}

	base := filepath.Dir(cfgPath)
	resolvePath := func(p string) string {
		if p == "" || filepath.IsAbs(p) {
			return p
		}
		return filepath.Join(base, p)
	}

	store, err := storage.New(resolvePath(cfg.DataDir))
	if err != nil {
		log.Fatalf("init storage: %v", err)
	}

	pythonPath := cfg.Localizer.PythonPath
	if pythonPath == "" {
		pythonPath = "python3"
	}

	pipelineCfg := inference.Config{
		RMBGPath:            resolvePath(cfg.Models.RMBGPath),
		SiamesePath:         resolvePath(cfg.Models.SiamesePath),
		LocalizerPath:       resolvePath(cfg.Models.LocalizerPath),
		SiameseThreshold:    cfg.Thresholds.SiameseExistence,
		LocalizerAbstain:    cfg.Thresholds.LocalizerAbstain,
		OrtLibPath:          cfg.OrtLibPath,
		TimingLogPath:       resolvePath(cfg.TimingLogPath),
		CUDAEnabled:         cfg.CUDA.Enabled,
		CUDADeviceID:        cfg.CUDA.DeviceID,
		PythonPath:          pythonPath,
		LocalizerScriptPath: resolvePath(cfg.Localizer.ScriptPath),
	}

	log.Println("Loading ONNX models and starting Python worker…")
	pipeline, err := inference.NewPipeline(pipelineCfg, store)
	if err != nil {
		log.Fatalf("init pipeline: %v", err)
	}
	defer pipeline.Destroy()

	router := api.NewRouter(pipeline, store, resolvePath(cfg.DataDir))

	addr := fmt.Sprintf(":%d", cfg.Server.Port)
	log.Printf("Server listening on %s", addr)
	if err := router.Run(addr); err != nil {
		log.Fatalf("server: %v", err)
	}
}
