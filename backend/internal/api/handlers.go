package api

import (
	"fmt"
	"image"
	"mime/multipart"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/pewpewnor/looking-glass/backend/internal/imageutil"
	"github.com/pewpewnor/looking-glass/backend/internal/inference"
	"github.com/pewpewnor/looking-glass/backend/internal/storage"
)

type Handlers struct {
	pipeline *inference.Pipeline
	store    *storage.Store
}

func NewHandlers(p *inference.Pipeline, s *storage.Store) *Handlers {
	return &Handlers{pipeline: p, store: s}
}

func (h *Handlers) UploadCategory(c *gin.Context) {
	categoryName := c.PostForm("category_name")
	if categoryName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "category_name is required"})
		return
	}

	form, err := c.MultipartForm()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid multipart form"})
		return
	}

	files := form.File["images[]"]
	if len(files) == 0 {
		files = form.File["images"]
	}
	if len(files) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "at least one image is required (field: images[] or images)"})
		return
	}
	if len(files) > 10 {
		files = files[:10]
	}

	imgs, err := decodeFileHeaders(files)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	saved, err := h.pipeline.ProcessAndSaveSupportImages(imgs, categoryName)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"category":     categoryName,
		"images_saved": saved,
	})
}

func (h *Handlers) Query(c *gin.Context) {
	categoryName := c.PostForm("category_name")
	if categoryName == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "category_name is required"})
		return
	}

	file, err := c.FormFile("query_image")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "query_image is required"})
		return
	}

	f, err := file.Open()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "cannot open query_image"})
		return
	}
	defer f.Close()

	queryImg, err := imageutil.DecodeImage(f)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "cannot decode query_image: " + err.Error()})
		return
	}

	siameseThresh := parseFloat32(c.PostForm("siamese_threshold"))
	abstainThresh := parseFloat32(c.PostForm("localizer_abstain_threshold"))

	result, err := h.pipeline.Query(categoryName, queryImg, siameseThresh, abstainThresh)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if !result.Found {
		c.JSON(http.StatusOK, gin.H{
			"found":        false,
			"siamese_prob": result.SiameseProb,
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"found":           true,
		"image":           result.ImageBase64,
		"siamese_prob":    result.SiameseProb,
		"localizer_score": result.LocalizerScore,
	})
}

func (h *Handlers) ListCategories(c *gin.Context) {
	names, err := h.store.ListCategories()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, names)
}

func (h *Handlers) GetCategoryImages(c *gin.Context) {
	name := c.Param("name")
	urls, err := h.store.ListCategoryImageURLs(name)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, urls)
}

func decodeFileHeaders(files []*multipart.FileHeader) ([]image.Image, error) {
	imgs := make([]image.Image, 0, len(files))
	for i, fh := range files {
		f, err := fh.Open()
		if err != nil {
			return nil, fmt.Errorf("open file %d: %w", i, err)
		}
		img, err := imageutil.DecodeImage(f)
		f.Close()
		if err != nil {
			return nil, fmt.Errorf("decode file %d (%s): %w", i, fh.Filename, err)
		}
		imgs = append(imgs, img)
	}
	return imgs, nil
}

func parseFloat32(s string) float32 {
	if s == "" {
		return 0
	}
	v, err := strconv.ParseFloat(s, 32)
	if err != nil {
		return 0
	}
	return float32(v)
}
