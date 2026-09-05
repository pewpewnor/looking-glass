package api

import (
	"net/http"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/pewpewnor/looking-glass/backend/internal/inference"
	"github.com/pewpewnor/looking-glass/backend/internal/storage"
)

// NewRouter creates and returns the configured Gin engine.
// dataDir is used to serve stored category images as static files.
func NewRouter(p *inference.Pipeline, s *storage.Store, dataDir string) *gin.Engine {
	r := gin.Default()

	r.Use(cors.New(cors.Config{
		AllowAllOrigins:  true,
		AllowMethods:     []string{"GET", "POST", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept"},
		AllowCredentials: false,
	}))

	// Serve saved support images: GET /images/categories/{cat}/{file}
	r.Static("/images", dataDir)

	h := NewHandlers(p, s)

	api := r.Group("/api")
	{
		api.POST("/categories/upload", h.UploadCategory)
		api.GET("/categories", h.ListCategories)
		api.GET("/categories/:name/images", h.GetCategoryImages)
		api.POST("/query", h.Query)
	}

	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	return r
}
