package storage

import (
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
	"github.com/pewpewnor/looking-glass/backend/internal/imageutil"
)

// Store manages the on-disk category / support-image hierarchy.
type Store struct {
	dataDir string
}

func New(dataDir string) (*Store, error) {
	if err := os.MkdirAll(filepath.Join(dataDir, "categories"), 0755); err != nil {
		return nil, err
	}
	return &Store{dataDir: dataDir}, nil
}

// SaveCroppedImage saves img as JPEG under data/categories/{categoryName}/.
// Returns the relative URL path (/images/categories/{category}/{file}).
func (s *Store) SaveCroppedImage(categoryName string, img image.Image) (string, error) {
	dir := s.categoryDir(categoryName)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}

	filename := uuid.New().String() + ".jpg"
	fullPath := filepath.Join(dir, filename)

	f, err := os.Create(fullPath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	if err := jpeg.Encode(f, img, &jpeg.Options{Quality: 92}); err != nil {
		return "", err
	}

	return "/images/categories/" + categoryName + "/" + filename, nil
}

// LoadCategoryImages loads all JPEG images stored for a category.
func (s *Store) LoadCategoryImages(categoryName string) ([]image.Image, error) {
	dir := s.categoryDir(categoryName)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("category %q not found", categoryName)
		}
		return nil, err
	}

	var imgs []image.Image
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !isImageFile(name) {
			continue
		}
		f, err := os.Open(filepath.Join(dir, name))
		if err != nil {
			return nil, err
		}
		img, err := imageutil.DecodeImage(f)
		f.Close()
		if err != nil {
			continue
		}
		imgs = append(imgs, img)
	}
	return imgs, nil
}

// ListCategories returns all category names that have at least one image.
func (s *Store) ListCategories() ([]string, error) {
	base := filepath.Join(s.dataDir, "categories")
	entries, err := os.ReadDir(base)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, err
	}

	var names []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		images, _ := s.ListCategoryImageURLs(e.Name())
		if len(images) > 0 {
			names = append(names, e.Name())
		}
	}
	if names == nil {
		names = []string{}
	}
	return names, nil
}

// ListCategoryImageURLs returns the relative URL paths for a category's images.
func (s *Store) ListCategoryImageURLs(categoryName string) ([]string, error) {
	dir := s.categoryDir(categoryName)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, err
	}

	var urls []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if isImageFile(e.Name()) {
			urls = append(urls, "/images/categories/"+categoryName+"/"+e.Name())
		}
	}
	if urls == nil {
		urls = []string{}
	}
	return urls, nil
}

// CategoryDir returns the absolute path to the directory holding support
// images for categoryName. The directory is not guaranteed to exist.
func (s *Store) CategoryDir(name string) string {
	return s.categoryDir(name)
}

func (s *Store) categoryDir(name string) string {
	return filepath.Join(s.dataDir, "categories", name)
}

func isImageFile(name string) bool {
	lower := strings.ToLower(name)
	return strings.HasSuffix(lower, ".jpg") || strings.HasSuffix(lower, ".jpeg") || strings.HasSuffix(lower, ".png")
}
