import React, { useState } from 'react';
import './App.css';

const mockModels = [
  { id: 'm1', name: 'ResNet50 không finetune' },
  { id: 'm2', name: 'ResNet50 có finetune' },
  { id: 'm3', name: 'ResNet50 + SVM không finetune' },
  { id: 'm4', name: 'ResNet50 + SVM có finetune' },
  { id: 'm5', name: 'MobileNetV2 không finetune' },
  { id: 'm6', name: 'MobileNetV2 có finetune' },
  { id: 'm7', name: 'MobileNetV2 + SVM không finetune' },
  { id: 'm8', name: 'MobileNetV2 + SVM có finetune' },
  { id: 'm9', name: 'VGG16 không finetune' },
  { id: 'm10', name: 'VGG16 có finetune' },
  { id: 'm11', name: 'VGG16 + SVM không finetune' },
  { id: 'm12', name: 'VGG16 + SVM có finetune' },
  { id: 'm13', name: 'Mô hình 1' },
  { id: 'm14', name: 'Mô hình 2' },
  { id: 'm15', name: 'Mô hình 3' },
  { id: 'm16', name: 'Mô hình 4' },
  { id: 'm17', name: 'Mô hình 5' },
  { id: 'm18', name: 'Mô hình 6' },
];


function App() {
  const [images, setImages] = useState([]);
  const [models] = useState(mockModels);
  const [selectedModels, setSelectedModels] = useState(new Set());
  const [results, setResults] = useState([]);
  const [showResults, setShowResults] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleUpload = (event) => {
    const files = Array.from(event.target.files);
    const newImages = files.map((file, index) => ({
      id: Date.now() + index,
      name: file.name,
      url: URL.createObjectURL(file),
      file: file,
    }));
    setImages(prev => [...prev, ...newImages]);
  };

  const handleModelToggle = (modelId) => {
    const newSelection = new Set(selectedModels);
    if (newSelection.has(modelId)) {
      newSelection.delete(modelId);
    } else {
      newSelection.add(modelId);
    }
    setSelectedModels(newSelection);
  };

  const handleRemoveImage = (idToRemove) => {
    setImages(prev => prev.filter(image => image.id !== idToRemove));
  };

  const handleStart = async () => {
    const chosenModels = models.filter(model => selectedModels.has(model.id));
    if (images.length === 0 || chosenModels.length === 0) return;

    setShowResults(true);
    setProgress(0);

    const total = images.length;
    const resultsArray = [];

    for (let i = 0; i < images.length; i++) {
      const formData = new FormData();
      formData.append('images', images[i].file);
      chosenModels.forEach(model => {
        formData.append('models', model.name);
      });

      try {
        const res = await fetch("http://localhost:8000/predict/", {
          method: "POST",
          body: formData,
        });
        const data = await res.json();
        resultsArray.push(...data.results);
      } catch (error) {
        console.error("Lỗi khi gọi API:", error);
      }

      setProgress(Math.round(((i + 1) / total) * 100));
    }

    setResults(resultsArray);
  };

  const handleSelectAllModels = () => {
    if (selectedModels.size === models.length) {
      // Nếu đã chọn hết thì bỏ chọn tất cả
      setSelectedModels(new Set());
    } else {
      // Chọn tất cả
      const all = new Set(models.map((m) => m.id));
      setSelectedModels(all);
    }
  };

  const getColorClassForLabel = (label) => {
    const lower = label.toLowerCase();
    if (lower.includes("glioma")) return "label-glioma";
    if (lower.includes("meningioma")) return "label-meningioma";
    if (lower.includes("pituitary")) return "label-pituitary";
    if (lower.includes("notumor") || lower.includes("no tumor")) return "label-notumor";
    return "label-default";
  };

  return (
    <div className="container">
      <header className="main-header">
        <div className="header-content">
          <img src="/images/logo.png" alt="Logo" className="logo-icon" />
          <span className="logo-text">VisionIS</span>
        </div>
      </header>

      <div className="top-section">
        {/* Upload Panel */}
        <div className="panel upload-panel">
          <div className="panel-header">
            <h2>Upload image ({images.length})</h2>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <label htmlFor="upload-input" className="upload-btn">Upload</label>
                  {images.length > 0 && (
                    <label onClick={() => setImages([])} className="upload-btn" style={{ cursor: 'pointer' }}>
                      Xoá tất cả
                    </label>
                  )}
                </div>
            <input
              type="file"
              id="upload-input"
              accept="image/*"
              multiple
              style={{ display: 'none' }}
              onChange={handleUpload}
            />
          </div>
          <ul className="item-list">
            {images.map((image) => (
              <li key={image.id} className="list-item image-item">
                <div className="item-left">
                  <img src={image.url} alt={image.name} className="image-thumb" />
                  <span className="image-name">{image.name}</span>
                </div>
                <button className="delete-btn" onClick={() => handleRemoveImage(image.id)}>−</button>
              </li>
            ))}
          </ul>
        </div>

        {/* Model Panel */}
        <div className="panel model-panel">
          <div className="panel-header">
            <h2>Mô hình</h2>
            <label onClick={handleSelectAllModels} className="upload-btn" style={{ cursor: 'pointer' }}>
              {selectedModels.size === models.length ? "Bỏ chọn tất cả" : "Chọn tất cả"}
            </label>
          </div>

          <ul className="item-list">
            {models.map((model) => (
              <li key={model.id} className="list-item">
                <input
                  type="checkbox"
                  id={model.id}
                  checked={selectedModels.has(model.id)}
                  onChange={() => handleModelToggle(model.id)}
                />
                <label htmlFor={model.id}>{model.name}</label>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <button className="start-btn" onClick={handleStart}>Start</button>
      {showResults && (
        <div className="progress-bar-container">
          <div className="progress-bar" style={{ width: `${progress}%` }}></div>
          <span>{progress}%</span>
        </div>
      )}

      {showResults && (
        <div className="results-panel">
          <h2>Kết quả phân loại</h2>
          {results.map((result, index) => (
            <div key={index} className="result-group">
              <div className="result-image-info">
                <img src={images.find(img => img.name === result.imageName)?.url} alt={result.imageName} className="result-image" />
                <span>{result.imageName}</span>
              </div>
              <div className="result-details">
                {result.classifications.map((classification, cIndex) => (
                  <div key={cIndex} className="result-item">
                    <span>{classification.modelName}</span>
                    <button className={`result-btn ${getColorClassForLabel(classification.result)}`}>
                      {classification.result}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;
