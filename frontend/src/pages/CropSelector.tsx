// =============================================================================
// AgriLite-Hybrid Frontend
// src/pages/CropSelector.tsx - Crop Selection Page
// 
// Allows user to select which crop they want to analyze before uploading image.
// =============================================================================

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Loader2, AlertCircle, Leaf } from 'lucide-react';
import { useDetection } from '../contexts/DetectionContext';
import { cropsAPI } from '../services/api';
import type { Crop } from '../types';

// ---------------------------------------------------------------------------
// Crop Card Component
// ---------------------------------------------------------------------------
interface CropCardProps {
  crop: Crop;
  isSelected: boolean;
  onSelect: (crop: Crop) => void;
}

const CropCard: React.FC<CropCardProps> = ({ crop, isSelected, onSelect }) => {
  const colorMap: Record<string, string> = {
    tomato: 'from-red-400 to-red-600',
    chilli: 'from-orange-400 to-orange-600',
  };

  const emojiMap: Record<string, string> = {
    tomato: '🍅',
    chilli: '🌶️',
  };

  const gradient = colorMap[crop.name.toLowerCase()] || 'from-green-400 to-green-600';
  const emoji = emojiMap[crop.name.toLowerCase()] || '🌱';

  return (
    <button
      onClick={() => onSelect(crop)}
      className={`
        card p-6 text-left transition-all duration-200 hover:shadow-lg hover:-translate-y-1
        ${isSelected ? 'ring-2 ring-primary-500 ring-offset-2' : ''}
      `}
    >
      <div className={`w-16 h-16 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center text-3xl mb-4`}>
        {emoji}
      </div>
      <h3 className="text-xl font-semibold text-gray-900 mb-1">{crop.display_name || crop.name}</h3>
      <p className="text-sm text-gray-600 mb-4 line-clamp-2">
        {crop.description || `Detect diseases in ${crop.name.toLowerCase()} plants`}
      </p>
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-500">
          {crop.diseases?.length || 0} diseases
        </span>
        {isSelected && (
          <span className="text-primary-600 font-medium flex items-center">
            Selected
            <ArrowRight className="w-4 h-4 ml-1" />
          </span>
        )}
      </div>
    </button>
  );
};

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
const CropSelector: React.FC = () => {
  const navigate = useNavigate();
  const { selectedCrop, selectCrop } = useDetection();
  const [crops, setCrops] = useState<Crop[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // ---------------------------------------------------------------------------
  // Fetch crops on mount
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const fetchCrops = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await cropsAPI.getAll();
        if (response.success && response.data?.crops) {
          setCrops(response.data.crops);
        } else {
          // Fallback data if API fails
          setCrops([
            {
              name: 'tomato',
              display_name: 'Tomato',
              description: 'Detect Early Blight, Late Blight, Leaf Mold, Septoria, and more.',
              image: '/images/crops/tomato.jpg',
              diseases: ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot', 'Spider Mites (Two-spotted)', 'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy', 'Powdery Mildew'],
              color: '#EF4444',
            },
            {
              name: 'chilli',
              display_name: 'Chilli',
              description: 'Detect Anthracnose, Leaf Curl, Damping Off, Whitefly, and more.',
              image: '/images/crops/chilli.jpg',
              diseases: ['Anthracnose', 'Damping Off', 'Leaf Curl Virus', 'Leaf Spot', 'Veinal Mottle Virus', 'Whitefly', 'Yellowish', 'Healthy'],
              color: '#F97316',
            },
          ]);
        }
      } catch (err) {
        console.error('Failed to fetch crops:', err);
        setError('Failed to load crops. Using default data.');
        // Set fallback crops
        setCrops([
          {
            name: 'tomato',
            display_name: 'Tomato',
            description: 'Detect Early Blight, Late Blight, Leaf Mold, Septoria, and more.',
            image: '/images/crops/tomato.jpg',
            diseases: ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot', 'Spider Mites (Two-spotted)', 'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic Virus', 'Healthy', 'Powdery Mildew'],
            color: '#EF4444',
          },
          {
            name: 'chilli',
            display_name: 'Chilli',
            description: 'Detect Anthracnose, Leaf Curl, Damping Off, Whitefly, and more.',
            image: '/images/crops/chilli.jpg',
            diseases: ['Anthracnose', 'Damping Off', 'Leaf Curl Virus', 'Leaf Spot', 'Veinal Mottle Virus', 'Whitefly', 'Yellowish', 'Healthy'],
            color: '#F97316',
          },
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchCrops();
  }, []);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------
  const handleSelectCrop = (crop: Crop) => {
    selectCrop(crop);
  };

  const handleContinue = () => {
    if (selectedCrop) {
      navigate('/upload');
    }
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="max-w-5xl mx-auto px-4 py-8 sm:py-12">
      {/* Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-100 text-primary-600 mb-4">
          <Leaf className="w-8 h-8" />
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-3">
          Select Your Crop
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Choose the crop you want to analyze. Our AI model will identify diseases
          specific to the selected crop.
        </p>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg flex items-center text-yellow-800">
          <AlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Loading State */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
          <span className="ml-3 text-gray-600">Loading crops...</span>
        </div>
      ) : (
        <>
          {/* Crop Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
            {crops.map((crop) => (
              <CropCard
                key={crop.name}
                crop={crop}
                isSelected={selectedCrop?.name === crop.name}
                onSelect={handleSelectCrop}
              />
            ))}
          </div>

          {/* Continue Button */}
          <div className="text-center">
            <button
              onClick={handleContinue}
              disabled={!selectedCrop}
              className={`
                btn text-lg px-8 py-3 font-medium inline-flex items-center
                ${selectedCrop
                  ? 'btn-primary'
                  : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                }
              `}
            >
              Continue to Upload
              <ArrowRight className="w-5 h-5 ml-2" />
            </button>
            {!selectedCrop && (
              <p className="text-sm text-gray-500 mt-3">
                Please select a crop to continue
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
};

export default CropSelector;
