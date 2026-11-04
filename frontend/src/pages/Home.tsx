// =============================================================================
// AgriLite-Hybrid Frontend
// src/pages/Home.tsx - Landing Page
// 
// Main landing page with hero section, features overview, and supported crops.
// =============================================================================

import React from 'react';
import { Link } from 'react-router-dom';
import {
  Leaf,
  Camera,
  Zap,
  Shield,
  Smartphone,
  ArrowRight,
  CheckCircle2,
} from 'lucide-react';

const Home: React.FC = () => {
  // ---------------------------------------------------------------------------
  // Feature data
  // ---------------------------------------------------------------------------
  const features = [
    {
      icon: Camera,
      title: 'Easy Image Capture',
      description:
        'Simply take a photo of the affected plant leaf using your camera or upload an existing image.',
    },
    {
      icon: Zap,
      title: 'Instant Analysis',
      description:
        'Our AI model analyzes the image in seconds and provides accurate disease identification.',
    },
    {
      icon: Shield,
      title: 'Treatment Recommendations',
      description:
        'Get actionable treatment recommendations with chemical and organic options.',
    },
    {
      icon: Smartphone,
      title: 'Works Everywhere',
      description:
        'Optimized for mobile devices and can work offline on Raspberry Pi edge devices.',
    },
  ];

  // ---------------------------------------------------------------------------
  // Supported crops data
  // ---------------------------------------------------------------------------
  const crops = [
    {
      name: 'Tomato',
      nameLocal: 'टमाटर',
      diseases: 11,
      color: 'bg-red-500',
      image: '🍅',
    },
    {
      name: 'Chilli',
      nameLocal: 'मिर्च',
      diseases: 8,
      color: 'bg-orange-500',
      image: '🌶️',
    },
  ];

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-primary-600 to-primary-800 text-white py-20 lg:py-32 overflow-hidden">
        {/* Background Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0" style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          }} />
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="text-center max-w-3xl mx-auto">
            <div className="flex justify-center mb-6">
              <div className="p-3 bg-white/20 rounded-full">
                <Leaf className="w-12 h-12" />
              </div>
            </div>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold mb-6 leading-tight">
              AI-Powered Plant Disease Detection
            </h1>
            <p className="text-lg sm:text-xl text-primary-100 mb-8 leading-relaxed">
              Protect your crops with instant disease identification. Simply upload
              a photo of your plant's leaf and get accurate diagnosis with treatment
              recommendations.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/select-crop"
                className="btn bg-white text-primary-700 hover:bg-primary-50 text-lg px-8 py-3 font-semibold inline-flex items-center justify-center"
              >
                Start Detection
                <ArrowRight className="w-5 h-5 ml-2" />
              </Link>
              <Link
                to="/register"
                className="btn bg-primary-700 hover:bg-primary-900 text-white border border-white/30 text-lg px-8 py-3"
              >
                Create Account
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 lg:py-24 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              How It Works
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our advanced deep learning model can identify plant diseases with high
              accuracy, helping farmers take timely action.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <div
                key={feature.title}
                className="card p-6 text-center hover:shadow-lg transition-shadow"
              >
                <div className="flex justify-center mb-4">
                  <div className="p-3 bg-primary-100 rounded-full text-primary-600">
                    <feature.icon className="w-8 h-8" />
                  </div>
                </div>
                <div className="flex justify-center items-center mb-2">
                  <span className="text-sm font-medium text-primary-600 bg-primary-50 px-2 py-1 rounded">
                    Step {index + 1}
                  </span>
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600 text-sm">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Supported Crops Section */}
      <section className="py-16 lg:py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Supported Crops
            </h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our model is trained to detect diseases in these commonly grown
              vegetable crops in India.
            </p>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
            {crops.map((crop) => (
              <Link
                key={crop.name}
                to="/select-crop"
                className="card p-6 text-center hover:shadow-lg transition-all hover:-translate-y-1 group"
              >
                <div className="text-5xl mb-4">{crop.image}</div>
                <h3 className="text-lg font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">
                  {crop.name}
                </h3>
                <p className="text-sm text-gray-500 mb-2">{crop.nameLocal}</p>
                <div className="flex items-center justify-center text-sm text-gray-600">
                  <span className={`w-2 h-2 ${crop.color} rounded-full mr-2`}></span>
                  {crop.diseases} diseases detected
                </div>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* Key Benefits Section */}
      <section className="py-16 lg:py-24 bg-primary-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold text-gray-900 mb-6">
                Why Choose AgriLite-Hybrid?
              </h2>
              <ul className="space-y-4">
                {[
                  'High accuracy disease detection using deep learning',
                  'Severity assessment (mild, moderate, severe)',
                  'Grad-CAM heatmaps showing affected areas',
                  'Chemical and organic treatment recommendations',
                  'Detection history tracking for registered users',
                  'Works on mobile devices and Raspberry Pi',
                  'Supports Hindi and English',
                ].map((benefit, index) => (
                  <li key={index} className="flex items-start space-x-3">
                    <CheckCircle2 className="w-6 h-6 text-primary-600 flex-shrink-0 mt-0.5" />
                    <span className="text-gray-700">{benefit}</span>
                  </li>
                ))}
              </ul>
              <div className="mt-8">
                <Link
                  to="/select-crop"
                  className="btn btn-primary text-lg px-6 py-3"
                >
                  Try It Now
                  <ArrowRight className="w-5 h-5 ml-2 inline" />
                </Link>
              </div>
            </div>
            <div className="relative">
              <div className="bg-white rounded-2xl shadow-xl p-8">
                <div className="text-center mb-6">
                  <div className="inline-block p-4 bg-primary-100 rounded-full mb-4">
                    <Leaf className="w-12 h-12 text-primary-600" />
                  </div>
                  <h3 className="text-xl font-semibold text-gray-900">
                    AgriLite-Hybrid Model
                  </h3>
                  <p className="text-gray-600 mt-2">
                    Lightweight CNN optimized for edge deployment
                  </p>
                </div>
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-600">Model Size</span>
                    <span className="font-semibold text-gray-900">&lt; 10 MB</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-600">Inference Time</span>
                    <span className="font-semibold text-gray-900">&lt; 500 ms</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-600">Accuracy</span>
                    <span className="font-semibold text-primary-600">95%+</span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                    <span className="text-gray-600">Classes</span>
                    <span className="font-semibold text-gray-900">31+ diseases</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 bg-gradient-to-r from-primary-600 to-primary-800 text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold mb-4">
            Ready to Protect Your Crops?
          </h2>
          <p className="text-lg text-primary-100 mb-8">
            Start detecting plant diseases today. No registration required for
            basic detection.
          </p>
          <Link
            to="/select-crop"
            className="btn bg-white text-primary-700 hover:bg-primary-50 text-lg px-8 py-3 font-semibold"
          >
            Start Detection Now
          </Link>
        </div>
      </section>
    </div>
  );
};

export default Home;
