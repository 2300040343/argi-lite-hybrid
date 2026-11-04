// =============================================================================
// AgriLite-Hybrid Frontend
// src/components/Footer.tsx - Application Footer Component
// 
// Simple footer with project info and links.
// =============================================================================

import React from 'react';
import { Link } from 'react-router-dom';
import { Leaf, Github, ExternalLink } from 'lucide-react';

const Footer: React.FC = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-gray-50 border-t border-gray-200 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand Section */}
          <div className="space-y-4">
            <Link to="/" className="flex items-center space-x-2 text-primary-600">
              <Leaf className="w-6 h-6" />
              <span className="font-bold text-lg">AgriLite-Hybrid</span>
            </Link>
            <p className="text-sm text-gray-600">
              AI-powered plant disease detection for Indian agriculture.
              Supporting farmers with early disease identification and treatment
              recommendations.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <Link
                  to="/"
                  className="text-sm text-gray-600 hover:text-primary-600 transition-colors"
                >
                  Home
                </Link>
              </li>
              <li>
                <Link
                  to="/select-crop"
                  className="text-sm text-gray-600 hover:text-primary-600 transition-colors"
                >
                  Detect Disease
                </Link>
              </li>
              <li>
                <Link
                  to="/history"
                  className="text-sm text-gray-600 hover:text-primary-600 transition-colors"
                >
                  Detection History
                </Link>
              </li>
            </ul>
          </div>

          {/* Supported Crops */}
          <div>
            <h3 className="font-semibold text-gray-900 mb-4">Supported Crops</h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-center space-x-2">
                <span className="w-2 h-2 bg-red-500 rounded-full"></span>
                <span>Tomato</span>
              </li>
              <li className="flex items-center space-x-2">
                <span className="w-2 h-2 bg-orange-500 rounded-full"></span>
                <span>Chilli</span>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gray-200 mt-8 pt-6 flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0">
          <p className="text-sm text-gray-500">
            &copy; {currentYear} AgriLite-Hybrid. Final Year Project & Research.
          </p>
          <div className="flex items-center space-x-4">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-500 hover:text-gray-700 transition-colors"
              aria-label="GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
            <a
              href="#"
              className="text-sm text-gray-500 hover:text-primary-600 transition-colors flex items-center space-x-1"
            >
              <span>Documentation</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
