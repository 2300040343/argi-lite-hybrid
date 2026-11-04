// =============================================================================
// AgriLite-Hybrid Frontend
// src/pages/History.tsx - Detection History Page
// 
// Displays user's past detection results with filtering and statistics.
// =============================================================================

import React, { useEffect, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  History as HistoryIcon,
  Loader2,
  AlertCircle,
  Trash2,
  Eye,
  Calendar,
  Filter,
  ChevronLeft,
  ChevronRight,
  RefreshCw,
  Leaf,
  BarChart3,
  XCircle,
} from 'lucide-react';
import { historyAPI } from '../services/api';
import type { DetectionHistory, HistoryStats, PaginationMeta } from '../types';
import toast from 'react-hot-toast';

// ---------------------------------------------------------------------------
// Severity Badge Component
// ---------------------------------------------------------------------------
const SeverityBadge: React.FC<{ severity: string }> = ({ severity }) => {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-full severity-${severity.toLowerCase()}`}
    >
      {severity}
    </span>
  );
};

// ---------------------------------------------------------------------------
// Stats Card Component
// ---------------------------------------------------------------------------
interface StatsCardProps {
  stats: HistoryStats | null;
  loading: boolean;
}

const StatsCard: React.FC<StatsCardProps> = ({ stats, loading }) => {
  if (loading) {
    return (
      <div className="card p-6 flex items-center justify-center">
        <Loader2 className="w-6 h-6 animate-spin text-primary-600" />
      </div>
    );
  }

  if (!stats) return null;

  return (
    <div className="card p-6 mb-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
        <BarChart3 className="w-5 h-5 mr-2 text-primary-600" />
        Detection Statistics
      </h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center p-3 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">{stats.total_detections}</div>
          <div className="text-sm text-gray-500">Total Detections</div>
        </div>
        <div className="text-center p-3 bg-green-50 rounded-lg">
          <div className="text-2xl font-bold text-green-600">{stats.healthy_count}</div>
          <div className="text-sm text-gray-500">Healthy</div>
        </div>
        <div className="text-center p-3 bg-yellow-50 rounded-lg">
          <div className="text-2xl font-bold text-yellow-600">
            {(stats.severity_distribution?.mild || 0) + (stats.severity_distribution?.moderate || 0)}
          </div>
          <div className="text-sm text-gray-500">Mild/Moderate</div>
        </div>
        <div className="text-center p-3 bg-red-50 rounded-lg">
          <div className="text-2xl font-bold text-red-600">
            {stats.severity_distribution?.severe || 0}
          </div>
          <div className="text-sm text-gray-500">Severe</div>
        </div>
      </div>
      {stats.crop_distribution && Object.keys(stats.crop_distribution).length > 0 && (
        <div className="mt-4 pt-4 border-t">
          <p className="text-sm text-gray-500 mb-2">By Crop:</p>
          <div className="flex flex-wrap gap-2">
            {Object.entries(stats.crop_distribution).map(([crop, count]) => (
              <span
                key={crop}
                className="px-3 py-1 bg-primary-50 text-primary-700 text-sm rounded-full"
              >
                {crop}: {count}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// History Item Component
// ---------------------------------------------------------------------------
interface HistoryItemProps {
  item: DetectionHistory;
  onDelete: (id: number) => void;
  onView: (item: DetectionHistory) => void;
}

const HistoryItem: React.FC<HistoryItemProps> = ({ item, onDelete, onView }) => {
  const formattedDate = new Date(item.created_at).toLocaleString('en-IN', {
    dateStyle: 'medium',
    timeStyle: 'short',
  });

  return (
    <div className="card p-4 hover:shadow-md transition-shadow">
      <div className="flex items-start gap-4">
        {/* Thumbnail */}
        <div className="w-20 h-20 flex-shrink-0 rounded-lg overflow-hidden bg-gray-100">
          {item.original_image ? (
            <img
              src={
                item.original_image.startsWith('data:')
                  ? item.original_image
                  : `data:image/jpeg;base64,${item.original_image}`
              }
              alt={`${item.crop} - ${item.disease}`}
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <Leaf className="w-8 h-8 text-gray-400" />
            </div>
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <div>
              <h4 className="font-semibold text-gray-900 truncate">{item.disease}</h4>
              <p className="text-sm text-gray-600">{item.crop}</p>
            </div>
            <SeverityBadge severity={item.severity} />
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-gray-500">
            <span className="flex items-center">
              <Calendar className="w-3.5 h-3.5 mr-1" />
              {formattedDate}
            </span>
            <span>
              Confidence: {Math.round(item.confidence * 100)}%
            </span>
          </div>

          {item.notes && (
            <p className="mt-2 text-sm text-gray-600 line-clamp-1">{item.notes}</p>
          )}
        </div>

        {/* Actions */}
        <div className="flex flex-col gap-2">
          <button
            onClick={() => onView(item)}
            className="p-2 text-gray-500 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
            title="View details"
          >
            <Eye className="w-5 h-5" />
          </button>
          <button
            onClick={() => onDelete(item.id)}
            className="p-2 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            title="Delete"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Detail Modal Component
// ---------------------------------------------------------------------------
interface DetailModalProps {
  item: DetectionHistory | null;
  onClose: () => void;
}

const DetailModal: React.FC<DetailModalProps> = ({ item, onClose }) => {
  if (!item) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex items-start justify-between mb-4">
            <h3 className="text-xl font-semibold text-gray-900">Detection Details</h3>
            <button
              onClick={onClose}
              className="p-1 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <XCircle className="w-6 h-6 text-gray-500" />
            </button>
          </div>

          {/* Images */}
          <div className="grid grid-cols-2 gap-4 mb-6">
            {item.original_image && (
              <div>
                <p className="text-sm text-gray-500 mb-2">Original Image</p>
                <img
                  src={
                    item.original_image.startsWith('data:')
                      ? item.original_image
                      : `data:image/jpeg;base64,${item.original_image}`
                  }
                  alt="Original"
                  className="w-full rounded-lg"
                />
              </div>
            )}
            {item.heatmap_image && (
              <div>
                <p className="text-sm text-gray-500 mb-2">Heatmap Analysis</p>
                <img
                  src={
                    item.heatmap_image.startsWith('data:')
                      ? item.heatmap_image
                      : `data:image/jpeg;base64,${item.heatmap_image}`
                  }
                  alt="Heatmap"
                  className="w-full rounded-lg"
                />
              </div>
            )}
          </div>

          {/* Details */}
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Crop:</span>
              <span className="ml-2 font-medium text-gray-900">{item.crop}</span>
            </div>
            <div>
              <span className="text-gray-500">Disease:</span>
              <span className="ml-2 font-medium text-gray-900">{item.disease}</span>
            </div>
            <div>
              <span className="text-gray-500">Severity:</span>
              <span className="ml-2">
                <SeverityBadge severity={item.severity} />
              </span>
            </div>
            <div>
              <span className="text-gray-500">Confidence:</span>
              <span className="ml-2 font-medium text-gray-900">
                {Math.round(item.confidence * 100)}%
              </span>
            </div>
            <div className="col-span-2">
              <span className="text-gray-500">Date:</span>
              <span className="ml-2 font-medium text-gray-900">
                {new Date(item.created_at).toLocaleString('en-IN', {
                  dateStyle: 'full',
                  timeStyle: 'medium',
                })}
              </span>
            </div>
          </div>

          {item.notes && (
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500 mb-1">Notes:</p>
              <p className="text-gray-700">{item.notes}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
const History: React.FC = () => {
  // State
  const [history, setHistory] = useState<DetectionHistory[]>([]);
  const [stats, setStats] = useState<HistoryStats | null>(null);
  const [pagination, setPagination] = useState<PaginationMeta | null>(null);
  const [loading, setLoading] = useState(true);
  const [statsLoading, setStatsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedItem, setSelectedItem] = useState<DetectionHistory | null>(null);

  // Filters
  const [currentPage, setCurrentPage] = useState(1);
  const [selectedCrop, setSelectedCrop] = useState<string>('');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('');

  const crops = ['Tomato', 'Chilli'];
  const severities = ['healthy', 'mild', 'moderate', 'severe'];

  // ---------------------------------------------------------------------------
  // Fetch History
  // ---------------------------------------------------------------------------
  const fetchHistory = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await historyAPI.getAll({
        page: currentPage,
        per_page: 10,
        crop: selectedCrop || undefined,
        severity: selectedSeverity || undefined,
      });

      if (response.success) {
        setHistory(response.data || []);
        setPagination(response.pagination || null);
      } else {
        throw new Error(response.error || 'Failed to load history');
      }
    } catch (err) {
      console.error('Failed to fetch history:', err);
      setError('Failed to load detection history');
    } finally {
      setLoading(false);
    }
  }, [currentPage, selectedCrop, selectedSeverity]);

  // ---------------------------------------------------------------------------
  // Fetch Stats
  // ---------------------------------------------------------------------------
  const fetchStats = useCallback(async () => {
    try {
      setStatsLoading(true);
      const response = await historyAPI.getStats();
      if (response.success && response.data) {
        setStats(response.data);
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    } finally {
      setStatsLoading(false);
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Initial Load
  // ---------------------------------------------------------------------------
  useEffect(() => {
    fetchHistory();
    fetchStats();
  }, [fetchHistory, fetchStats]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------
  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this detection record?')) {
      return;
    }

    try {
      await historyAPI.delete(id);
      setHistory((prev) => prev.filter((item) => item.id !== id));
      toast.success('Record deleted');
      fetchStats(); // Refresh stats
    } catch {
      toast.error('Failed to delete record');
    }
  };

  const handleView = (item: DetectionHistory) => {
    setSelectedItem(item);
  };

  const handleFilterChange = () => {
    setCurrentPage(1);
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="max-w-6xl mx-auto px-4 py-8 sm:py-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 flex items-center">
            <HistoryIcon className="w-8 h-8 mr-3 text-primary-600" />
            Detection History
          </h1>
          <p className="text-gray-600 mt-1">
            View and manage your past disease detections
          </p>
        </div>
        <Link to="/select-crop" className="btn btn-primary flex items-center">
          <Leaf className="w-5 h-5 mr-2" />
          New Detection
        </Link>
      </div>

      {/* Stats */}
      <StatsCard stats={stats} loading={statsLoading} />

      {/* Filters */}
      <div className="card p-4 mb-6">
        <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
          <div className="flex items-center text-gray-600">
            <Filter className="w-5 h-5 mr-2" />
            <span className="font-medium">Filters:</span>
          </div>

          <select
            value={selectedCrop}
            onChange={(e) => {
              setSelectedCrop(e.target.value);
              handleFilterChange();
            }}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          >
            <option value="">All Crops</option>
            {crops.map((crop) => (
              <option key={crop} value={crop}>
                {crop}
              </option>
            ))}
          </select>

          <select
            value={selectedSeverity}
            onChange={(e) => {
              setSelectedSeverity(e.target.value);
              handleFilterChange();
            }}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
          >
            <option value="">All Severities</option>
            {severities.map((severity) => (
              <option key={severity} value={severity}>
                {severity.charAt(0).toUpperCase() + severity.slice(1)}
              </option>
            ))}
          </select>

          <button
            onClick={() => {
              setSelectedCrop('');
              setSelectedSeverity('');
              handleFilterChange();
            }}
            className="text-sm text-primary-600 hover:text-primary-700"
          >
            Clear filters
          </button>

          <button
            onClick={fetchHistory}
            className="ml-auto p-2 text-gray-500 hover:text-primary-600 hover:bg-primary-50 rounded-lg transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Error State */}
      {error && (
        <div className="card p-6 text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">Error</h3>
          <p className="text-gray-600 mb-4">{error}</p>
          <button onClick={fetchHistory} className="btn btn-primary">
            Try Again
          </button>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="w-8 h-8 animate-spin text-primary-600" />
          <span className="ml-3 text-gray-600">Loading history...</span>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && history.length === 0 && (
        <div className="card p-12 text-center">
          <HistoryIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-xl font-medium text-gray-900 mb-2">
            No Detection History
          </h3>
          <p className="text-gray-600 mb-6">
            {selectedCrop || selectedSeverity
              ? 'No records match your filters. Try adjusting them.'
              : "You haven't made any detections yet. Start by analyzing a plant image."}
          </p>
          <Link to="/select-crop" className="btn btn-primary">
            Start Detection
          </Link>
        </div>
      )}

      {/* History List */}
      {!loading && !error && history.length > 0 && (
        <div className="space-y-4">
          {history.map((item) => (
            <HistoryItem
              key={item.id}
              item={item}
              onDelete={handleDelete}
              onView={handleView}
            />
          ))}
        </div>
      )}

      {/* Pagination */}
      {pagination && pagination.pages > 1 && (
        <div className="mt-6 flex items-center justify-center gap-2">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="p-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
          >
            <ChevronLeft className="w-5 h-5" />
          </button>

          <span className="px-4 py-2 text-sm text-gray-600">
            Page {currentPage} of {pagination.pages}
          </span>

          <button
            onClick={() => setCurrentPage((p) => Math.min(pagination.pages, p + 1))}
            disabled={currentPage === pagination.pages}
            className="p-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Detail Modal */}
      <DetailModal item={selectedItem} onClose={() => setSelectedItem(null)} />
    </div>
  );
};

export default History;
