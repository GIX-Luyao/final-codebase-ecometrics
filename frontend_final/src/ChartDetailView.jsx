import React, { useState } from 'react';
import { ChevronLeft, PieChart, BarChart2, LineChart, BarChart3 } from 'lucide-react';

const ChartDetailView = ({ chart, onClose, onConvert, onFilterChange, filters, calculatedData }) => {
  const [selectedChartType, setSelectedChartType] = useState(chart.type);

  const chartTypes = [
    { id: 'pie', icon: PieChart, label: 'Pie' },
    { id: 'donut', icon: PieChart, label: 'Donut' },
    { id: 'bar', icon: BarChart2, label: 'Bar' },
    { id: 'line', icon: LineChart, label: 'Line' },
    { id: 'stacked-bar', icon: BarChart3, label: 'Stacked' },
    { id: 'grouped-bar', icon: BarChart3, label: 'Grouped' }
  ];

  const facilities = calculatedData?.rawData
    ? [...new Set(calculatedData.rawData.map(row => row.LowOrg).filter(Boolean))]
    : [];

  const activityTypes = calculatedData?.rawData
    ? [...new Set(calculatedData.rawData.map(row => row.Subtype).filter(Boolean))]
    : [];

  const years = calculatedData?.rawData
    ? [...new Set(calculatedData.rawData.map(row => row.Year).filter(Boolean))].sort()
    : [];

  const handleCreateChart = () => {
    if (selectedChartType !== chart.type) {
      onConvert(chart.id, selectedChartType);
      onClose();
    }
  };

  const handleFacilityToggle = (facility) => {
    const newFacilities = filters.facilities.includes(facility)
      ? filters.facilities.filter(f => f !== facility)
      : [...filters.facilities, facility];
    onFilterChange({ ...filters, facilities: newFacilities });
  };

  const handleActivityTypeToggle = (type) => {
    const newTypes = filters.activityTypes.includes(type)
      ? filters.activityTypes.filter(t => t !== type)
      : [...filters.activityTypes, type];
    onFilterChange({ ...filters, activityTypes: newTypes });
  };

  const handleYearToggle = (year) => {
    onFilterChange({ ...filters, year });
  };

  const renderChart = () => {
    // Render the chart based on type (simplified for detail view)
    if (selectedChartType === 'pie' || selectedChartType === 'donut') {
      return renderLargePieChart();
    } else if (selectedChartType === 'bar' || selectedChartType === 'stacked-bar') {
      return renderLargeBarChart();
    } else if (selectedChartType === 'line') {
      return renderLargeLineChart();
    }
  };

  const renderLargePieChart = () => {
    const data = chart.data;
    if (!data || !data.data) return null;

    const colors = ['#5AB4AC', '#7ECCC4', '#A2E4DD', '#C6F3EF', '#8AB8B3'];
    
    return (
      <div className="flex items-center justify-center h-full">
        <div className="w-96 h-96 relative">
          {/* Simplified pie chart visualization */}
          <div className="grid grid-cols-2 gap-4 mt-8">
            {data.data.map((item, index) => (
              <div key={index} className="flex items-center gap-3">
                <div
                  className="w-6 h-6 rounded-full"
                  style={{ backgroundColor: colors[index % colors.length] }}
                ></div>
                <div>
                  <p className="text-sm font-medium">{item.label}</p>
                  <p className="text-xs text-gray-500">
                    {item.value.toFixed(2)} mtCO2e ({item.percentage}%)
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const renderLargeBarChart = () => {
    const data = chart.data;
    if (!data || !data.datasets) return null;

    const maxValue = Math.max(...data.datasets[0].data);

    return (
      <div className="h-full flex items-end justify-around px-16 pb-12">
        {data.labels.map((label, index) => {
          const value = data.datasets[0].data[index];
          const height = (value / maxValue) * 100;

          return (
            <div key={index} className="flex flex-col items-center flex-1 mx-2">
              <div className="w-full flex items-end justify-center" style={{ height: '400px' }}>
                <div
                  className="w-full bg-[#5AB4AC] rounded-t transition-all hover:bg-[#4A9D96]"
                  style={{ height: `${height}%` }}
                >
                  <span className="text-xs text-white p-2 block text-center">
                    {value.toFixed(2)}
                  </span>
                </div>
              </div>
              <span className="text-sm text-gray-700 mt-3 text-center">
                {label}
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  const renderLargeLineChart = () => {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-gray-400 text-lg">Line Chart (Coming Soon)</p>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-[#F5F5F5]">
      {/* Header */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-[1400px] mx-auto px-8 py-4">
          <button
            onClick={onClose}
            className="flex items-center gap-2 text-[#5AB4AC] hover:text-[#4A9D96] font-medium"
          >
            <ChevronLeft className="w-5 h-5" />
            Dashboard
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-[1400px] mx-auto px-8 py-8">
        <div className="grid grid-cols-[1fr_300px] gap-6">
          {/* Chart Display */}
          <div className="bg-white rounded-[10px] p-8">
            <h2 className="text-2xl font-bold text-[#365D60] mb-6">{chart.title}</h2>
            <div className="h-[500px]">
              {renderChart()}
            </div>
          </div>

          {/* Right Sidebar */}
          <div className="space-y-4">
            {/* Facility Filter */}
            <div className="bg-white rounded-[10px] p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-3">Facility</h3>
              <div className="space-y-2">
                {facilities.map((facility) => (
                  <button
                    key={facility}
                    onClick={() => handleFacilityToggle(facility)}
                    className={`w-full px-3 py-2 rounded text-sm text-left transition-colors ${
                      filters.facilities.includes(facility)
                        ? 'bg-[#5AB4AC] text-white'
                        : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    {facility}
                  </button>
                ))}
              </div>
            </div>

            {/* Activity Type Filter */}
            <div className="bg-white rounded-[10px] p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-3">Activity Type</h3>
              <div className="space-y-2">
                {activityTypes.map((type) => (
                  <button
                    key={type}
                    onClick={() => handleActivityTypeToggle(type)}
                    className={`w-full px-3 py-2 rounded text-sm text-left transition-colors ${
                      filters.activityTypes.includes(type)
                        ? 'bg-[#5AB4AC] text-white'
                        : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    {type}
                  </button>
                ))}
              </div>
            </div>

            {/* Year Filter */}
            <div className="bg-white rounded-[10px] p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-3">Year</h3>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleYearToggle('all')}
                  className={`px-3 py-2 rounded text-sm flex-1 ${
                    filters.year === 'all'
                      ? 'bg-[#5AB4AC] text-white'
                      : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                  }`}
                >
                  All
                </button>
                <select
                  value={filters.year === 'all' ? '' : filters.year}
                  onChange={(e) => handleYearToggle(e.target.value || 'all')}
                  className="px-3 py-2 rounded text-sm bg-gray-50 hover:bg-gray-100 border-none outline-none flex-1"
                >
                  <option value="">Year</option>
                  {years.map((year) => (
                    <option key={year} value={year}>{year}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Chart Type Selector */}
            <div className="bg-white rounded-[10px] p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-3">Chart Type</h3>
              <div className="grid grid-cols-2 gap-2">
                {chartTypes.map((type) => {
                  const Icon = type.icon;
                  return (
                    <button
                      key={type.id}
                      onClick={() => setSelectedChartType(type.id)}
                      className={`flex flex-col items-center gap-2 p-3 rounded transition-colors ${
                        selectedChartType === type.id
                          ? 'bg-[#5AB4AC] text-white'
                          : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      <Icon className="w-6 h-6" />
                      <span className="text-xs">{type.label}</span>
                    </button>
                  );
                })}
              </div>

              {selectedChartType !== chart.type && (
                <button
                  onClick={handleCreateChart}
                  className="w-full mt-3 px-4 py-2 bg-[#5AB4AC] text-white rounded hover:bg-[#4A9D96] transition-colors font-medium"
                >
                  Create New Chart
                </button>
              )}
            </div>

            {/* Description */}
            <div className="bg-gray-50 rounded-[10px] p-4">
              <p className="text-xs text-gray-500 leading-relaxed">
                description description description description description description description
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChartDetailView;