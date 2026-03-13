import React, { useState } from 'react';
import { MoreVertical, Trash2, BarChart2, PieChart, LineChart, BarChart3 } from 'lucide-react';

const ChartCard = ({ chart, onConvert, onDelete, onClick, filteredData }) => {
  const [showMenu, setShowMenu] = useState(false);
  const [isHovered, setIsHovered] = useState(false);

  const chartTypeIcons = {
    pie: { icon: PieChart, label: 'Pie Chart' },
    donut: { icon: PieChart, label: 'Donut Chart' },
    bar: { icon: BarChart2, label: 'Bar Chart' },
    'stacked-bar': { icon: BarChart3, label: 'Stacked Bar' },
    'grouped-bar': { icon: BarChart3, label: 'Grouped Bar' },
    line: { icon: LineChart, label: 'Line Chart' },
    area: { icon: LineChart, label: 'Area Chart' }
  };

  // Available conversions based on Frame_3039.png
  const getAvailableConversions = (currentType) => {
    const conversions = {
      'bar': ['pie', 'donut', 'line', 'stacked-bar'],
      'stacked-bar': ['pie', 'donut', 'bar'],
      'grouped-bar': ['line', 'area'],
      'pie': ['bar', 'donut'],
      'donut': ['pie', 'bar'],
      'line': ['bar', 'area'],
      'area': ['line', 'bar']
    };
    return conversions[currentType] || [];
  };

  const availableConversions = getAvailableConversions(chart.type);

  const handleConvertClick = (newType, e) => {
    e.stopPropagation();
    onConvert(chart.id, newType);
    setShowMenu(false);
  };

  const handleDeleteClick = (e) => {
    e.stopPropagation();
    if (window.confirm(`Delete "${chart.title}"?`)) {
      onDelete(chart.id);
    }
  };

  const renderChart = () => {
    const data = filteredData || chart.data;
    
    if (chart.type === 'pie' || chart.type === 'donut') {
      return renderPieChart(data);
    } else if (chart.type === 'bar' || chart.type === 'stacked-bar') {
      return renderBarChart(data);
    } else if (chart.type === 'line' || chart.type === 'area') {
      return renderLineChart(data);
    }
  };

  const renderPieChart = (data) => {
    if (!data || !data.data) return null;
    const colors = ['#5AB4AC', '#7ECCC4', '#A2E4DD', '#C6F3EF'];
    
    return (
      <div className="flex items-center justify-center h-64">
        <div className="space-y-2">
          {data.data.map((item, i) => (
            <div key={i} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full" style={{ backgroundColor: colors[i] }}></div>
              <span className="text-sm">{item.label}: {item.percentage}%</span>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderBarChart = (data) => {
    if (!data || !data.datasets) return null;
    const maxValue = Math.max(...data.datasets[0].data);
    
    return (
      <div className="h-64 flex items-end justify-around px-8">
        {data.labels.map((label, i) => {
          const value = data.datasets[0].data[i];
          const height = (value / maxValue) * 100;
          return (
            <div key={i} className="flex flex-col items-center flex-1 mx-1">
              <div className="w-full bg-[#5AB4AC] rounded-t" style={{ height: `${height}%` }}></div>
              <span className="text-xs mt-2">{label}</span>
            </div>
          );
        })}
      </div>
    );
  };

  const renderLineChart = (data) => {
    if (!data || !data.datasets) return null;
    
    return (
      <div className="h-64 flex items-center justify-center">
        <p className="text-gray-400">Line Chart (Coming Soon)</p>
      </div>
    );
  };

  const CurrentIcon = chartTypeIcons[chart.type]?.icon || BarChart2;

  return (
    <div
      className="bg-white rounded-[10px] shadow-sm hover:shadow-md transition-shadow cursor-pointer relative"
      onClick={() => onClick(chart.id)}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => {
        setIsHovered(false);
        setShowMenu(false);
      }}
    >
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <CurrentIcon className="w-5 h-5 text-[#365D60]" />
          <h3 className="font-medium">{chart.title}</h3>
        </div>
        
        {isHovered && (
          <div className="flex items-center gap-2">
            <div className="flex gap-1 bg-gray-50 rounded p-1">
              {availableConversions.map((type) => {
                const Icon = chartTypeIcons[type]?.icon;
                return (
                  <button
                    key={type}
                    onClick={(e) => handleConvertClick(type, e)}
                    className="p-2 rounded hover:bg-white"
                    title={chartTypeIcons[type]?.label}
                  >
                    <Icon className="w-4 h-4 text-[#5AB4AC]" />
                  </button>
                );
              })}
            </div>
            
            <button
              onClick={(e) => {
                e.stopPropagation();
                setShowMenu(!showMenu);
              }}
              className="p-2 rounded hover:bg-gray-100"
            >
              <MoreVertical className="w-4 h-4" />
            </button>
            
            {showMenu && (
              <div className="absolute right-4 top-16 bg-white rounded-lg shadow-lg border py-1 z-20">
                <button
                  onClick={handleDeleteClick}
                  className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
                >
                  <Trash2 className="w-4 h-4" />
                  Delete
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="p-4">
        {renderChart()}
      </div>
    </div>
  );
};

export default ChartCard;