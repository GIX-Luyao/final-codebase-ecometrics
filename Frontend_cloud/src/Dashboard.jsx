import React, { useState, useEffect } from 'react';
import { getCalculatedData, getAnalyticsSummary, getChartData as fetchChartData } from './services/api';

function SearchIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="11" cy="11" r="8" strokeWidth="2"/>
      <path d="m21 21-4.35-4.35" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

function GridIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <rect x="3" y="3" width="7" height="7" strokeWidth="2" rx="1"/>
      <rect x="14" y="3" width="7" height="7" strokeWidth="2" rx="1"/>
      <rect x="3" y="14" width="7" height="7" strokeWidth="2" rx="1"/>
      <rect x="14" y="14" width="7" height="7" strokeWidth="2" rx="1"/>
    </svg>
  );
}

function ListIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <line x1="4" y1="6" x2="20" y2="6" strokeWidth="2" strokeLinecap="round"/>
      <line x1="4" y1="12" x2="20" y2="12" strokeWidth="2" strokeLinecap="round"/>
      <line x1="4" y1="18" x2="20" y2="18" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

function SettingsIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="12" cy="12" r="3" strokeWidth="2"/>
      <path d="M12 1v6m0 6v6M1 12h6m6 0h6" strokeWidth="2" strokeLinecap="round"/>
      <path d="M4.22 4.22l4.24 4.24m7.08 0l4.24-4.24M4.22 19.78l4.24-4.24m7.08 0l4.24 4.24" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

function ChevronLeftIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <path d="M15 18l-6-6 6-6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function CheckIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <polyline points="20,6 9,17 4,12" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function FormulaIcon({ className, onClick }) {
  return (
    <svg 
      className={className} 
      viewBox="0 0 24 24" 
      fill="none" 
      stroke="currentColor"
      onClick={onClick}
    >
      <path d="M7 8L3 12L7 16M17 8L21 12L17 16M14 4L10 20" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function CloseIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <line x1="18" y1="6" x2="6" y2="18" strokeWidth="2" strokeLinecap="round"/>
      <line x1="6" y1="6" x2="18" y2="18" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

// Real data chart components
function PieChartReal({ data }) {
  if (!data || !data.data || data.data.length === 0) {
    return <p className="text-[#7E7E7E]">No data available</p>;
  }

  const colors = ['#29ABB5', '#365D60', '#50B461', '#7E7E7E', '#F0B429'];
  let currentAngle = 0;

  return (
    <div className="flex flex-col items-center w-full">
      <svg viewBox="0 0 200 200" className="w-full max-w-[200px] h-[200px] mb-4">
        <g transform="translate(100, 100)">
          {data.data.map((item, idx) => {
            const angle = (item.percentage / 100) * 360;
            const startAngle = (currentAngle * Math.PI) / 180;
            const endAngle = ((currentAngle + angle) * Math.PI) / 180;
            const x1 = 80 * Math.cos(startAngle);
            const y1 = 80 * Math.sin(startAngle);
            const x2 = 80 * Math.cos(endAngle);
            const y2 = 80 * Math.sin(endAngle);
            const largeArc = angle > 180 ? 1 : 0;

            const path = `M 0 0 L ${x1} ${y1} A 80 80 0 ${largeArc} 1 ${x2} ${y2} Z`;
            currentAngle += angle;

            return (
              <path
                key={idx}
                d={path}
                fill={colors[idx % colors.length]}
                stroke="white"
                strokeWidth="2"
              />
            );
          })}
        </g>
      </svg>
      <div className="grid grid-cols-2 gap-2 w-full">
        {data.data.map((item, idx) => (
          <div key={idx} className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-sm flex-shrink-0" 
              style={{ backgroundColor: colors[idx % colors.length] }}
            />
            <span className="text-xs text-[#575757]">
              {item.label}: {item.percentage}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function BarChartReal({ data }) {
  if (!data || !data.labels || data.labels.length === 0) {
    return <p className="text-[#7E7E7E]">No data available</p>;
  }

  const values = data.datasets[0]?.data || [];
  const maxValue = Math.max(...values);
  
  return (
    <div className="w-full h-[200px]">
      <svg viewBox="0 0 400 200" className="w-full h-full">
        {/* X axis */}
        <line x1="40" y1="180" x2="380" y2="180" stroke="#D9D9D9" strokeWidth="2"/>
        {/* Y axis */}
        <line x1="40" y1="20" x2="40" y2="180" stroke="#D9D9D9" strokeWidth="2"/>
        
        {/* Bars */}
        {values.map((value, idx) => {
          const barWidth = 300 / values.length - 10;
          const barHeight = (value / maxValue) * 140;
          const x = 60 + (idx * (300 / values.length));
          const y = 180 - barHeight;
          
          return (
            <rect
              key={idx}
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              fill="#29ABB5"
              rx="2"
            />
          );
        })}
        
        {/* Y-axis label */}
        <text x="10" y="100" fontSize="10" fill="#575757" transform="rotate(-90, 10, 100)">
          mtCO2e
        </text>
      </svg>
    </div>
  );
}

// Chart component wrapper with formula icon
function ChartCard({ title, subtitle, children, onClick, onFormulaClick }) {
  return (
    <div className="bg-white rounded-[10px] border border-[#EEEEEE] p-[24px] h-full flex flex-col">
      <div className="mb-[16px] flex items-start justify-between">
        <div className="flex-1">
          <h3 className="font-['Noto_Sans',sans-serif] font-bold text-[18px] text-[#13383B] mb-[8px]">
            {title}
          </h3>
          {subtitle && (
            <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] leading-[1.5]">
              {subtitle}
            </p>
          )}
        </div>
        {onFormulaClick && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              onFormulaClick();
            }}
            className="ml-2 p-2 bg-[#29ABB5]/10 rounded-full hover:bg-[#29ABB5]/20 transition group"
            title="View Formula"
          >
            <FormulaIcon className="w-[20px] h-[20px] text-[#29ABB5] group-hover:scale-110 transition" />
          </button>
        )}
      </div>
      <div 
        onClick={onClick}
        className="flex-1 flex items-center justify-center bg-[#F5F5F5] rounded-[8px] p-4 cursor-pointer hover:bg-[#EEEEEE] transition"
      >
        {children}
      </div>
    </div>
  );
}

export default function Dashboard({ onBack, sessionId, calculationResult: propResult, onGoToStep }) {
  const [viewMode, setViewMode] = useState('grid');
  const [selectedChart, setSelectedChart] = useState(null);
  const [activeFilters, setActiveFilters] = useState(['EQ_EQUIP_NO', 'Department_NO']);
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [showFormulaModal, setShowFormulaModal] = useState(false);
  const [selectedFormula, setSelectedFormula] = useState(null);

  const filters = ['EQ_EQUIP_NO', 'Department_NO', 'Label 4', 'Label 5', 'Label 6'];

  const formulas = {
    pieChart: {
      title: "Pie Chart Formula",
      description: "Groups emissions by fuel type (Subtype)",
      formula: "Total mtCO2e per Subtype = SUM(mtCO2e_calc) GROUP BY Subtype",
      calculation: "For each fuel type:\n• Gasoline: Sum of all Gasoline mtCO2e values\n• Diesel: Sum of all Diesel mtCO2e values\n• Biodiesel: Sum of all Biodiesel mtCO2e values",
      baseFormula: "mtCO2e_calc = Consumption × GHG_MTperUnit × GWP"
    },
    barChart: {
      title: "Bar Chart Formula",
      description: "Groups emissions by fill date",
      formula: "Total mtCO2e per Date = SUM(mtCO2e_calc) GROUP BY FillDate",
      calculation: "For each date:\n• Sum all mtCO2e values for that specific date\n• Display as bars chronologically",
      baseFormula: "mtCO2e_calc = Consumption × GHG_MTperUnit × GWP"
    }
  };

  useEffect(() => {
    async function loadData() {
      try {
        if (sessionId) {
          // Try loading from backend analytics API
          const [summaryResult, pieResult, barResult] = await Promise.all([
            getAnalyticsSummary(sessionId),
            fetchChartData(sessionId, 'pie', 'mtCO2e_calc', 'Subtype'),
            fetchChartData(sessionId, 'bar', 'mtCO2e_calc', 'Year'),
          ]);

          const summary = summaryResult.summary || {};
          setChartData({
            sessionId,
            calculationTimestamp: new Date().toISOString(),
            summary: {
              totalMtCO2e: summary.totalMtCO2e || 0,
              recordCount: summary.totalSources || 0,
            },
            charts: {
              pieChart: pieResult,
              barChart: {
                labels: barResult.data?.map(d => d.label) || [],
                datasets: [{ data: barResult.data?.map(d => d.value) || [] }],
              },
            },
          });
          setLoading(false);
          return;
        }

        if (propResult) {
          // Use passed calculation result as fallback
          setChartData(propResult);
          setLoading(false);
          return;
        }

        setLoading(false);
      } catch (error) {
        console.error('Failed to load analytics, trying legacy endpoint:', error);
        // Fallback to legacy endpoint
        try {
          if (propResult) {
            setChartData(propResult);
          } else if (sessionId) {
            const result = await getCalculatedData(sessionId);
            setChartData(result);
          }
        } catch (legacyError) {
          console.error('Legacy endpoint also failed:', legacyError);
        }
        setLoading(false);
      }
    }

    loadData();
  }, [sessionId, propResult]);

  const handleChartClick = (chartId) => {
    console.log('🖱️ Chart clicked:', chartId);
    setSelectedChart(chartId);
  };

  const handleFormulaClick = (formulaKey) => {
    setSelectedFormula(formulas[formulaKey]);
    setShowFormulaModal(true);
  };

  const toggleFilter = (filter) => {
    setActiveFilters(prev => 
      prev.includes(filter) 
        ? prev.filter(f => f !== filter)
        : [...prev, filter]
    );
  };

  if (loading) {
    return (
      <div className="bg-white w-full min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin w-[48px] h-[48px] border-4 border-[#29ABB5] border-t-transparent rounded-full mx-auto mb-4"></div>
          <p className="font-['Noto_Sans',sans-serif] text-[18px] text-[#7E7E7E]">
            Loading dashboard...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white fixed inset-0 overflow-hidden">
      {/* Toolbar */}
      <div className="absolute left-0 top-0 w-full h-[62px] bg-[#365D60] flex items-center justify-between px-[64px]">
        <div className="flex items-center gap-[12px]">
          <img src="/logo.png" alt="Logo" className="h-[40px] w-auto" />
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[24px] text-white">
            EcoMetrics
          </p>
        </div>
        <div className="bg-white rounded-[10px] w-[331px] h-[40px] flex items-center px-[12px]">
          <SearchIcon className="w-[20px] h-[20px] text-[#7E7E7E]" />
        </div>
      </div>

      {/* Progress Steps - 4 steps */}
      <div className="absolute left-[64px] right-[64px] top-[81px] flex items-center">
        {/* Step 1 - Completed (green, clickable) */}
        <div className="flex items-center gap-[8px] cursor-pointer hover:opacity-80 transition" onClick={() => onGoToStep(1)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <CheckIcon className="w-[16px] h-[16px] text-white" />
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Upload Your Data
          </p>
        </div>
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #50B461)' }}></div>

        {/* Step 2 - Completed (green, clickable) */}
        <div className="flex items-center gap-[8px] cursor-pointer hover:opacity-80 transition" onClick={() => onGoToStep(2)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <CheckIcon className="w-[16px] h-[16px] text-white" />
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Data Cleaning
          </p>
        </div>
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #50B461)' }}></div>

        {/* Step 3 - Completed (green, clickable) */}
        <div className="flex items-center gap-[8px] cursor-pointer hover:opacity-80 transition" onClick={() => onGoToStep(3)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <CheckIcon className="w-[16px] h-[16px] text-white" />
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Data Processing
          </p>
        </div>
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #29ABB5)' }}></div>

        {/* Step 4 - Active (current) */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-[#29ABB5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">4</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Data Visualization
          </p>
        </div>
      </div>

      {/* Main Content */}
      {!selectedChart ? (
        // Dashboard Grid View
        <div className="absolute left-[64px] right-[64px] top-[132px] bottom-[32px] overflow-auto scrollbar-hide" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
          {/* Summary Stats */}
          {chartData && chartData.summary && (
            <div className="grid grid-cols-3 gap-[24px] mb-[32px]">
              <div className="bg-[#29ABB5]/10 rounded-[10px] p-[24px] border-l-4 border-[#29ABB5]">
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E] mb-[8px]">
                  Total Emissions
                </p>
                <p className="font-['Noto_Sans',sans-serif] font-bold text-[28px] text-[#13383B]">
                  {chartData.summary.totalMtCO2e.toFixed(5)} <span className="text-[16px] text-[#7E7E7E]">mtCO2e</span>
                </p>
              </div>
              <div className="bg-[#50B461]/10 rounded-[10px] p-[24px] border-l-4 border-[#50B461]">
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E] mb-[8px]">
                  Records Processed
                </p>
                <p className="font-['Noto_Sans',sans-serif] font-bold text-[28px] text-[#13383B]">
                  {chartData.summary.recordCount}
                </p>
              </div>
              <div className="bg-[#365D60]/10 rounded-[10px] p-[24px] border-l-4 border-[#365D60]">
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E] mb-[8px]">
                  Session ID
                </p>
                <p className="font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#13383B] break-all">
                  {chartData.sessionId}
                </p>
              </div>
            </div>
          )}

          {/* Header with filters and view controls */}
          <div className="flex items-center justify-between mb-[24px]">
            {/* Filter Pills */}
            <div className="flex gap-[12px] flex-wrap">
              {filters.map(filter => (
                <button
                  key={filter}
                  onClick={() => toggleFilter(filter)}
                  className={`px-[16px] py-[8px] rounded-[4px] font-['Noto_Sans',sans-serif] text-[14px] transition border ${
                    activeFilters.includes(filter)
                      ? 'bg-[#365D60] text-white border-[#365D60]'
                      : 'bg-white text-[#575757] border-[#D9D9D9] hover:bg-[#F5F5F5]'
                  }`}
                >
                  {filter}
                </button>
              ))}
            </div>

            {/* View Controls */}
            <div className="flex items-center gap-[12px]">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-[10px] rounded-[8px] transition ${
                  viewMode === 'grid'
                    ? 'bg-[#29ABB5] text-white'
                    : 'bg-white text-[#575757] border border-[#D9D9D9] hover:bg-[#F5F5F5]'
                }`}
                title="Grid View"
              >
                <GridIcon className="w-[20px] h-[20px]" />
              </button>
              <button
                onClick={() => setViewMode('horizontal')}
                className={`p-[10px] rounded-[8px] transition ${
                  viewMode === 'horizontal'
                    ? 'bg-[#29ABB5] text-white'
                    : 'bg-white text-[#575757] border border-[#D9D9D9] hover:bg-[#F5F5F5]'
                }`}
                title="Horizontal Scroll View"
              >
                <ListIcon className="w-[20px] h-[20px]" />
              </button>
              <button className="p-[10px] rounded-[8px] bg-white text-[#575757] border border-[#D9D9D9] hover:bg-[#F5F5F5] transition">
                <SettingsIcon className="w-[20px] h-[20px]" />
              </button>
            </div>
          </div>

          {/* Charts Display */}
          {chartData && chartData.charts ? (
            viewMode === 'grid' ? (
              <div className="grid grid-cols-2 gap-[24px]">
                <ChartCard
                  title="Emissions by Fuel Type"
                  subtitle={`Total: ${chartData.summary.totalMtCO2e.toFixed(5)} mtCO2e`}
                  onClick={() => handleChartClick(1)}
                  onFormulaClick={() => handleFormulaClick('pieChart')}
                >
                  <PieChartReal data={chartData.charts.pieChart} />
                </ChartCard>
                <ChartCard
                  title="Emissions Over Time"
                  subtitle="Grouped by fill date"
                  onClick={() => handleChartClick(2)}
                  onFormulaClick={() => handleFormulaClick('barChart')}
                >
                  <BarChartReal data={chartData.charts.barChart} />
                </ChartCard>
                <ChartCard
                  title="Data Summary"
                  subtitle={`${chartData.summary.recordCount} records processed`}
                  onClick={() => handleChartClick(3)}
                >
                  <div className="text-center">
                    <p className="font-['Noto_Sans',sans-serif] font-bold text-[32px] text-[#29ABB5] mb-2">
                      {chartData.summary.recordCount}
                    </p>
                    <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">
                      Total Records
                    </p>
                    <p className="font-['Noto_Sans',sans-serif] font-bold text-[24px] text-[#365D60] mt-4">
                      {chartData.charts.pieChart.data.length}
                    </p>
                    <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">
                      Fuel Types
                    </p>
                  </div>
                </ChartCard>
                <ChartCard
                  title="Calculation Timestamp"
                  subtitle="Latest data processing time"
                  onClick={() => handleChartClick(4)}
                >
                  <div className="text-center">
                    <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E] mb-2">
                      Calculated at:
                    </p>
                    <p className="font-['Noto_Sans',sans-serif] font-medium text-[16px] text-[#13383B]">
                      {new Date(chartData.calculationTimestamp).toLocaleString()}
                    </p>
                  </div>
                </ChartCard>
              </div>
            ) : (
              <div className="relative">
                <div className="overflow-x-auto pb-4">
                  <div className="flex gap-[24px] min-w-max">
                    <div className="w-[800px] flex-shrink-0">
                      <ChartCard
                        title="Emissions by Fuel Type"
                        subtitle={`Total: ${chartData.summary.totalMtCO2e.toFixed(5)} mtCO2e`}
                        onClick={() => handleChartClick(1)}
                        onFormulaClick={() => handleFormulaClick('pieChart')}
                      >
                        <PieChartReal data={chartData.charts.pieChart} />
                      </ChartCard>
                    </div>
                    <div className="w-[800px] flex-shrink-0">
                      <ChartCard
                        title="Emissions Over Time"
                        subtitle="Grouped by fill date"
                        onClick={() => handleChartClick(2)}
                        onFormulaClick={() => handleFormulaClick('barChart')}
                      >
                        <BarChartReal data={chartData.charts.barChart} />
                      </ChartCard>
                    </div>
                  </div>
                </div>
                <div className="absolute left-0 top-0 bottom-0 w-[64px] bg-gradient-to-r from-white to-transparent pointer-events-none"></div>
                <div className="absolute right-0 top-0 bottom-0 w-[64px] bg-gradient-to-l from-white to-transparent pointer-events-none"></div>
              </div>
            )
          ) : (
            <div className="text-center py-[64px]">
              <p className="font-['Noto_Sans',sans-serif] text-[18px] text-[#7E7E7E]">
                No calculation data available. Please run calculation first.
              </p>
            </div>
          )}
        </div>
      ) : (
        // Detailed Chart View
        <div className="absolute left-[64px] right-[64px] top-[132px] bottom-[32px] overflow-auto scrollbar-hide" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
          <button
            onClick={() => {
              console.log('⬅️ Back button clicked');
              setSelectedChart(null);
            }}
            className="flex items-center gap-[12px] px-[24px] py-[12px] bg-[#29ABB5] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-medium text-[16px] text-white hover:bg-[#238d96] transition mb-[24px]"
          >
            <ChevronLeftIcon className="w-[20px] h-[20px]" />
            Back to Dashboard
          </button>

          {chartData && chartData.charts && (
            <div className="bg-white rounded-[10px] border border-[#EEEEEE] p-[48px]">
              <div className="mb-[32px]">
                <h2 className="font-['Noto_Sans',sans-serif] font-bold text-[28px] text-[#13383B] mb-[16px]">
                  {selectedChart === 1 ? 'Emissions by Fuel Type' :
                   selectedChart === 2 ? 'Emissions Over Time' :
                   selectedChart === 3 ? 'Data Summary' : 'Calculation Timestamp'}
                </h2>
                {selectedChart === 1 && (
                  <p className="font-['Noto_Sans',sans-serif] text-[16px] text-[#7E7E7E]">
                    Total: {chartData.summary.totalMtCO2e.toFixed(5)} mtCO2e
                  </p>
                )}
              </div>

              <div className="bg-[#F5F5F5] rounded-[8px] p-[48px] min-h-[400px] flex items-center justify-center">
                {selectedChart === 1 && (
                  <div className="w-full max-w-[500px]">
                    <PieChartReal data={chartData.charts.pieChart} />
                  </div>
                )}
                {selectedChart === 2 && (
                  <div className="w-full max-w-[800px]">
                    <BarChartReal data={chartData.charts.barChart} />
                  </div>
                )}
                {selectedChart === 3 && (
                  <div className="text-center">
                    <p className="font-['Noto_Sans',sans-serif] font-bold text-[48px] text-[#29ABB5] mb-4">
                      {chartData.summary.recordCount}
                    </p>
                    <p className="font-['Noto_Sans',sans-serif] text-[18px] text-[#7E7E7E]">
                      Total Records Processed
                    </p>
                  </div>
                )}
                {selectedChart === 4 && (
                  <div className="text-center">
                    <p className="font-['Noto_Sans',sans-serif] text-[16px] text-[#7E7E7E] mb-4">
                      Calculated at:
                    </p>
                    <p className="font-['Noto_Sans',sans-serif] font-medium text-[20px] text-[#13383B]">
                      {new Date(chartData.calculationTimestamp).toLocaleString()}
                    </p>
                  </div>
                )}
              </div>

              <div className="mt-[32px] bg-[#F5F5F5] rounded-[8px] p-[24px]">
                <h3 className="font-['Noto_Sans',sans-serif] font-bold text-[16px] text-[#13383B] mb-[16px]">
                  Chart Details
                </h3>
                <div className="grid grid-cols-3 gap-[24px]">
                  <div>
                    <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] mb-[4px]">Total Emissions</p>
                    <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B]">
                      {chartData.summary.totalMtCO2e.toFixed(5)} mtCO2e
                    </p>
                  </div>
                  <div>
                    <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] mb-[4px]">Data Source</p>
                    <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B]">Calculated Results</p>
                  </div>
                  <div>
                    <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] mb-[4px]">Last Updated</p>
                    <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B]">
                      {new Date(chartData.calculationTimestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Formula Modal */}
      {showFormulaModal && selectedFormula && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowFormulaModal(false)}>
          <div className="bg-white rounded-[10px] p-[32px] w-[600px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)] max-h-[80vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-[24px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[20px] text-[#575757]">
                {selectedFormula.title}
              </p>
              <button onClick={() => setShowFormulaModal(false)}>
                <CloseIcon className="w-[24px] h-[24px] text-[#575757] cursor-pointer hover:text-[#29ABB5]" />
              </button>
            </div>

            <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E] mb-[24px]">
              {selectedFormula.description}
            </p>

            <div className="bg-[#365D60] rounded-[8px] p-[24px] mb-[16px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white mb-[12px]">
                Aggregation Formula:
              </p>
              <p className="font-['Noto_Sans',sans-serif] text-[14px] text-white font-mono bg-white/20 p-[12px] rounded">
                {selectedFormula.formula}
              </p>
            </div>

            <div className="bg-[#F5F5F5] rounded-[8px] p-[16px] mb-[16px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#13383B] mb-[8px]">
                Calculation Steps:
              </p>
              <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#575757] whitespace-pre-line">
                {selectedFormula.calculation}
              </p>
            </div>

            <div className="bg-[#29ABB5]/10 border-l-4 border-[#29ABB5] rounded p-[16px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#13383B] mb-[8px]">
                Base Formula (per row):
              </p>
              <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#365D60] font-mono">
                {selectedFormula.baseFormula}
              </p>
            </div>

            <button
              onClick={() => setShowFormulaModal(false)}
              className="w-full mt-[24px] h-[40px] bg-[#29ABB5] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-white hover:bg-[#238d96] transition"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
