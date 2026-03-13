import React, { useState } from 'react';

// Icon Components
function SearchIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="11" cy="11" r="8" strokeWidth="2"/>
      <path d="m21 21-4.35-4.35" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

function InfoIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="12" cy="12" r="10" strokeWidth="2"/>
      <line x1="12" y1="16" x2="12" y2="12" strokeWidth="2" strokeLinecap="round"/>
      <circle cx="12" cy="8" r="0.5" fill="currentColor" stroke="none"/>
    </svg>
  );
}

function ArrowRightIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <path d="M5 12h14M12 5l7 7-7 7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
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

export default function ProcessedResults({ onNext, onBack, calculationResult, onGoToStep }) {
  const [activeTab, setActiveTab] = useState('processed');
  const [hoveredRow, setHoveredRow] = useState(null);

  if (!calculationResult) {
    return (
      <div className="bg-white w-full min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="font-['Noto_Sans',sans-serif] text-[18px] text-[#7E7E7E] mb-4">
            No calculation results available.
          </p>
          <button
            onClick={onBack}
            className="px-[32px] py-[12px] bg-[#29ABB5] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-medium text-[16px] text-white hover:bg-[#238d96] transition"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  const processedData = calculationResult.rawData || [];
  const columns = processedData.length > 0 ? Object.keys(processedData[0]).filter(key => key !== 'formulaReference' && key !== 'EF_ID') : [];
  
  // Show first 20 rows
  const displayData = processedData.slice(0, 20);

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

      {/* Progress Steps - 5 steps */}
      <div className="absolute left-[64px] right-[64px] top-[81px] flex items-center">
        {/* Step 1 - Completed (green, clickable) */}
        <div className="flex items-center gap-[8px] cursor-pointer hover:opacity-80 transition" onClick={() => onGoToStep(1)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <CheckIcon className="w-[16px] h-[16px] text-white" />
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Upload Data
          </p>
        </div>
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #50B461)' }}></div>

        {/* Step 2 - Completed (green, clickable) */}
        <div className="flex items-center gap-[8px] cursor-pointer hover:opacity-80 transition" onClick={() => onGoToStep(2)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <CheckIcon className="w-[16px] h-[16px] text-white" />
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Select Data
          </p>
        </div>
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #50B461)' }}></div>

        {/* Step 3 - Completed (green, clickable) */}
        <div className="flex items-center gap-[8px] cursor-pointer hover:opacity-80 transition" onClick={() => onGoToStep(3)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <CheckIcon className="w-[16px] h-[16px] text-white" />
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Data Cleaning
          </p>
        </div>
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #29ABB5)' }}></div>

        {/* Step 4 - Active (current) */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-[#29ABB5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">4</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Data Processing
          </p>
        </div>
        <div className="flex-1 h-[2px] bg-[#ACC2C5] mx-[12px]"></div>

        {/* Step 5 - Inactive */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-white border-2 border-[#ACC2C5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#7E7E7E]">5</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] whitespace-nowrap">
            Visualization
          </p>
        </div>
      </div>

      {/* Main Content - flex column layout, no page scroll */}
      <div className="absolute left-[64px] right-[64px] top-[132px] bottom-[32px] flex flex-col overflow-hidden">
        {/* Header with summary */}
        <div className="flex items-center justify-between mb-[20px] flex-shrink-0">
          <h1 className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B]">
            Processed Results
          </h1>
          <div className="flex items-center gap-[16px]">
            <div className="bg-[#29ABB5]/10 px-[16px] py-[8px] rounded-[8px]">
              <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#365D60]">
                Total: <span className="font-bold">{calculationResult.summary.totalMtCO2e.toFixed(5)} mtCO2e</span>
              </p>
            </div>
            <div className="bg-[#50B461]/10 px-[16px] py-[8px] rounded-[8px]">
              <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#365D60]">
                Records: <span className="font-bold">{calculationResult.summary.recordCount}</span>
              </p>
            </div>
          </div>
        </div>

        {/* Info Banner */}
        <div className="bg-[#29ABB5]/10 border-l-4 border-[#29ABB5] p-[16px] rounded-[8px] mb-[20px] flex items-start gap-[12px] flex-shrink-0">
          <InfoIcon className="w-[20px] h-[20px] text-[#29ABB5] flex-shrink-0 mt-[2px]" />
          <div>
            <p className="font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#13383B] mb-[4px]">
              Hover over any row to see the calculation formula
            </p>
            <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E]">
              Each mtCO2e value is calculated using: Consumption × GHG_MTperUnit × GWP
            </p>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex gap-[12px] mb-[20px] flex-shrink-0">
          <button
            onClick={() => setActiveTab('processed')}
            className={`px-[24px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] font-medium text-[14px] transition ${
              activeTab === 'processed'
                ? 'bg-[#29ABB5] text-white'
                : 'bg-[#F5F5F5] text-[#7E7E7E] hover:bg-[#EEEEEE]'
            }`}
          >
            Processed Results ({processedData.length} rows)
          </button>
          <button
            onClick={() => setActiveTab('raw')}
            className={`px-[24px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] font-medium text-[14px] transition ${
              activeTab === 'raw'
                ? 'bg-[#29ABB5] text-white'
                : 'bg-[#F5F5F5] text-[#7E7E7E] hover:bg-[#EEEEEE]'
            }`}
          >
            Raw Datasets
          </button>
        </div>

        {/* Processed Results Tab - fixed height table */}
        {activeTab === 'processed' && (
          <div className="min-h-0 h-[461.5px] mb-[16px] flex-shrink-0">
            <div className="bg-white rounded-[10px] border border-[#EEEEEE] overflow-hidden h-full">
              <div className="overflow-auto h-full">
                <table className="w-full">
                  <thead className="bg-[#D9D9D9] sticky top-0 z-10">
                    <tr>
                      {columns.map((col, idx) => (
                        <th key={idx} className="px-[16px] py-[12px] text-left font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#13383B] whitespace-nowrap">
                          {col}
                          {col === 'mtCO2e_calc' && (
                            <span className="ml-2 text-[#29ABB5]">ⓘ</span>
                          )}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {displayData.map((row, rowIdx) => (
                      <tr
                        key={rowIdx}
                        className="border-t border-[#EEEEEE] hover:bg-[#29ABB5]/5 transition cursor-pointer relative"
                        onMouseEnter={() => setHoveredRow(rowIdx)}
                        onMouseLeave={() => setHoveredRow(null)}
                      >
                        {columns.map((col, colIdx) => (
                          <td
                            key={colIdx}
                            className={`px-[16px] py-[12px] font-['Noto_Sans',sans-serif] text-[14px] whitespace-nowrap ${
                              col === 'mtCO2e_calc' ? 'font-bold text-[#29ABB5]' : 'text-[#13383B]'
                            }`}
                          >
                            {typeof row[col] === 'number' ? row[col].toFixed(5) : String(row[col] || '')}
                          </td>
                        ))}
                        {/* Semi-transparent Formula at bottom-right */}
                        {hoveredRow === rowIdx && row.formulaReference && (
                          <div className="absolute bottom-0 right-2 z-50 transform translate-y-full mt-2">
                            <div className="bg-[#365D60]/90 backdrop-blur-sm text-white p-[12px] rounded-[8px] shadow-lg w-[350px]">
                              <p className="font-['Noto_Sans',sans-serif] font-bold text-[12px] mb-[6px]">
                                Formula
                              </p>
                              <p className="font-['Noto_Sans',sans-serif] text-[11px] mb-[6px] font-mono bg-white/20 px-[8px] py-[4px] rounded">
                                {row.formulaReference.formula}
                              </p>
                              <p className="font-['Noto_Sans',sans-serif] text-[10px] mb-[4px] opacity-80">
                                Calculation:
                              </p>
                              <p className="font-['Noto_Sans',sans-serif] text-[11px] font-mono bg-white/20 px-[8px] py-[4px] rounded mb-[6px]">
                                {row.formulaReference.calculation}
                              </p>
                              <p className="font-['Noto_Sans',sans-serif] text-[10px] opacity-70">
                                Source: {row.formulaReference.efidSource}
                              </p>
                            </div>
                          </div>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Raw Datasets Tab - fixed height to match */}
        {activeTab === 'raw' && (
          <div className="min-h-0 h-[461.5px] mb-[16px] flex-shrink-0">
            <div className="bg-white rounded-[10px] border border-[#EEEEEE] p-[32px] h-full overflow-auto">
              <div className="grid grid-cols-2 gap-[24px]">
                {/* EFID Reference Data */}
                <div className="bg-[#F5F5F5] rounded-[10px] p-[24px]">
                  <h3 className="font-['Noto_Sans',sans-serif] font-bold text-[18px] text-[#13383B] mb-[16px]">
                    EFID Reference Data
                  </h3>
                  <div className="space-y-[12px]">
                    <div className="flex justify-between items-center">
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">Filename:</p>
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] font-medium">EFID_Reference.csv</p>
                    </div>
                    <div className="flex justify-between items-center">
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">Type:</p>
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] font-medium">Reference Data</p>
                    </div>
                    <div className="flex justify-between items-center">
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">Description:</p>
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] font-medium">Emission Factors & GWP</p>
                    </div>
                  </div>
                </div>

                {/* Fleet Fuel Data */}
                <div className="bg-[#F5F5F5] rounded-[10px] p-[24px]">
                  <h3 className="font-['Noto_Sans',sans-serif] font-bold text-[18px] text-[#13383B] mb-[16px]">
                    Fleet Fuel Data
                  </h3>
                  <div className="space-y-[12px]">
                    <div className="flex justify-between items-center">
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">Filename:</p>
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] font-medium">FleetFuel_2012_2022.csv</p>
                    </div>
                    <div className="flex justify-between items-center">
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">Type:</p>
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] font-medium">Consumption Data</p>
                    </div>
                    <div className="flex justify-between items-center">
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">Description:</p>
                      <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] font-medium">Fleet Fuel Consumption</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Data Source Info */}
              <div className="mt-[24px] bg-[#29ABB5]/10 border-l-4 border-[#29ABB5] p-[16px] rounded-[8px]">
                <p className="font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#13383B] mb-[8px]">
                  Data Sources
                </p>
                <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E]">
                  The processed results combine data from both EFID Reference (emission factors and global warming potentials)
                  and Fleet Fuel consumption records. Each row in the processed results represents a calculated emission value
                  using the formula: mtCO2e = Consumption × GHG_MTperUnit × GWP
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Dashboard Button - pinned at bottom */}
        <div className="flex justify-end items-center flex-shrink-0 mt-auto">
          <button
            onClick={onNext}
            className="px-[32px] py-[12px] bg-[#29ABB5] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-bold text-[18px] text-white hover:bg-[#238d96] transition"
          >
            Dashboard
          </button>
        </div>
      </div>
    </div>
  );
}
