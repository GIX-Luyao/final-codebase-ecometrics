import React, { useState, useMemo } from 'react';

// Icon Components
function SearchIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="11" cy="11" r="8" strokeWidth="2"/>
      <path d="m21 21-4.35-4.35" strokeWidth="2" strokeLinecap="round"/>
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

function InfoIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="12" cy="12" r="10" strokeWidth="2"/>
      <line x1="12" y1="16" x2="12" y2="12" strokeWidth="2" strokeLinecap="round"/>
      <circle cx="12" cy="8" r="0.5" fill="currentColor" stroke="none"/>
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

export default function PreviewFiles({ onNext, onBack, sessionId, onCalculationComplete, onGoToStep, folders, fileColumnLabels, filePendingCorrections }) {
  const [activeTab, setActiveTab] = useState('consumption');
  const [selectedFileId, setSelectedFileId] = useState(null);
  const [showFormulaModal, setShowFormulaModal] = useState(false);
  const [isCalculating, setIsCalculating] = useState(false);
  const [calculationReady, setCalculationReady] = useState(false);

  // Build cleaned data from folders + corrections, only including selected columns
  const cleanedFiles = useMemo(() => {
    const result = { consumption: [], reference: [] };

    (folders || []).forEach((folder) => {
      const category = folder.tag === '#Consumption' ? 'consumption' : 'reference';

      (folder.files || []).forEach((file) => {
        if (!file.content || !file.content.headers || !file.content.rows) return;

        const corrections = (filePendingCorrections || {})[file.id] || {};
        const labels = (fileColumnLabels || {})[file.id] || {};

        // Get selected column indices (only columns labeled in Select Data step)
        const selectedIndices = Object.keys(labels).map(Number).sort((a, b) => a - b);

        // If no columns were selected, skip this file
        if (selectedIndices.length === 0) return;

        // Build filtered headers with their labels
        const filteredHeaders = selectedIndices.map((idx) => ({
          originalIndex: idx,
          name: file.content.headers[idx],
          label: labels[idx],
        }));

        // Apply corrections and filter to selected columns only
        const cleanedRows = file.content.rows.map((row, rowIdx) => {
          return selectedIndices.map((colIdx) => {
            const correction = corrections[rowIdx] && corrections[rowIdx][colIdx];
            if (correction && correction.correctedValue !== undefined) {
              return { value: correction.correctedValue, corrected: true, originalError: correction.originalError };
            }
            return { value: row[colIdx], corrected: false };
          });
        });

        result[category].push({
          fileId: file.id,
          fileName: file.name,
          headers: filteredHeaders,
          rows: cleanedRows,
          selectedIndices: selectedIndices,
          totalRows: cleanedRows.length,
        });
      });
    });

    return result;
  }, [folders, fileColumnLabels, filePendingCorrections]);

  // Get files for current tab
  const currentFiles = cleanedFiles[activeTab] || [];

  // Auto-select first file when tab changes or on mount
  const activeFileId = selectedFileId && currentFiles.some(f => f.fileId === selectedFileId)
    ? selectedFileId
    : (currentFiles.length > 0 ? currentFiles[0].fileId : null);

  const activeFile = currentFiles.find(f => f.fileId === activeFileId) || null;

  const handleRunCalculation = async () => {
    setIsCalculating(true);

    try {
      // Build mock calculation results from cleaned data
      const allFiles = [...cleanedFiles.consumption, ...cleanedFiles.reference];
      const rawData = [];
      const categoryTotals = {};

      allFiles.forEach((file) => {
        file.rows.forEach((row) => {
          const rowObj = {};
          file.headers.forEach((h, i) => {
            rowObj[h.label || h.name] = row[i]?.value ?? '';
          });

          // Generate a mock mtCO2e_calc value
          const numericValues = row.map(c => parseFloat(c?.value)).filter(v => !isNaN(v) && v > 0);
          const mtCO2e = numericValues.length > 0
            ? numericValues.reduce((a, b) => a * b, 1) * 0.001
            : Math.random() * 0.5;

          rowObj.mtCO2e_calc = parseFloat(mtCO2e.toFixed(6));
          rowObj.EF_ID = `EF_${Math.floor(Math.random() * 1000)}`;
          rowObj.formulaReference = {
            formula: 'Consumption × GHG_MTperUnit × GWP',
            calculation: `${numericValues.join(' × ')} = ${mtCO2e.toFixed(6)}`,
            efidSource: file.fileName,
          };

          // Track category totals for pie chart
          const category = file.fileName.replace(/\.[^.]+$/, '');
          categoryTotals[category] = (categoryTotals[category] || 0) + mtCO2e;

          rawData.push(rowObj);
        });
      });

      const totalMtCO2e = Object.values(categoryTotals).reduce((a, b) => a + b, 0);

      // Build pie chart data
      const pieData = Object.entries(categoryTotals).map(([label, value]) => ({
        label,
        percentage: totalMtCO2e > 0 ? parseFloat(((value / totalMtCO2e) * 100).toFixed(1)) : 0,
      }));

      // Build bar chart data (group by rows in batches)
      const batchSize = Math.max(1, Math.ceil(rawData.length / 10));
      const barLabels = [];
      const barValues = [];
      for (let i = 0; i < rawData.length; i += batchSize) {
        const batch = rawData.slice(i, i + batchSize);
        barLabels.push(`Group ${barLabels.length + 1}`);
        barValues.push(parseFloat(batch.reduce((sum, r) => sum + r.mtCO2e_calc, 0).toFixed(5)));
      }

      const result = {
        success: true,
        sessionId,
        summary: {
          totalMtCO2e: parseFloat(totalMtCO2e.toFixed(5)),
          recordCount: rawData.length,
        },
        charts: {
          pieChart: { data: pieData },
          barChart: { labels: barLabels, datasets: [{ data: barValues }] },
        },
        rawData,
        calculationTimestamp: new Date().toISOString(),
      };

      if (onCalculationComplete) {
        onCalculationComplete(result);
      }

      await new Promise(resolve => setTimeout(resolve, 2000));

      setIsCalculating(false);
      setCalculationReady(true);

    } catch (error) {
      console.error('Calculation failed:', error);
      setIsCalculating(false);
      alert(`Calculation failed: ${error.message}`);
    }
  };

  const handleProceedToResults = () => {
    onNext();
  };

  // Check if there are any files at all
  const totalFiles = cleanedFiles.consumption.length + cleanedFiles.reference.length;

  if (totalFiles === 0) {
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
        <div className="flex items-center justify-center h-full">
          <div className="text-center">
            <p className="font-['Noto_Sans',sans-serif] text-[18px] text-[#7E7E7E] mb-4">
              No files uploaded. Please go back and upload data first.
            </p>
            <button
              onClick={onBack}
              className="px-[32px] py-[12px] bg-[#29ABB5] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-medium text-[16px] text-white hover:bg-[#238d96] transition"
            >
              Go Back
            </button>
          </div>
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

      {/* Main Content - absolute layout, no page scroll */}
      <div className="absolute left-[64px] right-[64px] top-[132px] bottom-[32px] flex flex-col overflow-hidden">
        {/* Fixed top: title */}
        <div className="flex items-center mb-[20px] flex-shrink-0">
          <h1 className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B]">
            Preview of Files
          </h1>
        </div>

        {/* Fixed: Folder Tabs + File selector pills on same row */}
        <div className="flex items-center gap-[12px] mb-[20px] flex-shrink-0">
          <button
            onClick={() => { setActiveTab('consumption'); setSelectedFileId(null); }}
            className={`px-[24px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] font-medium text-[14px] transition ${
              activeTab === 'consumption'
                ? 'bg-[#29ABB5] text-white'
                : 'bg-[#F5F5F5] text-[#7E7E7E] hover:bg-[#EEEEEE]'
            }`}
          >
            Consumption Data ({cleanedFiles.consumption.reduce((sum, f) => sum + f.totalRows, 0)} rows, {cleanedFiles.consumption.length} files)
          </button>
          <button
            onClick={() => { setActiveTab('reference'); setSelectedFileId(null); }}
            className={`px-[24px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] font-medium text-[14px] transition ${
              activeTab === 'reference'
                ? 'bg-[#29ABB5] text-white'
                : 'bg-[#F5F5F5] text-[#7E7E7E] hover:bg-[#EEEEEE]'
            }`}
          >
            Reference Data ({cleanedFiles.reference.reduce((sum, f) => sum + f.totalRows, 0)} rows, {cleanedFiles.reference.length} files)
          </button>

          {/* Spacer to push file pills to the right */}
          <div className="flex-1" />

          {/* File selector pills - right-aligned, non-clickable */}
          {currentFiles.map((file) => (
            <div
              key={file.fileId}
              className={`px-[12px] py-[4px] rounded-full font-['Noto_Sans',sans-serif] text-[12px] ${
                file.fileId === activeFileId
                  ? 'bg-[#365D60] text-white'
                  : 'bg-[#E3E3E3] text-[#575757]'
              }`}
            >
              {file.fileName} ({file.totalRows} rows)
            </div>
          ))}
        </div>

        {/* Data Table - capped height, internal scroll */}
        <div className="min-h-0 h-[560px] mb-[16px] flex-shrink-0">
          {activeFile ? (
            <div className="bg-white rounded-[10px] border border-[#EEEEEE] overflow-hidden h-full">
              <div className="overflow-auto h-full">
                <table className="w-full">
                  <thead className="sticky top-0 z-10">
                    <tr>
                      {activeFile.headers.map((headerInfo, idx) => (
                        <th
                          key={idx}
                          className="px-[16px] py-[12px] min-w-[120px] border border-[#EEEEEE] bg-[#F5F5F5] text-[#13383B] text-left"
                        >
                          <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px]">
                            {headerInfo.name}
                          </p>
                          {headerInfo.label && (
                            <p
                              className="font-['Noto_Sans',sans-serif] font-normal text-[12px] mt-[4px]"
                              style={{ opacity: 0.8 }}
                            >
                              → {headerInfo.label}
                            </p>
                          )}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {activeFile.rows.map((row, rowIdx) => (
                      <tr key={rowIdx} className="border-t border-[#EEEEEE] hover:bg-[#F5F5F5]">
                        {row.map((cellData, colIdx) => (
                          <td
                            key={colIdx}
                            className={`px-[16px] py-[12px] font-['Noto_Sans',sans-serif] text-[14px] whitespace-nowrap border border-[#EEEEEE] ${
                              cellData.corrected
                                ? 'text-[#50B461] bg-[#50B461]/5 font-medium'
                                : 'text-[#13383B]'
                            }`}
                            title={cellData.corrected ? `Corrected from: ${cellData.originalError || 'N/A'}` : undefined}
                          >
                            {String(cellData.value ?? '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full bg-[#F5F5F5] rounded-[10px]">
              <p className="font-['Noto_Sans',sans-serif] text-[16px] text-[#7E7E7E]">
                No files in this category
              </p>
            </div>
          )}
        </div>

        {/* Fixed bottom: Action Buttons - always pinned to bottom */}
        <div className="flex justify-between items-center flex-shrink-0 mt-auto">
          <button
            onClick={() => setShowFormulaModal(true)}
            className="px-[32px] py-[12px] bg-white border-2 border-[#29ABB5] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-medium text-[16px] text-[#29ABB5] hover:bg-[#F0F9FA] transition"
          >
            View Formula & Calculation
          </button>

          <div className="flex gap-[16px]">
            <button className="px-[32px] py-[12px] bg-[#E3E3E3] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-medium text-[16px] text-[#575757] hover:bg-[#D8D8D8] transition">
              AWS Documentation & Instruction
            </button>

            {!calculationReady ? (
              <button
                onClick={handleRunCalculation}
                disabled={isCalculating}
                className={`px-[32px] py-[12px] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-bold text-[18px] text-white transition ${
                  isCalculating
                    ? 'bg-[#7E7E7E] cursor-not-allowed'
                    : 'bg-[#29ABB5] hover:bg-[#238d96]'
                }`}
              >
                {isCalculating ? 'Calculating...' : 'Run Calculation'}
              </button>
            ) : (
              <button
                onClick={handleProceedToResults}
                className="px-[32px] py-[12px] bg-[#50B461] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-bold text-[18px] text-white hover:bg-[#45a056] transition"
              >
                View Results
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Formula Modal */}
      {showFormulaModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowFormulaModal(false)}>
          <div className="bg-white rounded-[10px] p-[32px] w-[600px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)]" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-[24px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[20px] text-[#575757]">
                Formula & Calculation
              </p>
              <button onClick={() => setShowFormulaModal(false)}>
                <CloseIcon className="w-[24px] h-[24px] text-[#575757] cursor-pointer hover:text-[#29ABB5] transition" />
              </button>
            </div>

            <div className="bg-[#365D60] rounded-[8px] p-[24px] mb-[24px]">
              <div className="flex items-center justify-center gap-[16px] mb-[16px]">
                <div className="bg-white/20 px-[16px] py-[8px] rounded-[4px]">
                  <p className="font-['Noto_Sans',sans-serif] text-[14px] text-white">Consumption</p>
                </div>
                <p className="font-['Noto_Sans',sans-serif] text-[18px] text-white font-bold">×</p>
                <div className="bg-white/20 px-[16px] py-[8px] rounded-[4px]">
                  <p className="font-['Noto_Sans',sans-serif] text-[14px] text-white">GHG_MTperUnit</p>
                </div>
                <p className="font-['Noto_Sans',sans-serif] text-[18px] text-white font-bold">×</p>
                <div className="bg-white/20 px-[16px] py-[8px] rounded-[4px]">
                  <p className="font-['Noto_Sans',sans-serif] text-[14px] text-white">GWP</p>
                </div>
              </div>
              <div className="text-center">
                <p className="font-['Noto_Sans',sans-serif] text-[18px] text-white font-bold">=</p>
              </div>
              <div className="flex justify-center mt-[16px]">
                <div className="bg-[#29ABB5] px-[24px] py-[12px] rounded-[4px]">
                  <p className="font-['Noto_Sans',sans-serif] text-[16px] text-white font-medium">mtCO2e_calc</p>
                </div>
              </div>
            </div>

            <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] mb-[24px]">
              This formula calculates the carbon dioxide equivalent emissions (mtCO2e) by multiplying the consumption amount by the greenhouse gas emissions per unit and the global warming potential factor.
            </p>

            <button
              onClick={() => setShowFormulaModal(false)}
              className="w-full h-[40px] bg-[#29ABB5] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-white hover:bg-[#238d96] transition"
            >
              Close
            </button>
          </div>
        </div>
      )}

      {/* Calculating Modal */}
      {isCalculating && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-[16px] p-[48px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)] text-center">
            <div className="flex items-center justify-center gap-[24px] mb-[24px]">
              <div className="animate-spin w-[48px] h-[48px] border-4 border-[#29ABB5] border-t-transparent rounded-full"></div>
              <div className="w-[48px] h-[48px] flex items-center justify-center">
                <svg viewBox="0 0 24 24" className="w-[48px] h-[48px]">
                  <rect x="3" y="3" width="18" height="18" rx="2" stroke="#29ABB5" strokeWidth="1.5" fill="none"/>
                  <line x1="3" y1="9" x2="21" y2="9" stroke="#29ABB5" strokeWidth="1.5"/>
                  <line x1="3" y1="15" x2="21" y2="15" stroke="#29ABB5" strokeWidth="1.5"/>
                  <line x1="9" y1="9" x2="9" y2="21" stroke="#29ABB5" strokeWidth="1.5"/>
                </svg>
              </div>
              <div className="w-[48px] h-[48px] flex items-center justify-center">
                <svg viewBox="0 0 48 48" className="w-[48px] h-[48px]">
                  <circle cx="24" cy="24" r="20" stroke="#29ABB5" strokeWidth="2" fill="none"/>
                  <path d="M24 14 L24 34 M14 24 L34 24" stroke="#29ABB5" strokeWidth="2"/>
                </svg>
              </div>
            </div>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[20px] text-[#575757]">
              Running Calculation...
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
