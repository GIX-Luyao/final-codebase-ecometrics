import React, { useState, useEffect } from 'react';
import { runCalculation, previewSource, listSources } from './services/api';

// Icons
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
      <line x1="12" y1="8" x2="12.01" y2="8" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

function FileIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" strokeWidth="2"/>
      <polyline points="13 2 13 9 20 9" strokeWidth="2"/>
    </svg>
  );
}

function CalculatorIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <rect x="4" y="2" width="16" height="20" rx="2" strokeWidth="2"/>
      <line x1="8" y1="6" x2="16" y2="6" strokeWidth="2"/>
      <line x1="8" y1="10" x2="12" y2="10" strokeWidth="2"/>
      <line x1="8" y1="14" x2="12" y2="14" strokeWidth="2"/>
      <line x1="16" y1="10" x2="16" y2="14" strokeWidth="2"/>
    </svg>
  );
}

// Sample data
const efidData = [
  { ef_id: 'Fuel_DSL_20XX_CO2_gal', ghg_mt: '0.0102100000', unit: 'CO2 MT/gal', ghg: 'CO2', gwp: '1', description: 'Fleet Combustion', scope: 'Scope 1' },
  { ef_id: 'Fuel_DSL_20XX_CH4_gal', ghg_mt: '0.0000002083', unit: 'CH4 MT/gal', ghg: 'CH4', gwp: '28', description: 'Fleet Combustion', scope: 'Scope 1' },
  { ef_id: 'Fuel_DSL_20XX_N2O_gal', ghg_mt: '0.0000002236', unit: 'N2O MT/gal', ghg: 'N2O', gwp: '265', description: 'Fleet Combustion', scope: 'Scope 1' },
  { ef_id: 'Fuel_BIO_20XX_CO2_gal', ghg_mt: '0.0094500000', unit: 'CO2 MT/gal', ghg: 'CO2', gwp: '1', description: 'Fleet Combustion', scope: 'Scope 1' },
  { ef_id: 'Fuel_BIO_20XX_CH4_gal', ghg_mt: '0.0000001928', unit: 'CH4 MT/gal', ghg: 'CH4', gwp: '28', description: 'Fleet Combustion', scope: 'Scope 1' },
  { ef_id: 'Fuel_BIO_20XX_N2O_gal', ghg_mt: '0.0000002070', unit: 'N2O MT/gal', ghg: 'N2O', gwp: '265', description: 'Fleet Combustion', scope: 'Scope 1' },
];

const gwpData = [
  { ghg_id: 'CH4', ghg_name: 'CH4', ar5_gwp: '28', ghg_longname: 'Methane', ar2_sar: '21', ar3_tar: '23', ar4: '25', source: 'TCR' },
  { ghg_id: 'CO2', ghg_name: 'CO2', ar5_gwp: '1', ghg_longname: 'Carbon dioxide', ar2_sar: '1', ar3_tar: '1', ar4: '1', source: 'TCR' },
  { ghg_id: 'CO2e', ghg_name: 'CO2e', ar5_gwp: '1', ghg_longname: 'Carbon dioxide equivalent', ar2_sar: '1', ar3_tar: '1', ar4: '1', source: 'TCR' },
  { ghg_id: 'N2O', ghg_name: 'N2O', ar5_gwp: '265', ghg_longname: 'Nitrous oxide', ar2_sar: '310', ar3_tar: '296', ar4: '298', source: 'TCR' },
  { ghg_id: 'NF3', ghg_name: 'NF3', ar5_gwp: '16100', ghg_longname: 'Nitrogen trifluoride', ar2_sar: 'n/a', ar3_tar: '10800', ar4: '17200', source: 'TCR' },
  { ghg_id: 'R-134A', ghg_name: 'HFC-134A', ar5_gwp: '1300', ghg_longname: '1,1,1,2-tetrafluoroethane', ar2_sar: '1300', ar3_tar: '1300', ar4: '1430', source: 'TCR' },
  { ghg_id: 'R-22', ghg_name: 'HCFC-22', ar5_gwp: '1760', ghg_longname: 'Chlorodifluoromethane (Freon)', ar2_sar: '1500', ar3_tar: '1700', ar4: '1810', source: 'GHGP/CARB' },
];

const FIELD_DESCRIPTIONS = {
  'EF_ID': 'Emission Factor ID - unique identifier combining fuel type, year range, GHG type, and unit',
  'GHG_MTperUnit': 'Greenhouse Gas in Metric Tons per Unit - the emission factor value',
  'Unit': 'Unit of measurement for the emission factor',
  'GHG': 'Type of greenhouse gas emitted (CO2, CH4, N2O, etc.)',
  'GWP': 'Global Warming Potential - multiplier for converting to CO2 equivalent',
  'Description': 'Description of the emission source or activity',
  'Scope': 'GHG Protocol scope classification (Scope 1 = Direct, Scope 2 = Indirect from purchased energy)',
  'GHG_ID': 'Greenhouse Gas identifier code',
  'GHG_Name': 'Chemical formula or common name',
  'AR5_GWP': 'GWP from IPCC Fifth Assessment Report (100-year timeframe)',
  'GHG_LongName': 'Full descriptive name of the greenhouse gas',
  'AR2_SAR': 'GWP from IPCC Second Assessment Report',
  'AR3_TAR': 'GWP from IPCC Third Assessment Report',
  'AR4': 'GWP from IPCC Fourth Assessment Report',
  'Source': 'Data source reference (TCR = The Climate Registry)',
};

export default function PreviewFiles({
  sessionId,
  onNext,
  onBack,
  onCalculationComplete,
  onGoToStep,
  onLogoClick,
}) {
  const [activeTab, setActiveTab] = useState('reference');
  const [selectedFile, setSelectedFile] = useState('efids');
  const [showFormulaModal, setShowFormulaModal] = useState(false);
  const [isCalculating, setIsCalculating] = useState(false);
  const [calculationReady, setCalculationReady] = useState(false);
  const [hoveredColumn, setHoveredColumn] = useState(null);

  // Consumption tab state
  const [consumptionSources, setConsumptionSources] = useState([]);
  const [consumptionPreview, setConsumptionPreview] = useState(null);
  const [selectedSource, setSelectedSource] = useState(null);
  const [loadingConsumption, setLoadingConsumption] = useState(false);

  // Load source list when consumption tab opens
  useEffect(() => {
    if (activeTab !== 'consumption') return;
    listSources()
      .then(d => {
        const srcs = d.sources || [];
        setConsumptionSources(srcs);
        if (srcs.length > 0 && !selectedSource) setSelectedSource(srcs[0]);
      })
      .catch(err => console.error('[PreviewFiles] listSources error:', err));
  }, [activeTab, selectedSource]);

  // Load preview when selected source changes
  useEffect(() => {
    if (!selectedSource) return;
    setLoadingConsumption(true);
    setConsumptionPreview(null);
    previewSource(selectedSource.sourceId, selectedSource.sheets?.[0])
      .then(d => { setConsumptionPreview(d); setLoadingConsumption(false); })
      .catch(err => { console.error('[PreviewFiles] previewSource error:', err); setLoadingConsumption(false); });
  }, [selectedSource]);

  const handleRunCalculation = async () => {
    if (!sessionId) {
      console.error('[PreviewFiles] No sessionId available');
      return;
    }
    setIsCalculating(true);
    try {
      const result = await runCalculation(sessionId, undefined, 'AR5');
      setIsCalculating(false);
      setCalculationReady(true);
      if (onCalculationComplete) onCalculationComplete(result);
    } catch (err) {
      console.error('[PreviewFiles] runCalculation error:', err);
      setIsCalculating(false);
    }
  };

  const handleProceedToResults = () => { onNext(); };

  const thStyle = "px-[16px] py-[12px] text-left font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#13383B]";
  const tdStyle = "px-[16px] py-[12px] font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B]";
  const handleLogoClick = () => {
    if (onLogoClick) onLogoClick();
  };
  const handleStepClick = (step, fallback) => {
    if (onGoToStep) {
      onGoToStep(step);
      return;
    }
    if (fallback) fallback();
  };

  return (
    <div className="bg-white relative w-full min-h-screen overflow-auto">
      {/* Toolbar */}
      <div className="sticky top-0 left-0 w-full h-[62px] bg-[#365D60] flex items-center justify-between px-[64px] z-20">
        <button
          type="button"
          onClick={handleLogoClick}
          className="flex items-center gap-[12px] bg-transparent border-0 p-0 cursor-pointer"
        >
          <img src="/logo.png" alt="Logo" className="h-[40px] w-auto" />
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[24px] text-white">EcoMetrics</p>
        </button>
        <div className="bg-white rounded-[10px] w-[331px] h-[40px] flex items-center px-[12px]">
          <SearchIcon className="w-[20px] h-[20px] text-[#7E7E7E]" />
        </div>
      </div>

      {/* Progress Steps */}
      <div className="sticky top-[62px] bg-white left-[64px] right-[64px] py-[19px] px-[64px] flex items-center z-10 border-b border-[#EEEEEE]">
        <div className="flex items-center gap-[12px] cursor-pointer" onClick={() => handleStepClick(4, onBack)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">✓</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">Upload Your Data</p>
        </div>
        <div className="flex-1 h-[2px] bg-[#50B461] mx-[16px]"></div>
        <div className="flex items-center gap-[12px] cursor-pointer" onClick={() => handleStepClick(3)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">✓</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">Data Cleaning</p>
        </div>
        <div className="flex-1 h-[2px] bg-[#50B461] mx-[16px]"></div>
        <div className="flex items-center gap-[12px] cursor-pointer" onClick={() => handleStepClick(5)}>
          <div className="w-[32px] h-[32px] rounded-full bg-[#29ABB5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">3</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">Data Processing</p>
        </div>
        <div className="flex-1 h-[2px] bg-[#ACC2C5] mx-[16px]"></div>
        <div className="flex items-center gap-[12px] cursor-pointer" onClick={() => handleStepClick(6)}>
          <div className="w-[32px] h-[32px] rounded-full bg-white border-2 border-[#ACC2C5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#7E7E7E]">4</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] whitespace-nowrap">Data Visualization</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="px-[64px] py-[32px]">
        <h1 className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B] mb-[24px]">
          Preview of Files
        </h1>

        {/* Tabs */}
        <div className="flex gap-[12px] mb-[24px]">
          <button onClick={() => setActiveTab('reference')}
            className={`px-[24px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] font-medium text-[14px] transition ${
              activeTab === 'reference' ? 'bg-[#29ABB5] text-white' : 'bg-[#F5F5F5] text-[#7E7E7E] hover:bg-[#EEEEEE]'
            }`}>Reference Data</button>
          <button onClick={() => setActiveTab('consumption')}
            className={`px-[24px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] font-medium text-[14px] transition ${
              activeTab === 'consumption' ? 'bg-[#29ABB5] text-white' : 'bg-[#F5F5F5] text-[#7E7E7E] hover:bg-[#EEEEEE]'
            }`}>Consumption Data</button>
        </div>

        {/* Reference Data Content */}
        {activeTab === 'reference' && (
          <>
            {/* File selector pills */}
            <div className="flex gap-[12px] mb-[24px]">
              <button onClick={() => setSelectedFile('efids')}
                className={`px-[16px] py-[8px] rounded-[4px] font-['Noto_Sans',sans-serif] text-[14px] transition ${
                  selectedFile === 'efids' ? 'bg-[#365D60] text-white' : 'bg-[#E3E3E3] text-[#575757] hover:bg-[#D8D8D8]'
                }`}>EFIDs.xlsx</button>
              <button onClick={() => setSelectedFile('gwp')}
                className={`px-[16px] py-[8px] rounded-[4px] font-['Noto_Sans',sans-serif] text-[14px] transition ${
                  selectedFile === 'gwp' ? 'bg-[#365D60] text-white' : 'bg-[#E3E3E3] text-[#575757] hover:bg-[#D8D8D8]'
                }`}>GWP.xlsx</button>
            </div>

            {/* Data Table */}
            <div className="bg-white rounded-[10px] border border-[#EEEEEE] overflow-hidden mb-[24px]">
              <div className="overflow-x-auto max-h-[420px]">
                {selectedFile === 'efids' ? (
                  <table className="w-full">
                    <thead className="bg-[#D9D9D9] sticky top-0 z-[5]">
                      <tr>
                        {['EF_ID','GHG_MTperUnit','Unit','GHG','GWP','Description','Scope'].map(col => (
                          <th key={col} className={`${thStyle} relative group`}
                            onMouseEnter={() => setHoveredColumn(col)} onMouseLeave={() => setHoveredColumn(null)}>
                            <div className="flex items-center gap-[6px]">
                              {col === 'GHG_MTperUnit' ? 'GHG_MTperUnit' : col}
                              <InfoIcon className="w-[14px] h-[14px] text-[#7E7E7E] opacity-0 group-hover:opacity-100 transition"/>
                            </div>
                            {hoveredColumn === col && FIELD_DESCRIPTIONS[col] && (
                              <div className="absolute top-full left-0 mt-[4px] bg-[#365D60] text-white text-[12px] px-[12px] py-[8px] rounded-[6px] w-[280px] z-10 shadow-lg font-normal">
                                <p className="leading-[1.4]">{FIELD_DESCRIPTIONS[col]}</p>
                              </div>
                            )}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {efidData.map((row, idx) => (
                        <tr key={idx} className="border-t border-[#EEEEEE] hover:bg-[#E8F4F3] transition">
                          <td className={tdStyle}>{row.ef_id}</td>
                          <td className={tdStyle}>{row.ghg_mt}</td>
                          <td className={tdStyle}>{row.unit}</td>
                          <td className={tdStyle}>{row.ghg}</td>
                          <td className={tdStyle}>{row.gwp}</td>
                          <td className={tdStyle}>{row.description}</td>
                          <td className={tdStyle}>{row.scope}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <table className="w-full">
                    <thead className="bg-[#D9D9D9] sticky top-0 z-[5]">
                      <tr>
                        {['GHG_ID','GHG_Name','AR5_GWP','GHG_LongName','AR2_SAR','AR3_TAR','AR4','Source'].map(col => (
                          <th key={col} className={`${thStyle} relative group`}
                            onMouseEnter={() => setHoveredColumn(col)} onMouseLeave={() => setHoveredColumn(null)}>
                            <div className="flex items-center gap-[6px]">
                              {col}
                              <InfoIcon className="w-[14px] h-[14px] text-[#7E7E7E] opacity-0 group-hover:opacity-100 transition"/>
                            </div>
                            {hoveredColumn === col && FIELD_DESCRIPTIONS[col] && (
                              <div className="absolute top-full left-0 mt-[4px] bg-[#365D60] text-white text-[12px] px-[12px] py-[8px] rounded-[6px] w-[280px] z-10 shadow-lg font-normal">
                                <p className="leading-[1.4]">{FIELD_DESCRIPTIONS[col]}</p>
                              </div>
                            )}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {gwpData.map((row, idx) => (
                        <tr key={idx} className="border-t border-[#EEEEEE] hover:bg-[#E8F4F3] transition">
                          <td className={tdStyle}>{row.ghg_id}</td>
                          <td className={tdStyle}>{row.ghg_name}</td>
                          <td className={tdStyle}>{row.ar5_gwp}</td>
                          <td className={tdStyle}>{row.ghg_longname}</td>
                          <td className={tdStyle}>{row.ar2_sar}</td>
                          <td className={tdStyle}>{row.ar3_tar}</td>
                          <td className={tdStyle}>{row.ar4}</td>
                          <td className={tdStyle}>{row.source}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex justify-between items-center">
              <button onClick={() => setShowFormulaModal(true)}
                className="px-[32px] py-[12px] bg-white border-2 border-[#29ABB5] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-medium text-[16px] text-[#29ABB5] hover:bg-[#E8F4F3] transition">
                View Formula & Calculation
              </button>

              <div className="flex gap-[16px]">
                <button className="px-[32px] py-[12px] bg-[#E3E3E3] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-medium text-[16px] text-[#575757] hover:bg-[#D8D8D8] transition">
                  AWS Documentation & Instruction
                </button>
                
                {!calculationReady ? (
                  <button onClick={handleRunCalculation} disabled={isCalculating}
                    className={`px-[32px] py-[12px] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-bold text-[18px] text-white transition flex items-center gap-[12px] ${
                      isCalculating ? 'bg-[#7E7E7E] cursor-not-allowed' : 'bg-[#29ABB5] hover:bg-[#238d96]'
                    }`}>
                    {isCalculating ? (
                      <><svg className="w-[20px] h-[20px] animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M21 12a9 9 0 11-6.219-8.56" strokeWidth="2" strokeLinecap="round"/>
                      </svg>Calculating...</>
                    ) : (
                      <><CalculatorIcon className="w-[20px] h-[20px]"/>Run Calculation</>
                    )}
                  </button>
                ) : (
                  <button onClick={handleProceedToResults}
                    className="px-[32px] py-[12px] bg-[#50B461] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-bold text-[18px] text-white hover:bg-[#45a056] transition flex items-center gap-[8px]">
                    View Results
                    <svg className="w-[20px] h-[20px]" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M5 12h14M12 5l7 7-7 7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </button>
                )}
              </div>
            </div>
          </>
        )}

        {/* Consumption Data Content */}
        {activeTab === 'consumption' && (
          <div>
            {/* Source file pills */}
            <div className="flex gap-[12px] mb-[24px] flex-wrap">
              {consumptionSources.length === 0 && !loadingConsumption && (
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">No source files found.</p>
              )}
              {consumptionSources.map(src => (
                <button key={src.sourceId}
                  onClick={() => setSelectedSource(src)}
                  className={`px-[16px] py-[8px] rounded-[4px] font-['Noto_Sans',sans-serif] text-[14px] transition ${
                    selectedSource?.sourceId === src.sourceId
                      ? 'bg-[#365D60] text-white'
                      : 'bg-[#E3E3E3] text-[#575757] hover:bg-[#D8D8D8]'
                  }`}>{src.name}</button>
              ))}
            </div>

            {loadingConsumption && (
              <div className="text-center py-[64px]">
                <div className="animate-spin w-[40px] h-[40px] border-4 border-[#29ABB5] border-t-transparent rounded-full mx-auto mb-[16px]"></div>
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">Loading preview...</p>
              </div>
            )}

            {consumptionPreview && !loadingConsumption && (
              <>
                <div className="flex items-center gap-[16px] mb-[16px]">
                  <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">
                    <span className="font-medium text-[#13383B]">{consumptionPreview.totalRows?.toLocaleString()}</span> total rows
                  </p>
                  {consumptionPreview.sheet && (
                    <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">
                      Sheet: <span className="font-medium text-[#13383B]">{consumptionPreview.sheet}</span>
                    </p>
                  )}
                </div>
                <div className="bg-white rounded-[10px] border border-[#EEEEEE] overflow-hidden mb-[24px]">
                  <div className="overflow-x-auto max-h-[420px]">
                    <table className="w-full">
                      <thead className="bg-[#D9D9D9] sticky top-0 z-[5]">
                        <tr>
                          {consumptionPreview.columns?.map(col => (
                            <th key={col.name} className={`${thStyle} relative group`}
                              onMouseEnter={() => setHoveredColumn(col.name)}
                              onMouseLeave={() => setHoveredColumn(null)}>
                              <div className="flex items-center gap-[6px]">
                                {col.name}
                                <InfoIcon className="w-[14px] h-[14px] text-[#7E7E7E] opacity-0 group-hover:opacity-100 transition"/>
                              </div>
                              {hoveredColumn === col.name && (
                                <div className="absolute top-full left-0 mt-[4px] bg-[#365D60] text-white text-[12px] px-[12px] py-[8px] rounded-[6px] w-[220px] z-10 shadow-lg font-normal">
                                  <p className="leading-[1.4]">Type: {col.type} · Nulls: {col.nullCount ?? 0}</p>
                                  {col.sampleValues && <p className="mt-[4px] opacity-80">Samples: {col.sampleValues.slice(0,3).join(', ')}</p>}
                                </div>
                              )}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {consumptionPreview.preview?.map((row, idx) => (
                          <tr key={idx} className="border-t border-[#EEEEEE] hover:bg-[#E8F4F3] transition">
                            {consumptionPreview.columns?.map(col => (
                              <td key={col.name} className={tdStyle}>{String(row[col.name] ?? '')}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Formula Modal */}
      {showFormulaModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" 
          onClick={() => setShowFormulaModal(false)}
          style={{ animation: 'fadeIn 0.2s ease-out' }}>
          <div className="bg-white rounded-[10px] p-[32px] w-[600px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)]" 
            onClick={(e) => e.stopPropagation()}
            style={{ animation: 'slideUp 0.3s ease-out' }}>
            <div className="flex items-center justify-between mb-[24px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[20px] text-[#13383B]">
                Formula & Calculation
              </p>
              <CloseIcon className="w-[24px] h-[24px] text-[#575757] cursor-pointer hover:text-[#13383B] transition" onClick={() => setShowFormulaModal(false)} />
            </div>

            <div className="bg-[#365D60] rounded-[8px] p-[24px] mb-[24px]">
              <div className="flex items-center justify-center gap-[16px] mb-[16px] flex-wrap">
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

            <div className="bg-[#F5F5F5] rounded-[8px] p-[16px] mb-[24px]">
              <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] leading-relaxed">
                This formula calculates the carbon dioxide equivalent emissions (mtCO2e) by multiplying the <strong>consumption amount</strong> by the <strong>greenhouse gas emissions per unit</strong> (emission factor) and the <strong>global warming potential</strong> factor.
              </p>
              <div className="mt-[12px] border-t border-[#D9D9D9] pt-[12px]">
                <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E]">
                  <strong>Example:</strong> 100 gal × 0.0102 CO2 MT/gal × 1 GWP = 1.02 mtCO2e
                </p>
              </div>
            </div>

            <button onClick={() => setShowFormulaModal(false)}
              className="w-full h-[40px] bg-[#29ABB5] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-white hover:bg-[#238d96] transition">
              Close
            </button>
          </div>
        </div>
      )}

      {/* Calculating Modal */}
      {isCalculating && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-[16px] p-[48px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)] text-center"
            style={{ animation: 'slideUp 0.3s ease-out' }}>
            <div className="flex items-center justify-center gap-[24px] mb-[24px]">
              <div className="animate-spin w-[48px] h-[48px] border-4 border-[#29ABB5] border-t-transparent rounded-full"></div>
              <FileIcon className="w-[40px] h-[40px] text-[#29ABB5]"/>
              <CalculatorIcon className="w-[40px] h-[40px] text-[#29ABB5]"/>
            </div>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[20px] text-[#575757]">
              Running Calculation...
            </p>
            <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E] mt-[8px]">
              Processing emission factors and computing results
            </p>
          </div>
        </div>
      )}

      <style>{`
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}
