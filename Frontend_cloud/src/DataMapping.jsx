import React, { useState, useEffect, useRef } from 'react';
import { listSources, previewSource, saveSourceConfig, runCalculation } from './services/api';

function SearchIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="11" cy="11" r="8" strokeWidth="2"/>
      <path d="m21 21-4.35-4.35" strokeWidth="2" strokeLinecap="round"/>
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

function ChevronDownIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <path d="M6 9l6 6 6-6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
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

function PlusIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <line x1="12" y1="5" x2="12" y2="19" strokeWidth="2" strokeLinecap="round"/>
      <line x1="5" y1="12" x2="19" y2="12" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

const REQUIRED_FIELDS = ['ACCT_ID', 'Year', 'Consumption', 'Unit', 'Subtype', 'Sector', 'GHG', 'EF_ID'];

function getMappingSummary(mapping) {
  if (!mapping) return null;
  if (mapping.sourceColumn) return mapping.sourceColumn;
  if (mapping.staticValue) return mapping.staticValue;
  if (mapping.pattern) return mapping.pattern;
  if (mapping.ghgType) return mapping.ghgType.join(', ');
  if (mapping.derivedFrom) return `${mapping.derivedFrom} (${mapping.extractType})`;
  if (mapping.referenceTable) return `Lookup: ${mapping.columnJoined}`;
  return null;
}

// Configuration Modal component
function ConfigModal({ fieldName, sourceColumns, mapping, onSave, onClose, isOptional, referenceSources }) {
  const [mode, setMode] = useState(mapping?.sourceColumn ? 'column' : 'fill');
  const [selectedColumn, setSelectedColumn] = useState(mapping?.sourceColumn || '');
  const [fillType, setFillType] = useState(() => {
    if (mapping?.pattern) return 'pattern';
    if (mapping?.staticValue) return 'fixed';
    if (mapping?.ghgType) return 'ghg';
    if (mapping?.derivedFrom) return 'extract';
    if (mapping?.referenceTable) return 'lookup';
    return 'fixed';
  });
  const [fixedValue, setFixedValue] = useState(mapping?.staticValue || '');
  const [patternValue, setPatternValue] = useState(mapping?.pattern || '');
  const [ghgSelections, setGhgSelections] = useState(mapping?.ghgType || []);
  const [extractColumn, setExtractColumn] = useState(mapping?.derivedFrom || '');
  const [extractType, setExtractType] = useState(mapping?.extractType || 'year');
  const [refTable, setRefTable] = useState(mapping?.referenceTable || '');
  const [refColumnJoined, setRefColumnJoined] = useState(mapping?.columnJoined || '');
  const [refSourceKey, setRefSourceKey] = useState(mapping?.sourceKey || '');
  const [refKey, setRefKey] = useState(mapping?.refKey || '');
  const [optionalName, setOptionalName] = useState(fieldName || '');

  const modalRef = useRef(null);
  const [modalPosition, setModalPosition] = useState({ openBelow: true, alignLeft: true });

  useEffect(() => {
    function handleClickOutside(e) {
      if (modalRef.current && !modalRef.current.contains(e.target)) {
        onClose();
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [onClose]);

  // Calculate modal position to stay within viewport
  useEffect(() => {
    if (modalRef.current) {
      const parent = modalRef.current.parentElement;
      if (parent) {
        const rect = parent.getBoundingClientRect();
        const spaceBelow = window.innerHeight - rect.bottom;
        const spaceRight = window.innerWidth - rect.left;
        setModalPosition({
          openBelow: spaceBelow >= 420,
          alignLeft: spaceRight >= 500,
        });
      }
    }
  }, []);

  const handleSave = () => {
    const name = isOptional ? optionalName.trim() : fieldName;
    if (isOptional && !name) return;

    let result;
    if (mode === 'column') {
      result = { sourceColumn: selectedColumn };
    } else {
      switch (fillType) {
        case 'fixed':
          result = { staticValue: fixedValue };
          break;
        case 'pattern':
          result = { pattern: patternValue };
          break;
        case 'ghg':
          result = { ghgType: ghgSelections };
          break;
        case 'extract':
          result = { derivedFrom: extractColumn, extractType };
          break;
        case 'lookup':
          result = { referenceTable: refTable, refKey, sourceKey: refSourceKey, columnJoined: refColumnJoined };
          break;
        default:
          return;
      }
    }
    onSave(name, result);
  };

  const toggleGhg = (gas) => {
    setGhgSelections(prev =>
      prev.includes(gas) ? prev.filter(g => g !== gas) : [...prev, gas]
    );
  };

  return (
    <div
      ref={modalRef}
      className="absolute z-50 bg-white rounded-[10px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.15)] p-[24px] w-[480px]"
      style={{
        ...(modalPosition.openBelow
          ? { top: '100%', marginTop: '8px' }
          : { bottom: '100%', marginBottom: '8px' }),
        ...(modalPosition.alignLeft
          ? { left: '0' }
          : { right: '0' }),
      }}
    >
      {/* Optional field name input */}
      {isOptional && (
        <div className="flex items-center gap-[12px] mb-[16px] pb-[16px] border-b border-[#EEEEEE]">
          <label className="font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#575757] whitespace-nowrap">
            Name:
          </label>
          <input
            type="text"
            value={optionalName}
            onChange={(e) => setOptionalName(e.target.value)}
            placeholder="e.g. LowOrg, Month"
            className="flex-1 h-[36px] px-[12px] border border-[#D9D9D9] rounded-[6px] font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] focus:outline-none focus:border-[#29ABB5]"
          />
        </div>
      )}

      {/* Radio: Select Column from Raw File */}
      <label className="flex items-center gap-[8px] mb-[8px] cursor-pointer">
        <input
          type="radio"
          name={`mode-${fieldName}`}
          checked={mode === 'column'}
          onChange={() => setMode('column')}
          className="w-[16px] h-[16px] accent-[#365D60]"
        />
        <span className="font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#13383B]">
          Select Column from Raw File
        </span>
      </label>
      {mode === 'column' && (
        <div className="ml-[24px] mb-[16px]">
          <select
            value={selectedColumn}
            onChange={(e) => setSelectedColumn(e.target.value)}
            className="w-full h-[36px] px-[12px] border border-[#D9D9D9] rounded-[6px] font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] bg-white focus:outline-none focus:border-[#29ABB5]"
          >
            <option value="">-- Select column --</option>
            {sourceColumns.map(col => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
        </div>
      )}

      {/* Radio: Fill in Information */}
      <label className="flex items-center gap-[8px] mb-[8px] cursor-pointer">
        <input
          type="radio"
          name={`mode-${fieldName}`}
          checked={mode === 'fill'}
          onChange={() => setMode('fill')}
          className="w-[16px] h-[16px] accent-[#365D60]"
        />
        <span className="font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#13383B]">
          Fill in Information
        </span>
      </label>

      {mode === 'fill' && (
        <div className="ml-[24px]">
          <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E] mb-[12px]">
            How to fill this field?
          </p>

          {/* Grid of fill sub-options */}
          <div className="grid grid-cols-2 gap-x-[24px] gap-y-[16px]">
            {/* Enter Fixed Value */}
            <div>
              <label className="flex items-center gap-[6px] mb-[6px] cursor-pointer">
                <input
                  type="radio"
                  name={`fill-${fieldName}`}
                  checked={fillType === 'fixed'}
                  onChange={() => setFillType('fixed')}
                  className="w-[14px] h-[14px] accent-[#365D60]"
                />
                <span className="font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B]">
                  Enter Fixed Value
                </span>
              </label>
              {fillType === 'fixed' && (
                <input
                  type="text"
                  value={fixedValue}
                  onChange={(e) => setFixedValue(e.target.value)}
                  placeholder="e.g. Fleet, gal, KWH"
                  className="w-full h-[32px] px-[8px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B] focus:outline-none focus:border-[#29ABB5]"
                />
              )}
            </div>

            {/* Name a Pattern */}
            <div>
              <label className="flex items-center gap-[6px] mb-[6px] cursor-pointer">
                <input
                  type="radio"
                  name={`fill-${fieldName}`}
                  checked={fillType === 'pattern'}
                  onChange={() => setFillType('pattern')}
                  className="w-[14px] h-[14px] accent-[#365D60]"
                />
                <span className="font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B]">
                  Name a Pattern
                </span>
              </label>
              {fillType === 'pattern' && (
                <div>
                  <input
                    type="text"
                    value={patternValue}
                    onChange={(e) => setPatternValue(e.target.value)}
                    placeholder="e.g. Fuel_{Subtype}_20XX_{GHG}"
                    className="w-full h-[32px] px-[8px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B] focus:outline-none focus:border-[#29ABB5]"
                  />
                  <p className="mt-[4px] font-['Noto_Sans',sans-serif] text-[11px] text-[#7E7E7E]">
                    Use {'{ColumnName}'} as placeholders
                  </p>
                </div>
              )}
            </div>

            {/* Select Greenhouse Gases */}
            <div>
              <label className="flex items-center gap-[6px] mb-[6px] cursor-pointer">
                <input
                  type="radio"
                  name={`fill-${fieldName}`}
                  checked={fillType === 'ghg'}
                  onChange={() => setFillType('ghg')}
                  className="w-[14px] h-[14px] accent-[#365D60]"
                />
                <span className="font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B]">
                  Select Greenhouse Gases
                </span>
              </label>
              {fillType === 'ghg' && (
                <div className="flex flex-col gap-[4px] ml-[20px]">
                  {['CO2', 'CH4', 'N2O', 'CO2e', 'SF6', 'R-22', 'R-404A', 'R-407C', 'R-410A', 'R-134A'].map(gas => (
                    <label key={gas} className="flex items-center gap-[6px] cursor-pointer">
                      <input
                        type="checkbox"
                        checked={ghgSelections.includes(gas)}
                        onChange={() => toggleGhg(gas)}
                        className="w-[14px] h-[14px] accent-[#29ABB5]"
                      />
                      <span className="font-['Noto_Sans',sans-serif] text-[13px] text-[#575757]">{gas}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>

            {/* Extract from Column */}
            <div>
              <label className="flex items-center gap-[6px] mb-[6px] cursor-pointer">
                <input
                  type="radio"
                  name={`fill-${fieldName}`}
                  checked={fillType === 'extract'}
                  onChange={() => setFillType('extract')}
                  className="w-[14px] h-[14px] accent-[#365D60]"
                />
                <span className="font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B]">
                  Extract from Column
                </span>
              </label>
              {fillType === 'extract' && (
                <div className="flex flex-col gap-[4px]">
                  <select
                    value={extractColumn}
                    onChange={(e) => setExtractColumn(e.target.value)}
                    className="w-full h-[28px] px-[6px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[12px] bg-white focus:outline-none focus:border-[#29ABB5]"
                  >
                    <option value="">Source Column</option>
                    {sourceColumns.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                  <select
                    value={extractType}
                    onChange={(e) => setExtractType(e.target.value)}
                    className="w-full h-[28px] px-[6px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[12px] bg-white focus:outline-none focus:border-[#29ABB5]"
                  >
                    <option value="year">Extract: Year</option>
                    <option value="month">Extract: Month</option>
                    <option value="day">Extract: Day</option>
                    <option value="quarter">Extract: Quarter</option>
                    <option value="dayofweek">Extract: Day of Week</option>
                  </select>
                </div>
              )}
            </div>

            {/* Look up from Table */}
            <div>
              <label className="flex items-center gap-[6px] mb-[6px] cursor-pointer">
                <input
                  type="radio"
                  name={`fill-${fieldName}`}
                  checked={fillType === 'lookup'}
                  onChange={() => setFillType('lookup')}
                  className="w-[14px] h-[14px] accent-[#365D60]"
                />
                <span className="font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B]">
                  Look up from Table
                </span>
              </label>
              {fillType === 'lookup' && (
                <div className="flex flex-col gap-[4px]">
                  <select
                    value={refTable}
                    onChange={(e) => setRefTable(e.target.value)}
                    className="w-full h-[28px] px-[6px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[12px] bg-white focus:outline-none focus:border-[#29ABB5]"
                  >
                    <option value="">Ref Table</option>
                    {referenceSources.map(src => (
                      <option key={src.sourceId} value={src.filePath}>{src.name || src.sourceId}</option>
                    ))}
                  </select>
                  <input
                    type="text"
                    value={refColumnJoined}
                    onChange={(e) => setRefColumnJoined(e.target.value)}
                    placeholder="Column to Add"
                    className="w-full h-[28px] px-[6px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[12px] focus:outline-none focus:border-[#29ABB5]"
                  />
                  <input
                    type="text"
                    value={refSourceKey}
                    onChange={(e) => setRefSourceKey(e.target.value)}
                    placeholder="Source Key (in raw data)"
                    className="w-full h-[28px] px-[6px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[12px] focus:outline-none focus:border-[#29ABB5]"
                  />
                  <input
                    type="text"
                    value={refKey}
                    onChange={(e) => setRefKey(e.target.value)}
                    placeholder="Mapping Key (in ref table)"
                    className="w-full h-[28px] px-[6px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[12px] focus:outline-none focus:border-[#29ABB5]"
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Save / Cancel buttons */}
      <div className="flex justify-end gap-[12px] mt-[20px] pt-[16px] border-t border-[#EEEEEE]">
        <button
          onClick={onClose}
          className="px-[16px] py-[6px] rounded-[6px] border border-[#D9D9D9] font-['Noto_Sans',sans-serif] text-[13px] text-[#575757] hover:bg-[#F5F5F5] transition"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          className="px-[16px] py-[6px] rounded-[6px] bg-[#29ABB5] font-['Noto_Sans',sans-serif] text-[13px] text-white hover:bg-[#238d96] transition"
        >
          Save
        </button>
      </div>
    </div>
  );
}

export default function DataMapping({ onBack, onComplete, onGoToStep, ensureSession }) {
  // Backend data state
  const [backendSources, setBackendSources] = useState([]);
  const [selectedSourceId, setSelectedSourceId] = useState('');
  const [sourcePreview, setSourcePreview] = useState(null);
  const [loadingSources, setLoadingSources] = useState(true);
  const [loadingPreview, setLoadingPreview] = useState(false);

  // Mapping state
  const [columnMappings, setColumnMappings] = useState({});
  const [optionalFields, setOptionalFields] = useState([]);
  const [activeModal, setActiveModal] = useState(null); // field name or '__new_optional__'
  const [errorMessage, setErrorMessage] = useState('');
  const [isCalculating, setIsCalculating] = useState(false);

  // Load sources on mount
  useEffect(() => {
    async function load() {
      try {
        const result = await listSources();
        setBackendSources(result.sources || []);
      } catch (err) {
        console.error('Failed to load sources:', err);
        setErrorMessage('Failed to connect to backend. Make sure the backend is running on port 8000.');
      } finally {
        setLoadingSources(false);
      }
    }
    load();
  }, []);

  // Load preview when source changes
  useEffect(() => {
    if (!selectedSourceId) {
      setSourcePreview(null);
      return;
    }
    async function loadPreview() {
      setLoadingPreview(true);
      try {
        const result = await previewSource(selectedSourceId);
        setSourcePreview(result);
      } catch (err) {
        console.error('Failed to load preview:', err);
        setErrorMessage(`Failed to load preview for source: ${selectedSourceId}`);
      } finally {
        setLoadingPreview(false);
      }
    }
    loadPreview();
  }, [selectedSourceId]);

  const sourceColumns = sourcePreview?.columns?.map(c => c.name) || [];
  const referenceSources = backendSources.filter(s =>
    s.filePath?.includes('REFERENCE') || s.sourceId?.toLowerCase().includes('ref')
  );
  const selectedSource = backendSources.find(s => s.sourceId === selectedSourceId);

  const handleFieldClick = (fieldName) => {
    setActiveModal(activeModal === fieldName ? null : fieldName);
    setErrorMessage('');
  };

  const handleSaveMapping = (fieldName, mapping) => {
    setColumnMappings(prev => ({ ...prev, [fieldName]: mapping }));
    if (activeModal === '__new_optional__' && !optionalFields.includes(fieldName)) {
      setOptionalFields(prev => [...prev, fieldName]);
    }
    setActiveModal(null);
  };

  const handleRemoveOptional = (fieldName) => {
    setOptionalFields(prev => prev.filter(f => f !== fieldName));
    setColumnMappings(prev => {
      const next = { ...prev };
      delete next[fieldName];
      return next;
    });
  };

  const handleRunCalculation = async () => {
    // Validate required fields
    const missingRequired = REQUIRED_FIELDS.filter(f => !columnMappings[f]);
    if (missingRequired.length > 0) {
      setErrorMessage(`Missing required fields: ${missingRequired.join(', ')}`);
      return;
    }

    if (!selectedSourceId) {
      setErrorMessage('Please select a data source first.');
      return;
    }

    setIsCalculating(true);
    setErrorMessage('');

    try {
      const sessionId = await ensureSession();

      // Build config from mappings
      const config = {
        sourceId: selectedSourceId,
        columnMappings: { ...columnMappings },
      };

      // If source has sheets, include the first sheet
      if (selectedSource?.sheets?.length > 0) {
        config.sourceSheet = selectedSource.sheets[0];
      }

      // Save config
      await saveSourceConfig(sessionId, selectedSourceId, config);

      // Run calculation
      const result = await runCalculation(sessionId, [selectedSourceId]);

      // Navigate to dashboard
      onComplete(result, sessionId);
    } catch (err) {
      console.error('Calculation failed:', err);
      setErrorMessage(err.message || 'Calculation failed. Check your mappings and try again.');
    } finally {
      setIsCalculating(false);
    }
  };

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
        {/* Step 1 - Completed */}
        <div
          className="flex items-center gap-[8px] cursor-pointer hover:opacity-80 transition"
          onClick={() => onGoToStep(1)}
        >
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <CheckIcon className="w-[16px] h-[16px] text-white" />
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Upload Your Data
          </p>
        </div>
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #50B461)' }}></div>

        {/* Step 2 - Completed */}
        <div
          className="flex items-center gap-[8px] cursor-pointer hover:opacity-80 transition"
          onClick={() => onGoToStep(2)}
        >
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <CheckIcon className="w-[16px] h-[16px] text-white" />
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Data Cleaning
          </p>
        </div>
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #29ABB5)' }}></div>

        {/* Step 3 - Active */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-[#29ABB5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">3</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Data Processing
          </p>
        </div>
        <div className="flex-1 h-[2px] bg-[#ACC2C5] mx-[12px]"></div>

        {/* Step 4 - Inactive */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-white border-2 border-[#ACC2C5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#7E7E7E]">4</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] whitespace-nowrap">
            Data Visualization
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="absolute left-[64px] right-[64px] top-[132px] bottom-[80px] overflow-auto">
        {/* Title */}
        <h1 className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B] mb-[24px]">
          Data Mapping
        </h1>

        {/* Source Selector */}
        <div className="mb-[24px]">
          <select
            value={selectedSourceId}
            onChange={(e) => {
              setSelectedSourceId(e.target.value);
              setColumnMappings({});
              setOptionalFields([]);
              setErrorMessage('');
            }}
            className="w-[400px] h-[40px] px-[12px] border border-[#D9D9D9] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] bg-white focus:outline-none focus:border-[#29ABB5]"
            disabled={loadingSources}
          >
            <option value="">
              {loadingSources ? 'Loading sources...' : '-- Select a data source --'}
            </option>
            {backendSources.map(src => (
              <option key={src.sourceId} value={src.sourceId}>
                {src.name || src.sourceId}
              </option>
            ))}
          </select>
        </div>

        {/* Required Fields + Optional section */}
        <div className="flex items-start justify-between mb-[24px]">
          {/* Required Fields */}
          <div>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[16px] text-[#13383B] mb-[8px]">
              Required Fields
            </p>
            <div className="flex flex-wrap gap-[12px]">
              {REQUIRED_FIELDS.map(field => {
                const isConfigured = !!columnMappings[field];
                const summary = getMappingSummary(columnMappings[field]);
                return (
                  <div key={field} className="relative">
                    <div className="flex flex-col items-start">
                      <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] mb-[4px]">
                        {field}
                      </p>
                      <button
                        onClick={() => handleFieldClick(field)}
                        className={`flex items-center gap-[8px] h-[36px] px-[12px] rounded-[18px] font-['Noto_Sans',sans-serif] text-[13px] transition ${
                          isConfigured
                            ? 'bg-[#29ABB5] text-white'
                            : 'bg-white text-[#7E7E7E] border border-[#D9D9D9] hover:border-[#29ABB5]'
                        }`}
                      >
                        <span className="max-w-[120px] truncate">
                          {isConfigured ? (summary || field) : field}
                        </span>
                        <ChevronDownIcon className="w-[14px] h-[14px] flex-shrink-0" />
                      </button>
                    </div>
                    {/* Modal */}
                    {activeModal === field && (
                      <ConfigModal
                        fieldName={field}
                        sourceColumns={sourceColumns}
                        mapping={columnMappings[field]}
                        onSave={handleSaveMapping}
                        onClose={() => setActiveModal(null)}
                        isOptional={false}
                        referenceSources={referenceSources}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Optional section */}
          <div>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[16px] text-[#13383B] mb-[8px]">
              Optional
            </p>
            <div className="flex flex-wrap gap-[8px] items-center">
              {optionalFields.map(field => {
                const summary = getMappingSummary(columnMappings[field]);
                return (
                  <div key={field} className="relative">
                    <div className="flex items-center gap-[4px]">
                      <button
                        onClick={() => handleFieldClick(field)}
                        className="flex items-center gap-[6px] h-[36px] px-[12px] rounded-[18px] bg-[#29ABB5] text-white font-['Noto_Sans',sans-serif] text-[13px]"
                      >
                        <span className="max-w-[100px] truncate">{summary || field}</span>
                        <ChevronDownIcon className="w-[14px] h-[14px]" />
                      </button>
                      <button
                        onClick={() => handleRemoveOptional(field)}
                        className="w-[20px] h-[20px] flex items-center justify-center text-[#7E7E7E] hover:text-[#DC3545] transition"
                      >
                        <CloseIcon className="w-[12px] h-[12px]" />
                      </button>
                    </div>
                    {activeModal === field && (
                      <ConfigModal
                        fieldName={field}
                        sourceColumns={sourceColumns}
                        mapping={columnMappings[field]}
                        onSave={handleSaveMapping}
                        onClose={() => setActiveModal(null)}
                        isOptional={false}
                        referenceSources={referenceSources}
                      />
                    )}
                  </div>
                );
              })}
              {/* Add a Field button */}
              <div className="relative">
                <button
                  onClick={() => handleFieldClick('__new_optional__')}
                  className="flex items-center gap-[6px] h-[36px] px-[12px] rounded-[18px] border-2 border-dashed border-[#ACC2C5] text-[#7E7E7E] font-['Noto_Sans',sans-serif] text-[13px] hover:border-[#29ABB5] hover:text-[#29ABB5] transition"
                >
                  Add a Field
                  <PlusIcon className="w-[14px] h-[14px]" />
                </button>
                {activeModal === '__new_optional__' && (
                  <ConfigModal
                    fieldName=""
                    sourceColumns={sourceColumns}
                    mapping={null}
                    onSave={handleSaveMapping}
                    onClose={() => setActiveModal(null)}
                    isOptional={true}
                    referenceSources={referenceSources}
                  />
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Error message */}
        {errorMessage && (
          <div className="mb-[16px] p-[12px] bg-[#DC3545]/10 border border-[#DC3545]/30 rounded-[8px]">
            <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#DC3545]">{errorMessage}</p>
          </div>
        )}

        {/* Raw File Preview */}
        <div className="mb-[24px]">
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[16px] text-[#13383B] mb-[12px]">
            Raw File
          </p>
          {loadingPreview ? (
            <div className="flex items-center justify-center h-[200px] bg-[#F5F5F5] rounded-[10px]">
              <div className="animate-spin w-[32px] h-[32px] border-3 border-[#29ABB5] border-t-transparent rounded-full"></div>
            </div>
          ) : sourcePreview && sourcePreview.preview && sourcePreview.preview.length > 0 ? (
            <div className="border border-[#EEEEEE] rounded-[10px] overflow-hidden">
              <div className="overflow-x-auto" style={{ maxHeight: '360px', overflowY: 'auto' }}>
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-[#F5F5F5]">
                      {sourcePreview.columns.map(col => (
                        <th
                          key={col.name}
                          className="px-[12px] py-[8px] text-left font-['Noto_Sans',sans-serif] font-bold text-[12px] text-[#575757] whitespace-nowrap border-b border-[#EEEEEE]"
                        >
                          {col.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sourcePreview.preview.map((row, rowIdx) => (
                      <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-[#FAFAFA]'}>
                        {sourcePreview.columns.map(col => (
                          <td
                            key={col.name}
                            className="px-[12px] py-[6px] font-['Noto_Sans',sans-serif] text-[12px] text-[#575757] whitespace-nowrap border-b border-[#EEEEEE]"
                          >
                            {row[col.name] != null ? String(row[col.name]) : ''}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {sourcePreview.totalRows && (
                <div className="px-[12px] py-[8px] bg-[#F5F5F5] border-t border-[#EEEEEE]">
                  <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E]">
                    Showing {sourcePreview.preview.length} of {sourcePreview.totalRows} total rows
                  </p>
                </div>
              )}
            </div>
          ) : selectedSourceId ? (
            <div className="flex items-center justify-center h-[200px] bg-[#F5F5F5] rounded-[10px]">
              <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">
                No preview data available
              </p>
            </div>
          ) : (
            <div className="flex items-center justify-center h-[200px] bg-[#F5F5F5] rounded-[10px]">
              <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">
                Select a data source to see preview
              </p>
            </div>
          )}
        </div>

      </div>

      {/* Action Buttons - fixed at bottom */}
      <div className="absolute left-[64px] right-[64px] bottom-[24px] flex items-center justify-between">
        <button
          onClick={onBack}
          className="flex items-center gap-[8px] px-[24px] py-[10px] rounded-[24.5px] border border-[#D9D9D9] font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#575757] hover:bg-[#F5F5F5] transition"
        >
          Back
        </button>
        <button
          onClick={handleRunCalculation}
          disabled={isCalculating}
          className={`flex items-center gap-[8px] px-[32px] py-[10px] rounded-[24.5px] font-['Noto_Sans',sans-serif] font-medium text-[14px] text-white transition ${
            isCalculating
              ? 'bg-[#ACC2C5] cursor-not-allowed'
              : 'bg-[#29ABB5] hover:bg-[#238d96]'
          }`}
        >
          {isCalculating ? (
            <>
              <div className="animate-spin w-[16px] h-[16px] border-2 border-white border-t-transparent rounded-full"></div>
              Calculating...
            </>
          ) : (
            'Run Calculation'
          )}
        </button>
      </div>
    </div>
  );
}
