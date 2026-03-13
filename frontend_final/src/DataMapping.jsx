import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
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

function FolderIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <path d="M3 7V19C3 20.1046 3.89543 21 5 21H19C20.1046 21 21 20.1046 21 19V9C21 7.89543 20.1046 7 19 7H12L10 5H5C3.89543 5 3 5.89543 3 7Z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function DataTableIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <rect x="3" y="3" width="18" height="18" rx="2" strokeWidth="1.5"/>
      <line x1="3" y1="9" x2="21" y2="9" strokeWidth="1.5"/>
      <line x1="3" y1="15" x2="21" y2="15" strokeWidth="1.5"/>
      <line x1="9" y1="9" x2="9" y2="21" strokeWidth="1.5"/>
    </svg>
  );
}

function getDisplayFileName(sourceId) {
  if (!sourceId) return '';
  const withoutPrefix = sourceId.replace(/^consumption_/, '').replace(/^reference_/, '');
  const withoutSuffix = withoutPrefix.replace(/_[0-9]+$/, '');
  return withoutSuffix.replace(/_/g, ' ');
}

const REQUIRED_FIELDS = ['ACCT_ID', 'Year', 'Consumption', 'Unit', 'Subtype', 'Sector', 'GHG', 'EF_ID'];
const VALID_EXTRACT_TYPES = ['year', 'month', 'day', 'quarter', 'dayofweek'];


function normalizeExtractType(value) {
  if (!value) return '';
  const normalized = String(value).trim().toLowerCase();
  return VALID_EXTRACT_TYPES.includes(normalized) ? normalized : '';
}

function inferExtractTypeFromFieldName(fieldName) {
  const normalized = String(fieldName || '').trim().toLowerCase();
  if (!normalized) return 'year';
  if (normalized.includes('month')) return 'month';
  if (normalized.includes('day')) return 'day';
  if (normalized.includes('quarter') || normalized === 'qtr' || normalized === 'q') return 'quarter';
  if (normalized.includes('week')) return 'dayofweek';
  return 'year';
}

function buildDateFromParts(year, month, day) {
  const y = Number(year);
  const m = Number(month);
  const d = Number(day);
  if (!Number.isInteger(y) || !Number.isInteger(m) || !Number.isInteger(d)) return null;
  if (y < 1000 || y > 9999 || m < 1 || m > 12 || d < 1 || d > 31) return null;
  const dt = new Date(y, m - 1, d, 12);
  if (dt.getFullYear() !== y || dt.getMonth() !== m - 1 || dt.getDate() !== d) return null;
  return dt;
}

function parseFlexibleDate(rawValue) {
  if (rawValue == null) return null;
  if (rawValue instanceof Date && !Number.isNaN(rawValue.getTime())) return rawValue;

  const raw = String(rawValue).trim();
  if (!raw) return null;

  // Excel serial number support.
  if (/^\d+(\.\d+)?$/.test(raw)) {
    const serial = Number(raw);
    if (Number.isFinite(serial) && serial > 1 && serial < 2958465) {
      const excelEpochMs = Date.UTC(1899, 11, 30);
      const date = new Date(excelEpochMs + Math.round(serial) * 24 * 60 * 60 * 1000);
      if (!Number.isNaN(date.getTime())) return date;
    }
  }

  const cleaned = raw
    .replace(/(\d)(st|nd|rd|th)\b/gi, '$1')
    .replace(/,/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  const normalized = cleaned.replace(/[.]/g, '/').replace(/-/g, '/');

  // YYYY/MM/DD
  let match = normalized.match(/^(\d{4})\/(\d{1,2})\/(\d{1,2})(?:\s|T|$)/);
  if (match) {
    const dt = buildDateFromParts(match[1], match[2], match[3]);
    if (dt) return dt;
  }

  // MM/DD/YYYY or DD/MM/YYYY
  match = normalized.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})(?:\s|T|$)/);
  if (match) {
    const first = Number(match[1]);
    const second = Number(match[2]);
    const year = Number(match[3]);
    // Assumption: ambiguous dates (both <= 12) are interpreted as MM/DD/YYYY.
    const monthFirst = first <= 12 ? first : second;
    const daySecond = first <= 12 ? second : first;
    const dt = buildDateFromParts(year, monthFirst, daySecond);
    if (dt) return dt;
  }

  // YYYYMMDD
  match = cleaned.match(/^(\d{4})(\d{2})(\d{2})$/);
  if (match) {
    const dt = buildDateFromParts(match[1], match[2], match[3]);
    if (dt) return dt;
  }

  // Let JS parse month-name and datetime variants.
  const parsed = new Date(cleaned);
  if (!Number.isNaN(parsed.getTime())) return parsed;

  return null;
}

function extractDatePart(rawValue, extractType) {
  const parsed = parseFlexibleDate(rawValue);
  if (!parsed) return '';
  const normalizedType = normalizeExtractType(extractType) || 'year';
  if (normalizedType === 'year') return parsed.getFullYear();
  if (normalizedType === 'month') return parsed.getMonth() + 1;
  if (normalizedType === 'day') return parsed.getDate();
  if (normalizedType === 'quarter') return Math.floor(parsed.getMonth() / 3) + 1;
  if (normalizedType === 'dayofweek') return parsed.getDay();
  return '';
}

function parsePatternTokens(pattern) {
  const source = String(pattern || '');
  const regex = /\{([^{}]+)\}/g;
  const tokens = [];
  let lastIdx = 0;
  let match;

  while ((match = regex.exec(source)) !== null) {
    if (match.index > lastIdx) {
      tokens.push({ type: 'text', value: source.slice(lastIdx, match.index) });
    }
    tokens.push({ type: 'column', value: match[1] });
    lastIdx = match.index + match[0].length;
  }

  if (lastIdx < source.length) {
    tokens.push({ type: 'text', value: source.slice(lastIdx) });
  }

  if (!tokens.length) {
    tokens.push({ type: 'text', value: '' });
  }

  return tokens;
}

// Configuration Modal component
function ConfigModal({
  fieldName,
  sourceColumns,
  mapping,
  onSave,
  onClose,
  onClear,
  isOptional,
  referenceSources,
  availableSources,
  allSessionSources,
  consumptionSources,
  defaultSourceId,
  loadSourcePreview,
}) {
  const [mode, setMode] = useState(mapping?.sourceColumn ? 'column' : 'fill');
  const [selectedSourceId, setSelectedSourceId] = useState(
    mapping?.sourceId || defaultSourceId || availableSources?.[0]?.sourceId || ''
  );
  const [selectedColumn, setSelectedColumn] = useState(mapping?.sourceColumn || '');
  const [fillType, setFillType] = useState(() => {
    if (mapping?.pattern) return 'pattern';
    if (mapping?.staticValue) return 'fixed';
    if (mapping?.ghgType) return 'ghg';
    if (mapping?.derivedFrom) return 'extract';
    if (mapping?.lookupSourceId || mapping?.lookupKeyColumn || mapping?.lookupValueColumn) return 'lookup';
    return 'fixed';
  });
  const [fixedValue, setFixedValue] = useState(mapping?.staticValue || '');
  const [patternValue, setPatternValue] = useState(mapping?.pattern || '');
  const [ghgSelections, setGhgSelections] = useState(mapping?.ghgType || []);
  const [extractSourceId, setExtractSourceId] = useState(
    mapping?.extractSourceId || defaultSourceId || consumptionSources?.[0]?.sourceId || ''
  );
  const [extractColumn, setExtractColumn] = useState(mapping?.derivedFrom || '');
  const [extractType, setExtractType] = useState(() => {
    const normalized = normalizeExtractType(mapping?.extractType);
    if (normalized) return normalized;
    return inferExtractTypeFromFieldName(fieldName);
  });
  const [lookupQuerySourceId, setLookupQuerySourceId] = useState(
    mapping?.lookupQuerySourceId || defaultSourceId || consumptionSources?.[0]?.sourceId || ''
  );
  const [lookupQueryColumn, setLookupQueryColumn] = useState(mapping?.lookupQueryColumn || '');
  const [lookupSourceId, setLookupSourceId] = useState(
    mapping?.lookupSourceId || allSessionSources?.[0]?.sourceId || ''
  );
  const [lookupKeyColumn, setLookupKeyColumn] = useState(mapping?.lookupKeyColumn || '');
  const [lookupValueColumn, setLookupValueColumn] = useState(mapping?.lookupValueColumn || '');
  const [optionalName, setOptionalName] = useState(fieldName || '');
  const activeSource = availableSources?.find((s) => s.sourceId === selectedSourceId);
  const activeSourceColumns = activeSource?.columns || sourceColumns || [];
  const [showPatternBuilder, setShowPatternBuilder] = useState(false);
  const patternSourceOptions = useMemo(
    () => (consumptionSources?.length > 0 ? consumptionSources : (availableSources || [])),
    [consumptionSources, availableSources]
  );
  const [patternSourceId, setPatternSourceId] = useState(
    defaultSourceId || patternSourceOptions?.[0]?.sourceId || ''
  );
  const [patternPreviewLoading, setPatternPreviewLoading] = useState(false);
  const [patternPreviewError, setPatternPreviewError] = useState('');
  const [patternPreviewData, setPatternPreviewData] = useState({ columns: [], preview: [] });
  const patternEditorRef = useRef(null);
  const patternBuilderInitializedRef = useRef(false);
  const [showColumnPicker, setShowColumnPicker] = useState(false);
  const [pickerTitle, setPickerTitle] = useState('Select Column');
  const [pickerTarget, setPickerTarget] = useState('source');
  const [pickerSourceId, setPickerSourceId] = useState('');
  const [pickerLoading, setPickerLoading] = useState(false);
  const [pickerError, setPickerError] = useState('');
  const [pickerData, setPickerData] = useState({ columns: [], preview: [] });
  const [pickerHoveredColumn, setPickerHoveredColumn] = useState(null);
  const [pickerColumnPositions, setPickerColumnPositions] = useState([]);
  const pickerTableRef = useRef(null);

  const modalRef = useRef(null);
  const [modalPosition, setModalPosition] = useState({ openBelow: true, alignLeft: true });

  useEffect(() => {
    function handleClickOutside(e) {
      if (e.target?.closest?.('[data-mapping-toggle="true"]')) return;
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

  useEffect(() => {
    if (selectedSourceId) return;
    if (mapping?.sourceId) return;
    if (defaultSourceId) {
      setSelectedSourceId(defaultSourceId);
      return;
    }
    if (availableSources?.length > 0) {
      setSelectedSourceId(availableSources[0].sourceId);
    }
  }, [selectedSourceId, mapping?.sourceId, defaultSourceId, availableSources]);

  useEffect(() => {
    if (mapping?.extractType) return;
    if (fillType !== 'extract') return;
    const inferred = inferExtractTypeFromFieldName(isOptional ? optionalName : fieldName);
    setExtractType(inferred);
  }, [mapping?.extractType, fillType, isOptional, optionalName, fieldName]);

  useEffect(() => {
    if (patternSourceId) return;
    if (defaultSourceId) {
      setPatternSourceId(defaultSourceId);
      return;
    }
    if (patternSourceOptions.length > 0) {
      setPatternSourceId(patternSourceOptions[0].sourceId);
    }
  }, [patternSourceId, defaultSourceId, patternSourceOptions]);

  const handleSave = () => {
    const name = isOptional ? optionalName.trim() : fieldName;
    if (isOptional && !name) return;

    let result;
    if (mode === 'column') {
      if (!selectedSourceId || !selectedColumn) return;
      result = { sourceId: selectedSourceId, sourceColumn: selectedColumn };
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
          if (!extractSourceId || !extractColumn) return;
          {
            const normalizedExtractType = normalizeExtractType(extractType)
              || inferExtractTypeFromFieldName(name);
            result = {
              extractSourceId,
              derivedFrom: extractColumn,
              extractType: normalizedExtractType,
            };
          }
          break;
        case 'lookup':
          if (!lookupQuerySourceId || !lookupQueryColumn || !lookupSourceId || !lookupKeyColumn || !lookupValueColumn) return;
          result = {
            lookupQuerySourceId,
            lookupQueryColumn,
            lookupSourceId,
            lookupKeyColumn,
            lookupValueColumn,
          };
          break;
        default:
          return;
      }
    }
    onSave(name, result);
  };

  const openColumnPicker = async (sourceId, target, title) => {
    if (!sourceId) return;
    setPickerLoading(true);
    setPickerError('');
    setPickerTarget(target);
    setPickerSourceId(sourceId);
    setPickerTitle(title);
    try {
      const result = await loadSourcePreview(sourceId);
      const normalized = result || {};
      setPickerData({
        columns: normalized.columns || [],
        preview: normalized.preview || [],
      });
      setShowColumnPicker(true);
    } catch (err) {
      console.error('Failed to load source preview for picker:', err);
      setPickerError('Failed to load preview for selected file.');
    } finally {
      setPickerLoading(false);
    }
  };

  const createPatternTokenNode = useCallback((columnName) => {
    const token = document.createElement('span');
    token.setAttribute('data-pattern-column', columnName);
    token.setAttribute('contenteditable', 'false');
    token.setAttribute('draggable', 'false');
    token.textContent = columnName;
    token.style.display = 'inline-flex';
    token.style.alignItems = 'center';
    token.style.padding = '0 8px';
    token.style.margin = '0 2px';
    token.style.height = '22px';
    token.style.borderRadius = '11px';
    token.style.border = '1px solid #29ABB5';
    token.style.backgroundColor = 'rgba(41,171,181,0.12)';
    token.style.color = '#13383B';
    token.style.fontFamily = 'Noto Sans, sans-serif';
    token.style.fontSize = '12px';
    token.style.whiteSpace = 'nowrap';
    token.style.verticalAlign = 'middle';
    return token;
  }, []);

  const serializePatternFromEditor = useCallback(() => {
    const editor = patternEditorRef.current;
    if (!editor) return patternValue || '';

    let next = '';
    editor.childNodes.forEach((node) => {
      if (node.nodeType === 3) {
        next += node.textContent || '';
        return;
      }
      if (node.nodeType === 1) {
        const col = node.getAttribute?.('data-pattern-column');
        if (col) {
          next += `{${col}}`;
        } else {
          next += node.textContent || '';
        }
      }
    });
    return next.replace(/\u00a0/g, ' ');
  }, [patternValue]);

  const renderPatternToEditor = useCallback((pattern) => {
    const editor = patternEditorRef.current;
    if (!editor) return;
    while (editor.firstChild) {
      editor.removeChild(editor.firstChild);
    }
    const tokens = parsePatternTokens(pattern);
    tokens.forEach((token) => {
      if (token.type === 'column') {
        editor.appendChild(createPatternTokenNode(token.value));
      } else {
        editor.appendChild(document.createTextNode(token.value));
      }
    });
    if (!editor.firstChild) {
      editor.appendChild(document.createTextNode(''));
    }
  }, [createPatternTokenNode]);

  const loadPatternPreview = useCallback(async (sourceId) => {
    if (!sourceId) {
      setPatternPreviewData({ columns: [], preview: [] });
      return;
    }
    setPatternPreviewLoading(true);
    setPatternPreviewError('');
    try {
      const result = await loadSourcePreview(sourceId);
      const normalized = result || {};
      setPatternPreviewData({
        columns: normalized.columns || [],
        preview: normalized.preview || [],
      });
    } catch (err) {
      console.error('Failed to load source preview for pattern builder:', err);
      setPatternPreviewError('Failed to load preview for selected file.');
    } finally {
      setPatternPreviewLoading(false);
    }
  }, [loadSourcePreview]);

  useEffect(() => {
    if (!showPatternBuilder) return;
    if (!patternSourceId) return;
    loadPatternPreview(patternSourceId);
  }, [showPatternBuilder, patternSourceId, loadPatternPreview]);

  useEffect(() => {
    if (!showPatternBuilder) {
      patternBuilderInitializedRef.current = false;
      return;
    }
    if (patternBuilderInitializedRef.current) return;
    patternBuilderInitializedRef.current = true;
    const timer = setTimeout(() => {
      renderPatternToEditor(patternValue);
      const editor = patternEditorRef.current;
      if (!editor) return;
      editor.focus();
      const range = document.createRange();
      range.selectNodeContents(editor);
      range.collapse(false);
      const sel = window.getSelection();
      if (sel) {
        sel.removeAllRanges();
        sel.addRange(range);
      }
    }, 0);
    return () => clearTimeout(timer);
  }, [showPatternBuilder, patternValue, renderPatternToEditor]);

  const insertPatternToken = useCallback((columnName, point = null) => {
    if (!columnName) return;
    const editor = patternEditorRef.current;
    if (!editor) return;
    editor.focus();

    let range = null;
    if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) {
      if (document.caretRangeFromPoint) {
        range = document.caretRangeFromPoint(point.x, point.y);
      } else if (document.caretPositionFromPoint) {
        const caretPos = document.caretPositionFromPoint(point.x, point.y);
        if (caretPos) {
          range = document.createRange();
          range.setStart(caretPos.offsetNode, caretPos.offset);
          range.collapse(true);
        }
      }
    }

    if (!range) {
      const sel = window.getSelection();
      if (sel && sel.rangeCount > 0) {
        const candidate = sel.getRangeAt(0);
        if (editor.contains(candidate.commonAncestorContainer)) {
          range = candidate;
        }
      }
    }

    if (!range || !editor.contains(range.commonAncestorContainer)) {
      range = document.createRange();
      range.selectNodeContents(editor);
      range.collapse(false);
    }

    const findTokenAncestor = (node) => {
      let current = node;
      while (current && current !== editor) {
        if (current.nodeType === 1 && current.getAttribute?.('data-pattern-column')) {
          return current;
        }
        current = current.parentNode;
      }
      return null;
    };

    const tokenAncestor = findTokenAncestor(range.startContainer);
    if (tokenAncestor) {
      const tokenRange = document.createRange();
      if (point && Number.isFinite(point.x)) {
        const rect = tokenAncestor.getBoundingClientRect();
        const shouldPlaceAfter = point.x > rect.left + (rect.width / 2);
        if (shouldPlaceAfter) {
          tokenRange.setStartAfter(tokenAncestor);
        } else {
          tokenRange.setStartBefore(tokenAncestor);
        }
      } else {
        tokenRange.setStartAfter(tokenAncestor);
      }
      tokenRange.collapse(true);
      range = tokenRange;
    } else if (!range.collapsed) {
      range.deleteContents();
    }

    range.collapse(true);
    const tokenNode = createPatternTokenNode(columnName);
    range.insertNode(tokenNode);

    const spacer = document.createTextNode('');
    tokenNode.after(spacer);
    range.setStart(spacer, 0);
    range.collapse(true);
    const sel = window.getSelection();
    if (sel) {
      sel.removeAllRanges();
      sel.addRange(range);
    }

    setPatternValue(serializePatternFromEditor());
  }, [createPatternTokenNode, serializePatternFromEditor]);

  const handlePatternEditorInput = useCallback(() => {
    setPatternValue(serializePatternFromEditor());
  }, [serializePatternFromEditor]);

  const handlePickerColumnSelect = (col) => {
    switch (pickerTarget) {
      case 'source':
        setSelectedColumn(col);
        break;
      case 'extract':
        setExtractColumn(col);
        break;
      case 'lookup_query':
        setLookupQueryColumn(col);
        break;
      case 'lookup_key':
        setLookupKeyColumn(col);
        break;
      case 'lookup_value':
        setLookupValueColumn(col);
        break;
      default:
        break;
    }
    setShowColumnPicker(false);
  };

  const currentPickerSelectedColumn = (() => {
    switch (pickerTarget) {
      case 'source':
        return selectedColumn;
      case 'extract':
        return extractColumn;
      case 'lookup_query':
        return lookupQueryColumn;
      case 'lookup_key':
        return lookupKeyColumn;
      case 'lookup_value':
        return lookupValueColumn;
      default:
        return '';
    }
  })();

  useEffect(() => {
    if (!showColumnPicker) return;
    if (!pickerTableRef.current) return;

    const calculatePositions = () => {
      const table = pickerTableRef.current;
      if (!table) return;
      const headerCells = table.querySelectorAll('thead th');
      const positions = [];
      headerCells.forEach((cell) => {
        positions.push({
          left: cell.offsetLeft,
          width: cell.offsetWidth,
          height: table.offsetHeight,
        });
      });
      setPickerColumnPositions(positions);
    };

    const timer = setTimeout(calculatePositions, 10);
    window.addEventListener('resize', calculatePositions);
    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', calculatePositions);
    };
  }, [showColumnPicker, pickerData, pickerHoveredColumn]);

  const toggleGhg = (gas) => {
    setGhgSelections(prev =>
      prev.includes(gas) ? prev.filter(g => g !== gas) : [...prev, gas]
    );
  };

  return (
    <div
      ref={modalRef}
      className="absolute z-50 bg-white rounded-[10px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.15)] p-[24px] w-[480px] max-h-[80vh] overflow-y-auto"
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
            value={selectedSourceId}
            onChange={(e) => {
              setSelectedSourceId(e.target.value);
              setSelectedColumn('');
            }}
            className="w-full h-[36px] px-[12px] border border-[#D9D9D9] rounded-[6px] font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] bg-white focus:outline-none focus:border-[#29ABB5] mb-[8px]"
          >
            <option value="">-- Select file --</option>
            {(availableSources || []).map(src => (
              <option key={src.sourceId} value={src.sourceId}>
                {getDisplayFileName(src.sourceId)}
              </option>
            ))}
          </select>
          <button
            onClick={() => openColumnPicker(selectedSourceId, 'source', 'Select Column from Raw File')}
            disabled={!selectedSourceId || pickerLoading}
            className={`w-full h-[36px] px-[12px] border rounded-[6px] font-['Noto_Sans',sans-serif] text-[14px] text-left transition ${
              !selectedSourceId || pickerLoading
                ? 'border-[#D9D9D9] text-[#A0A0A0] bg-[#F5F5F5] cursor-not-allowed'
                : 'border-[#D9D9D9] text-[#13383B] bg-white hover:border-[#29ABB5]'
            }`}
          >
            {pickerLoading ? 'Loading columns...' : (selectedColumn || 'Select Column')}
          </button>
          {pickerError && (
            <p className="mt-[6px] font-['Noto_Sans',sans-serif] text-[12px] text-[#DC3545]">
              {pickerError}
            </p>
          )}
          {!!selectedColumn && (
            <p className="mt-[6px] font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E]">
              Selected column: <span className="text-[#13383B]">{selectedColumn}</span>
            </p>
          )}
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
                  onChange={() => {
                    setFillType('pattern');
                    setShowPatternBuilder(true);
                  }}
                  className="w-[14px] h-[14px] accent-[#365D60]"
                />
                <span className="font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B]">
                  Name a Pattern
                </span>
              </label>
              {fillType === 'pattern' && (
                <div>
                  <div className="w-full min-h-[32px] px-[8px] py-[6px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B] bg-white">
                    {patternValue || 'No pattern yet'}
                  </div>
                  <button
                    onClick={() => setShowPatternBuilder(true)}
                    className="mt-[6px] w-full h-[32px] px-[8px] rounded-[4px] border border-[#29ABB5] bg-[#29ABB5]/10 font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B] hover:bg-[#29ABB5]/20 transition"
                  >
                    Open Pattern Builder
                  </button>
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
                    value={extractType}
                    onChange={(e) => setExtractType(e.target.value)}
                    className="w-full h-[32px] px-[8px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] bg-white focus:outline-none focus:border-[#29ABB5]"
                  >
                    <option value="year">Extract Year</option>
                    <option value="month">Extract Month</option>
                    <option value="day">Extract Day</option>
                    <option value="quarter">Extract Quarter</option>
                    <option value="dayofweek">Extract Day of Week</option>
                  </select>
                  <select
                    value={extractSourceId}
                    onChange={(e) => {
                      setExtractSourceId(e.target.value);
                      setExtractColumn('');
                    }}
                    className="w-full h-[32px] px-[8px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] bg-white focus:outline-none focus:border-[#29ABB5]"
                  >
                    <option value="">-- Select file --</option>
                    {(consumptionSources || []).map(src => (
                      <option key={src.sourceId} value={src.sourceId}>{getDisplayFileName(src.sourceId)}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => openColumnPicker(extractSourceId, 'extract', 'Extract from Column')}
                    disabled={!extractSourceId || pickerLoading}
                    className={`w-full h-[32px] px-[8px] border rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] text-left transition ${
                      !extractSourceId || pickerLoading
                        ? 'border-[#D9D9D9] text-[#A0A0A0] bg-[#F5F5F5] cursor-not-allowed'
                        : 'border-[#D9D9D9] text-[#13383B] bg-white hover:border-[#29ABB5]'
                    }`}
                  >
                    {extractColumn || 'Select Column'}
                  </button>
                  <p className="font-['Noto_Sans',sans-serif] text-[11px] text-[#7E7E7E]">
                    Supports mixed date formats. Ambiguous numeric dates default to MM/DD/YYYY.
                  </p>
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
                <div className="flex flex-col gap-[6px]">
                  <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E]">Query (Consumption file + column)</p>
                  <select
                    value={lookupQuerySourceId}
                    onChange={(e) => {
                      setLookupQuerySourceId(e.target.value);
                      setLookupQueryColumn('');
                    }}
                    className="w-full h-[32px] px-[8px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] bg-white focus:outline-none focus:border-[#29ABB5]"
                  >
                    <option value="">-- Select consumption file --</option>
                    {(consumptionSources || []).map(src => (
                      <option key={src.sourceId} value={src.sourceId}>{getDisplayFileName(src.sourceId)}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => openColumnPicker(lookupQuerySourceId, 'lookup_query', 'Lookup Query Column')}
                    disabled={!lookupQuerySourceId || pickerLoading}
                    className={`w-full h-[32px] px-[8px] border rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] text-left transition ${
                      !lookupQuerySourceId || pickerLoading
                        ? 'border-[#D9D9D9] text-[#A0A0A0] bg-[#F5F5F5] cursor-not-allowed'
                        : 'border-[#D9D9D9] text-[#13383B] bg-white hover:border-[#29ABB5]'
                    }`}
                  >
                    {lookupQueryColumn || 'Select Query Column'}
                  </button>

                  <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] mt-[4px]">Lookup table (any file + key/value columns)</p>
                  <select
                    value={lookupSourceId}
                    onChange={(e) => {
                      setLookupSourceId(e.target.value);
                      setLookupKeyColumn('');
                      setLookupValueColumn('');
                    }}
                    className="w-full h-[32px] px-[8px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] bg-white focus:outline-none focus:border-[#29ABB5]"
                  >
                    <option value="">-- Select lookup file --</option>
                    {(allSessionSources || []).map(src => (
                      <option key={src.sourceId} value={src.sourceId}>{getDisplayFileName(src.sourceId)}</option>
                    ))}
                  </select>
                  <button
                    onClick={() => openColumnPicker(lookupSourceId, 'lookup_key', 'Lookup Key Column')}
                    disabled={!lookupSourceId || pickerLoading}
                    className={`w-full h-[32px] px-[8px] border rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] text-left transition ${
                      !lookupSourceId || pickerLoading
                        ? 'border-[#D9D9D9] text-[#A0A0A0] bg-[#F5F5F5] cursor-not-allowed'
                        : 'border-[#D9D9D9] text-[#13383B] bg-white hover:border-[#29ABB5]'
                    }`}
                  >
                    {lookupKeyColumn || 'Select Key Column'}
                  </button>
                  <button
                    onClick={() => openColumnPicker(lookupSourceId, 'lookup_value', 'Lookup Value Column')}
                    disabled={!lookupSourceId || pickerLoading}
                    className={`w-full h-[32px] px-[8px] border rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] text-left transition ${
                      !lookupSourceId || pickerLoading
                        ? 'border-[#D9D9D9] text-[#A0A0A0] bg-[#F5F5F5] cursor-not-allowed'
                        : 'border-[#D9D9D9] text-[#13383B] bg-white hover:border-[#29ABB5]'
                    }`}
                  >
                    {lookupValueColumn || 'Select Value Column'}
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Save / Cancel / Clear buttons */}
      <div className="flex justify-between items-center mt-[20px] pt-[16px] border-t border-[#EEEEEE]">
        <div>
          {mapping && onClear && (
            <button
              onClick={() => {
                onClear(fieldName);
                onClose();
              }}
              className="px-[16px] py-[6px] rounded-[6px] border border-[#DC3545] bg-white font-['Noto_Sans',sans-serif] text-[13px] text-[#DC3545] hover:bg-[#DC3545] hover:text-white transition"
            >
              Clear
            </button>
          )}
        </div>
        <div className="flex gap-[12px]">
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

      {showColumnPicker && (
        <div
          className="fixed inset-0 bg-black/40 z-[70] flex items-center justify-center"
          onClick={() => setShowColumnPicker(false)}
        >
          <div
            className="bg-white rounded-[10px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)] w-[900px] max-w-[90vw] h-[560px] p-[20px] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-[12px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[18px] text-[#13383B]">
                {pickerTitle}: {getDisplayFileName(pickerSourceId)}
              </p>
              <button
                onClick={() => setShowColumnPicker(false)}
                className="w-[28px] h-[28px] flex items-center justify-center rounded-[6px] hover:bg-[#F5F5F5]"
              >
                <CloseIcon className="w-[14px] h-[14px] text-[#575757]" />
              </button>
            </div>
            <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E] mb-[12px]">
              Click a column header to select it.
            </p>

            <div className="flex-1 bg-[#F5F5F5] border border-[#E0E0E0] overflow-auto">
              <div className="relative min-w-max">
                <table
                  ref={pickerTableRef}
                  className="border-collapse bg-white min-w-full"
                  style={{ minWidth: 'max-content' }}
                >
                  <thead className="sticky top-0 z-10">
                    <tr>
                      {(pickerData.columns.length > 0 ? pickerData.columns : activeSourceColumns).map((col, idx) => (
                        <th
                          key={col}
                          onMouseEnter={() => setPickerHoveredColumn(idx)}
                          onMouseLeave={() => setPickerHoveredColumn(null)}
                          onClick={() => handlePickerColumnSelect(col)}
                          className={`px-[16px] py-[12px] min-w-[120px] text-left cursor-pointer border border-[#EEEEEE] ${
                            currentPickerSelectedColumn === col
                              ? 'bg-[#29ABB5] text-white'
                              : 'bg-[#F5F5F5] text-[#13383B]'
                          }`}
                        >
                          <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px]">
                            {col}
                          </p>
                          <p
                            className="font-['Noto_Sans',sans-serif] font-normal text-[12px] mt-[4px]"
                            style={{
                              visibility: currentPickerSelectedColumn === col ? 'visible' : 'hidden',
                              opacity: currentPickerSelectedColumn === col ? 0.8 : 0,
                            }}
                          >
                            ✓ Selected
                          </p>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(pickerData.preview || []).map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {(pickerData.columns.length > 0 ? pickerData.columns : activeSourceColumns).map((col, idx) => (
                          <td
                            key={col}
                            onMouseEnter={() => setPickerHoveredColumn(idx)}
                            onMouseLeave={() => setPickerHoveredColumn(null)}
                            onClick={() => handlePickerColumnSelect(col)}
                            className={`px-[16px] py-[10px] font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] cursor-pointer border border-[#EEEEEE] ${
                              currentPickerSelectedColumn === col ? 'bg-[#29ABB5]/10' : 'bg-white'
                            }`}
                          >
                            {row[col] != null ? String(row[col]) : ''}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>

                {pickerHoveredColumn !== null &&
                  pickerColumnPositions[pickerHoveredColumn] &&
                  (pickerData.columns.length > 0 ? pickerData.columns : activeSourceColumns)[pickerHoveredColumn] !== currentPickerSelectedColumn && (
                    <div
                      className="absolute z-20"
                      style={{
                        left: pickerColumnPositions[pickerHoveredColumn].left,
                        top: 0,
                        width: pickerColumnPositions[pickerHoveredColumn].width,
                        height: pickerColumnPositions[pickerHoveredColumn].height,
                        border: '2px solid #29ABB5',
                        backgroundColor: 'rgba(41, 171, 181, 0.05)',
                        boxSizing: 'border-box',
                        pointerEvents: 'none',
                      }}
                    >
                      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
                        <div className="w-[40px] h-[40px] bg-[#29ABB5] rounded-full flex items-center justify-center shadow-[0px_4px_12px_0px_rgba(0,0,0,0.25)]">
                          <svg className="w-[20px] h-[20px] text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                            <line x1="12" y1="5" x2="12" y2="19" />
                            <line x1="5" y1="12" x2="19" y2="12" />
                          </svg>
                        </div>
                      </div>
                    </div>
                  )}
              </div>
            </div>
          </div>
        </div>
      )}

      {showPatternBuilder && (
        <div
          className="fixed inset-0 bg-black/40 z-[75] flex items-center justify-center"
          onClick={() => setShowPatternBuilder(false)}
        >
          <div
            className="bg-white rounded-[10px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)] w-[980px] max-w-[94vw] h-[620px] p-[20px] flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-[12px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[20px] text-[#13383B]">
                Pattern Builder
              </p>
              <button
                onClick={() => setShowPatternBuilder(false)}
                className="w-[28px] h-[28px] flex items-center justify-center rounded-[6px] hover:bg-[#F5F5F5]"
              >
                <CloseIcon className="w-[14px] h-[14px] text-[#575757]" />
              </button>
            </div>

            <div className="flex items-center gap-[12px] mb-[10px]">
              <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#575757] whitespace-nowrap">
                Raw data source:
              </p>
              <select
                value={patternSourceId}
                onChange={(e) => setPatternSourceId(e.target.value)}
                className="w-[320px] h-[32px] px-[8px] border border-[#D9D9D9] rounded-[4px] font-['Noto_Sans',sans-serif] text-[13px] bg-white focus:outline-none focus:border-[#29ABB5]"
              >
                <option value="">-- Select file --</option>
                {patternSourceOptions.map((src) => (
                  <option key={src.sourceId} value={src.sourceId}>
                    {getDisplayFileName(src.sourceId)}
                  </option>
                ))}
              </select>
            </div>

            <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] mb-[8px]">
              Drag a column chip into the editor. You can type any text around it.
            </p>

            <div className="flex flex-wrap gap-[8px] mb-[10px] min-h-[34px]">
              {(patternPreviewData.columns || []).map((col) => (
                <button
                  key={col}
                  type="button"
                  draggable
                  onDragStart={(e) => {
                    e.dataTransfer.setData('application/x-pattern-column', col);
                    e.dataTransfer.setData('text/plain', col);
                    e.dataTransfer.effectAllowed = 'copy';
                  }}
                  onClick={() => insertPatternToken(col)}
                  className="px-[10px] h-[28px] rounded-[14px] border border-[#29ABB5] bg-[#29ABB5]/10 font-['Noto_Sans',sans-serif] text-[12px] text-[#13383B] hover:bg-[#29ABB5]/20 cursor-grab active:cursor-grabbing"
                >
                  {col}
                </button>
              ))}
            </div>

            <div
              ref={patternEditorRef}
              contentEditable
              suppressContentEditableWarning
              onInput={handlePatternEditorInput}
              onDragOver={(e) => {
                e.preventDefault();
                e.dataTransfer.dropEffect = 'copy';
              }}
              onDrop={(e) => {
                e.preventDefault();
                const droppedColumn = e.dataTransfer.getData('application/x-pattern-column')
                  || e.dataTransfer.getData('text/plain');
                insertPatternToken(droppedColumn, { x: e.clientX, y: e.clientY });
              }}
              className="mb-[12px] min-h-[72px] max-h-[140px] overflow-y-auto border border-[#D9D9D9] rounded-[6px] px-[10px] py-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] focus:outline-none focus:border-[#29ABB5]"
            />

            {patternPreviewError && (
              <p className="mb-[8px] font-['Noto_Sans',sans-serif] text-[12px] text-[#DC3545]">
                {patternPreviewError}
              </p>
            )}

            <div className="flex-1 border border-[#EEEEEE] overflow-hidden">
              <div className="h-full overflow-auto">
                {patternPreviewLoading ? (
                  <div className="h-full flex items-center justify-center bg-[#F5F5F5]">
                    <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">Loading preview...</p>
                  </div>
                ) : (patternPreviewData.preview || []).length === 0 ? (
                  <div className="h-full flex items-center justify-center bg-[#F5F5F5]">
                    <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">No preview data available.</p>
                  </div>
                ) : (
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-[#F5F5F5] sticky top-0">
                        {(patternPreviewData.columns || []).map((col) => (
                          <th
                            key={col}
                            className="px-[12px] py-[8px] text-left font-['Noto_Sans',sans-serif] font-bold text-[12px] text-[#575757] whitespace-nowrap border-b border-[#EEEEEE]"
                          >
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(patternPreviewData.preview || []).map((row, rowIdx) => (
                        <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-[#FAFAFA]'}>
                          {(patternPreviewData.columns || []).map((col) => (
                            <td
                              key={col}
                              className="px-[12px] py-[6px] font-['Noto_Sans',sans-serif] text-[12px] text-[#575757] whitespace-nowrap border-b border-[#EEEEEE]"
                            >
                              {row[col] != null ? String(row[col]) : ''}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            </div>

            <div className="flex justify-end gap-[10px] mt-[12px]">
              <button
                onClick={() => setShowPatternBuilder(false)}
                className="px-[16px] py-[6px] rounded-[6px] border border-[#D9D9D9] font-['Noto_Sans',sans-serif] text-[13px] text-[#575757] hover:bg-[#F5F5F5] transition"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function DataMapping({
  onBack,
  onComplete,
  onGoToStep,
  ensureSession,
  onLogoClick,
}) {
  // Backend data state
  const [backendSources, setBackendSources] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [selectedSourceId, setSelectedSourceId] = useState('');
  const [sourcePreview, setSourcePreview] = useState(null);
  const [sourceDataCache, setSourceDataCache] = useState({});
  const [loadingSources, setLoadingSources] = useState(true);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [previewMode, setPreviewMode] = useState('cleaned');
  const [leftPanelWidth, setLeftPanelWidth] = useState(30);
  const [isResizingLeftPanel, setIsResizingLeftPanel] = useState(false);
  const mainContentRef = useRef(null);

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
        const sid = await ensureSession();
        setSessionId(sid);
        const result = await listSources();
        setBackendSources(result.sources || []);
      } catch (err) {
        console.error('Failed to load sources:', err);
        setErrorMessage('Failed to load source files from backend.');
      } finally {
        setLoadingSources(false);
      }
    }
    load();
  }, [ensureSession]);

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

  const ensureSourceLoaded = useCallback(async (sid, sourceId) => {
    if (!sid || !sourceId) return null;
    if (sourceDataCache[sourceId]) return sourceDataCache[sourceId];
    const result = await previewSource(sourceId);
    const normalized = {
      ...result,
      columns: (result.columns || []).map((c) => c.name),
      preview: result.preview || [],
    };
    setSourceDataCache(prev => ({ ...prev, [sourceId]: normalized }));
    return normalized;
  }, [sourceDataCache]);

  const sourceColumns = sourcePreview?.columns?.map(c => c.name) || [];
  const referenceSources = backendSources.filter(s =>
    s.filePath?.includes('REFERENCE') || s.sourceId?.toLowerCase().includes('ref')
  );
  const consumptionSources = backendSources.filter(s =>
    !(s.filePath?.includes('REFERENCE') || s.sourceId?.toLowerCase().includes('ref'))
  );
  const selectedSource = backendSources.find(s => s.sourceId === selectedSourceId);
  const allSessionSources = backendSources;

  const previewData = useMemo(() => {
    const rawRows = sourcePreview?.preview || [];
    if (!rawRows.length) return { rows: [], columns: [], duplicateMultiplier: 1 };

    let rows = rawRows.map((r) => ({ ...r }));
    const mappings = Object.entries(columnMappings || {});
    let duplicateMultiplier = 1;

    const resolveTemplate = (template, row) => {
      return String(template).replace(/\{([^}]+)\}/g, (_, key) => {
        if (Object.prototype.hasOwnProperty.call(row, key)) {
          const v = row[key];
          return v == null ? '' : String(v);
        }
        return `{${key}}`;
      });
    };

    mappings.forEach(([target, mapping]) => {
      if (!mapping) return;
      if (mapping.ghgType && Array.isArray(mapping.ghgType) && mapping.ghgType.length > 0) {
        duplicateMultiplier *= mapping.ghgType.length;
        rows = rows.flatMap((row) =>
          mapping.ghgType.map((gas) => ({ ...row, [target]: gas }))
        );
        return;
      }

      let lookupMap = null;
      let lookupQueryRows = null;
      if (mapping.lookupSourceId && mapping.lookupKeyColumn && mapping.lookupValueColumn) {
        const lookupRows = sourceDataCache[mapping.lookupSourceId]?.preview || [];
        lookupMap = new Map();
        lookupRows.forEach((r) => {
          const key = r?.[mapping.lookupKeyColumn];
          if (key != null && !lookupMap.has(String(key))) {
            lookupMap.set(String(key), r?.[mapping.lookupValueColumn] ?? '');
          }
        });
        const querySourceId = mapping.lookupQuerySourceId || selectedSourceId;
        lookupQueryRows = sourceDataCache[querySourceId]?.preview || [];
      }

      rows = rows.map((row, idx) => {
        const next = { ...row };
        if (mapping.sourceColumn) {
          const srcId = mapping.sourceId || selectedSourceId;
          const srcRows = sourceDataCache[srcId]?.preview || [];
          const srcRow = srcRows[idx] || {};
          next[target] = Object.prototype.hasOwnProperty.call(srcRow, mapping.sourceColumn)
            ? srcRow[mapping.sourceColumn]
            : '';
        } else if (mapping.derivedFrom) {
          const extractSourceId = mapping.extractSourceId || selectedSourceId;
          const extractRows = sourceDataCache[extractSourceId]?.preview || [];
          const extractRow = extractRows[idx] || {};
          const rawValue = Object.prototype.hasOwnProperty.call(extractRow, mapping.derivedFrom)
            ? extractRow[mapping.derivedFrom]
            : row[mapping.derivedFrom];
          const extractType = normalizeExtractType(mapping.extractType)
            || inferExtractTypeFromFieldName(target);
          next[target] = extractDatePart(rawValue, extractType);
        } else if (lookupMap && mapping.lookupQueryColumn) {
          const queryRow = lookupQueryRows?.[idx] || {};
          const queryVal = queryRow?.[mapping.lookupQueryColumn];
          next[target] = queryVal == null ? '' : (lookupMap.get(String(queryVal)) ?? '');
        } else if (mapping.staticValue !== undefined) {
          next[target] = mapping.staticValue;
        } else if (mapping.pattern) {
          next[target] = resolveTemplate(mapping.pattern, next);
        }
        return next;
      });
    });

    const mappedColumns = [...new Set(mappings.map(([key]) => key))];
    return { rows, columns: mappedColumns, duplicateMultiplier };
  }, [sourcePreview, columnMappings, sourceDataCache, selectedSourceId]);

  useEffect(() => {
    async function loadNeededSources() {
      if (!sessionId) return;
      const sourceIds = Object.values(columnMappings || {}).flatMap((m) => {
        if (!m) return [];
        return [
          m.sourceId,
          m.extractSourceId,
          m.lookupQuerySourceId,
          m.lookupSourceId,
        ].filter(Boolean);
      });
      const uniqueIds = [...new Set(sourceIds)];
      for (const sid of uniqueIds) {
        if (!sourceDataCache[sid]) {
          try {
            await ensureSourceLoaded(sessionId, sid, 300);
          } catch (e) {
            console.error(`Failed to preload source ${sid}`, e);
          }
        }
      }
    }
    loadNeededSources();
  }, [columnMappings, sessionId, sourceDataCache, ensureSourceLoaded]);

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

  const handleClearMapping = (fieldName) => {
    setColumnMappings(prev => {
      const next = { ...prev };
      delete next[fieldName];
      return next;
    });
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

  useEffect(() => {
    if (!isResizingLeftPanel) return;

    const handleMouseMove = (e) => {
      if (!mainContentRef.current) return;
      const rect = mainContentRef.current.getBoundingClientRect();
      const minPx = 420;
      const maxPx = rect.width * 0.65;
      const rawPx = e.clientX - rect.left;
      const clampedPx = Math.min(Math.max(rawPx, minPx), maxPx);
      const pct = (clampedPx / rect.width) * 100;
      setLeftPanelWidth(pct);
    };

    const handleMouseUp = () => setIsResizingLeftPanel(false);

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizingLeftPanel]);

  const canRunCalculation = REQUIRED_FIELDS.every((f) => columnMappings[f]);

  return (
    <div className="bg-white fixed inset-0 overflow-hidden">
      {/* Toolbar */}
      <div className="absolute left-0 top-0 w-full h-[62px] bg-[#365D60] flex items-center justify-between px-[64px]">
        <button
          type="button"
          onClick={onLogoClick}
          className="flex items-center gap-[12px] bg-transparent border-0 p-0 cursor-pointer"
        >
          <img src="/logo.png" alt="Logo" className="h-[40px] w-auto" />
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[24px] text-white">
            EcoMetrics
          </p>
        </button>
        <div className="bg-white rounded-[10px] w-[331px] h-[40px] flex items-center px-[12px]">
          <SearchIcon className="w-[20px] h-[20px] text-[#7E7E7E]" />
        </div>
      </div>

      {/* Progress Steps - 6 steps */}
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
            Upload Data
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
            Select Data
          </p>
        </div>
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #50B461)' }}></div>

        {/* Step 3 - Completed */}
        <div
          className="flex items-center gap-[8px] cursor-pointer hover:opacity-80 transition"
          onClick={() => onGoToStep(3)}
        >
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <CheckIcon className="w-[16px] h-[16px] text-white" />
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Data Cleaning
          </p>
        </div>
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #29ABB5)' }}></div>

        {/* Step 4 - Active */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-[#29ABB5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">4</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Data Mapping
          </p>
        </div>
        <div className="flex-1 h-[2px] bg-[#ACC2C5] mx-[12px]"></div>

        {/* Step 5 - Inactive */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-white border-2 border-[#ACC2C5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#7E7E7E]">5</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] whitespace-nowrap">
            Data Processing
          </p>
        </div>
        <div className="flex-1 h-[2px] bg-[#ACC2C5] mx-[12px]"></div>

        {/* Step 6 - Inactive */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-white border-2 border-[#ACC2C5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#7E7E7E]">6</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] whitespace-nowrap">
            Visualization
          </p>
        </div>
      </div>

      {/* Main Content Area - Two Columns */}
      <div ref={mainContentRef} className="absolute left-[64px] right-[64px] top-[132px] bottom-[32px] flex items-stretch">
        {/* Left Column - File List with Status (same style as Step 3) */}
        <div
          className="flex-shrink-0 h-full bg-[#F5F5F5] rounded-[10px] p-[24px] relative"
          style={{ width: `${leftPanelWidth}%`, minWidth: '420px' }}
        >
          <div className="flex items-center justify-between mb-[24px]">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B]">
              Data source to map
            </p>
          </div>

          <div className="overflow-y-auto pr-[8px]" style={{ height: 'calc(100% - 80px)' }}>
            {loadingSources ? (
              <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">Loading session files...</p>
            ) : consumptionSources.length === 0 ? (
              <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">
                No cleaning-stage files found in current session.
              </p>
            ) : (
              <div className="mb-[24px]">
                <div className="bg-white h-[49px] flex items-center pl-[13px] pr-[17px] justify-between relative border-l-4 border-[#29ABB5]">
                  <div className="flex items-center gap-[12px]">
                    <FolderIcon className="w-[24px] h-[24px] text-[#365D60]" />
                    <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
                      Consumption Data
                    </p>
                  </div>
                  <div className="bg-[#29ABB5]/10 px-[5px] py-0 rounded-[4px]">
                    <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#365D60] whitespace-nowrap">
                      #Consumption
                    </p>
                  </div>
                </div>

                {consumptionSources.map((src) => (
                  <div
                    key={src.sourceId}
                    onClick={() => {
                      setSelectedSourceId(src.sourceId);
                      setColumnMappings({});
                      setOptionalFields([]);
                      setErrorMessage('');
                    }}
                    className={`bg-white h-[49px] flex items-center pl-[17px] pr-[17px] justify-between border-t border-[#EEEEEE] relative transition cursor-pointer ${
                      selectedSourceId === src.sourceId ? 'bg-[#29ABB5]/20' : 'hover:bg-[#29ABB5]/10'
                    }`}
                  >
                    <div className="flex items-center gap-[12px] pl-[32px]">
                      <DataTableIcon className="w-[24px] h-[24px] text-[#7E7E7E]" />
                      <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B]">
                        {getDisplayFileName(src.sourceId)}
                      </p>
                    </div>
                    <div className="px-[8px] py-[2px] rounded-[4px] bg-[#365D60]/10 text-[#365D60]">
                      <p className="font-['Noto_Sans',sans-serif] font-normal text-[12px]">
                        {src.rowCount || 0} rows
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div
          onMouseDown={(e) => {
            e.preventDefault();
            setIsResizingLeftPanel(true);
          }}
          className="w-[12px] mx-[6px] cursor-col-resize flex items-center justify-center"
        >
          <div className="w-[2px] h-[60px] bg-[#D9D9D9] rounded-full" />
        </div>

        {/* Right Column - Mapping Panel */}
        <div className="flex-1 min-w-0 h-full pt-[24px] pb-[24px] pl-[12px] relative flex flex-col">
          <div className="flex-1 overflow-visible pr-[4px] pb-[60px]">
            <div className="flex items-start justify-between mb-[24px]">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B]">
              Data Mapping
            </p>
            <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] max-w-[360px] text-right leading-relaxed">
              Map cleaned columns from step 3 into required and optional emission fields.
            </p>
            </div>

            {!selectedSourceId ? (
              <div className="h-[300px] flex items-center justify-center bg-[#F5F5F5] rounded-[10px]">
                <p className="font-['Noto_Sans',sans-serif] text-[16px] text-[#7E7E7E]">
                  Select a cleaning output file from the left sidebar.
                </p>
              </div>
            ) : (
              <div className="flex flex-col">
              {/* Error message */}
              {errorMessage && (
                <div className="mb-[16px] p-[12px] bg-[#DC3545]/10 border border-[#DC3545]/30 rounded-[8px]">
                  <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#DC3545]">{errorMessage}</p>
                </div>
              )}

              {/* Cleaning Output Preview */}
              <div className="mb-[24px] order-2">
                <div className="flex items-center justify-between mb-[12px]">
                  <p className="font-['Noto_Sans',sans-serif] font-bold text-[16px] text-[#13383B]">
                    {previewMode === 'cleaned' ? 'Cleaned File Preview' : 'Mapped File Preview'}
                  </p>
                  <div className="flex rounded-[8px] overflow-hidden border border-[#D9D9D9]">
                    <button
                      onClick={() => setPreviewMode('cleaned')}
                      className={`px-[14px] h-[30px] font-['Noto_Sans',sans-serif] text-[12px] transition ${
                        previewMode === 'cleaned' ? 'bg-[#29ABB5] text-white' : 'bg-[#E3E3E3] text-[#575757]'
                      }`}
                    >
                      cleaned
                    </button>
                    <button
                      onClick={() => setPreviewMode('mapped')}
                      className={`px-[14px] h-[30px] font-['Noto_Sans',sans-serif] text-[12px] transition ${
                        previewMode === 'mapped' ? 'bg-[#29ABB5] text-white' : 'bg-[#E3E3E3] text-[#575757]'
                      }`}
                    >
                      mapped
                    </button>
                  </div>
                </div>
                {loadingPreview ? (
                  <div className="flex items-center justify-center h-[200px] bg-[#F5F5F5] rounded-[10px]">
                    <div className="animate-spin w-[32px] h-[32px] border-3 border-[#29ABB5] border-t-transparent rounded-full"></div>
                  </div>
                ) : sourcePreview && sourcePreview.preview && sourcePreview.preview.length > 0 && (previewMode === 'cleaned' || previewData.columns.length > 0) ? (
                  <div className="border border-[#EEEEEE] overflow-hidden">
                    <div className="overflow-x-auto" style={{ maxHeight: '360px', overflowY: 'auto' }}>
                      <table className="w-full border-collapse">
                        <thead>
                          <tr className="bg-[#F5F5F5]">
                            {(previewMode === 'cleaned' ? (sourcePreview.columns || []).map(c => c.name || c) : (previewData.columns || [])).map(col => (
                              <th
                                key={col}
                                className="px-[12px] py-[8px] text-left font-['Noto_Sans',sans-serif] font-bold text-[12px] text-[#575757] whitespace-nowrap border-b border-[#EEEEEE]"
                              >
                                {col}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {(previewMode === 'cleaned' ? sourcePreview.preview : previewData.rows).map((row, rowIdx) => (
                            <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-[#FAFAFA]'}>
                              {(previewMode === 'cleaned' ? (sourcePreview.columns || []).map(c => c.name || c) : (previewData.columns || [])).map(col => (
                                <td
                                  key={col}
                                  className="px-[12px] py-[6px] font-['Noto_Sans',sans-serif] text-[12px] text-[#575757] whitespace-nowrap border-b border-[#EEEEEE]"
                                >
                                  {row[col] != null ? String(row[col]) : ''}
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
                          {(() => {
                            if (previewMode === 'cleaned') {
                              return `Showing ${sourcePreview.preview.length} of ${sourcePreview.totalRows} total rows`;
                            }
                            const rawTotal = sourcePreview.totalRows || sourcePreview.preview.length;
                            const previewTotal = rawTotal * (previewData.duplicateMultiplier || 1);
                            return `Showing ${previewData.rows.length} of ${previewTotal} total rows`;
                          })()}
                        </p>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-[200px] bg-[#F5F5F5] rounded-[10px]">
                    <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">
                      {previewMode === 'mapped'
                        ? 'No mapped preview yet. Configure fields to generate preview columns.'
                        : 'No preview data available'}
                    </p>
                  </div>
                )}
              </div>

              {/* Required Fields + Optional section */}
              <div className="flex items-start w-full mb-[24px] order-1 gap-[24px]">
                {/* Required Fields */}
                <div className="flex-1 min-w-0">
                  <p className="font-['Noto_Sans',sans-serif] font-bold text-[16px] text-[#13383B] mb-[8px]">
                    Required Fields
                  </p>
                  <div className="flex flex-wrap gap-[12px]">
                    {REQUIRED_FIELDS.map(field => {
                      const isConfigured = !!columnMappings[field];
                      return (
                        <div key={field} className="relative">
                          <button
                            onClick={() => handleFieldClick(field)}
                            data-mapping-toggle="true"
                            className={`inline-flex items-center gap-[6px] px-[12px] py-[6px] rounded-[16px] border font-['Noto_Sans',sans-serif] text-[13px] transition ${
                              isConfigured
                                ? 'bg-[#29ABB5]/10 border-[#29ABB5] text-[#13383B]'
                                : 'bg-white text-[#7E7E7E] border-[#D9D9D9] hover:border-[#29ABB5]'
                            }`}
                          >
                            <span className="max-w-[120px] truncate">{field}</span>
                            <ChevronDownIcon className="w-[14px] h-[14px] flex-shrink-0" />
                          </button>
                          {activeModal === field && (
                            <ConfigModal
                              fieldName={field}
                              sourceColumns={sourceColumns}
                              mapping={columnMappings[field]}
                              onSave={handleSaveMapping}
                              onClose={() => setActiveModal(null)}
                              onClear={handleClearMapping}
                              isOptional={false}
                              referenceSources={referenceSources}
                              availableSources={['GHG', 'EF_ID'].includes(field) ? allSessionSources : consumptionSources}
                              allSessionSources={allSessionSources}
                              consumptionSources={consumptionSources}
                              defaultSourceId={selectedSourceId}
                              loadSourcePreview={(sourceId) => ensureSourceLoaded(sessionId, sourceId)}
                            />
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Optional section */}
                <div className="ml-auto flex flex-col items-start">
                  <p className="font-['Noto_Sans',sans-serif] font-bold text-[16px] text-[#13383B] mb-[8px]">
                    Optional
                  </p>
                  <div className="flex flex-wrap gap-[8px] items-center justify-start min-h-[34px]">
                    <div className="relative">
                      <button
                        onClick={() => handleFieldClick('__new_optional__')}
                        data-mapping-toggle="true"
                        className="inline-flex items-center gap-[6px] px-[12px] py-[6px] rounded-[16px] border border-[#D9D9D9] bg-white text-[#7E7E7E] font-['Noto_Sans',sans-serif] text-[13px] hover:border-[#29ABB5] transition"
                      >
                        Add a Field
                      </button>
                      {activeModal === '__new_optional__' && (
                        <ConfigModal
                          fieldName=""
                          sourceColumns={sourceColumns}
                          mapping={null}
                          onSave={handleSaveMapping}
                          onClose={() => setActiveModal(null)}
                          onClear={handleClearMapping}
                          isOptional={true}
                          referenceSources={referenceSources}
                          availableSources={consumptionSources}
                          allSessionSources={allSessionSources}
                          consumptionSources={consumptionSources}
                          defaultSourceId={selectedSourceId}
                          loadSourcePreview={(sourceId) => ensureSourceLoaded(sessionId, sourceId)}
                        />
                      )}
                    </div>
                    {optionalFields.map(field => {
                      const isConfigured = !!columnMappings[field];
                      return (
                        <div key={field} className="relative">
                          <div className="flex items-center gap-[4px]">
                            <button
                              onClick={() => handleFieldClick(field)}
                              data-mapping-toggle="true"
                              className={`inline-flex items-center gap-[6px] px-[12px] py-[6px] rounded-[16px] border font-['Noto_Sans',sans-serif] text-[13px] transition ${
                                isConfigured
                                  ? 'bg-[#29ABB5]/10 border-[#29ABB5] text-[#13383B]'
                                  : 'bg-white text-[#7E7E7E] border-[#D9D9D9] hover:border-[#29ABB5]'
                              }`}
                            >
                              <span className="max-w-[100px] truncate">{field}</span>
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
                              onClear={handleClearMapping}
                              isOptional={false}
                              referenceSources={referenceSources}
                              availableSources={consumptionSources}
                              allSessionSources={allSessionSources}
                              consumptionSources={consumptionSources}
                              defaultSourceId={selectedSourceId}
                              loadSourcePreview={(sourceId) => ensureSourceLoaded(sessionId, sourceId)}
                            />
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
            )}
          </div>

          {/* Bottom Action Button - same location as Step 3 Cleaning Completed */}
          <div className="absolute bottom-0 left-[12px] right-[4px]">
            <button
              onClick={!isCalculating && canRunCalculation ? handleRunCalculation : undefined}
              disabled={isCalculating || !canRunCalculation}
              className={`w-full rounded-[24.5px] h-[49px] flex items-center justify-center transition ${
                isCalculating || !canRunCalculation
                  ? 'bg-[#e3e3e3] cursor-not-allowed'
                  : 'bg-[#29ABB5] cursor-pointer hover:bg-[#50B461]'
              }`}
            >
              <p className={`font-['Noto_Sans',sans-serif] font-bold text-[18px] ${
                isCalculating || !canRunCalculation ? 'text-[#7E7E7E]' : 'text-white'
              }`}>
                {isCalculating ? 'Calculating...' : 'Run Calculation'}
              </p>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
