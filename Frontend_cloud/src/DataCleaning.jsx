import React, { useState, useRef, useEffect } from 'react';

// Icon Components
function IconEllipsis({ className }) {
  return (
    <div className={className} data-name="Icon / Ellipsis">
      <div className="absolute inset-[76.92%_0_0_0]">
        <div className="w-full h-full bg-[#575757] rounded-full"></div>
      </div>
      <div className="absolute inset-[38.46%_0]">
        <div className="w-full h-full bg-[#575757] rounded-full"></div>
      </div>
      <div className="absolute inset-[0_0_76.92%_0]">
        <div className="w-full h-full bg-[#575757] rounded-full"></div>
      </div>
    </div>
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

function SearchIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="11" cy="11" r="8" strokeWidth="2"/>
      <path d="m21 21-4.35-4.35" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

function ZoomInIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="11" cy="11" r="8" strokeWidth="2"/>
      <path d="m21 21-4.35-4.35" strokeWidth="2" strokeLinecap="round"/>
      <line x1="11" y1="8" x2="11" y2="14" strokeWidth="2" strokeLinecap="round"/>
      <line x1="8" y1="11" x2="14" y2="11" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

function ZoomOutIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="11" cy="11" r="8" strokeWidth="2"/>
      <path d="m21 21-4.35-4.35" strokeWidth="2" strokeLinecap="round"/>
      <line x1="8" y1="11" x2="14" y2="11" strokeWidth="2" strokeLinecap="round"/>
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

function CloseIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <line x1="18" y1="6" x2="6" y2="18" strokeWidth="2" strokeLinecap="round"/>
      <line x1="6" y1="6" x2="18" y2="18" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

export default function DataCleaning({
  folders,
  setFolders,
  fileStatuses,
  setFileStatuses,
  fileColumnLabels,
  setFileColumnLabels,
  filePendingCorrections,
  setFilePendingCorrections,
  onBack,
  onComplete,
  onGoToStep
}) {
  const [selectedFileId, setSelectedFileId] = useState(null);

  // State to store errors for each file: { fileId: { rowIndex: { colIndex: errorMessage } } }
  const [fileErrors, setFileErrors] = useState({});

  // State to store error counts for each file
  const [fileErrorCounts, setFileErrorCounts] = useState({});

  // Error correction modal states
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [selectedError, setSelectedError] = useState(null); // { rowIdx, colIdx, errorMessage, currentValue, isPending }
  const [correctedValue, setCorrectedValue] = useState('');

  // Ref to store error cell DOM elements for scrolling
  const errorCellRefs = React.useRef({});

  // State to track focused/highlighted cell
  const [focusedCell, setFocusedCell] = useState(null); // { rowIdx, colIdx }

  // State to track hovered cell for showing original value tooltip
  const [hoveredCell, setHoveredCell] = useState(null); // { rowIdx, colIdx, originalValue, x, y }

  // Confirmation modal for marking as correct
  const [showMarkCorrectConfirmation, setShowMarkCorrectConfirmation] = useState(false);

  // Simulate backend error detection when entering Data Cleaning page
  React.useEffect(() => {
    const detectErrors = () => {
      const errors = {};
      const errorCounts = {};

      console.log('=== Starting Error Detection ===');
      console.log('Folders:', folders);
      console.log('fileColumnLabels:', fileColumnLabels);

      folders.forEach(folder => {
        folder.files.forEach(file => {
          console.log(`\nProcessing file: ${file.name} (ID: ${file.id})`);
          console.log('  Full file object:', file);
          console.log('  file.content:', file.content);

          // Use file.content instead of file.preview
          const fileData = file.content;
          console.log('  fileData:', fileData);

          if (!fileData || !fileData.rows || fileData.rows.length === 0 || !fileData.headers) {
            console.log('  -> No data or headers, skipping');
            console.log('  -> Checks:', {
              hasFileData: !!fileData,
              hasRows: !!fileData?.rows,
              rowsLength: fileData?.rows?.length,
              hasHeaders: !!fileData?.headers
            });
            errors[file.id] = {};
            errorCounts[file.id] = 0;
            return;
          }

          console.log('  Headers:', fileData.headers);
          console.log('  Row count:', fileData.rows.length);
          console.log('  First 3 rows:', fileData.rows.slice(0, 3));

          // Get selected columns from fileColumnLabels
          const selectedCols = fileColumnLabels[file.id] || {};
          const selectedColIndices = Object.keys(selectedCols).map(k => parseInt(k)).sort((a, b) => a - b);

          console.log('  Selected columns:', selectedColIndices);
          console.log('  Selected column labels:', selectedCols);

          if (selectedColIndices.length === 0) {
            console.log('  -> No selected columns, skipping');
            errors[file.id] = {};
            errorCounts[file.id] = 0;
            return;
          }

          const fileErrorMap = {};
          let totalErrors = 0;

          // Helper function to check if a value is empty
          const isEmpty = (value) => {
            return value === null || value === undefined ||
                   (typeof value === 'string' && value.trim() === '');
          };

          // Helper function to detect data type
          const getDataType = (value) => {
            if (isEmpty(value)) return 'empty';
            const trimmed = String(value).trim();

            // Check if pure number (including decimals)
            if (/^\d+(\.\d+)?$/.test(trimmed)) return 'number';

            // Check if pure English letters (with optional spaces)
            if (/^[a-zA-Z\s]+$/.test(trimmed)) return 'text';

            // Mixed or other
            return 'mixed';
          };

          // Get pending corrections for this file
          const filePendingMap = filePendingCorrections[file.id] || {};

          // Check each selected column
          selectedColIndices.forEach((colIdx) => {
            const columnValues = fileData.rows.map(row => row[colIdx]);

            // Check each cell in this column
            columnValues.forEach((cellValue, rowIdx) => {
              // Skip if this cell is marked as correct (but not if it's just corrected and awaiting verification)
              const pendingInfo = filePendingMap[rowIdx]?.[colIdx];
              if (pendingInfo && pendingInfo.type === 'marked-correct') {
                return;
              }
              // If it's type 'corrected', continue checking so it stays in errors

              let errorMessage = null;

              // Rule 1: Check for empty cells
              if (isEmpty(cellValue)) {
                errorMessage = 'Missing value';
              } else {
                // Rule 2: Check for inconsistent data types
                const cellType = getDataType(cellValue);

                // Get types of all other non-empty cells in this column
                const otherTypes = columnValues
                  .filter((val, idx) => idx !== rowIdx && !isEmpty(val))
                  .map(val => getDataType(val));

                if (otherTypes.length > 0) {
                  // Find the most common type in the column
                  const typeCounts = {};
                  otherTypes.forEach(type => {
                    typeCounts[type] = (typeCounts[type] || 0) + 1;
                  });

                  const dominantType = Object.keys(typeCounts).reduce((a, b) =>
                    typeCounts[a] > typeCounts[b] ? a : b
                  );

                  // If current cell type doesn't match dominant type, mark as error
                  if (cellType !== dominantType && cellType !== 'mixed' && dominantType !== 'mixed') {
                    if (dominantType === 'number' && cellType === 'text') {
                      errorMessage = 'Expected number, found text';
                    } else if (dominantType === 'text' && cellType === 'number') {
                      errorMessage = 'Expected text, found number';
                    } else {
                      errorMessage = 'Inconsistent data type';
                    }
                  }
                }
              }

              // If error found, add to error map
              if (errorMessage) {
                if (!fileErrorMap[rowIdx]) {
                  fileErrorMap[rowIdx] = {};
                }
                fileErrorMap[rowIdx][colIdx] = errorMessage;
                totalErrors++;
              }
            });
          });

          // Preserve errors for 'corrected' pending items (reuse filePendingMap from above)
          // Only for columns that are currently selected
          Object.keys(filePendingMap).forEach(rowIdx => {
            const row = filePendingMap[rowIdx];
            Object.keys(row).forEach(colIdx => {
              const colIndex = parseInt(colIdx);
              const pendingInfo = row[colIdx];
              if (pendingInfo && pendingInfo.type === 'corrected' && selectedColIndices.includes(colIndex)) {
                // Preserve system errors (originalError) or failed manual edit errors (dataCheckError)
                const errorMsg = pendingInfo.originalError || (pendingInfo.wasManualEdit && pendingInfo.dataCheckError) || null;
                if (errorMsg) {
                  if (!fileErrorMap[rowIdx]) {
                    fileErrorMap[rowIdx] = {};
                  }
                  fileErrorMap[rowIdx][colIdx] = errorMsg;
                  totalErrors++;
                }
              }
            });
          });

          errors[file.id] = fileErrorMap;
          errorCounts[file.id] = totalErrors;
        });
      });

      setFileErrors(errors);
      setFileErrorCounts(errorCounts);
    };

    // Detect errors when component mounts or folders change
    if (folders.length > 0) {
      detectErrors();
    }
  }, [folders, fileColumnLabels, filePendingCorrections]);

  // Auto-mark files with 0 errors as 'Completed' if they don't have a status yet
  useEffect(() => {
    const allFiles = folders.flatMap(folder => folder.files);
    const updates = {};
    allFiles.forEach(file => {
      const errorCount = fileErrorCounts[file.id] || 0;
      if (errorCount === 0 && !fileStatuses[file.id]) {
        updates[file.id] = 'Completed';
      }
    });
    if (Object.keys(updates).length > 0) {
      setFileStatuses(prev => ({ ...prev, ...updates }));
    }
  }, [folders, fileErrorCounts, fileStatuses, setFileStatuses]);

  // Update error counts based on actual error and pending tags (red + yellow, excluding green)
  useEffect(() => {
    const newErrorCounts = {};

    folders.forEach(folder => {
      folder.files.forEach(file => {
        const fileId = file.id;
        const currentFileErrors = fileErrors[fileId] || {};
        const currentFilePending = filePendingCorrections[fileId] || {};
        const currentSelectedCols = fileColumnLabels[fileId] || {};
        const currentSelectedColIndices = Object.keys(currentSelectedCols).map(k => parseInt(k));

        let errorCount = 0;
        const processedCells = new Set();

        // Count errors from fileErrors
        Object.keys(currentFileErrors).forEach(rowIdx => {
          Object.keys(currentFileErrors[rowIdx]).forEach(colIdx => {
            const cellKey = `${rowIdx}-${colIdx}`;
            const colIndex = parseInt(colIdx);

            // Only count if column is selected
            if (currentSelectedColIndices.includes(colIndex)) {
              processedCells.add(cellKey);
              const pendingInfo = currentFilePending[rowIdx]?.[colIdx];

              // Count as error if: no pending info, OR pending type is 'corrected' (yellow)
              // Do NOT count if pending type is 'marked-correct' (green) or if it's a manual edit
              if ((!pendingInfo || pendingInfo.type === 'corrected') && !(pendingInfo && pendingInfo.wasManualEdit)) {
                errorCount++;
              }
            }
          });
        });

        newErrorCounts[fileId] = errorCount;
      });
    });

    setFileErrorCounts(newErrorCounts);
  }, [folders, fileErrors, filePendingCorrections, fileColumnLabels]);

  const [zoom, setZoom] = useState(100);
  const previewContainerRef = useRef(null);

  // Menu states
  const [folderMenuOpen, setFolderMenuOpen] = useState(null);
  const [fileMenuOpen, setFileMenuOpen] = useState(null);

  // Modal states
  const [showMoveAllModal, setShowMoveAllModal] = useState(null);
  const [showDeleteAllModal, setShowDeleteAllModal] = useState(null);
  const [showMoveFileModal, setShowMoveFileModal] = useState(null);
  const [showRenameModal, setShowRenameModal] = useState(null);
  const [showDeleteFileModal, setShowDeleteFileModal] = useState(null);
  const [renameValue, setRenameValue] = useState('');

  // Data Title modal states
  const [showDataTitleModal, setShowDataTitleModal] = useState(false);
  const [pendingColumnIndex, setPendingColumnIndex] = useState(null);
  const [suggestedTitles, setSuggestedTitles] = useState([]);
  const [selectedSuggestedTitle, setSelectedSuggestedTitle] = useState(null);
  const [manualTitle, setManualTitle] = useState('');
  const [isRenameMode, setIsRenameMode] = useState(false);
  const [currentTitleBeforeRename, setCurrentTitleBeforeRename] = useState('');

  // Column action menu state (for selected columns)
  const [showColumnActionMenu, setShowColumnActionMenu] = useState(null);
  const [columnActionMenuPosition, setColumnActionMenuPosition] = useState({ x: 0, y: 0 });

  // Drag and drop states
  const [draggedFile, setDraggedFile] = useState(null);
  const [dragOverFolder, setDragOverFolder] = useState(null);

  // Get actual file data from the selected file
  const getSelectedFileData = () => {
    if (!selectedFileId) return null;
    for (const folder of folders) {
      const file = folder.files.find(f => f.id === selectedFileId);
      if (file && file.content) {
        return file.content;
      }
    }
    // Fallback mock data if file has no content
    return {
      headers: ['Column 1', 'Column 2', 'Column 3'],
      rows: [['No data available', '', '']]
    };
  };

  const selectedFileData = getSelectedFileData();

  // Selected columns state for current file
  const [selectedColumns, setSelectedColumns] = useState({});
  // Hovered column for preview selection effect
  const [hoveredColumn, setHoveredColumn] = useState(null);
  // Refs for column overlay positioning
  const tableRef = useRef(null);
  const [columnPositions, setColumnPositions] = useState([]);

  // File operations
  const handleMoveAllFiles = (fromFolderId, toFolderTag) => {
    setFolders(prev => prev.map(folder => {
      if (folder.id === fromFolderId) {
        return { ...folder, files: [] };
      }
      if (folder.tag === toFolderTag) {
        const fromFolder = prev.find(f => f.id === fromFolderId);
        return { ...folder, files: [...folder.files, ...fromFolder.files] };
      }
      return folder;
    }));
    setShowMoveAllModal(null);
    setFolderMenuOpen(null);
  };

  const handleDeleteAllFiles = (folderId) => {
    const folder = folders.find(f => f.id === folderId);
    folder.files.forEach(file => {
      if (selectedFileId === file.id) {
        setSelectedFileId(null);
        setSelectedColumns({});
      }
    });
    setFolders(prev => prev.map(f => {
      if (f.id === folderId) {
        return { ...f, files: [] };
      }
      return f;
    }));
    setShowDeleteAllModal(null);
    setFolderMenuOpen(null);
  };

  const handleMoveFile = (fileId, fromFolderId, toFolderTag) => {
    setFolders(prev => prev.map(folder => {
      if (folder.id === fromFolderId) {
        return { ...folder, files: folder.files.filter(f => f.id !== fileId) };
      }
      if (folder.tag === toFolderTag) {
        const fromFolder = prev.find(f => f.id === fromFolderId);
        const fileToMove = fromFolder.files.find(f => f.id === fileId);
        return { ...folder, files: [...folder.files, fileToMove] };
      }
      return folder;
    }));
    setShowMoveFileModal(null);
    setFileMenuOpen(null);
  };

  const handleRenameFile = (fileId, folderId) => {
    if (!renameValue.trim()) {
      alert('Please enter a valid file name');
      return;
    }
    setFolders(prev => prev.map(folder => {
      if (folder.id === folderId) {
        return {
          ...folder,
          files: folder.files.map(f =>
            f.id === fileId ? { ...f, name: renameValue } : f
          )
        };
      }
      return folder;
    }));
    setShowRenameModal(null);
    setFileMenuOpen(null);
    setRenameValue('');
  };

  const handleDeleteFile = (fileId, folderId) => {
    if (selectedFileId === fileId) {
      setSelectedFileId(null);
      setSelectedColumns({});
    }
    setFolders(prev => prev.map(folder => {
      if (folder.id === folderId) {
        return { ...folder, files: folder.files.filter(f => f.id !== fileId) };
      }
      return folder;
    }));
    setShowDeleteFileModal(null);
    setFileMenuOpen(null);
  };

  // Drag and drop handlers
  const handleDragStart = (e, file, folderId) => {
    setDraggedFile({ file, folderId });
    e.dataTransfer.effectAllowed = 'move';
  };

  const handleDragEnd = () => {
    setDraggedFile(null);
    setDragOverFolder(null);
  };

  const handleDragOver = (e, folderId) => {
    e.preventDefault();
    if (draggedFile && draggedFile.folderId !== folderId) {
      setDragOverFolder(folderId);
    }
  };

  const handleDragLeave = () => {
    setDragOverFolder(null);
  };

  const handleDrop = (e, targetFolderId) => {
    e.preventDefault();
    if (draggedFile && draggedFile.folderId !== targetFolderId) {
      const targetFolder = folders.find(f => f.id === targetFolderId);
      handleMoveFile(draggedFile.file.id, draggedFile.folderId, targetFolder.tag);
    }
    setDraggedFile(null);
    setDragOverFolder(null);
  };

  // Handle file selection
  const handleFileSelect = (fileId) => {
    // Save current column labels if there was a previous selection
    if (selectedFileId && Object.keys(selectedColumns).length > 0) {
      setFileColumnLabels(prev => ({
        ...prev,
        [selectedFileId]: selectedColumns
      }));
    }

    // Update previous file status back to Pending if it wasn't completed
    if (selectedFileId && fileStatuses[selectedFileId] === 'Processing') {
      setFileStatuses(prev => ({
        ...prev,
        [selectedFileId]: 'Pending'
      }));
    }

    // Set new file as Processing (unless already Completed)
    if (fileStatuses[fileId] !== 'Completed') {
      setFileStatuses(prev => ({
        ...prev,
        [fileId]: 'Processing'
      }));
    }

    setSelectedFileId(fileId);
    // Restore column labels if previously labeled
    setSelectedColumns(fileColumnLabels[fileId] || {});
  };

  // Generate suggested titles based on the original column header
  const generateSuggestedTitles = (originalTitle) => {
    // Common title suggestions based on data patterns
    const suggestions = [
      'Date',
      'Amount',
      'Category',
      'Department',
      'Product',
      'Quantity',
      'Price',
      'Total',
      'Name',
      'ID',
      'Description',
      'Status',
      'Location',
      'Type',
      'Value',
      'Count',
      'Revenue',
      'Cost',
      'Unit',
      'Period'
    ];

    // Filter out the original title and get random suggestions
    const filteredSuggestions = suggestions.filter(s =>
      s.toLowerCase() !== originalTitle.toLowerCase()
    );

    // Shuffle and pick 2 random suggestions
    const shuffled = filteredSuggestions.sort(() => Math.random() - 0.5);
    return shuffled.slice(0, 2);
  };

  // Handle column click - open Data Title modal for new selection
  const handleColumnToggle = (columnIndex, event) => {
    if (selectedColumns[columnIndex]) {
      // If already selected, do nothing on click (menu shows on hover)
      return;
    } else {
      // Open Data Title modal for new selection
      const originalTitle = selectedFileData?.headers[columnIndex] || `Column ${columnIndex + 1}`;
      const otherSuggestions = generateSuggestedTitles(originalTitle);

      setPendingColumnIndex(columnIndex);
      setSuggestedTitles([originalTitle, ...otherSuggestions]);
      setSelectedSuggestedTitle(null);
      setManualTitle('');
      setIsRenameMode(false);
      setCurrentTitleBeforeRename('');
      setShowDataTitleModal(true);
    }
  };

  // Handle hover on selected column - show action menu below header
  const handleColumnHeaderHover = (columnIndex) => {
    if (selectedColumns[columnIndex] && tableRef.current) {
      // Get the header cell position
      const headerCell = tableRef.current.querySelector(`thead th:nth-child(${columnIndex + 1})`);
      if (headerCell) {
        const rect = headerCell.getBoundingClientRect();
        setColumnActionMenuPosition({
          x: rect.left + rect.width / 2,
          y: rect.bottom
        });
        setShowColumnActionMenu(columnIndex);
      }
    }
  };

  // Handle mouse leave from column overlay
  const handleColumnHeaderLeave = (event) => {
    // Check if mouse is moving to the action menu
    const relatedTarget = event.relatedTarget;
    if (relatedTarget && relatedTarget.closest && relatedTarget.closest('[data-column-action-menu]')) {
      return; // Don't close if moving to menu
    }
    setShowColumnActionMenu(null);
  };

  // Handle deselect column
  const handleDeselectColumn = (columnIndex) => {
    const newSelectedColumns = { ...selectedColumns };
    delete newSelectedColumns[columnIndex];
    setSelectedColumns(newSelectedColumns);

    // Save column labels for current file
    if (selectedFileId) {
      setFileColumnLabels(prev => ({
        ...prev,
        [selectedFileId]: newSelectedColumns
      }));

      const hasSelection = Object.keys(newSelectedColumns).length > 0;
      setFileStatuses(prev => ({
        ...prev,
        [selectedFileId]: hasSelection ? 'Completed' : 'Processing'
      }));
    }

    setShowColumnActionMenu(null);
  };

  // Handle click on column tag - scroll to column and show action menu
  const handleColumnTagClick = (columnIndex) => {
    if (!tableRef.current || !previewContainerRef.current) return;

    const headerCell = tableRef.current.querySelector(`thead th:nth-child(${columnIndex + 1})`);
    if (headerCell) {
      // Get current zoom level
      const currentZoom = zoom / 100;

      // Calculate the scroll position to center the column
      const containerWidth = previewContainerRef.current.clientWidth;
      const cellLeft = headerCell.offsetLeft * currentZoom;
      const cellWidth = headerCell.offsetWidth * currentZoom;
      const scrollLeft = cellLeft - (containerWidth / 2) + (cellWidth / 2);

      // Set the column we're scrolling to
      setHoveredColumn(columnIndex);

      // Scroll to the column
      previewContainerRef.current.scrollTo({
        left: Math.max(0, scrollLeft),
        behavior: 'smooth'
      });

      // Use scrollend event if supported, otherwise use timeout with position check
      const container = previewContainerRef.current;
      let scrollEndTimer;

      const showMenuAfterScroll = () => {
        // Re-get the header cell position after scroll
        const updatedHeaderCell = tableRef.current?.querySelector(`thead th:nth-child(${columnIndex + 1})`);
        if (updatedHeaderCell) {
          const rect = updatedHeaderCell.getBoundingClientRect();
          setColumnActionMenuPosition({
            x: rect.left + rect.width / 2,
            y: rect.bottom
          });
          setShowColumnActionMenu(columnIndex);
        }
      };

      const handleScrollEnd = () => {
        clearTimeout(scrollEndTimer);
        scrollEndTimer = setTimeout(() => {
          showMenuAfterScroll();
          container.removeEventListener('scroll', handleScrollEnd);
        }, 100);
      };

      container.addEventListener('scroll', handleScrollEnd);
      // Fallback timeout in case scroll doesn't trigger
      setTimeout(() => {
        container.removeEventListener('scroll', handleScrollEnd);
        showMenuAfterScroll();
      }, 500);
    }
  };

  // Handle rename column title
  const handleRenameColumnTitle = (columnIndex) => {
    const originalTitle = selectedFileData?.headers[columnIndex] || `Column ${columnIndex + 1}`;
    const currentTitle = selectedColumns[columnIndex];
    const otherSuggestions = generateSuggestedTitles(originalTitle);

    setPendingColumnIndex(columnIndex);
    setSuggestedTitles([originalTitle, ...otherSuggestions]);
    setSelectedSuggestedTitle(null);
    setManualTitle('');
    setIsRenameMode(true);
    setCurrentTitleBeforeRename(currentTitle);
    setShowDataTitleModal(true);
    setShowColumnActionMenu(null);
  };

  // Handle refresh suggestions (keep original, change others)
  const handleRefreshSuggestions = () => {
    if (pendingColumnIndex === null || !selectedFileData) return;
    const originalTitle = selectedFileData.headers[pendingColumnIndex] || `Column ${pendingColumnIndex + 1}`;
    const otherSuggestions = generateSuggestedTitles(originalTitle);
    setSuggestedTitles([originalTitle, ...otherSuggestions]);
    setSelectedSuggestedTitle(null);
  };

  // Handle confirm Data Title selection
  const handleConfirmDataTitle = () => {
    if (pendingColumnIndex === null) return;

    const finalTitle = manualTitle.trim() || selectedSuggestedTitle;
    if (!finalTitle) return;

    const newSelectedColumns = {
      ...selectedColumns,
      [pendingColumnIndex]: finalTitle
    };

    setSelectedColumns(newSelectedColumns);

    // Save column labels for current file
    if (selectedFileId) {
      setFileColumnLabels(prev => ({
        ...prev,
        [selectedFileId]: newSelectedColumns
      }));

      setFileStatuses(prev => ({
        ...prev,
        [selectedFileId]: 'Completed'
      }));
    }

    // Close modal and reset states
    setShowDataTitleModal(false);
    setPendingColumnIndex(null);
    setSuggestedTitles([]);
    setSelectedSuggestedTitle(null);
    setManualTitle('');
    setIsRenameMode(false);
    setCurrentTitleBeforeRename('');
  };

  // Handle cancel Data Title modal
  const handleCancelDataTitle = () => {
    setShowDataTitleModal(false);
    setPendingColumnIndex(null);
    setSuggestedTitles([]);
    setSelectedSuggestedTitle(null);
    setManualTitle('');
    setIsRenameMode(false);
    setCurrentTitleBeforeRename('');
  };

  // Get all file IDs
  const allFileIds = folders.flatMap(folder => folder.files.map(f => f.id));

  // Check if all files are completed (no errors remaining)
  const allFilesCompleted = allFileIds.length > 0 && allFileIds.every(id => {
    const errorCount = fileErrorCounts[id] || 0;
    return errorCount === 0 && fileStatuses[id] === 'Completed';
  });

  // Check if there are any pending corrections (yellow or blue tags)
  const hasPendingCorrections = allFileIds.some(fileId => {
    const corrections = filePendingCorrections[fileId];
    if (!corrections) return false;

    // Check if there are any corrections with type 'corrected' (yellow pending)
    // or 'manual-edit' (blue pending) that haven't been marked as correct
    return Object.values(corrections).some(rowCorrections =>
      Object.values(rowCorrections).some(correction =>
        (correction.type === 'corrected' || correction.type === 'manual-edit') &&
        !correction.markedCorrect
      )
    );
  });

  // Calculate file statistics (errors, pending, correct)
  const getFileStats = (fileId) => {
    const errors = fileErrors[fileId] || {};
    const pending = filePendingCorrections[fileId] || {};

    let errorCount = 0;
    let pendingCount = 0;
    let correctCount = 0;

    // Count errors (red cells)
    Object.keys(errors).forEach(rowIdx => {
      Object.keys(errors[rowIdx]).forEach(colIdx => {
        // Check if this error also has a pending correction
        const pendingInfo = pending[rowIdx]?.[colIdx];
        if (!pendingInfo || pendingInfo.type === 'corrected' || pendingInfo.type === 'manual-edit') {
          // It's either a pure error or a pending correction (yellow/blue)
          if (pendingInfo && (pendingInfo.type === 'corrected' || pendingInfo.type === 'manual-edit')) {
            pendingCount++;
          } else {
            errorCount++;
          }
        } else if (pendingInfo && pendingInfo.type === 'marked-correct') {
          // It's marked as correct (green)
          correctCount++;
        }
      });
    });

    // Count pending corrections that are not in errors (manual edits without original errors)
    Object.keys(pending).forEach(rowIdx => {
      Object.keys(pending[rowIdx]).forEach(colIdx => {
        const pendingInfo = pending[rowIdx][colIdx];
        // If it's not already counted in errors
        if (!errors[rowIdx]?.[colIdx]) {
          if (pendingInfo.type === 'marked-correct') {
            correctCount++;
          } else if (pendingInfo.type === 'manual-edit') {
            pendingCount++;
          }
        }
      });
    });

    return { errorCount, pendingCount, correctCount };
  };

  // Handle zoom
  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 10, 200));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 10, 50));
  };

  // Handle error correction confirmation
  const handleConfirmCorrection = () => {
    if (!selectedError || !selectedFileId) return;

    const { rowIdx, colIdx, isManualEdit, originalValue, pendingType, originalError } = selectedError;

    // Check if the corrected value is the same as the original value
    if (correctedValue === originalValue || correctedValue.toString() === originalValue.toString()) {
      // If same, do nothing and just close the modal
      setShowErrorModal(false);
      setSelectedError(null);
      setCorrectedValue('');
      return;
    }

    // Don't update the file data - keep original data intact
    // Only store the correction in pending corrections

    // If this was previously marked-correct and had an original error, restore the error
    if (pendingType === 'marked-correct' && originalError) {
      setFileErrors(prev => {
        const newErrors = { ...prev };
        if (!newErrors[selectedFileId]) {
          newErrors[selectedFileId] = {};
        }
        if (!newErrors[selectedFileId][rowIdx]) {
          newErrors[selectedFileId][rowIdx] = {};
        }
        // Restore the original error message
        newErrors[selectedFileId][rowIdx][colIdx] = originalError;
        return newErrors;
      });
    } else if (!isManualEdit) {
      // For system-detected errors: keep the error in fileErrors
      setFileErrors(prev => {
        const newErrors = { ...prev };
        if (!newErrors[selectedFileId]) {
          newErrors[selectedFileId] = {};
        }
        if (!newErrors[selectedFileId][rowIdx]) {
          newErrors[selectedFileId][rowIdx] = {};
        }
        // Keep the original error message
        newErrors[selectedFileId][rowIdx][colIdx] = selectedError.errorMessage || selectedError.originalError;
        return newErrors;
      });
    }

    // Add to pending corrections
    // If the cell was previously marked as correct (green), change it back to pending (yellow/blue)
    // For manual edits: type is 'manual-edit'
    // For system errors: type is 'corrected'
    setFilePendingCorrections(prev => {
      const newPending = { ...prev };
      if (!newPending[selectedFileId]) {
        newPending[selectedFileId] = {};
      }
      if (!newPending[selectedFileId][rowIdx]) {
        newPending[selectedFileId][rowIdx] = {};
      }

      // Determine the correct type: if it was marked-correct with an original error, it should go back to 'corrected'
      // Otherwise use the logic based on isManualEdit
      let newType;
      if (pendingType === 'marked-correct' && originalError) {
        newType = 'corrected'; // Was a system error, go back to yellow pending
      } else {
        newType = isManualEdit ? 'manual-edit' : 'corrected';
      }

      newPending[selectedFileId][rowIdx][colIdx] = {
        originalError: originalError || (isManualEdit ? null : (selectedError.errorMessage || selectedError.originalError)),
        correctedValue: correctedValue, // Store the corrected value
        type: newType,  // Set to appropriate pending type (not marked-correct)
        wasManualEdit: isManualEdit || selectedError.wasManualEdit || false
      };
      return newPending;
    });

    // Close modal
    setShowErrorModal(false);
    setSelectedError(null);
    setCorrectedValue('');
  };

  // Handle cancel error correction
  const handleCancelCorrection = () => {
    setShowErrorModal(false);
    setSelectedError(null);
    setCorrectedValue('');
  };

  // Handle mark as correct button click - show confirmation
  const handleMarkAsCorrectClick = () => {
    setShowMarkCorrectConfirmation(true);
  };

  // Handle confirm mark as correct
  const handleConfirmMarkAsCorrect = () => {
    if (!selectedError || !selectedFileId) return;

    const { rowIdx, colIdx } = selectedError;

    // Remove error from fileErrors (without changing the data)
    setFileErrors(prev => {
      const newErrors = { ...prev };
      if (newErrors[selectedFileId] && newErrors[selectedFileId][rowIdx]) {
        const newRowErrors = { ...newErrors[selectedFileId][rowIdx] };
        delete newRowErrors[colIdx];

        if (Object.keys(newRowErrors).length === 0) {
          const newFileErrors = { ...newErrors[selectedFileId] };
          delete newFileErrors[rowIdx];
          newErrors[selectedFileId] = newFileErrors;
        } else {
          newErrors[selectedFileId] = {
            ...newErrors[selectedFileId],
            [rowIdx]: newRowErrors
          };
        }
      }
      return newErrors;
    });

    // Add to pending corrections (type: marked-correct - no verification needed)
    setFilePendingCorrections(prev => {
      const newPending = { ...prev };
      if (!newPending[selectedFileId]) {
        newPending[selectedFileId] = {};
      }
      if (!newPending[selectedFileId][rowIdx]) {
        newPending[selectedFileId][rowIdx] = {};
      }
      newPending[selectedFileId][rowIdx][colIdx] = {
        originalError: selectedError.errorMessage || selectedError.originalError,
        type: 'marked-correct'
      };
      return newPending;
    });

    // Close both modals
    setShowMarkCorrectConfirmation(false);
    setShowErrorModal(false);
    setSelectedError(null);
    setCorrectedValue('');
  };

  // Handle cancel mark as correct confirmation
  const handleCancelMarkAsCorrect = () => {
    setShowMarkCorrectConfirmation(false);
  };

  // Handle unmark as correct (revert back to error state)
  const handleUnmarkAsCorrect = () => {
    if (!selectedError || !selectedFileId) return;

    const { rowIdx, colIdx } = selectedError;

    // Remove from pending corrections
    setFilePendingCorrections(prev => {
      const newPending = { ...prev };
      if (newPending[selectedFileId]?.[rowIdx]?.[colIdx]) {
        const filePending = { ...newPending[selectedFileId] };
        const rowPending = { ...filePending[rowIdx] };
        delete rowPending[colIdx];

        if (Object.keys(rowPending).length === 0) {
          delete filePending[rowIdx];
        } else {
          filePending[rowIdx] = rowPending;
        }

        if (Object.keys(filePending).length === 0) {
          delete newPending[selectedFileId];
        } else {
          newPending[selectedFileId] = filePending;
        }
      }
      return newPending;
    });

    // Add back to errors with original error message
    setFileErrors(prev => {
      const newErrors = { ...prev };
      if (!newErrors[selectedFileId]) {
        newErrors[selectedFileId] = {};
      }
      if (!newErrors[selectedFileId][rowIdx]) {
        newErrors[selectedFileId][rowIdx] = {};
      }
      newErrors[selectedFileId][rowIdx][colIdx] = selectedError.originalError || selectedError.errorMessage;
      return newErrors;
    });

    // Close modal
    setShowErrorModal(false);
    setSelectedError(null);
    setCorrectedValue('');
  };

  // Helper function to detect data type
  const detectDataType = (value) => {
    if (value === null || value === undefined || value === '') return 'empty';
    const strValue = String(value).trim();
    if (strValue === '') return 'empty';
    // Check if it's a number (including decimals)
    if (!isNaN(strValue) && !isNaN(parseFloat(strValue))) return 'number';
    return 'string';
  };

  // Handle Run Data Check
  const handleRunDataCheck = () => {
    // Get all files
    const allFiles = folders.flatMap(folder => folder.files);

    allFiles.forEach(file => {
      const fileId = file.id;
      const fileData = file.content;
      const pendingCorrections = filePendingCorrections[fileId];

      if (!pendingCorrections || !fileData) return;

      // For each column, collect all values to determine the column type
      const columnTypes = {};
      fileData.rows.forEach((row, rowIdx) => {
        row.forEach((cell, colIdx) => {
          if (!columnTypes[colIdx]) {
            columnTypes[colIdx] = [];
          }
          // Use original value unless there's a pending correction
          const pendingInfo = pendingCorrections[rowIdx]?.[colIdx];
          const valueToCheck = pendingInfo?.correctedValue !== undefined ? pendingInfo.correctedValue : cell;
          const type = detectDataType(valueToCheck);
          if (type !== 'empty') {
            columnTypes[colIdx].push(type);
          }
        });
      });

      // Determine majority type for each column
      const columnMajorityTypes = {};
      Object.keys(columnTypes).forEach(colIdx => {
        const types = columnTypes[colIdx];
        const typeCounts = types.reduce((acc, type) => {
          acc[type] = (acc[type] || 0) + 1;
          return acc;
        }, {});
        // Get the type with highest count
        columnMajorityTypes[colIdx] = Object.keys(typeCounts).reduce((a, b) =>
          typeCounts[a] > typeCounts[b] ? a : b
        );
      });

      // Check all pending corrections
      Object.keys(pendingCorrections).forEach(rowIdx => {
        Object.keys(pendingCorrections[rowIdx]).forEach(colIdx => {
          const pendingInfo = pendingCorrections[rowIdx][colIdx];

          // Only check cells with 'corrected' or 'manual-edit' type (yellow/blue pending)
          if (pendingInfo.type !== 'corrected' && pendingInfo.type !== 'manual-edit') return;

          const correctedValue = pendingInfo.correctedValue;
          const valueType = detectDataType(correctedValue);
          const expectedType = columnMajorityTypes[colIdx];

          let hasError = false;
          let errorMessage = '';

          // Check 1: Empty value
          if (valueType === 'empty') {
            hasError = true;
            errorMessage = 'Value cannot be empty';
          }
          // Check 2: Type mismatch with column majority
          else if (valueType !== expectedType) {
            hasError = true;
            errorMessage = `Type mismatch: expected ${expectedType}, got ${valueType}`;
          }

          if (hasError) {
            // Change to error state (red) but keep the pending correction value
            setFileErrors(prev => {
              const newErrors = { ...prev };
              if (!newErrors[fileId]) {
                newErrors[fileId] = {};
              }
              if (!newErrors[fileId][rowIdx]) {
                newErrors[fileId][rowIdx] = {};
              }
              newErrors[fileId][rowIdx][colIdx] = errorMessage;
              return newErrors;
            });

            // Keep the pending correction with its corrected value so the cell
            // still displays the user's edit, but change type back to 'corrected'
            // so it shows as needing re-correction (yellow)
            setFilePendingCorrections(prev => {
              const newPending = { ...prev };
              if (newPending[fileId]?.[rowIdx]?.[colIdx]) {
                const filePending = { ...newPending[fileId] };
                const rowPending = { ...filePending[rowIdx] };
                rowPending[colIdx] = {
                  ...rowPending[colIdx],
                  type: 'corrected',
                  passedDataCheck: false,
                  wasManualEdit: rowPending[colIdx].type === 'manual-edit' || rowPending[colIdx].wasManualEdit || false,
                  dataCheckError: errorMessage
                };
                filePending[rowIdx] = rowPending;
                newPending[fileId] = filePending;
              }
              return newPending;
            });
          } else {
            // Change to marked-correct state (green)
            setFilePendingCorrections(prev => {
              const newPending = { ...prev };
              if (!newPending[fileId]) {
                newPending[fileId] = {};
              }
              if (!newPending[fileId][rowIdx]) {
                newPending[fileId][rowIdx] = {};
              }
              newPending[fileId][rowIdx][colIdx] = {
                ...pendingInfo,
                type: 'marked-correct',
                passedDataCheck: true,
                wasManualEdit: pendingInfo.type === 'manual-edit' || pendingInfo.wasManualEdit || false
              };
              return newPending;
            });

            // Remove from errors if it was there
            setFileErrors(prev => {
              const newErrors = { ...prev };
              if (newErrors[fileId]?.[rowIdx]?.[colIdx]) {
                const fileErrors = { ...newErrors[fileId] };
                const rowErrors = { ...fileErrors[rowIdx] };
                delete rowErrors[colIdx];

                if (Object.keys(rowErrors).length === 0) {
                  delete fileErrors[rowIdx];
                } else {
                  fileErrors[rowIdx] = rowErrors;
                }

                if (Object.keys(fileErrors).length === 0) {
                  delete newErrors[fileId];
                } else {
                  newErrors[fileId] = fileErrors;
                }
              }
              return newErrors;
            });
          }
        });
      });
    });
  };

  // Handle revert to original value (for manual-edit)
  const handleRevertToOriginal = () => {
    if (!selectedError || !selectedFileId) return;

    const { rowIdx, colIdx } = selectedError;

    // Remove from pending corrections
    setFilePendingCorrections(prev => {
      const newPending = { ...prev };
      if (newPending[selectedFileId]?.[rowIdx]?.[colIdx]) {
        const filePending = { ...newPending[selectedFileId] };
        const rowPending = { ...filePending[rowIdx] };
        delete rowPending[colIdx];

        if (Object.keys(rowPending).length === 0) {
          delete filePending[rowIdx];
        } else {
          filePending[rowIdx] = rowPending;
        }

        if (Object.keys(filePending).length === 0) {
          delete newPending[selectedFileId];
        } else {
          newPending[selectedFileId] = filePending;
        }
      }
      return newPending;
    });

    // Also remove from fileErrors (in case a failed manual edit added an error entry)
    setFileErrors(prev => {
      const newErrors = { ...prev };
      if (newErrors[selectedFileId]?.[rowIdx]?.[colIdx]) {
        const fileErrs = { ...newErrors[selectedFileId] };
        const rowErrs = { ...fileErrs[rowIdx] };
        delete rowErrs[colIdx];

        if (Object.keys(rowErrs).length === 0) {
          delete fileErrs[rowIdx];
        } else {
          fileErrs[rowIdx] = rowErrs;
        }

        if (Object.keys(fileErrs).length === 0) {
          delete newErrors[selectedFileId];
        } else {
          newErrors[selectedFileId] = fileErrs;
        }
      }
      return newErrors;
    });

    // Close modal
    setShowErrorModal(false);
    setSelectedError(null);
    setCorrectedValue('');
  };

  // Get status tag color
  const getStatusColor = (status) => {
    switch (status) {
      case 'Pending':
        return 'bg-[#7E7E7E]/10 text-[#7E7E7E]';
      case 'Processing':
        return 'bg-[#B58E42]/10 text-[#B58E42]';
      case 'Completed':
        return 'bg-[#50B461]/10 text-[#50B461]';
      default:
        return 'bg-[#7E7E7E]/10 text-[#7E7E7E]';
    }
  };

  // Close menus when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      if (folderMenuOpen !== null || fileMenuOpen !== null) {
        setFolderMenuOpen(null);
        setFileMenuOpen(null);
      }
      if (showColumnActionMenu !== null) {
        setShowColumnActionMenu(null);
      }
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, [folderMenuOpen, fileMenuOpen, showColumnActionMenu]);

  // Calculate column positions for overlay (using offset values to work with zoom)
  useEffect(() => {
    if (tableRef.current && selectedFileData) {
      const calculatePositions = () => {
        const table = tableRef.current;
        if (!table) return;

        const headerCells = table.querySelectorAll('thead th');
        const positions = [];

        headerCells.forEach((cell) => {
          positions.push({
            left: cell.offsetLeft,
            width: cell.offsetWidth,
            height: table.offsetHeight
          });
        });

        setColumnPositions(positions);
      };

      // Small delay to ensure DOM is ready after zoom change
      const timer = setTimeout(calculatePositions, 50);

      // Recalculate on resize
      window.addEventListener('resize', calculatePositions);
      return () => {
        clearTimeout(timer);
        window.removeEventListener('resize', calculatePositions);
      };
    }
  }, [selectedFileData, zoom, hoveredColumn]);

  // Handle scroll - update menu position and check if column is visible
  useEffect(() => {
    const container = previewContainerRef.current;
    if (!container) return;

    const handleScroll = () => {
      // If there's a column action menu showing, update its position or hide it
      if (showColumnActionMenu !== null && tableRef.current) {
        const headerCell = tableRef.current.querySelector(`thead th:nth-child(${showColumnActionMenu + 1})`);
        if (headerCell) {
          const cellRect = headerCell.getBoundingClientRect();
          const containerRect = container.getBoundingClientRect();

          // Check if column is visible in the container
          const isVisible = (
            cellRect.left < containerRect.right &&
            cellRect.right > containerRect.left
          );

          if (isVisible) {
            // Update menu position
            setColumnActionMenuPosition({
              x: cellRect.left + cellRect.width / 2,
              y: cellRect.bottom
            });
          } else {
            // Column scrolled out of view, hide menu and hover
            setShowColumnActionMenu(null);
            setHoveredColumn(null);
          }
        }
      }
    };

    container.addEventListener('scroll', handleScroll);
    return () => container.removeEventListener('scroll', handleScroll);
  }, [showColumnActionMenu]);

  return (
    <div className="bg-white fixed inset-0 overflow-hidden" data-name="DATA_LABELING">
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
        {/* Step 1 - Completed (Clickable to go back) */}
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
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #29ABB5)' }}></div>

        {/* Step 2 - Active */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-[#29ABB5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">2</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Data Cleaning
          </p>
        </div>
        <div className="flex-1 h-[2px] bg-[#ACC2C5] mx-[12px]"></div>

        {/* Step 3 - Inactive */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-white border-2 border-[#ACC2C5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#7E7E7E]">3</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] whitespace-nowrap">
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

      {/* Main Content Area - Two Columns */}
      <div className="absolute left-[64px] right-[64px] top-[132px] bottom-[32px] flex gap-[24px]">
        {/* Left Column - File List with Status */}
        <div className="w-[40%] flex-shrink-0 h-full bg-[#F5F5F5] rounded-[10px] p-[24px] relative">
          <div className="flex items-center justify-between mb-[24px]">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B]">
              Raw Data staging area
            </p>
          </div>

          {/* Scrollable list container */}
          <div
            className="overflow-y-auto pr-[8px]"
            style={{ height: 'calc(100% - 80px - 49px - 24px)' }}
            onClick={(e) => {
              // Only deselect if clicking directly on the scrollable container (empty area)
              if (e.target === e.currentTarget) {
                if (selectedFileId && fileStatuses[selectedFileId] === 'Processing') {
                  setFileStatuses(prev => ({ ...prev, [selectedFileId]: 'Pending' }));
                }
                setSelectedFileId(null);
                setSelectedColumns({});
              }
            }}
          >
            {folders.map((folder) => (
              <div key={folder.id} className="mb-[24px]">
                {/* Folder Header */}
                <div
                  className={`bg-white h-[49px] flex items-center pl-[13px] pr-[17px] justify-between relative border-l-4 border-[#29ABB5] transition ${
                    dragOverFolder === folder.id ? 'bg-[#29ABB5]/20 border-l-[6px]' : ''
                  }`}
                  onDragOver={(e) => handleDragOver(e, folder.id)}
                  onDragLeave={handleDragLeave}
                  onDrop={(e) => handleDrop(e, folder.id)}
                >
                  <div className="flex items-center gap-[12px]">
                    <FolderIcon className="w-[24px] h-[24px] text-[#365D60]" />
                    <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B]">
                      {folder.name}
                    </p>
                  </div>
                  <div className="flex items-center gap-[8px]">
                    <div className="bg-[#29ABB5]/10 px-[5px] py-0 rounded-[4px]">
                      <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#365D60]">
                        {folder.tag}
                      </p>
                    </div>
                    <div className="relative -mr-[9px]">
                      <div
                        onClick={(e) => {
                          e.stopPropagation();
                          setFolderMenuOpen(folderMenuOpen === folder.id ? null : folder.id);
                        }}
                        className="px-[12px] py-[8px] cursor-pointer hover:bg-[#f5f5f5] rounded-[4px] transition"
                      >
                        <IconEllipsis className="w-[3px] h-[13px] relative" />
                      </div>
                      {folderMenuOpen === folder.id && (
                        <div className="absolute right-0 top-[20px] bg-white rounded-[8px] shadow-[0px_2px_8px_0px_rgba(0,0,0,0.15)] py-[8px] w-[180px] z-10" onClick={(e) => e.stopPropagation()}>
                          <button
                            onClick={() => {
                              setShowMoveAllModal(folder.id);
                              setFolderMenuOpen(null);
                            }}
                            className="w-full px-[16px] py-[8px] text-left font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#f5f5f5] transition"
                          >
                            Move all files to
                          </button>
                          <button
                            onClick={() => {
                              setShowDeleteAllModal(folder.id);
                              setFolderMenuOpen(null);
                            }}
                            className="w-full px-[16px] py-[8px] text-left font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#f5f5f5] transition"
                          >
                            Delete all files
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Files in folder */}
                {folder.files.length > 0 ? (
                  folder.files.map((file) => (
                    <div
                      key={file.id}
                      draggable
                      onDragStart={(e) => handleDragStart(e, file, folder.id)}
                      onDragEnd={handleDragEnd}
                      onClick={() => handleFileSelect(file.id)}
                      className={`bg-white h-[49px] flex items-center pl-[17px] pr-[17px] justify-between border-t border-[#EEEEEE] relative transition cursor-pointer ${
                        selectedFileId === file.id ? 'bg-[#29ABB5]/20' : 'hover:bg-[#29ABB5]/10'
                      } ${draggedFile?.file.id === file.id ? 'opacity-50' : ''}`}
                    >
                      <div className="flex items-center gap-[12px] pl-[32px]">
                        <DataTableIcon className="w-[24px] h-[24px] text-[#7E7E7E]" />
                        <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B]">
                          {file.name}
                        </p>
                      </div>
                      <div className="flex items-center gap-[8px]">
                        {(() => {
                          const stats = getFileStats(file.id);
                          const { errorCount, pendingCount, correctCount } = stats;
                          const totalIssues = errorCount + pendingCount + correctCount;

                          // If no issues at all, show nothing or a placeholder
                          if (totalIssues === 0) {
                            return null;
                          }

                          // If all are correct, show "Verified"
                          if (errorCount === 0 && pendingCount === 0 && correctCount > 0) {
                            return (
                              <div className="px-[8px] py-[2px] rounded-[4px] bg-[#50B461]/10 text-[#50B461]">
                                <p className="font-['Noto_Sans',sans-serif] font-normal text-[12px]">
                                  Verified
                                </p>
                              </div>
                            );
                          }

                          // Show error and/or pending tags
                          return (
                            <>
                              {errorCount > 0 && (
                                <div className="px-[8px] py-[2px] rounded-[4px] bg-[#DC3545]/10 text-[#DC3545]">
                                  <p className="font-['Noto_Sans',sans-serif] font-normal text-[12px]">
                                    {errorCount} error{errorCount !== 1 ? 's' : ''}
                                  </p>
                                </div>
                              )}
                              {pendingCount > 0 && (
                                <div className="px-[8px] py-[2px] rounded-[4px] bg-[#B58E42]/10 text-[#B58E42]">
                                  <p className="font-['Noto_Sans',sans-serif] font-normal text-[12px]">
                                    {pendingCount} pending
                                  </p>
                                </div>
                              )}
                            </>
                          );
                        })()}
                        <div className="relative -mr-[8px]">
                          <div
                            onClick={(e) => {
                              e.stopPropagation();
                              setFileMenuOpen(fileMenuOpen === file.id ? null : file.id);
                            }}
                            className="px-[12px] py-[8px] cursor-pointer hover:bg-[#e5e5e5] rounded-[4px] transition"
                          >
                            <IconEllipsis className="w-[3px] h-[13px] relative" />
                          </div>
                          {fileMenuOpen === file.id && (
                            <div className="absolute right-0 top-[20px] bg-white rounded-[8px] shadow-[0px_2px_8px_0px_rgba(0,0,0,0.15)] py-[8px] w-[140px] z-10" onClick={(e) => e.stopPropagation()}>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setShowMoveFileModal({ fileId: file.id, folderId: folder.id });
                                  setFileMenuOpen(null);
                                }}
                                className="w-full px-[16px] py-[8px] text-left font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#f5f5f5] transition"
                              >
                                Move to
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setShowRenameModal({ fileId: file.id, folderId: folder.id });
                                  setRenameValue(file.name);
                                  setFileMenuOpen(null);
                                }}
                                className="w-full px-[16px] py-[8px] text-left font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#f5f5f5] transition"
                              >
                                Rename
                              </button>
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setShowDeleteFileModal({ fileId: file.id, folderId: folder.id });
                                  setFileMenuOpen(null);
                                }}
                                className="w-full px-[16px] py-[8px] text-left font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#f5f5f5] transition"
                              >
                                Delete
                              </button>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                ) : null}
              </div>
            ))}
          </div>

          {/* Custom scrollbar styling */}
          <style jsx>{`
            div::-webkit-scrollbar {
              width: 11px;
            }
            div::-webkit-scrollbar-track {
              background: rgba(175, 175, 175, 0.2);
              border-radius: 5px;
            }
            div::-webkit-scrollbar-thumb {
              background: #afafaf;
              border-radius: 2.5px;
              width: 3px;
            }
          `}</style>
        </div>

        {/* Right Column - Data Preview (White background like Upload Your Data) */}
        <div className="flex-1 min-w-0 h-full pt-[24px] pb-[24px] pl-[24px] relative flex flex-col">
          {/* Header with title and description side by side */}
          <div className="flex items-start justify-between mb-[24px]">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B] h-[39px]">
              Fix Data Errors
            </p>
            <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] max-w-[360px] text-right leading-relaxed h-[39px] flex items-center justify-end">
              Review all data and fix both system-flagged<br/>and manually identified errors.
            </p>
          </div>

          {selectedFileId ? (
            <>
              {/* Table Preview Wrapper - relative container for fixed zoom controls */}
              <div className="relative flex-shrink-0" style={{ height: 'calc((100% - 80px - 49px - 16px) * 0.8)' }}>
                {/* Scrollable Table Container */}
                <div
                  ref={previewContainerRef}
                  className="h-full bg-[#F5F5F5] overflow-auto border border-[#E0E0E0]"
                >
                  {/* Table with zoom - using CSS zoom instead of transform for sticky to work */}
                  <div
                    className="zoom-wrapper"
                    style={{
                      zoom: zoom / 100,
                      minWidth: 'max-content'
                    }}
                  >
                    {selectedFileData && (
                      <div className="relative">
                        <table
                          ref={tableRef}
                          className="border-collapse bg-white"
                          style={{ minWidth: 'max-content' }}
                        >
                          <thead className="sticky top-0 z-10">
                            <tr>
                              {selectedFileData.headers.map((header, idx) => (
                                <th
                                  key={idx}
                                  className="px-[16px] py-[12px] min-w-[120px] border border-[#EEEEEE] relative bg-[#F5F5F5] text-[#13383B]"
                                  style={{ opacity: selectedColumns[idx] ? 1 : 0.3 }}
                                >
                                  <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px]">
                                    {header}
                                  </p>
                                  {selectedColumns[idx] && (
                                    <p
                                      className="font-['Noto_Sans',sans-serif] font-normal text-[12px] mt-[4px]"
                                      style={{ opacity: 0.8 }}
                                    >
                                      {typeof selectedColumns[idx] === 'string' ? `→ ${selectedColumns[idx]}` : ''}
                                    </p>
                                  )}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {selectedFileData.rows.map((row, rowIdx) => {
                              const currentFileErrors = fileErrors[selectedFileId] || {};
                              const rowErrors = currentFileErrors[rowIdx] || {};
                              const currentFilePending = filePendingCorrections[selectedFileId] || {};
                              const rowPending = currentFilePending[rowIdx] || {};

                              return (
                                <tr key={rowIdx}>
                                  {row.map((cell, cellIdx) => {
                                    const hasError = rowErrors[cellIdx];
                                    const errorMessage = hasError ? rowErrors[cellIdx] : null;
                                    const pendingInfo = rowPending[cellIdx];
                                    const isPending = !!pendingInfo;
                                    const pendingType = pendingInfo?.type;

                                    // Determine cell style and behavior
                                    let bgClass = 'bg-white';
                                    let isClickable = false;
                                    let cellTitle = '';
                                    let iconType = null; // 'error', 'marked-correct', 'corrected', 'manual-edit'
                                    let focusBgColor = ''; // Background color when focused (hex value with opacity)
                                    let isManualEdit = false; // Track if this is a manual edit (no system error)

                                    if (hasError && isPending && pendingType === 'corrected') {
                                      // Corrected but awaiting verification - yellow
                                      bgClass = 'bg-[#B58E42]/10 hover:bg-[#B58E42]/20 cursor-pointer';
                                      isClickable = true;
                                      cellTitle = 'Corrected - awaiting verification';
                                      iconType = 'corrected';
                                      focusBgColor = '#B58E42';
                                    } else if (hasError) {
                                      // Has error - red
                                      bgClass = 'bg-[#DC3545]/10 hover:bg-[#DC3545]/20 cursor-pointer';
                                      isClickable = true;
                                      cellTitle = errorMessage;
                                      iconType = 'error';
                                      focusBgColor = '#DC3545';
                                    } else if (isPending && pendingType === 'marked-correct') {
                                      // Marked as correct - green
                                      // Different styles based on whether it passed data check
                                      const passedDataCheck = pendingInfo?.passedDataCheck;
                                      if (passedDataCheck) {
                                        // Passed data check: no fill, green border only
                                        bgClass = 'bg-white border-2 border-[#50B461] hover:bg-[#50B461]/5 cursor-pointer';
                                      } else {
                                        // Manually marked: green fill
                                        bgClass = 'bg-[#50B461]/10 hover:bg-[#50B461]/20 cursor-pointer';
                                      }
                                      isClickable = true;
                                      cellTitle = passedDataCheck ? 'Verified correct (data check)' : 'Marked as correct';
                                      iconType = 'marked-correct';
                                      focusBgColor = '#50B461';
                                    } else if (isPending && pendingType === 'manual-edit') {
                                      // Manually edited by user (no system error) - saturated blue
                                      bgClass = 'bg-[#4D6EC7]/10 hover:bg-[#4D6EC7]/20 cursor-pointer';
                                      isClickable = true;
                                      cellTitle = 'Manually edited - awaiting verification';
                                      iconType = 'manual-edit';
                                      focusBgColor = '#4D6EC7';
                                      isManualEdit = true;
                                    } else if (selectedColumns[cellIdx]) {
                                      // Normal cell in selected column - allow manual editing
                                      bgClass = 'bg-white hover:bg-gray-50 cursor-pointer';
                                      isClickable = true;
                                      cellTitle = 'Click to edit';
                                    }

                                    // Check if this cell is currently focused
                                    const isFocused = focusedCell && focusedCell.rowIdx === rowIdx && focusedCell.colIdx === cellIdx;

                                    return (
                                      <td
                                        key={cellIdx}
                                        ref={(el) => {
                                          // Store ref for error cells
                                          if (isClickable && el) {
                                            if (!errorCellRefs.current[selectedFileId]) {
                                              errorCellRefs.current[selectedFileId] = {};
                                            }
                                            if (!errorCellRefs.current[selectedFileId][rowIdx]) {
                                              errorCellRefs.current[selectedFileId][rowIdx] = {};
                                            }
                                            errorCellRefs.current[selectedFileId][rowIdx][cellIdx] = el;
                                          }
                                        }}
                                        className={`px-[16px] py-[10px] font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] border border-[#EEEEEE] relative ${isFocused ? '' : bgClass}`}
                                        style={{
                                          opacity: selectedColumns[cellIdx] ? 1 : 0.3,
                                          transition: 'all 0.3s ease',
                                          ...(isFocused && focusBgColor ? {
                                            backgroundColor: `${focusBgColor}33`
                                          } : {})
                                        }}
                                        title={cellTitle}
                                        onMouseEnter={(e) => {
                                          // Show tooltip for yellow pending cells with corrected value
                                          if (isPending && (pendingType === 'corrected' || pendingType === 'manual-edit') && pendingInfo.correctedValue !== undefined) {
                                            const rect = e.currentTarget.getBoundingClientRect();
                                            setHoveredCell({
                                              rowIdx,
                                              colIdx: cellIdx,
                                              originalValue: cell, // The original raw value
                                              x: rect.left + rect.width / 2,
                                              y: rect.top - 10
                                            });
                                          }
                                        }}
                                        onMouseLeave={() => {
                                          setHoveredCell(null);
                                        }}
                                        onClick={() => {
                                          if (isClickable) {
                                            const originalError = pendingInfo?.originalError || null;
                                            const displayValue = pendingInfo && pendingInfo.correctedValue !== undefined ? pendingInfo.correctedValue : cell;

                                            // Determine if this is a manual edit:
                                            // - No system error (hasError is false)
                                            // - OR it's already marked as manual-edit pending
                                            const isManualEditClick = !hasError || isManualEdit;

                                            setSelectedError({
                                              rowIdx,
                                              colIdx: cellIdx,
                                              errorMessage: hasError ? errorMessage : 'Manual edit',
                                              currentValue: displayValue,
                                              originalValue: cell, // Store the original raw value
                                              isPending,
                                              pendingType,
                                              originalError,
                                              isManualEdit: isManualEditClick, // Flag to identify manual edits
                                              passedDataCheck: pendingInfo?.passedDataCheck || false,
                                              wasManualEdit: pendingInfo?.wasManualEdit || false
                                            });
                                            setCorrectedValue(displayValue || '');
                                            setShowErrorModal(true);
                                          }
                                        }}
                                      >
                                        <div className="flex items-center gap-[8px]">
                                          {iconType === 'error' && (
                                            <svg className="w-[16px] h-[16px] text-[#DC3545] flex-shrink-0" viewBox="0 0 24 24" fill="currentColor">
                                              <circle cx="12" cy="12" r="10" opacity="0.2"/>
                                              <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                                            </svg>
                                          )}
                                          {iconType === 'marked-correct' && (
                                            <svg className="w-[16px] h-[16px] text-[#50B461] flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                              <circle cx="12" cy="12" r="10" strokeWidth="2"/>
                                              <path d="M9 12l2 2 4-4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                            </svg>
                                          )}
                                          {iconType === 'corrected' && (
                                            <svg className="w-[16px] h-[16px] text-[#B58E42] flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                              <circle cx="12" cy="12" r="10" strokeWidth="2"/>
                                              <path d="M12 6v6l4 2" strokeWidth="2" strokeLinecap="round"/>
                                            </svg>
                                          )}
                                          {iconType === 'manual-edit' && (
                                            <svg className="w-[16px] h-[16px] text-[#4D6EC7] flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                              <circle cx="12" cy="12" r="10" strokeWidth="2"/>
                                              <path d="M12 6v6l4 2" strokeWidth="2" strokeLinecap="round"/>
                                            </svg>
                                          )}
                                          <span>
                                            {/* Show corrected value if exists, otherwise show original cell value */}
                                            {pendingInfo && pendingInfo.correctedValue !== undefined ? pendingInfo.correctedValue : cell}
                                          </span>
                                        </div>
                                      </td>
                                    );
                                  })}
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </div>

                {/* Zoom controls - fixed to wrapper, not scrollable content */}
                <div className="absolute bottom-[20px] right-[20px] flex items-center gap-[8px] bg-white rounded-[8px] px-[8px] py-[6px] shadow-[0px_4px_12px_0px_rgba(0,0,0,0.25)] z-30 border border-[#E0E0E0]">
                  <button
                    onClick={handleZoomOut}
                    className="w-[28px] h-[28px] bg-[#F5F5F5] rounded-[6px] flex items-center justify-center hover:bg-[#EEEEEE] transition"
                  >
                    <ZoomOutIcon className="w-[16px] h-[16px] text-[#365D60]" />
                  </button>
                  <span className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] w-[40px] text-center">
                    {zoom}%
                  </span>
                  <button
                    onClick={handleZoomIn}
                    className="w-[28px] h-[28px] bg-[#F5F5F5] rounded-[6px] flex items-center justify-center hover:bg-[#EEEEEE] transition"
                  >
                    <ZoomInIcon className="w-[16px] h-[16px] text-[#365D60]" />
                  </button>
                </div>
              </div>

              {/* Errors Detected and Manual Edits Section */}
              <div className="mt-[24px] flex gap-[16px]">
                {/* Left: Errors Detected */}
                <div className="flex-1 min-w-0">
                  <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#13383B] mb-[12px]">
                    Errors Detected
                  </p>
                  <div className="overflow-x-auto pb-[8px]" style={{
                    scrollbarWidth: 'thin',
                    scrollbarColor: '#D1D5DB transparent'
                  }}>
                    <style>{`
                      .overflow-x-auto::-webkit-scrollbar {
                        height: 6px;
                      }
                      .overflow-x-auto::-webkit-scrollbar-track {
                        background: transparent;
                      }
                      .overflow-x-auto::-webkit-scrollbar-thumb {
                        background-color: #D1D5DB;
                        border-radius: 3px;
                      }
                      .overflow-x-auto::-webkit-scrollbar-thumb:hover {
                        background-color: #9CA3AF;
                      }
                    `}</style>
                    <div className="flex gap-[8px] min-w-min">
                  {selectedFileId ? (
                    (() => {
                      const currentFileErrors = fileErrors[selectedFileId] || {};
                      const currentFilePending = filePendingCorrections[selectedFileId] || {};
                      const issueList = [];

                      // Get selected columns for current file
                      const currentSelectedCols = fileColumnLabels[selectedFileId] || {};
                      const currentSelectedColIndices = Object.keys(currentSelectedCols).map(k => parseInt(k));

                      // Collect all issues (errors + pending corrections)
                      const processedCells = new Set();

                      // First, collect all cells from fileErrors (exclude manual edits — they go in Manual Edits section)
                      Object.keys(currentFileErrors).forEach(rowIdx => {
                        Object.keys(currentFileErrors[rowIdx]).forEach(colIdx => {
                          const cellKey = `${rowIdx}-${colIdx}`;
                          processedCells.add(cellKey);

                          const pendingInfo = currentFilePending[rowIdx]?.[colIdx];

                          // Skip manual edits — they belong in the Manual Edits section, not here
                          if (pendingInfo && pendingInfo.wasManualEdit) return;

                          let status = 'error'; // default
                          let statusPrefix = '';
                          let bgColor = 'bg-[#DC3545]/10';
                          let borderColor = 'border-[#DC3545]';
                          let hoverColor = 'hover:bg-[#DC3545]/20';

                          if (pendingInfo) {
                            if (pendingInfo.type === 'marked-correct') {
                              status = 'marked-correct';
                              statusPrefix = '[Marked Correct] ';
                              // Different styles based on whether it passed data check
                              if (pendingInfo.passedDataCheck) {
                                bgColor = 'bg-white';
                                borderColor = 'border-2 border-[#50B461]';
                                hoverColor = 'hover:bg-[#50B461]/5';
                              } else {
                                bgColor = 'bg-[#50B461]/10';
                                borderColor = 'border-[#50B461]';
                                hoverColor = 'hover:bg-[#50B461]/20';
                              }
                            } else if (pendingInfo.type === 'corrected') {
                              status = 'pending';
                              statusPrefix = '[Pending] ';
                              bgColor = 'bg-[#B58E42]/10';
                              borderColor = 'border-[#B58E42]';
                              hoverColor = 'hover:bg-[#B58E42]/20';
                            }
                          }

                          issueList.push({
                            rowIdx: parseInt(rowIdx),
                            colIdx: parseInt(colIdx),
                            message: currentFileErrors[rowIdx][colIdx],
                            status,
                            statusPrefix,
                            bgColor,
                            borderColor,
                            hoverColor
                          });
                        });
                      });

                      // Then, collect any pending corrections that are NOT in fileErrors (marked-correct cells from system errors only)
                      // Only for columns that are currently selected
                      Object.keys(currentFilePending).forEach(rowIdx => {
                        Object.keys(currentFilePending[rowIdx]).forEach(colIdx => {
                          const cellKey = `${rowIdx}-${colIdx}`;
                          const colIndex = parseInt(colIdx);
                          if (!processedCells.has(cellKey)) {
                            const pendingInfo = currentFilePending[rowIdx][colIdx];
                            // Only include system-error marked-correct (NOT manual edits)
                            if (pendingInfo.type === 'marked-correct' && !pendingInfo.wasManualEdit && currentSelectedColIndices.includes(colIndex)) {
                              // Different styles based on whether it passed data check
                              let bgColor, borderColor, hoverColor;
                              if (pendingInfo.passedDataCheck) {
                                bgColor = 'bg-white';
                                borderColor = 'border-2 border-[#50B461]';
                                hoverColor = 'hover:bg-[#50B461]/5';
                              } else {
                                bgColor = 'bg-[#50B461]/10';
                                borderColor = 'border-[#50B461]';
                                hoverColor = 'hover:bg-[#50B461]/20';
                              }

                              issueList.push({
                                rowIdx: parseInt(rowIdx),
                                colIdx: colIndex,
                                message: pendingInfo.originalError || 'Marked as correct',
                                status: 'marked-correct',
                                statusPrefix: '[Marked Correct] ',
                                bgColor,
                                borderColor,
                                hoverColor
                              });
                            }
                          }
                        });
                      });

                      // Sort issues: first by column (left to right), then by row (top to bottom)
                      issueList.sort((a, b) => {
                        if (a.colIdx !== b.colIdx) {
                          return a.colIdx - b.colIdx; // Sort by column first
                        }
                        return a.rowIdx - b.rowIdx; // Then by row within same column
                      });

                      return issueList.length > 0 ? (
                        issueList.map((issue, index) => (
                          <div
                            key={`${issue.rowIdx}-${issue.colIdx}`}
                            className={`inline-flex items-center justify-center w-[32px] h-[32px] ${issue.bgColor} border ${issue.borderColor} rounded-full cursor-pointer ${issue.hoverColor} transition`}
                            onClick={() => {
                              // Single click: scroll to cell and highlight it
                              const cellRef = errorCellRefs.current[selectedFileId]?.[issue.rowIdx]?.[issue.colIdx];
                              if (cellRef && previewContainerRef.current) {
                                // Set focused cell for highlighting
                                setFocusedCell({ rowIdx: issue.rowIdx, colIdx: issue.colIdx });

                                // Scroll the cell into view
                                cellRef.scrollIntoView({
                                  behavior: 'smooth',
                                  block: 'center',
                                  inline: 'center'
                                });

                                // Remove highlight after 2 seconds
                                setTimeout(() => {
                                  setFocusedCell(null);
                                }, 2000);
                              }
                            }}
                            onDoubleClick={() => {
                              // Double click: open error modal
                              const selectedFileData = folders
                                .flatMap(f => f.files)
                                .find(f => f.id === selectedFileId)?.content;

                              if (selectedFileData) {
                                const cell = selectedFileData.rows[issue.rowIdx][issue.colIdx];
                                const pendingInfo = currentFilePending[issue.rowIdx]?.[issue.colIdx];
                                const isPending = !!pendingInfo;
                                const pendingType = pendingInfo?.type;
                                const displayValue = pendingInfo && pendingInfo.correctedValue !== undefined ? pendingInfo.correctedValue : cell;

                                setSelectedError({
                                  rowIdx: issue.rowIdx,
                                  colIdx: issue.colIdx,
                                  errorMessage: issue.message,
                                  currentValue: displayValue,
                                  originalValue: cell, // Store the original raw value
                                  isPending,
                                  pendingType,
                                  originalError: pendingInfo?.originalError || null,
                                  passedDataCheck: pendingInfo?.passedDataCheck || false
                                });
                                setCorrectedValue(displayValue || '');
                                setShowErrorModal(true);
                              }
                            }}
                          >
                            {issue.status === 'error' && (
                              <svg className="w-[16px] h-[16px] text-[#DC3545] flex-shrink-0" viewBox="0 0 24 24" fill="currentColor">
                                <circle cx="12" cy="12" r="10" opacity="0.2"/>
                                <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" fill="none"/>
                              </svg>
                            )}
                            {issue.status === 'marked-correct' && (
                              <svg className="w-[16px] h-[16px] text-[#50B461] flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <circle cx="12" cy="12" r="10" strokeWidth="2"/>
                                <path d="M9 12l2 2 4-4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                              </svg>
                            )}
                            {issue.status === 'pending' && (
                              <svg className="w-[16px] h-[16px] text-[#B58E42] flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <circle cx="12" cy="12" r="10" strokeWidth="2"/>
                                <path d="M12 6v6l4 2" strokeWidth="2" strokeLinecap="round"/>
                              </svg>
                            )}
                          </div>
                        ))
                      ) : (
                        <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E]">
                          No errors detected
                        </p>
                      );
                    })()
                  ) : (
                    <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E]">
                      No errors detected
                    </p>
                  )}
                    </div>
                  </div>
                </div>

                {/* Right: Manual Edits */}
                <div className="flex-1 min-w-0">
                  <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#13383B] mb-[12px]">
                    Manual Edits
                  </p>
                  <div className="overflow-x-auto pb-[8px]" style={{
                    scrollbarWidth: 'thin',
                    scrollbarColor: '#D1D5DB transparent'
                  }}>
                    <div className="flex gap-[8px] min-w-min">
                    {selectedFileId ? (
                      (() => {
                        const currentFilePending = filePendingCorrections[selectedFileId] || {};
                        const manualEditList = [];

                        // Get selected columns for current file
                        const currentSelectedCols = fileColumnLabels[selectedFileId] || {};
                        const currentSelectedColIndices = Object.keys(currentSelectedCols).map(k => parseInt(k));

                        // Collect manual edits (all types that originated from manual edits)
                        Object.keys(currentFilePending).forEach(rowIdx => {
                          Object.keys(currentFilePending[rowIdx]).forEach(colIdx => {
                            const colIndex = parseInt(colIdx);
                            const pendingInfo = currentFilePending[rowIdx][colIdx];

                            // Include: manual-edit (blue), verified manual-edit (green), failed manual-edit (yellow)
                            const isManualEdit = pendingInfo.type === 'manual-edit';
                            const isVerifiedManualEdit = pendingInfo.type === 'marked-correct' && pendingInfo.wasManualEdit;
                            const isFailedManualEdit = pendingInfo.type === 'corrected' && pendingInfo.wasManualEdit;

                            if ((isManualEdit || isVerifiedManualEdit || isFailedManualEdit) && currentSelectedColIndices.includes(colIndex)) {
                              let bgColor, borderColor, hoverColor, message;
                              if (isVerifiedManualEdit) {
                                bgColor = 'bg-[#50B461]/10';
                                borderColor = 'border-[#50B461]';
                                hoverColor = 'hover:bg-[#50B461]/20';
                                message = 'Manual edit (verified)';
                              } else if (isFailedManualEdit) {
                                bgColor = 'bg-[#B58E42]/10';
                                borderColor = 'border-[#B58E42]';
                                hoverColor = 'hover:bg-[#B58E42]/20';
                                message = 'Manual edit (needs correction)';
                              } else {
                                bgColor = 'bg-[#4D6EC7]/10';
                                borderColor = 'border-[#4D6EC7]';
                                hoverColor = 'hover:bg-[#4D6EC7]/20';
                                message = 'Manual edit';
                              }

                              manualEditList.push({
                                rowIdx: parseInt(rowIdx),
                                colIdx: colIndex,
                                message,
                                bgColor,
                                borderColor,
                                hoverColor,
                                isVerified: isVerifiedManualEdit,
                                isFailed: isFailedManualEdit
                              });
                            }
                          });
                        });

                        // Sort by column then row
                        manualEditList.sort((a, b) => {
                          if (a.colIdx !== b.colIdx) {
                            return a.colIdx - b.colIdx;
                          }
                          return a.rowIdx - b.rowIdx;
                        });

                        return manualEditList.length > 0 ? (
                          manualEditList.map((edit, index) => (
                            <div
                              key={`${edit.rowIdx}-${edit.colIdx}`}
                              className={`inline-flex items-center justify-center w-[32px] h-[32px] ${edit.bgColor} border ${edit.borderColor} rounded-full cursor-pointer ${edit.hoverColor} transition`}
                              onClick={() => {
                                // Single click: scroll to cell and highlight it
                                const cellRef = errorCellRefs.current[selectedFileId]?.[edit.rowIdx]?.[edit.colIdx];
                                if (cellRef && previewContainerRef.current) {
                                  setFocusedCell({ rowIdx: edit.rowIdx, colIdx: edit.colIdx });
                                  cellRef.scrollIntoView({
                                    behavior: 'smooth',
                                    block: 'center',
                                    inline: 'center'
                                  });
                                  setTimeout(() => {
                                    setFocusedCell(null);
                                  }, 2000);
                                }
                              }}
                              onDoubleClick={() => {
                                // Double click: open error modal
                                const selectedFileData = folders
                                  .flatMap(f => f.files)
                                  .find(f => f.id === selectedFileId)?.content;

                                if (selectedFileData) {
                                  const cell = selectedFileData.rows[edit.rowIdx][edit.colIdx];
                                  const pendingInfo = currentFilePending[edit.rowIdx]?.[edit.colIdx];

                                  setSelectedError({
                                    rowIdx: edit.rowIdx,
                                    colIdx: edit.colIdx,
                                    errorMessage: edit.message,
                                    currentValue: pendingInfo?.correctedValue !== undefined ? pendingInfo.correctedValue : cell,
                                    originalValue: cell,
                                    isPending: true,
                                    pendingType: pendingInfo?.type || 'manual-edit',
                                    originalError: null,
                                    isManualEdit: true,
                                    passedDataCheck: pendingInfo?.passedDataCheck || false,
                                    wasManualEdit: pendingInfo?.wasManualEdit || false
                                  });
                                  setCorrectedValue(pendingInfo?.correctedValue || cell || '');
                                  setShowErrorModal(true);
                                }
                              }}
                            >
                              {edit.isVerified ? (
                                <svg className="w-[16px] h-[16px] text-[#50B461] flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                  <circle cx="12" cy="12" r="10" strokeWidth="2"/>
                                  <path d="M8 12l3 3 5-5" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                                </svg>
                              ) : edit.isFailed ? (
                                <svg className="w-[16px] h-[16px] text-[#B58E42] flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                  <circle cx="12" cy="12" r="10" strokeWidth="2"/>
                                  <path d="M12 8v4m0 4h.01" strokeWidth="2" strokeLinecap="round"/>
                                </svg>
                              ) : (
                                <svg className="w-[16px] h-[16px] text-[#4D6EC7] flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                  <circle cx="12" cy="12" r="10" strokeWidth="2"/>
                                  <path d="M12 6v6l4 2" strokeWidth="2" strokeLinecap="round"/>
                                </svg>
                              )}
                            </div>
                          ))
                        ) : (
                          <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E]">
                            No manual edits
                          </p>
                        );
                      })()
                    ) : (
                      <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E]">
                        No manual edits
                      </p>
                    )}
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <p className="font-['Noto_Sans',sans-serif] font-normal text-[16px] text-[#7E7E7E]">
                Select a file from the left panel to preview the data
              </p>
            </div>
          )}

          {/* Bottom Action Buttons - Always at bottom, aligned with left panel */}
          <div className="absolute bottom-0 left-[24px] right-0 flex flex-col gap-[6px]">
            {/* Run Data Check Button */}
            <button
              onClick={hasPendingCorrections ? handleRunDataCheck : undefined}
              disabled={!hasPendingCorrections}
              className={`w-full rounded-[24.5px] h-[49px] flex items-center justify-center transition ${
                hasPendingCorrections
                  ? 'bg-[#29ABB5] cursor-pointer hover:bg-[#50B461]'
                  : 'bg-[#e3e3e3] cursor-not-allowed'
              }`}
            >
              <p className={`font-['Noto_Sans',sans-serif] font-bold text-[18px] ${
                hasPendingCorrections ? 'text-white' : 'text-[#7E7E7E]'
              }`}>
                Run Data Check
              </p>
            </button>

            {/* Cleaning Completed Button */}
            <button
              onClick={allFilesCompleted ? onComplete : undefined}
              disabled={!allFilesCompleted}
              className={`w-full rounded-[24.5px] h-[49px] flex items-center justify-center transition ${
                allFilesCompleted
                  ? 'bg-[#29ABB5] cursor-pointer hover:bg-[#50B461]'
                  : 'bg-[#e3e3e3] cursor-not-allowed'
              }`}
            >
              <p className={`font-['Noto_Sans',sans-serif] font-bold text-[18px] ${
                allFilesCompleted ? 'text-white' : 'text-[#7E7E7E]'
              }`}>
                Cleaning Completed
              </p>
            </button>
          </div>
        </div>
      </div>

      {/* Move All Files Modal */}
      {showMoveAllModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowMoveAllModal(null)}>
          <div className="bg-white rounded-[10px] p-[32px] w-[400px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)]" onClick={(e) => e.stopPropagation()}>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[18px] text-[#575757] mb-[16px]">
              Move All Files
            </p>
            <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] mb-[24px]">
              Select the destination folder:
            </p>
            <div className="flex flex-col gap-[8px] mb-[24px]">
              {folders.filter(f => f.id !== showMoveAllModal).map(folder => (
                <button
                  key={folder.id}
                  onClick={() => handleMoveAllFiles(showMoveAllModal, folder.tag)}
                  className="w-full px-[16px] py-[12px] bg-[#e3e3e3] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#d8d8d8] transition text-left"
                >
                  {folder.name} ({folder.tag})
                </button>
              ))}
            </div>
            <button
              onClick={() => setShowMoveAllModal(null)}
              className="w-full h-[40px] bg-[#e3e3e3] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#d8d8d8] transition"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Delete All Files Modal */}
      {showDeleteAllModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowDeleteAllModal(null)}>
          <div className="bg-white rounded-[10px] p-[32px] w-[400px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)]" onClick={(e) => e.stopPropagation()}>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[18px] text-[#575757] mb-[16px]">
              Delete All Files
            </p>
            <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] mb-[24px]">
              Are you sure you want to delete all files in this folder? This action cannot be undone.
            </p>
            <div className="flex gap-[12px]">
              <button
                onClick={() => setShowDeleteAllModal(null)}
                className="flex-1 h-[40px] bg-[#e3e3e3] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#d8d8d8] transition"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeleteAllFiles(showDeleteAllModal)}
                className="flex-1 h-[40px] bg-[#d32f2f] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-white hover:bg-[#b71c1c] transition"
              >
                Delete All
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Move File Modal */}
      {showMoveFileModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowMoveFileModal(null)}>
          <div className="bg-white rounded-[10px] p-[32px] w-[400px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)]" onClick={(e) => e.stopPropagation()}>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[18px] text-[#575757] mb-[16px]">
              Move File
            </p>
            <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] mb-[24px]">
              Select the destination folder:
            </p>
            <div className="flex flex-col gap-[8px] mb-[24px]">
              {folders.filter(f => f.id !== showMoveFileModal.folderId).map(folder => (
                <button
                  key={folder.id}
                  onClick={() => handleMoveFile(showMoveFileModal.fileId, showMoveFileModal.folderId, folder.tag)}
                  className="w-full px-[16px] py-[12px] bg-[#e3e3e3] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#d8d8d8] transition text-left"
                >
                  {folder.name} ({folder.tag})
                </button>
              ))}
            </div>
            <button
              onClick={() => setShowMoveFileModal(null)}
              className="w-full h-[40px] bg-[#e3e3e3] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#d8d8d8] transition"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Rename File Modal */}
      {showRenameModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => { setShowRenameModal(null); setRenameValue(''); }}>
          <div className="bg-white rounded-[10px] p-[32px] w-[400px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)]" onClick={(e) => e.stopPropagation()}>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[18px] text-[#575757] mb-[16px]">
              Rename File
            </p>
            <input
              type="text"
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              className="w-full border border-[#d9d9d9] rounded-[8px] px-[12px] py-[8px] font-['Noto_Sans',sans-serif] text-[14px] focus:outline-none focus:border-[#29ABB5] mb-[24px]"
              placeholder="Enter new file name"
            />
            <div className="flex gap-[12px]">
              <button
                onClick={() => {
                  setShowRenameModal(null);
                  setRenameValue('');
                }}
                className="flex-1 h-[40px] bg-[#e3e3e3] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#d8d8d8] transition"
              >
                Cancel
              </button>
              <button
                onClick={() => handleRenameFile(showRenameModal.fileId, showRenameModal.folderId)}
                className="flex-1 h-[40px] bg-[#29ABB5] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-white hover:bg-[#238d96] transition"
              >
                Rename
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete File Modal */}
      {showDeleteFileModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowDeleteFileModal(null)}>
          <div className="bg-white rounded-[10px] p-[32px] w-[400px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)]" onClick={(e) => e.stopPropagation()}>
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[18px] text-[#575757] mb-[16px]">
              Delete File
            </p>
            <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] mb-[24px]">
              Are you sure you want to delete this file? This action cannot be undone.
            </p>
            <div className="flex gap-[12px]">
              <button
                onClick={() => setShowDeleteFileModal(null)}
                className="flex-1 h-[40px] bg-[#e3e3e3] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#d8d8d8] transition"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeleteFile(showDeleteFileModal.fileId, showDeleteFileModal.folderId)}
                className="flex-1 h-[40px] bg-[#d32f2f] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-white hover:bg-[#b71c1c] transition"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Column Action Menu - Hover-triggered popup */}
      {showColumnActionMenu !== null && (
        <div
          data-column-action-menu="true"
          className="fixed z-[60] bg-white rounded-[8px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.15)] py-[8px] w-[160px]"
          style={{
            left: columnActionMenuPosition.x,
            top: columnActionMenuPosition.y + 4,
            transform: 'translateX(-50%)'
          }}
          onMouseLeave={() => setShowColumnActionMenu(null)}
        >
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleRenameColumnTitle(showColumnActionMenu);
              }}
              className="w-full px-[16px] py-[10px] text-left font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] hover:bg-[#F5F5F5] transition flex items-center gap-[8px]"
            >
              <svg className="w-[16px] h-[16px] text-[#7E7E7E]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
              </svg>
              Rename Title
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDeselectColumn(showColumnActionMenu);
              }}
              className="w-full px-[16px] py-[10px] text-left font-['Noto_Sans',sans-serif] text-[14px] text-[#d32f2f] hover:bg-[#F5F5F5] transition flex items-center gap-[8px]"
            >
              <svg className="w-[16px] h-[16px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
              Deselect
            </button>
        </div>
      )}

      {/* Data Title Modal */}
      {showDataTitleModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={handleCancelDataTitle}>
          <div className="bg-white rounded-[10px] p-[32px] w-[480px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)]" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div className="flex items-center justify-between mb-[24px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[20px] text-[#13383B]">
                Data Title
              </p>
              <button
                onClick={handleCancelDataTitle}
                className="w-[32px] h-[32px] flex items-center justify-center hover:bg-[#F5F5F5] rounded-full transition"
              >
                <svg className="w-[20px] h-[20px] text-[#7E7E7E]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            {/* Suggested Titles Section */}
            <div className="mb-[24px]">
              <div className="flex items-center justify-between mb-[12px]">
                <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#13383B]">
                  Suggested Titles
                </p>
                <button
                  onClick={handleRefreshSuggestions}
                  className="flex items-center gap-[4px] font-['Noto_Sans',sans-serif] text-[12px] text-[#29ABB5] hover:text-[#238d96] transition"
                >
                  <svg className="w-[14px] h-[14px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 4v6h-6" />
                    <path d="M1 20v-6h6" />
                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                  </svg>
                  More Suggestions
                </button>
              </div>

              <div className="flex flex-col gap-[8px]">
                {/* First row: Original and Current (if in rename mode) side by side */}
                {isRenameMode && currentTitleBeforeRename ? (
                  <>
                    {/* Original + Current in horizontal layout */}
                    <div className="flex gap-[8px]">
                      <button
                        onClick={() => {
                          setSelectedSuggestedTitle(suggestedTitles[0]);
                          setManualTitle('');
                        }}
                        className={`flex-1 px-[16px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-left transition border-2 ${
                          selectedSuggestedTitle === suggestedTitles[0]
                            ? 'border-[#29ABB5] bg-[#29ABB5]/10 text-[#13383B]'
                            : 'border-[#E0E0E0] bg-white text-[#575757] hover:border-[#29ABB5]/50'
                        }`}
                      >
                        <span className="flex items-center gap-[8px]">
                          <span className="text-[12px] text-[#7E7E7E] font-normal">Original:</span>
                          <span className="truncate">{suggestedTitles[0]}</span>
                          {selectedSuggestedTitle === suggestedTitles[0] && (
                            <CheckIcon className="w-[16px] h-[16px] text-[#29ABB5] ml-auto flex-shrink-0" />
                          )}
                        </span>
                      </button>
                      <button
                        onClick={() => {
                          setSelectedSuggestedTitle(currentTitleBeforeRename);
                          setManualTitle('');
                        }}
                        className={`flex-1 px-[16px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-left transition border-2 ${
                          selectedSuggestedTitle === currentTitleBeforeRename
                            ? 'border-[#29ABB5] bg-[#29ABB5]/10 text-[#13383B]'
                            : 'border-[#E0E0E0] bg-white text-[#575757] hover:border-[#29ABB5]/50'
                        }`}
                      >
                        <span className="flex items-center gap-[8px]">
                          <span className="text-[12px] text-[#7E7E7E] font-normal">Current:</span>
                          <span className="truncate">{currentTitleBeforeRename}</span>
                          {selectedSuggestedTitle === currentTitleBeforeRename && (
                            <CheckIcon className="w-[16px] h-[16px] text-[#29ABB5] ml-auto flex-shrink-0" />
                          )}
                        </span>
                      </button>
                    </div>
                    {/* Other suggestions */}
                    {suggestedTitles.slice(1).map((title, idx) => (
                      <button
                        key={idx + 1}
                        onClick={() => {
                          setSelectedSuggestedTitle(title);
                          setManualTitle('');
                        }}
                        className={`w-full px-[16px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-left transition border-2 ${
                          selectedSuggestedTitle === title
                            ? 'border-[#29ABB5] bg-[#29ABB5]/10 text-[#13383B]'
                            : 'border-[#E0E0E0] bg-white text-[#575757] hover:border-[#29ABB5]/50'
                        }`}
                      >
                        <span className="flex items-center gap-[8px]">
                          {title}
                          {selectedSuggestedTitle === title && (
                            <CheckIcon className="w-[16px] h-[16px] text-[#29ABB5] ml-auto" />
                          )}
                        </span>
                      </button>
                    ))}
                  </>
                ) : (
                  /* Normal mode: all suggestions in vertical layout */
                  suggestedTitles.map((title, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setSelectedSuggestedTitle(title);
                        setManualTitle('');
                      }}
                      className={`w-full px-[16px] py-[12px] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-left transition border-2 ${
                        selectedSuggestedTitle === title
                          ? 'border-[#29ABB5] bg-[#29ABB5]/10 text-[#13383B]'
                          : 'border-[#E0E0E0] bg-white text-[#575757] hover:border-[#29ABB5]/50'
                      }`}
                    >
                      <span className="flex items-center gap-[8px]">
                        {idx === 0 && (
                          <span className="text-[12px] text-[#7E7E7E] font-normal">Original:</span>
                        )}
                        {title}
                        {selectedSuggestedTitle === title && (
                          <CheckIcon className="w-[16px] h-[16px] text-[#29ABB5] ml-auto" />
                        )}
                      </span>
                    </button>
                  ))
                )}
              </div>
            </div>

            {/* Divider */}
            <div className="flex items-center gap-[12px] mb-[24px]">
              <div className="flex-1 h-[1px] bg-[#E0E0E0]"></div>
              <span className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E]">or</span>
              <div className="flex-1 h-[1px] bg-[#E0E0E0]"></div>
            </div>

            {/* Manual Input Section */}
            <div className="mb-[32px]">
              <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#13383B] mb-[12px]">
                Input Title Manually
              </p>
              <input
                type="text"
                value={manualTitle}
                onChange={(e) => {
                  setManualTitle(e.target.value);
                  if (e.target.value.trim()) {
                    setSelectedSuggestedTitle(null);
                  }
                }}
                onFocus={() => setSelectedSuggestedTitle(null)}
                className="w-full border-2 border-[#E0E0E0] rounded-[8px] px-[16px] py-[12px] font-['Noto_Sans',sans-serif] text-[14px] focus:outline-none focus:border-[#29ABB5] transition"
                placeholder="Enter your custom title..."
              />
            </div>

            {/* Action Buttons */}
            <div className="flex gap-[12px]">
              <button
                onClick={handleCancelDataTitle}
                className="flex-1 h-[44px] bg-[#E3E3E3] rounded-[8px] font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#575757] hover:bg-[#D8D8D8] transition"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDataTitle}
                disabled={!selectedSuggestedTitle && !manualTitle.trim()}
                className={`flex-1 h-[44px] rounded-[8px] font-['Noto_Sans',sans-serif] font-semibold text-[14px] transition ${
                  selectedSuggestedTitle || manualTitle.trim()
                    ? 'bg-[#29ABB5] text-white hover:bg-[#238d96] cursor-pointer'
                    : 'bg-[#E0E0E0] text-[#A0A0A0] cursor-not-allowed'
                }`}
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error Correction Modal */}
      {showErrorModal && selectedError && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={handleCancelCorrection}>
          <div className="bg-white rounded-[10px] p-[32px] w-[480px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)]" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div className="flex items-center justify-between mb-[24px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[20px] text-[#13383B]">
                {selectedError.isPending
                  ? (selectedError.pendingType === 'marked-correct'
                      ? (selectedError.passedDataCheck
                          ? 'Verified Correct Cell (Data Check)'
                          : 'Manually Marked Correct')
                      : 'Pending Verification')
                  : (selectedError.isManualEdit ? 'Manual Edit Undetected Error' : 'Invalid Value Detected')
                }
              </p>
              <button
                onClick={handleCancelCorrection}
                className="w-[32px] h-[32px] flex items-center justify-center hover:bg-[#F5F5F5] rounded-full transition"
              >
                <svg className="w-[20px] h-[20px] text-[#7E7E7E]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            {/* Status Section (for marked-correct) */}
            {selectedError.isPending && selectedError.pendingType === 'marked-correct' && (
              <div className="mb-[24px]">
                <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] mb-[12px] text-[#50B461]">
                  Status
                </p>
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#50B461] mb-[16px]">
                  {selectedError.passedDataCheck
                    ? 'This value has passed the data check and is verified as correct.'
                    : 'This value has been marked as correct.'}
                </p>
                {/* Show Unmark button only for manually marked correct (not data check verified) */}
                {!selectedError.passedDataCheck && (
                  <button
                    onClick={handleUnmarkAsCorrect}
                    className="w-full h-[44px] rounded-[8px] font-['Noto_Sans',sans-serif] font-semibold text-[14px] bg-white text-[#DC3545] border-2 border-[#DC3545] hover:bg-[#DC3545]/10 transition"
                  >
                    Unmark as Correct
                  </button>
                )}
              </div>
            )}

            {/* Status Section (for corrected - awaiting verification) */}
            {selectedError.isPending && selectedError.pendingType === 'corrected' && (
              <div className="mb-[24px]">
                <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] mb-[12px] text-[#B58E42]">
                  Status
                </p>
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#B58E42]">
                  This value has been corrected and is awaiting verification through data check.
                </p>
              </div>
            )}

            {/* Status Section (for manual-edit - awaiting verification) */}
            {selectedError.isPending && selectedError.pendingType === 'manual-edit' && (
              <div className="mb-[24px]">
                <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] mb-[12px] text-[#4D6EC7]">
                  Status
                </p>
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#4D6EC7]">
                  This value has been manually edited and is awaiting verification through data check.
                </p>
              </div>
            )}

            {/* Original Manual Edit (for manual-edit pending) */}
            {selectedError.isPending && selectedError.pendingType === 'manual-edit' && (
              <div className="mb-[24px]">
                <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] mb-[12px] text-[#7E7E7E]">
                  Original Manual Edit
                </p>
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">
                  <span className="font-semibold">Original Value: </span>
                  {selectedError.originalValue === '' || selectedError.originalValue === null || selectedError.originalValue === undefined ? (
                    <span className="italic">(empty)</span>
                  ) : selectedError.originalValue.toString().trim() === '' ? (
                    <span className="italic">(whitespace)</span>
                  ) : (
                    selectedError.originalValue
                  )}
                </p>
              </div>
            )}

            {/* Original Error (for pending) or Issue Description (for errors) */}
            {/* Don't show this section for manual-edit */}
            {(selectedError.originalError && selectedError.isPending && selectedError.pendingType !== 'manual-edit') ? (
              <div className="mb-[24px]">
                <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] mb-[12px] text-[#7E7E7E]">
                  Original System Detection
                </p>
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">
                  <span className="font-semibold">Original Value: </span>
                  {selectedError.originalValue === '' || selectedError.originalValue === null || selectedError.originalValue === undefined ? (
                    <span className="italic">(empty)</span>
                  ) : selectedError.originalValue.toString().trim() === '' ? (
                    <span className="italic">(whitespace)</span>
                  ) : (
                    selectedError.originalValue
                  )}
                  <br />
                  <span className="font-semibold">Error: </span>
                  {selectedError.originalError}
                </p>
              </div>
            ) : (!selectedError.isPending && !selectedError.isManualEdit) ? (
              <div className="mb-[24px]">
                <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] mb-[12px] text-[#DC3545]">
                  Issue Description
                </p>
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#DC3545]">
                  <span className="font-semibold">Original Value: </span>
                  {selectedError.originalValue === '' || selectedError.originalValue === null || selectedError.originalValue === undefined ? (
                    <span className="italic">(empty)</span>
                  ) : selectedError.originalValue.toString().trim() === '' ? (
                    <span className="italic">(whitespace)</span>
                  ) : (
                    selectedError.originalValue
                  )}
                  <br />
                  <span className="font-semibold">Error: </span>
                  {selectedError.errorMessage}
                </p>
              </div>
            ) : (selectedError.isManualEdit && !selectedError.isPending) ? (
              <div className="mb-[24px]">
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B]">
                  <span className="font-semibold">Original Value: </span>
                  {selectedError.originalValue === '' || selectedError.originalValue === null || selectedError.originalValue === undefined ? (
                    <span className="italic">(empty)</span>
                  ) : selectedError.originalValue.toString().trim() === '' ? (
                    <span className="italic">(whitespace)</span>
                  ) : (
                    selectedError.originalValue
                  )}
                </p>
              </div>
            ) : null}

            {/* Manual Input Section */}
            <div className="mb-[24px]">
              <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#13383B] mb-[12px]">
                {selectedError.isPending && selectedError.pendingType === 'marked-correct' && selectedError.passedDataCheck
                  ? 'Enter a Different Value'
                  : 'Enter Corrected Value'}
              </p>
              <input
                type="text"
                value={correctedValue}
                onChange={(e) => setCorrectedValue(e.target.value)}
                className="w-full border-2 border-[#E0E0E0] rounded-[8px] px-[16px] py-[12px] font-['Noto_Sans',sans-serif] text-[14px] focus:outline-none focus:border-[#29ABB5] transition"
                placeholder="Enter the correct value..."
                autoFocus
              />
            </div>

            {/* Divider and Mark as Correct Option (hide for marked-correct) */}
            {(!selectedError.isPending || selectedError.pendingType === 'corrected') && !selectedError.isManualEdit && (
              <>
                {/* Divider */}
                <div className="flex items-center gap-[12px] mb-[24px]">
                  <div className="flex-1 h-[1px] bg-[#E0E0E0]"></div>
                  <span className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E]">or</span>
                  <div className="flex-1 h-[1px] bg-[#E0E0E0]"></div>
                </div>

                {/* Mark as Correct Option */}
                <div className="mb-[32px]">
                  <button
                    onClick={handleMarkAsCorrectClick}
                    className="w-full h-[44px] rounded-[8px] font-['Noto_Sans',sans-serif] font-semibold text-[14px] bg-white text-[#50B461] border-2 border-[#50B461] hover:bg-[#50B461]/10 transition"
                  >
                    Mark as Correct
                  </button>
                </div>
              </>
            )}

            {/* Divider and Revert to Original Option (for any manual-edit origin) */}
            {selectedError.isManualEdit && selectedError.isPending && (selectedError.pendingType === 'manual-edit' || selectedError.wasManualEdit) && (
              <>
                {/* Divider */}
                <div className="flex items-center gap-[12px] mb-[24px]">
                  <div className="flex-1 h-[1px] bg-[#E0E0E0]"></div>
                  <span className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E]">or</span>
                  <div className="flex-1 h-[1px] bg-[#E0E0E0]"></div>
                </div>

                {/* Revert to Original Option */}
                <div className="mb-[32px]">
                  <button
                    onClick={handleRevertToOriginal}
                    className="w-full h-[44px] rounded-[8px] font-['Noto_Sans',sans-serif] font-semibold text-[14px] bg-white text-[#50B461] border-2 border-[#50B461] hover:bg-[#50B461]/10 transition"
                  >
                    Revert to Original Value
                  </button>
                </div>
              </>
            )}

            {/* Add spacing for marked-correct state */}
            {selectedError.isPending && selectedError.pendingType === 'marked-correct' && <div className="mb-[32px]"></div>}

            {/* Action Buttons */}
            <div className="flex gap-[12px]">
              <button
                onClick={handleCancelCorrection}
                className="flex-1 h-[44px] bg-[#E3E3E3] rounded-[8px] font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#575757] hover:bg-[#D8D8D8] transition"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmCorrection}
                className="flex-1 h-[44px] rounded-[8px] font-['Noto_Sans',sans-serif] font-semibold text-[14px] bg-[#29ABB5] text-white hover:bg-[#238d96] cursor-pointer transition"
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Mark as Correct Confirmation Modal */}
      {showMarkCorrectConfirmation && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[60]" onClick={handleCancelMarkAsCorrect}>
          <div className="bg-white rounded-[10px] p-[32px] w-[480px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)]" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div className="flex items-center justify-between mb-[24px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[20px] text-[#13383B]">
                Confirm Action
              </p>
              <button
                onClick={handleCancelMarkAsCorrect}
                className="w-[32px] h-[32px] flex items-center justify-center hover:bg-[#F5F5F5] rounded-full transition"
              >
                <svg className="w-[20px] h-[20px] text-[#7E7E7E]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            {/* Warning Message */}
            <div className="mb-[32px]">
              <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#13383B] leading-relaxed">
                Are you sure you want to mark this value as correct? This will remove the error flag, but{' '}
                <span className="font-semibold text-[#DC3545]">future calculations may be affected by potentially incorrect data</span>.
              </p>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-[12px]">
              <button
                onClick={handleCancelMarkAsCorrect}
                className="flex-1 h-[44px] bg-[#E3E3E3] rounded-[8px] font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#575757] hover:bg-[#D8D8D8] transition"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmMarkAsCorrect}
                className="flex-1 h-[44px] rounded-[8px] font-['Noto_Sans',sans-serif] font-semibold text-[14px] bg-[#50B461] text-white hover:bg-[#449954] cursor-pointer transition"
              >
                Yes, Mark as Correct
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tooltip for showing original value on hover */}
      {hoveredCell && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{
            left: `${hoveredCell.x}px`,
            top: `${hoveredCell.y}px`,
            transform: 'translate(-50%, -100%)'
          }}
        >
          <div className="bg-white rounded-[8px] shadow-[0px_4px_12px_0px_rgba(0,0,0,0.15)] px-[12px] py-[8px] border border-[#E0E0E0]">
            <div className="font-['Noto_Sans',sans-serif] text-[11px] text-[#7E7E7E] mb-[2px]">
              Original Value:
            </div>
            <div className="font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B] font-medium">
              {hoveredCell.originalValue === '' || hoveredCell.originalValue === null || hoveredCell.originalValue === undefined ? (
                <span className="italic text-[#7E7E7E]">(empty)</span>
              ) : hoveredCell.originalValue.toString().trim() === '' ? (
                <span className="italic text-[#7E7E7E]">(whitespace)</span>
              ) : (
                hoveredCell.originalValue
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
