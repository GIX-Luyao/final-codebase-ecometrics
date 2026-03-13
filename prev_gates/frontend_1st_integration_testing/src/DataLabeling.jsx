import React, { useState, useRef, useEffect } from 'react';
import { updateData } from './services/api';

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

export default function DataLabeling({
  folders,
  setFolders,
  fileStatuses,
  setFileStatuses,
  fileColumnLabels,
  setFileColumnLabels,
  onBack,
  onComplete
}) {
  const [selectedFileId, setSelectedFileId] = useState(null);

  // API integration states
  const [sessionId, setSessionId] = useState(null);
  const [apiLoading, setApiLoading] = useState(false);
  const [apiError, setApiError] = useState(null);

  // Initialize file statuses and ensure only selected file is Processing
  React.useEffect(() => {
    const newStatuses = { ...fileStatuses };
    let hasChanges = false;
    folders.forEach(folder => {
      folder.files.forEach(file => {
        if (!newStatuses[file.id]) {
          // New file - set to Pending
          newStatuses[file.id] = 'Pending';
          hasChanges = true;
        } else if (newStatuses[file.id] === 'Processing' && file.id !== selectedFileId) {
          // File is Processing but not selected - reset to Pending
          newStatuses[file.id] = 'Pending';
          hasChanges = true;
        }
      });
    });
    if (hasChanges) {
      setFileStatuses(newStatuses);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [folders, selectedFileId]);
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
  const handleConfirmDataTitle = async () => {
    if (pendingColumnIndex === null) return;

    const finalTitle = manualTitle.trim() || selectedSuggestedTitle;
    if (!finalTitle) return;

    // Get the original column header name
    const originalHeader = selectedFileData?.headers[pendingColumnIndex] || `Column ${pendingColumnIndex + 1}`;

    // Only call API if the title is different from the original header
    if (finalTitle !== originalHeader) {
      setApiLoading(true);
      setApiError(null);

      try {
        // Call backend API to save the column rename
        const response = await updateData(
          'fleetfuel',
          [{
            changeType: 'column_rename',
            oldName: originalHeader,
            newName: finalTitle
          }],
          sessionId
        );

        if (response.success) {
          // Store sessionId for subsequent calls
          setSessionId(response.sessionId);
          console.log('Column rename saved successfully:', response);
        } else {
          // Handle validation or other errors
          setApiError(response.error?.message || 'Failed to save changes');
          console.error('API Error:', response.error);
          setApiLoading(false);
          return; // Don't update local state if API failed
        }
      } catch (error) {
        setApiError(error.message || 'Network error');
        console.error('Network Error:', error);
        setApiLoading(false);
        return; // Don't update local state if API failed
      }

      setApiLoading(false);
    }

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

  // Check if all files are completed
  const allFilesCompleted = allFileIds.length > 0 && allFileIds.every(id => fileStatuses[id] === 'Completed');

  // Handle zoom
  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 10, 200));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 10, 50));
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

      {/* Progress Steps - 5 steps */}
      <div className="absolute left-[64px] right-[64px] top-[81px] flex items-center">
        {/* Step 1 - Completed (Clickable to go back) */}
        <div
          className="flex items-center gap-[8px] cursor-pointer hover:opacity-80 transition"
          onClick={onBack}
        >
          <div className="w-[32px] h-[32px] rounded-full bg-[#50B461] flex items-center justify-center flex-shrink-0">
            <CheckIcon className="w-[16px] h-[16px] text-white" />
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Upload Data
          </p>
        </div>
        <div className="flex-1 h-[2px] mx-[12px]" style={{ background: 'linear-gradient(to right, #50B461, #29ABB5)' }}></div>

        {/* Step 2 - Active */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-[#29ABB5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">2</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Select Data
          </p>
        </div>
        <div className="flex-1 h-[2px] bg-[#ACC2C5] mx-[12px]"></div>

        {/* Step 3 - Inactive */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-white border-2 border-[#ACC2C5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#7E7E7E]">3</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] whitespace-nowrap">
            Data Cleaning
          </p>
        </div>
        <div className="flex-1 h-[2px] bg-[#ACC2C5] mx-[12px]"></div>

        {/* Step 4 - Inactive */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-white border-2 border-[#ACC2C5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#7E7E7E]">4</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] whitespace-nowrap">
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
            style={{ height: 'calc(100% - 80px)' }}
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
                        <div className={`px-[8px] py-[2px] rounded-[4px] ${getStatusColor(fileStatuses[file.id] || 'Pending')}`}>
                          <p className="font-['Noto_Sans',sans-serif] font-normal text-[12px]">
                            {fileStatuses[file.id] || 'Pending'}
                          </p>
                        </div>
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
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B] mb-[16px]">
            Select Data
          </p>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] mb-[24px]">
            Select a file from the left panel to preview. Click on column headers to mark the data you need.
          </p>

          {selectedFileId ? (
            <>
              {/* Table Preview Wrapper - relative container for fixed zoom controls */}
              <div className="relative flex-shrink-0" style={{ height: 'calc((100% - 80px - 49px - 16px) * 0.8)' }}>
                {/* Scrollable Table Container */}
                <div
                  ref={previewContainerRef}
                  className="h-full bg-[#F5F5F5] overflow-auto border border-[#E0E0E0]"
                  onMouseMove={(e) => {
                    // If menu is showing from tag click, check if mouse is over that column or the menu
                    if (showColumnActionMenu !== null && tableRef.current) {
                      // Check if mouse is over the action menu
                      const menuElement = document.querySelector('[data-column-action-menu]');
                      if (menuElement) {
                        const menuRect = menuElement.getBoundingClientRect();
                        if (
                          e.clientX >= menuRect.left &&
                          e.clientX <= menuRect.right &&
                          e.clientY >= menuRect.top &&
                          e.clientY <= menuRect.bottom
                        ) {
                          // Mouse is over the menu, don't hide
                          return;
                        }
                      }

                      const headerCell = tableRef.current.querySelector(`thead th:nth-child(${showColumnActionMenu + 1})`);
                      if (headerCell) {
                        const cellRect = headerCell.getBoundingClientRect();

                        // Check if mouse is within the column's horizontal bounds
                        if (e.clientX < cellRect.left || e.clientX > cellRect.right) {
                          // Mouse moved to a different column area, hide menu
                          setShowColumnActionMenu(null);
                          setHoveredColumn(null);
                        }
                      }
                    }
                  }}
                  onMouseLeave={(e) => {
                    // Check if mouse is moving to the action menu
                    const relatedTarget = e.relatedTarget;
                    if (relatedTarget && relatedTarget.closest && relatedTarget.closest('[data-column-action-menu]')) {
                      return; // Don't close if moving to menu
                    }
                    // When mouse leaves the preview container, hide menu and hover
                    if (showColumnActionMenu !== null) {
                      setShowColumnActionMenu(null);
                      setHoveredColumn(null);
                    }
                  }}
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
                                  onMouseEnter={() => setHoveredColumn(idx)}
                                  onMouseLeave={() => setHoveredColumn(null)}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleColumnToggle(idx, e);
                                  }}
                                  className={`px-[16px] py-[12px] min-w-[120px] cursor-pointer border border-[#EEEEEE] relative ${
                                    selectedColumns[idx]
                                      ? 'bg-[#29ABB5] text-white'
                                      : 'bg-[#F5F5F5] text-[#13383B]'
                                  }`}
                                >
                                  <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px]">
                                    {header}
                                  </p>
                                  <p
                                    className="font-['Noto_Sans',sans-serif] font-normal text-[12px] mt-[4px]"
                                    style={{
                                      visibility: selectedColumns[idx] ? 'visible' : 'hidden',
                                      opacity: selectedColumns[idx] ? 0.8 : 0
                                    }}
                                  >
                                    ✓ {typeof selectedColumns[idx] === 'string' ? selectedColumns[idx] : 'Selected'}
                                  </p>
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {selectedFileData.rows.map((row, rowIdx) => (
                              <tr key={rowIdx}>
                                {row.map((cell, cellIdx) => (
                                  <td
                                    key={cellIdx}
                                    onMouseEnter={() => setHoveredColumn(cellIdx)}
                                    onMouseLeave={() => setHoveredColumn(null)}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleColumnToggle(cellIdx, e);
                                    }}
                                    className={`px-[16px] py-[10px] font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] cursor-pointer border border-[#EEEEEE] ${
                                      selectedColumns[cellIdx] ? 'bg-[#29ABB5]/10' : 'bg-white'
                                    }`}
                                  >
                                    {cell}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>

                        {/* Column Hover Overlay - Only show for unselected columns on hover */}
                        {hoveredColumn !== null && columnPositions[hoveredColumn] && !selectedColumns[hoveredColumn] && (
                          <div
                            className="absolute z-20"
                            style={{
                              left: columnPositions[hoveredColumn].left,
                              top: 0,
                              width: columnPositions[hoveredColumn].width,
                              height: columnPositions[hoveredColumn].height,
                              border: '2px solid #29ABB5',
                              backgroundColor: 'rgba(41, 171, 181, 0.05)',
                              boxSizing: 'border-box',
                              pointerEvents: 'none'
                            }}
                          >
                            {/* Add icon in center of column overlay */}
                            <div
                              className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 cursor-pointer"
                              style={{ pointerEvents: 'auto' }}
                              onMouseEnter={() => setHoveredColumn(hoveredColumn)}
                              onClick={(e) => {
                                e.stopPropagation();
                                handleColumnToggle(hoveredColumn, e);
                              }}
                            >
                              <div className="w-[40px] h-[40px] bg-[#29ABB5] rounded-full flex items-center justify-center shadow-[0px_4px_12px_0px_rgba(0,0,0,0.25)]">
                                <svg className="w-[20px] h-[20px] text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                                  <line x1="12" y1="5" x2="12" y2="19" />
                                  <line x1="5" y1="12" x2="19" y2="12" />
                                </svg>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* Selected Column Hover Overlay - Only show action menu trigger on hover */}
                        {hoveredColumn !== null && columnPositions[hoveredColumn] && selectedColumns[hoveredColumn] && (
                          <div
                            className="absolute z-20"
                            style={{
                              left: columnPositions[hoveredColumn].left,
                              top: 0,
                              width: columnPositions[hoveredColumn].width,
                              height: columnPositions[hoveredColumn].height,
                              border: '2px solid #29ABB5',
                              backgroundColor: 'transparent',
                              boxSizing: 'border-box',
                              pointerEvents: 'auto',
                              cursor: 'pointer'
                            }}
                            onMouseEnter={() => {
                              setHoveredColumn(hoveredColumn);
                              handleColumnHeaderHover(hoveredColumn);
                            }}
                            onMouseLeave={(e) => {
                              handleColumnHeaderLeave(e);
                            }}
                          />
                        )}
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

              {/* Named Column Selected Section */}
              <div className="mt-[24px]">
                <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#13383B] mb-[12px]">
                  Named Column Selected
                </p>
                <div className="flex flex-wrap gap-[8px]">
                  {Object.keys(selectedColumns).length > 0 ? (
                    Object.entries(selectedColumns).map(([colIndex, title]) => (
                      <div
                        key={colIndex}
                        onClick={() => handleColumnTagClick(parseInt(colIndex))}
                        className="inline-flex items-center gap-[6px] px-[12px] py-[6px] bg-[#29ABB5]/10 border border-[#29ABB5] rounded-[16px] cursor-pointer hover:bg-[#29ABB5]/20 transition"
                      >
                        <span className="font-['Noto_Sans',sans-serif] text-[13px] text-[#13383B]">
                          {title}
                        </span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeselectColumn(parseInt(colIndex));
                          }}
                          className="w-[16px] h-[16px] flex items-center justify-center hover:bg-[#29ABB5]/30 rounded-full transition"
                        >
                          <svg className="w-[10px] h-[10px] text-[#7E7E7E]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                            <line x1="18" y1="6" x2="6" y2="18" />
                            <line x1="6" y1="6" x2="18" y2="18" />
                          </svg>
                        </button>
                      </div>
                    ))
                  ) : (
                    <p className="font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E]">
                      No columns selected yet
                    </p>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <p className="font-['Noto_Sans',sans-serif] font-normal text-[16px] text-[#7E7E7E]">
                Select a file from the left panel to preview and select columns
              </p>
            </div>
          )}

          {/* Selection Completed Button - Always at bottom, aligned with left panel */}
          <button
            onClick={allFilesCompleted ? onComplete : undefined}
            disabled={!allFilesCompleted}
            className={`absolute bottom-0 left-[24px] right-0 rounded-[24.5px] h-[49px] flex items-center justify-center transition ${
              allFilesCompleted
                ? 'bg-[#29ABB5] cursor-pointer hover:bg-[#50B461]'
                : 'bg-[#e3e3e3] cursor-not-allowed'
            }`}
          >
            <p className={`font-['Noto_Sans',sans-serif] font-bold text-[18px] ${
              allFilesCompleted ? 'text-white' : 'text-[#7E7E7E]'
            }`}>
              Selection Completed
            </p>
          </button>
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

            {/* API Error Display */}
            {apiError && (
              <div className="mb-[16px] p-[12px] bg-[#FFEBEE] border border-[#EF5350] rounded-[8px]">
                <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#C62828]">
                  {apiError}
                </p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-[12px]">
              <button
                onClick={handleCancelDataTitle}
                disabled={apiLoading}
                className="flex-1 h-[44px] bg-[#E3E3E3] rounded-[8px] font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#575757] hover:bg-[#D8D8D8] transition disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDataTitle}
                disabled={(!selectedSuggestedTitle && !manualTitle.trim()) || apiLoading}
                className={`flex-1 h-[44px] rounded-[8px] font-['Noto_Sans',sans-serif] font-semibold text-[14px] transition ${
                  (selectedSuggestedTitle || manualTitle.trim()) && !apiLoading
                    ? 'bg-[#29ABB5] text-white hover:bg-[#238d96] cursor-pointer'
                    : 'bg-[#E0E0E0] text-[#A0A0A0] cursor-not-allowed'
                }`}
              >
                {apiLoading ? 'Saving...' : 'Confirm'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
