import React, { useState, useEffect } from 'react';
import * as XLSX from 'xlsx';

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

function AddCircleIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <circle cx="12" cy="12" r="10" strokeWidth="2"/>
      <line x1="12" y1="8" x2="12" y2="16" strokeWidth="2" strokeLinecap="round"/>
      <line x1="8" y1="12" x2="16" y2="12" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

function MonitorUploadIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 48 48" fill="none" stroke="currentColor">
      <rect x="6" y="8" width="36" height="24" rx="2" strokeWidth="2"/>
      <line x1="24" y1="32" x2="24" y2="40" strokeWidth="2"/>
      <line x1="16" y1="40" x2="32" y2="40" strokeWidth="2"/>
      <path d="M24 16V24M24 16L20 20M24 16L28 20" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function SharePointIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 64 64" fill="currentColor">
      <path d="M32,1A31,31,0,1,0,63,32,31.035,31.035,0,0,0,32,1Zm0,60c-5.008,0-9.7-5.054-12.468-13H44.468C41.705,55.946,37.008,61,32,61ZM23,46H18.9q-.282-.977-.527-2H45.63q-.245,1.023-.527,2ZM3.025,33h10.7A5.93,5.93,0,0,0,11,38a6.006,6.006,0,0,0,5.307,5.957Q16.544,45,16.817,46H6.614A28.8,28.8,0,0,1,3.025,33ZM32,3c4.23,0,8.078,3.466,10.819,9H21.191C23.977,6.393,27.861,3,32,3ZM43.723,14a44.52,44.52,0,0,1,2.945,11.977A8.968,8.968,0,0,0,40,23a8.874,8.874,0,0,0-2.883.479,11.972,11.972,0,0,0-18.35-5.032A38.594,38.594,0,0,1,20.278,14ZM47,42H36V36h2a1,1,0,0,0,.759-1.65l-6-7a1.031,1.031,0,0,0-1.518,0l-6,7A1,1,0,0,0,26,36h2v6H17a3.99,3.99,0,0,1-.6-7.936,1,1,0,0,0,.737-1.451A9.87,9.87,0,0,1,16,28a10,10,0,0,1,19.567-2.907,1,1,0,0,0,1.393.61A6.92,6.92,0,0,1,40,25a6.946,6.946,0,0,1,6.926,7.958,1,1,0,0,0,.808,1.118A4,4,0,0,1,47,42ZM29,34h-.826L32,29.536,35.826,34H35a1,1,0,0,0-1,1v7H30V35A1,1,0,0,0,29,34ZM16.023,21.343A11.878,11.878,0,0,0,14.392,31H3.025A28.847,28.847,0,0,1,9.288,14h8.833A44.651,44.651,0,0,0,16.023,21.343Zm31.67,22.614A5.979,5.979,0,0,0,50.276,33h10.7a28.8,28.8,0,0,1-3.589,13h-10.2Q47.457,45,47.693,43.957ZM48.987,31A49.954,49.954,0,0,0,45.9,14h8.815a28.847,28.847,0,0,1,6.263,17ZM52.97,12H45.057a23.328,23.328,0,0,0-5.752-8.067A29.019,29.019,0,0,1,52.97,12ZM24.7,3.932A23.094,23.094,0,0,0,18.959,12H11.03A29.013,29.013,0,0,1,24.7,3.932ZM7.829,48h9.586c1.742,5.338,4.294,9.506,7.319,12.078A29.1,29.1,0,0,1,7.829,48ZM39.266,60.078c3.025-2.572,5.577-6.74,7.319-12.078h9.586A29.1,29.1,0,0,1,39.266,60.078Z"/>
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

function CloseIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor">
      <line x1="18" y1="6" x2="6" y2="18" strokeWidth="2" strokeLinecap="round"/>
      <line x1="6" y1="6" x2="18" y2="18" strokeWidth="2" strokeLinecap="round"/>
    </svg>
  );
}

export default function Input({ folders, setFolders, onComplete }) {
  // Local state management
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileTag, setFileTag] = useState('');
  const [fileName, setFileName] = useState('');

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

  // Drag and drop states
  const [draggedFile, setDraggedFile] = useState(null);
  const [dragOverFolder, setDragOverFolder] = useState(null);

  // Parse CSV content with better handling
  const parseCSV = (text) => {
    // Remove BOM if present
    let cleanText = text;
    if (cleanText.charCodeAt(0) === 0xFEFF) {
      cleanText = cleanText.slice(1);
    }

    // Normalize line endings and split
    const lines = cleanText.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trim().split('\n');
    if (lines.length === 0) return { headers: [], rows: [] };

    // Helper function to parse a CSV line (handles quoted fields with commas)
    const parseLine = (line) => {
      const result = [];
      let current = '';
      let inQuotes = false;

      for (let i = 0; i < line.length; i++) {
        const char = line[i];

        if (char === '"') {
          if (inQuotes && line[i + 1] === '"') {
            // Escaped quote
            current += '"';
            i++;
          } else {
            // Toggle quote mode
            inQuotes = !inQuotes;
          }
        } else if (char === ',' && !inQuotes) {
          result.push(current.trim());
          current = '';
        } else {
          current += char;
        }
      }
      result.push(current.trim());

      return result;
    };

    // Parse headers
    const headers = parseLine(lines[0]);

    // Parse rows
    const rows = lines.slice(1)
      .filter(line => line.trim()) // Skip empty lines
      .map(line => parseLine(line));

    return { headers, rows };
  };

  // Handle file selection
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setFileName(file.name.replace(/\.[^/.]+$/, '')); // Remove extension
      setShowUploadModal(true);
    }
    // Reset input value to allow selecting the same file again
    e.target.value = '';
  };

  // Parse Excel file
  const parseExcel = (arrayBuffer) => {
    try {
      const workbook = XLSX.read(arrayBuffer, { type: 'array' });
      const firstSheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[firstSheetName];

      // Convert to array of arrays
      const data = XLSX.utils.sheet_to_json(worksheet, { header: 1 });

      if (data.length === 0) return { headers: [], rows: [] };

      // First row is headers
      const headers = data[0].map(h => String(h || ''));

      // Rest are data rows
      const rows = data.slice(1)
        .filter(row => row.some(cell => cell !== null && cell !== undefined && cell !== ''))
        .map(row => row.map(cell => String(cell || '')));

      return { headers, rows };
    } catch (error) {
      console.error('Error parsing Excel file:', error);
      alert('Error parsing Excel file. Please make sure the file is a valid Excel format.');
      return { headers: [], rows: [] };
    }
  };

  // Handle upload confirmation
  const handleUploadConfirm = () => {
    if (!fileTag || !fileName) {
      alert('Please select a tag and provide a file name');
      return;
    }

    const reader = new FileReader();
    const fileExtension = selectedFile.name.split('.').pop().toLowerCase();

    reader.onload = (e) => {
      try {
        let parsedData;

        if (fileExtension === 'xlsx' || fileExtension === 'xls') {
          // Parse Excel file
          parsedData = parseExcel(e.target.result);
        } else {
          // Parse CSV file
          parsedData = parseCSV(e.target.result);
        }

        // Check if parsing was successful
        if (!parsedData || parsedData.headers.length === 0) {
          alert('The file appears to be empty or could not be parsed.');
          return;
        }

        const newFile = {
          id: Date.now(),
          name: fileName,
          originalName: selectedFile.name,
          content: parsedData,
        };

        setFolders(prev => prev.map(folder => {
          if (folder.tag === fileTag) {
            return {
              ...folder,
              files: [...folder.files, newFile]
            };
          }
          return folder;
        }));

        // Reset states
        setShowUploadModal(false);
        setSelectedFile(null);
        setFileTag('');
        setFileName('');
      } catch (error) {
        console.error('Error processing file:', error);
        alert('Error processing file. Please try again.');
      }
    };

    reader.onerror = () => {
      console.error('Error reading file');
      alert('Error reading file. Please try again.');
    };

    // Use different read method based on file type
    if (fileExtension === 'xlsx' || fileExtension === 'xls') {
      reader.readAsArrayBuffer(selectedFile);
    } else {
      reader.readAsText(selectedFile);
    }
  };

  // Handle upload cancel
  const handleUploadCancel = () => {
    setShowUploadModal(false);
    setSelectedFile(null);
    setFileTag('');
    setFileName('');
  };

  // Folder operations
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
    setFolders(prev => prev.map(folder => {
      if (folder.id === folderId) {
        return { ...folder, files: [] };
      }
      return folder;
    }));
    setShowDeleteAllModal(null);
    setFolderMenuOpen(null);
  };

  // File operations
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

  // Close menus when clicking outside
  useEffect(() => {
    const handleClickOutside = () => {
      setFolderMenuOpen(null);
      setFileMenuOpen(null);
    };

    if (folderMenuOpen || fileMenuOpen) {
      document.addEventListener('click', handleClickOutside);
      return () => {
        document.removeEventListener('click', handleClickOutside);
      };
    }
  }, [folderMenuOpen, fileMenuOpen]);

  return (
    <div className="bg-white fixed inset-0 overflow-hidden" data-name="INPUT">
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

      {/* Progress Steps */}
      <div className="absolute left-[64px] right-[64px] top-[81px] flex items-center">
        {/* Step 1 - Active */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-[#29ABB5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-white">1</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B] whitespace-nowrap">
            Upload Data
          </p>
        </div>
        <div className="flex-1 h-[2px] bg-[#ACC2C5] mx-[12px]"></div>

        {/* Step 2 - Inactive */}
        <div className="flex items-center gap-[8px]">
          <div className="w-[32px] h-[32px] rounded-full bg-white border-2 border-[#ACC2C5] flex items-center justify-center flex-shrink-0">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[14px] text-[#7E7E7E]">2</p>
          </div>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] whitespace-nowrap">
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
        {/* Left Column - Raw Data Staging Area */}
        <div className="w-[58%] h-full bg-[#F5F5F5] rounded-[10px] p-[24px] relative">
          <div className="flex items-center justify-between mb-[24px]">
            <p className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B]">
              Raw Data staging area
            </p>
          </div>

          {/* Scrollable list container */}
          <div className="overflow-y-auto pr-[8px]" style={{ height: 'calc(100% - 80px)' }}>
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
                      className={`bg-white h-[49px] flex items-center pl-[17px] pr-[17px] justify-between border-t border-[#EEEEEE] relative hover:bg-[#29ABB5]/10 transition cursor-grab active:cursor-grabbing ${
                        draggedFile?.file.id === file.id ? 'opacity-50' : ''
                      }`}
                    >
                      <div className="flex items-center gap-[12px] pl-[32px]">
                        <DataTableIcon className="w-[24px] h-[24px] text-[#7E7E7E]" />
                        <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#13383B]">
                          {file.name}
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
                              onClick={() => {
                                setShowMoveFileModal({ fileId: file.id, folderId: folder.id });
                                setFileMenuOpen(null);
                              }}
                              className="w-full px-[16px] py-[8px] text-left font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#f5f5f5] transition"
                            >
                              Move to
                            </button>
                            <button
                              onClick={() => {
                                setShowRenameModal({ fileId: file.id, folderId: folder.id });
                                setRenameValue(file.name);
                                setFileMenuOpen(null);
                              }}
                              className="w-full px-[16px] py-[8px] text-left font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#f5f5f5] transition"
                            >
                              Rename
                            </button>
                            <button
                              onClick={() => {
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

        {/* Right Column - Upload Area */}
        <div className="flex-1 h-full pt-[24px] pb-[24px] pl-[24px] relative">
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B] mb-[16px]">
            Upload Your Data
          </p>
          <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E] mb-[36px]">
            Once uploaded, the data file will appear in the Raw Data staging area on the left where you can edit it
          </p>

          {/* Upload Cards */}
          <div className="grid grid-cols-2 gap-[12px] mb-[24px]">
            <label className="h-[213px] bg-white border-2 border-[#29ABB5] rounded-[10px] shadow-[0px_2px_4px_0px_rgba(0,0,0,0.1)] cursor-pointer hover:bg-[#f0f9fa] transition">
              <input
                type="file"
                className="hidden"
                onChange={handleFileSelect}
                accept=".csv,.xlsx,.xls"
              />
              <div className="h-[176px] flex items-center justify-center">
                <MonitorUploadIcon className="w-[48px] h-[48px] text-[#29ABB5]" />
              </div>
              <div className="bg-[#29ABB5] h-[37px] rounded-bl-[10px] rounded-br-[10px] flex items-center justify-center">
                <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-white">
                  Upload from Local
                </p>
              </div>
            </label>

            <div className="h-[213px] bg-[#EEEEEE] rounded-[10px] shadow-[0px_2px_4px_0px_rgba(0,0,0,0.15)] opacity-50 cursor-not-allowed">
              <div className="h-[176px] flex items-center justify-center">
                <SharePointIcon className="w-[44px] h-[44px] text-[#7E7E7E]" />
              </div>
              <div className="bg-white h-[37px] rounded-bl-[10px] rounded-br-[10px] flex items-center justify-center">
                <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-[#7E7E7E]">
                  Upload SharePoint Folder
                </p>
              </div>
            </div>
          </div>

          {/* Note */}
          <div className="mt-[24px]">
            <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-black mb-[8px]">
              Note:
            </p>
            <p className="font-['Noto_Sans',sans-serif] font-normal text-[14px] text-black">
              You may upload folders (e.g., separating consumption data, GWP, and EFID) to help the system classify different data types. <span className="italic underline">However, the files inside each folder must be in CSV or Excel format.</span>
            </p>
          </div>

          {/* Upload Completed Button */}
          <div className="absolute bottom-0 left-[24px] right-0">
            <div
              onClick={() => onComplete && onComplete()}
              className="bg-[#29ABB5] rounded-[24.5px] h-[49px] w-full flex items-center justify-center cursor-pointer hover:bg-[#50B461] transition"
            >
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[18px] text-white">
                Upload Completed
              </p>
            </div>
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

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={handleUploadCancel}>
          <div className="bg-white rounded-[10px] p-[32px] w-[500px] shadow-[0px_4px_16px_0px_rgba(0,0,0,0.25)]" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-[24px]">
              <p className="font-['Noto_Sans',sans-serif] font-bold text-[20px] text-[#575757]">
                Upload File
              </p>
              <CloseIcon
                className="w-[24px] h-[24px] text-[#575757] cursor-pointer"
                onClick={handleUploadCancel}
              />
            </div>

            {/* File name input */}
            <div className="mb-[24px]">
              <label className="font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#575757] block mb-[8px]">
                File Name <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                value={fileName}
                onChange={(e) => setFileName(e.target.value)}
                className="w-full border border-[#d9d9d9] rounded-[8px] px-[12px] py-[8px] font-['Noto_Sans',sans-serif] text-[14px] focus:outline-none focus:border-[#b0b0b0]"
                placeholder="Enter file name"
              />
            </div>

            {/* Tag selection */}
            <div className="mb-[32px]">
              <label className="font-['Noto_Sans',sans-serif] font-medium text-[14px] text-[#575757] block mb-[8px]">
                Select Tag <span className="text-red-500">*</span>
              </label>
              <div className="flex gap-[12px]">
                {['#Consumption', '#Reference'].map((tag) => (
                  <button
                    key={tag}
                    onClick={() => setFileTag(tag)}
                    className={`flex-1 h-[40px] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] transition ${
                      fileTag === tag
                        ? 'bg-[#29ABB5]/10 text-[#365D60] border-2 border-[#29ABB5]'
                        : 'bg-[#e3e3e3] text-[#575757] hover:bg-[#d8d8d8] border-2 border-transparent'
                    }`}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            </div>

            {/* Action buttons */}
            <div className="flex gap-[12px]">
              <button
                onClick={handleUploadCancel}
                className="flex-1 h-[40px] bg-[#e3e3e3] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] text-[#575757] hover:bg-[#d8d8d8] transition"
              >
                Cancel
              </button>
              <button
                onClick={handleUploadConfirm}
                disabled={!fileTag}
                className={`flex-1 h-[40px] rounded-[8px] font-['Noto_Sans',sans-serif] text-[14px] transition ${
                  fileTag
                    ? 'bg-[#29ABB5] text-white hover:bg-[#238d96] cursor-pointer'
                    : 'bg-[#e3e3e3] text-[#7E7E7E] cursor-not-allowed'
                }`}
              >
                Confirm Upload
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
