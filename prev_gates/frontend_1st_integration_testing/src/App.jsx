import React, { useState } from 'react';
import Input from './Input';
import DataLabeling from './DataLabeling';

export default function App() {
  const [currentPage, setCurrentPage] = useState('input');

  // Unified state for folders - persists across page navigation
  const [folders, setFolders] = useState([
    { id: 1, name: 'Consumption Data', tag: '#Consumption', files: [] },
    { id: 2, name: 'Reference Data', tag: '#Reference', files: [] },
  ]);

  // State for file labeling status and column selections - persists across navigation
  const [fileStatuses, setFileStatuses] = useState({});
  const [fileColumnLabels, setFileColumnLabels] = useState({});

  const handleUploadComplete = () => {
    setCurrentPage('labeling');
  };

  const handleBackToInput = () => {
    setCurrentPage('input');
  };

  const handleLabelComplete = () => {
    // For now, just log - can be extended for next page
    console.log('Labeling completed!');
    // Could navigate to next step here
  };

  if (currentPage === 'input') {
    return (
      <Input
        folders={folders}
        setFolders={setFolders}
        onComplete={handleUploadComplete}
      />
    );
  }

  if (currentPage === 'labeling') {
    return (
      <DataLabeling
        folders={folders}
        setFolders={setFolders}
        fileStatuses={fileStatuses}
        setFileStatuses={setFileStatuses}
        fileColumnLabels={fileColumnLabels}
        setFileColumnLabels={setFileColumnLabels}
        onBack={handleBackToInput}
        onComplete={handleLabelComplete}
      />
    );
  }

  return null;
}
