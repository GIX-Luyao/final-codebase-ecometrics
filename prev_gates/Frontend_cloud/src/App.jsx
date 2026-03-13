import React, { useState, useCallback, useRef } from 'react';
import Input from './Input';
import DataCleaning from './DataCleaning';
import DataMapping from './DataMapping';
import Dashboard from './Dashboard';
import { createSession } from './services/api';

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

  // State for pending corrections - persists across navigation
  const [filePendingCorrections, setFilePendingCorrections] = useState({});

  // Backend session state
  const [backendSessionId, setBackendSessionId] = useState(null);
  const sessionPromiseRef = useRef(null);

  // Lazily create a backend session (only created once)
  const ensureSession = useCallback(async () => {
    if (backendSessionId) return backendSessionId;

    // Prevent duplicate session creation if called concurrently
    if (sessionPromiseRef.current) return sessionPromiseRef.current;

    sessionPromiseRef.current = (async () => {
      const result = await createSession('EcoMetrics Analysis');
      const id = result.sessionId;
      setBackendSessionId(id);
      return id;
    })();

    try {
      return await sessionPromiseRef.current;
    } finally {
      sessionPromiseRef.current = null;
    }
  }, [backendSessionId]);

  // Calculation result passed from DataMapping to Dashboard
  const [calculationResult, setCalculationResult] = useState(null);

  // Step number to page mapping for status bar navigation (4 steps)
  const stepToPage = {
    1: 'input',
    2: 'cleaning',
    3: 'dataMapping',
    4: 'dashboard',
  };

  const handleGoToStep = (step) => {
    if (stepToPage[step]) {
      setCurrentPage(stepToPage[step]);
    }
  };

  // Navigation handlers
  const handleUploadComplete = () => setCurrentPage('cleaning');
  const handleBackToInput = () => setCurrentPage('input');

  const handleCleaningComplete = () => setCurrentPage('dataMapping');
  const handleBackToCleaning = () => setCurrentPage('cleaning');

  const handleMappingComplete = (result, sessionId) => {
    setCalculationResult(result);
    if (sessionId) setBackendSessionId(sessionId);
    setCurrentPage('dashboard');
  };
  const handleBackToMapping = () => setCurrentPage('dataMapping');

  // Step 1: Upload Data
  if (currentPage === 'input') {
    return (
      <Input
        folders={folders}
        setFolders={setFolders}
        onComplete={handleUploadComplete}
        onGoToStep={handleGoToStep}
      />
    );
  }

  // Step 2: Data Cleaning
  if (currentPage === 'cleaning') {
    return (
      <DataCleaning
        folders={folders}
        setFolders={setFolders}
        fileStatuses={fileStatuses}
        setFileStatuses={setFileStatuses}
        fileColumnLabels={fileColumnLabels}
        setFileColumnLabels={setFileColumnLabels}
        filePendingCorrections={filePendingCorrections}
        setFilePendingCorrections={setFilePendingCorrections}
        onBack={handleBackToInput}
        onComplete={handleCleaningComplete}
        onGoToStep={handleGoToStep}
      />
    );
  }

  // Step 3: Data Mapping
  if (currentPage === 'dataMapping') {
    return (
      <DataMapping
        onBack={handleBackToCleaning}
        onComplete={handleMappingComplete}
        onGoToStep={handleGoToStep}
        ensureSession={ensureSession}
      />
    );
  }

  // Step 4: Visualization Dashboard
  if (currentPage === 'dashboard') {
    return (
      <Dashboard
        sessionId={backendSessionId}
        calculationResult={calculationResult}
        onBack={handleBackToMapping}
        onGoToStep={handleGoToStep}
      />
    );
  }

  return null;
}
