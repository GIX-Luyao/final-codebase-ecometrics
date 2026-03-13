import { useState, useCallback, useRef, useEffect } from 'react';
import Input from './Input';
import DataLabeling from './DataLabeling';
import DataCleaning from './DataCleaning';
import DataMapping from './DataMapping';
import PreviewFiles from './PreviewFiles';
import ProcessedResults from './ProcessedResults';
import Dashboard from './Dashboard';
import { createSession } from './services/api';

const buildDefaultFolders = () => ([
  { id: 1, name: 'Consumption Data', tag: '#Consumption', files: [] },
  { id: 2, name: 'Reference Data', tag: '#Reference', files: [] },
]);

export default function App() {
  const [currentPage, setCurrentPage] = useState('input');

  // Unified state for folders - persists across page navigation
  const [folders, setFolders] = useState(buildDefaultFolders());

  // State for file labeling status and column selections - persists across navigation
  const [fileStatuses, setFileStatuses] = useState({});
  const [fileColumnLabels, setFileColumnLabels] = useState({});

  // State for pending corrections - persists across navigation
  const [filePendingCorrections, setFilePendingCorrections] = useState({});

  // Backend session state — created once on mount
  const [backendSessionId, setBackendSessionId] = useState(null);
  const sessionPromiseRef = useRef(null);
  const [calculationResult, setCalculationResult] = useState(null);

  // Auto-create a session immediately on app mount
  useEffect(() => {
    if (sessionPromiseRef.current) return;
    const timestamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
    sessionPromiseRef.current = createSession(`EcoMetrics Analysis ${timestamp}`)
      .then((result) => {
        setBackendSessionId(result.sessionId);
      })
      .catch((err) => {
        console.error('[App] Failed to create session on mount:', err);
      });
  }, []);

  // ensureSession — returns existing session or waits for the in-flight creation
  const ensureSession = useCallback(async () => {
    if (backendSessionId) return backendSessionId;
    if (sessionPromiseRef.current) {
      // Wait for the in-flight mount creation, then return whatever was set
      await sessionPromiseRef.current;
    }
    // Fallback: create a new session if mount attempt failed
    const timestamp = new Date().toISOString().slice(0, 16).replace('T', ' ');
    const result = await createSession(`EcoMetrics Analysis ${timestamp}`);
    setBackendSessionId(result.sessionId);
    return result.sessionId;
  }, [backendSessionId]);

  // Step number to page mapping for status bar navigation
  const stepToPage = {
    1: 'input',
    2: 'labeling',
    3: 'cleaning',
    4: 'dataMapping',
    5: 'previewFiles',
    6: 'dashboard',
  };

  const handleGoToStep = (step) => {
    if (stepToPage[step]) {
      setCurrentPage(stepToPage[step]);
    }
  };

  // Logo click → go back to Step 1 (Upload)
  const handleLogoClick = () => {
    setCurrentPage('input');
  };

  // Navigation handlers
  const handleUploadComplete = () => setCurrentPage('labeling');
  const handleBackToInput = () => setCurrentPage('input');

  const handleLabelComplete = () => setCurrentPage('cleaning');
  const handleBackToLabeling = () => setCurrentPage('labeling');

  const handleCleaningComplete = () => setCurrentPage('dataMapping');
  const handleBackToCleaning = () => setCurrentPage('cleaning');

  const handleMappingComplete = (result, sessionId) => {
    setCalculationResult(result);
    if (sessionId) setBackendSessionId(sessionId);
    setCurrentPage('previewFiles');
  };
  const handleBackToMapping = () => setCurrentPage('dataMapping');

  const handlePreviewNext = () => setCurrentPage('processedResults');
  const handleBackToPreview = () => setCurrentPage('previewFiles');

  const handleResultsNext = () => setCurrentPage('dashboard');
  const handleBackToResults = () => setCurrentPage('processedResults');

  // Step 1: Upload Data
  if (currentPage === 'input') {
    return (
      <Input
        folders={folders}
        setFolders={setFolders}
        onComplete={handleUploadComplete}
        onGoToStep={handleGoToStep}
        onLogoClick={handleLogoClick}
      />
    );
  }

  // Step 2: Select Data
  if (currentPage === 'labeling') {
    return (
      <DataLabeling
        folders={folders}
        setFolders={setFolders}
        fileStatuses={fileStatuses}
        setFileStatuses={setFileStatuses}
        fileColumnLabels={fileColumnLabels}
        setFileColumnLabels={setFileColumnLabels}
        filePendingCorrections={filePendingCorrections}
        setFilePendingCorrections={setFilePendingCorrections}
        onBack={handleBackToInput}
        onComplete={handleLabelComplete}
        onGoToStep={handleGoToStep}
        onLogoClick={handleLogoClick}
      />
    );
  }

  // Step 3: Data Cleaning
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
        onBack={handleBackToLabeling}
        onComplete={handleCleaningComplete}
        onGoToStep={handleGoToStep}
        onLogoClick={handleLogoClick}
      />
    );
  }

  // Step 4: Data Mapping
  if (currentPage === 'dataMapping') {
    return (
      <DataMapping
        onBack={handleBackToCleaning}
        onComplete={handleMappingComplete}
        onGoToStep={handleGoToStep}
        ensureSession={ensureSession}
        onLogoClick={handleLogoClick}
        folders={folders}
        fileColumnLabels={fileColumnLabels}
      />
    );
  }

  // Step 5a: Data Processing - Preview Files
  if (currentPage === 'previewFiles') {
    return (
      <PreviewFiles
        sessionId={backendSessionId}
        onNext={handlePreviewNext}
        onBack={handleBackToMapping}
        onCalculationComplete={setCalculationResult}
        onGoToStep={handleGoToStep}
        folders={folders}
        fileColumnLabels={fileColumnLabels}
        filePendingCorrections={filePendingCorrections}
        onLogoClick={handleLogoClick}
      />
    );
  }

  // Step 5b: Data Processing - Processed Results
  if (currentPage === 'processedResults') {
    return (
      <ProcessedResults
        sessionId={backendSessionId}
        calculationResult={calculationResult}
        onNext={handleResultsNext}
        onBack={handleBackToPreview}
        onGoToStep={handleGoToStep}
        onLogoClick={handleLogoClick}
      />
    );
  }

  // Step 6: Visualization Dashboard
  if (currentPage === 'dashboard') {
    return (
      <Dashboard
        sessionId={backendSessionId}
        calculationResult={calculationResult}
        onBack={handleBackToResults}
        onGoToStep={handleGoToStep}
        onLogoClick={handleLogoClick}
      />
    );
  }

  return null;
}
