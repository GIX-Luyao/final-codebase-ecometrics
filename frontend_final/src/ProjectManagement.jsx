import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { createProject, deleteProject, listProjects, renameProject } from './services/api';

export default function ProjectManagement({ onOpenProject, currentProjectId }) {
  const [projects, setProjects] = useState([]);
  const [selectedProjectId, setSelectedProjectId] = useState('');
  const [newProjectName, setNewProjectName] = useState('');
  const [renameProjectValue, setRenameProjectValue] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState('');

  const selectedProject = useMemo(
    () => projects.find((project) => project.projectId === selectedProjectId) || null,
    [projects, selectedProjectId]
  );

  const formatCreatedDate = (value) => {
    if (!value) return '-';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '-';
    return date.toISOString().slice(0, 10);
  };

  const loadProjects = useCallback(async () => {
    setIsLoading(true);
    const result = await listProjects();
    const list = result.projects || [];
    setProjects(list);
    setSelectedProjectId((prev) => {
      if (currentProjectId && list.some((p) => p.projectId === currentProjectId)) {
        return currentProjectId;
      }
      if (prev && list.some((p) => p.projectId === prev)) return prev;
      return list[0]?.projectId || '';
    });
    setIsLoading(false);
  }, [currentProjectId]);

  useEffect(() => {
    (async () => {
      try {
        setError('');
        await loadProjects();
      } catch (e) {
        setError(e.message || 'Failed to load projects');
        setIsLoading(false);
      }
    })();
  }, [loadProjects]);

  useEffect(() => {
    setRenameProjectValue(selectedProject?.name || '');
  }, [selectedProject]);

  const handleCreateProject = async () => {
    const projectName = newProjectName.trim();
    if (!projectName) return;
    setIsBusy(true);
    setError('');
    try {
      const created = await createProject(projectName);
      setNewProjectName('');
      await loadProjects();
      setSelectedProjectId(created.projectId);
    } catch (e) {
      setError(e.message || 'Failed to create project');
    } finally {
      setIsBusy(false);
    }
  };

  const handleRenameProject = async () => {
    if (!selectedProjectId) return;
    const name = renameProjectValue.trim();
    if (!name) return;
    setIsBusy(true);
    setError('');
    try {
      await renameProject(selectedProjectId, name);
      await loadProjects();
    } catch (e) {
      setError(e.message || 'Failed to rename project');
    } finally {
      setIsBusy(false);
    }
  };

  const handleDeleteProject = async () => {
    if (!selectedProjectId) return;
    if (!window.confirm('Delete this project and all associated history/artifacts?')) return;
    setIsBusy(true);
    setError('');
    try {
      await deleteProject(selectedProjectId);
      await loadProjects();
    } catch (e) {
      setError(e.message || 'Failed to delete project');
    } finally {
      setIsBusy(false);
    }
  };

  const handleContinue = () => {
    if (!selectedProjectId || !onOpenProject) return;
    onOpenProject(selectedProjectId);
  };

  return (
    <div className="bg-white fixed inset-0 overflow-hidden">
      <div className="absolute left-0 top-0 w-full h-[62px] bg-[#365D60] flex items-center justify-between px-[64px]">
        <div className="flex items-center gap-[12px]">
          <img src="/logo.png" alt="Logo" className="h-[40px] w-auto" />
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[24px] text-white">EcoMetrics</p>
        </div>
        <div className="bg-white rounded-[10px] w-[331px] h-[40px] flex items-center px-[12px]">
          <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E]">Project Management</p>
        </div>
      </div>

      <div className="absolute left-[64px] right-[64px] top-[96px] bottom-[32px] flex gap-[16px]">
        <div className="w-[32%] bg-[#F5F5F5] rounded-[10px] p-[20px] flex flex-col">
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[20px] text-[#13383B] mb-[12px]">Projects</p>
          <div className="mb-[12px] flex gap-[8px]">
            <input
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              placeholder="New project name"
              className="flex-1 h-[40px] rounded-[8px] border border-[#D9D9D9] px-[10px] font-['Noto_Sans',sans-serif] text-[13px]"
            />
            <button
              onClick={handleCreateProject}
              disabled={isBusy || !newProjectName.trim()}
              className="h-[40px] px-[12px] rounded-[8px] bg-[#29ABB5] text-white font-['Noto_Sans',sans-serif] text-[13px] disabled:opacity-60"
            >
              Create
            </button>
          </div>
          <div className="flex-1 overflow-auto bg-white rounded-[8px] border border-[#E9E9E9]">
            {isLoading ? (
              <div className="p-[12px] font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E]">Loading projects...</div>
            ) : projects.length === 0 ? (
              <div className="p-[12px] font-['Noto_Sans',sans-serif] text-[13px] text-[#7E7E7E]">No projects yet. Create one first.</div>
            ) : (
              projects.map((project) => {
                const selected = project.projectId === selectedProjectId;
                const isCurrent = currentProjectId && project.projectId === currentProjectId;
                return (
                  <button
                    key={project.projectId}
                    onClick={() => setSelectedProjectId(project.projectId)}
                    className={`w-full text-left px-[12px] py-[10px] border-b border-[#EEEEEE] ${
                      selected ? 'bg-[#29ABB5]/10' : 'bg-white hover:bg-[#F9F9F9]'
                    }`}
                  >
                    <p className="font-['Noto_Sans',sans-serif] font-semibold text-[13px] text-[#13383B] truncate">
                      {project.name}
                      {isCurrent ? ' (current)' : ''}
                    </p>
                    <p className="font-['Noto_Sans',sans-serif] text-[11px] text-[#7E7E7E] truncate">
                      Created at {formatCreatedDate(project.createdAt)}
                    </p>
                  </button>
                );
              })
            )}
          </div>
        </div>

        <div className="flex-1 bg-white rounded-[10px] border border-[#EEEEEE] p-[24px] flex flex-col">
          <p className="font-['Noto_Sans',sans-serif] font-bold text-[26px] text-[#13383B] mb-[8px]">Project Operations</p>
          <p className="font-['Noto_Sans',sans-serif] text-[14px] text-[#7E7E7E] mb-[20px]">
            Choose a project on the left, then continue to Step 1 (Upload Data).
          </p>

          {error && (
            <div className="mb-[16px] rounded-[8px] bg-[#FDECEC] px-[12px] py-[10px] font-['Noto_Sans',sans-serif] text-[13px] text-[#A33A3A]">
              {error}
            </div>
          )}

          <div className="bg-[#F5F5F5] rounded-[10px] p-[16px] mb-[16px]">
            <p className="font-['Noto_Sans',sans-serif] text-[12px] text-[#7E7E7E] mb-[6px]">Selected Project</p>
            <p className="font-['Noto_Sans',sans-serif] font-semibold text-[18px] text-[#13383B]">
              {selectedProject?.name || 'No project selected'}
            </p>
          </div>

          <div className="bg-[#F5F5F5] rounded-[10px] p-[16px] mb-[16px]">
            <p className="font-['Noto_Sans',sans-serif] font-semibold text-[14px] text-[#13383B] mb-[8px]">Rename Project</p>
            <div className="flex gap-[10px]">
              <input
                value={renameProjectValue}
                onChange={(e) => setRenameProjectValue(e.target.value)}
                placeholder="New project name"
                className="flex-1 h-[40px] rounded-[8px] border border-[#D9D9D9] px-[10px] font-['Noto_Sans',sans-serif] text-[13px]"
              />
              <button
                onClick={handleRenameProject}
                disabled={isBusy || !selectedProjectId || !renameProjectValue.trim()}
                className="h-[40px] px-[14px] rounded-[8px] bg-[#29ABB5] text-white font-['Noto_Sans',sans-serif] text-[13px] disabled:opacity-60"
              >
                Rename
              </button>
            </div>
          </div>

          <div className="mt-auto flex items-center justify-between">
            <button
              onClick={handleDeleteProject}
              disabled={isBusy || !selectedProjectId}
              className="h-[40px] px-[14px] rounded-[8px] bg-[#FDECEC] text-[#A33A3A] font-['Noto_Sans',sans-serif] text-[13px] disabled:opacity-60"
            >
              Delete Project
            </button>
            <button
              onClick={handleContinue}
              disabled={!selectedProjectId}
              className="h-[42px] px-[20px] rounded-[24px] bg-[#29ABB5] text-white font-['Noto_Sans',sans-serif] font-semibold text-[14px] disabled:opacity-60"
            >
              Continue To Upload Data
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
